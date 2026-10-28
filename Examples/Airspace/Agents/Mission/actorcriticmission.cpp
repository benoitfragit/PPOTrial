/**
* @file Agents/mission/src/actorcriticmission.cpp
* @brief Implementation for actorcriticmission.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "actorcriticmission.hpp"

#include <torch/torch.h>
#include <tuple>
#include <string>
#include <stdexcept>
#include <cmath>

/**
* @brief Computes categorical log-probabilities for discrete actions.
* @param actions Chosen categorical actions.
* @param logits Raw categorical logits.
* @return Per-sample log-probability tensor.
*/
static torch::Tensor categorical_log_prob(
    const torch::Tensor& actions,
    const torch::Tensor& logits)
{
    auto logp_all = torch::log_softmax(logits, -1);
    auto a = actions.to(torch::kLong);
    if (a.dim() == 1) a = a.unsqueeze(1);
    return logp_all.gather(1, a).squeeze(1);
}

/**
* @brief Computes the entropy of a categorical distribution.
* @param logits Raw categorical logits.
* @return Per-sample entropy tensor.
*/
static torch::Tensor categorical_entropy(const torch::Tensor& logits)
{
    auto p    = torch::softmax(logits, -1);
    auto logp = torch::log_softmax(logits, -1);
    return -(p * logp).sum(-1);
}

/**
* @brief Pools valid track embeddings with masked mean and max operators.
* @param x Encoded track tensor.
* @param mask Track availability mask.
* @return Concatenated mean/max pooled tensor.
*/
static torch::Tensor masked_mean_max_pool(const torch::Tensor& x,
                                          const torch::Tensor& mask)
{
    auto m = mask.unsqueeze(-1);

    auto sum   = (x * m).sum(1);
    auto denom = m.sum(1).clamp_min(1e-6f);
    auto mean  = sum / denom;

    auto neg_inf = torch::full_like(x, -1e9f);
    auto x_masked = torch::where(m > 0.5f, x, neg_inf);
    auto maxv = std::get<0>(x_masked.max(1));

    return torch::cat({ mean, maxv }, -1);
}

constexpr double PI = 3.14159265358979323846;


/**
* @brief Computes Gaussian log-probabilities for continuous actions.
* @param x Sampled actions.
* @param mu Gaussian means.
* @param log_std Gaussian log-standard deviations.
* @return Per-sample log-probability tensor.
*/
static torch::Tensor gaussian_log_prob(
    const torch::Tensor& x,
    const torch::Tensor& mu,
    const torch::Tensor& log_std)
{
    auto var = torch::exp(2.0 * log_std);
    auto logp =
        -0.5 * (((x - mu).pow(2) / var)
            + 2.0 * log_std
            + std::log(2.0 * PI));
    return logp.sum(-1);
}

/**
* @brief Computes Gaussian entropy for continuous actions.
* @param log_std Gaussian log-standard deviations.
* @return Per-sample entropy tensor.
*/
static torch::Tensor gaussian_entropy(
    const torch::Tensor& log_std)
{
    auto c = 0.5 * (1.0 + std::log(2.0 * PI));
    return (log_std + c).sum(-1);
}

namespace rl {

namespace {

/**
* @brief Number of per-track mission features consumed by the network.
*/
constexpr int64_t kTrackFeatureCount =
    static_cast<int64_t>(MissionObservationLayout::kTrackFeatureCount);
/**
* @brief Maximum number of ranked mission tracks consumed by the network.
*/
constexpr int64_t kMaxTracks =
    static_cast<int64_t>(MissionObservationLayout::kMaxTracks);
/**
* @brief Number of global mission features appended after the track block.
*/
constexpr int64_t kGlobalFeatureCount =
    static_cast<int64_t>(MissionObservationLayout::kGlobalFeatureCount);
/**
* @brief Number of ammunition-related mission features.
*/
constexpr int64_t kAmmoFeatureCount =
    static_cast<int64_t>(MissionObservationLayout::kAmmoFeatureCount);
/**
* @brief Flattened mission observation size expected by the network.
*/
constexpr int64_t kStateDimension =
    static_cast<int64_t>(MissionObservationLayout::kStateDimension);
/**
* @brief Number of action values emitted by the mission policy.
*/
constexpr int64_t kActionCount =
    static_cast<int64_t>(MissionObservationLayout::kActionCount);
/**
* @brief Flattened size of the ranked-track block inside the mission observation.
*/
constexpr int64_t kTrackSectionSize = kMaxTracks * kTrackFeatureCount;

} // namespace

/**
* @brief Builds the mission actor-critic architecture and initializes weights.
*/
ActorCriticTrackAttention::ActorCriticTrackAttention(int64_t state_dim, int64_t action_dim)
{
    TORCH_CHECK(state_dim == kStateDimension && action_dim == kActionCount);

    // Encoder partagé par track
    enc1 = register_module("enc1", torch::nn::Linear(kTrackFeatureCount, 128));
    enc2 = register_module("enc2", torch::nn::Linear(128, d_model));

    // Attention
    attn_q = register_module("attn_q", torch::nn::Linear(d_model * 2, d_model));
    attn_k = register_module("attn_k", torch::nn::Linear(d_model, d_model));

    // Actor heads
    // Entrée = pooled(2*d) + ctx(d) + ammo(1) + impacts(P-1)
    const int64_t actor_in = d_model * 3 + kGlobalFeatureCount;

    mu_head     = register_module("mu_head",     torch::nn::Linear(actor_in, 2));
    logstd_head = register_module("logstd_head", torch::nn::Linear(actor_in, 2));
    fire_head   = register_module("fire_head",   torch::nn::Linear(actor_in, 1));

    // Critic
    critic1 = register_module("critic1", torch::nn::Linear(actor_in, d_model));
    critic2 = register_module("critic2", torch::nn::Linear(d_model, 1));

    // Orthogonal init (PPO-safe)
    for (auto& m : modules(false))
        if (auto* l = dynamic_cast<torch::nn::LinearImpl*>(m.get()))
        {
            torch::nn::init::orthogonal_(l->weight, std::sqrt(2.0));
            torch::nn::init::constant_(l->bias, 0.0);
        }

    torch::nn::init::orthogonal_(mu_head->weight, 0.01);
    torch::nn::init::orthogonal_(logstd_head->weight, 0.01);
    torch::nn::init::orthogonal_(fire_head->weight, 0.01);
    torch::nn::init::orthogonal_(critic2->weight, 1.0);
}

/**
* @brief Splits the flattened mission state into semantic tensor groups.
*/
ParsedInput
ActorCriticTrackAttention::parse_input(
    const torch::Tensor& input) const
{
    TORCH_CHECK(input.dim() == 2);
    TORCH_CHECK(input.size(1) == kStateDimension);

    ParsedInput out;
    const auto B = input.size(0);

    // ammo = dernière colonne
    out.ammo = input.slice(1, kTrackSectionSize, kTrackSectionSize + kAmmoFeatureCount).contiguous(); // [B,1]

    // impacts = dernières colonnes
    out.impacts = input.slice(
        1,
        kTrackSectionSize + kAmmoFeatureCount,
        kTrackSectionSize + kGlobalFeatureCount).contiguous(); // [B,P-1]

    // tracks flatten -> reshape
    auto flat = input.slice(1, 0, kTrackSectionSize).contiguous();
    out.tracks = flat.view({ B, kMaxTracks, kTrackFeatureCount }); // [B,N,M]

    return out;
}

/**
* @brief Applies masked dot-product attention over valid tracks only.
*/
torch::Tensor
ActorCriticTrackAttention::masked_attention_context(
    const torch::Tensor& feat,
    const torch::Tensor& query,
    const torch::Tensor& mask)
{
    auto K = attn_k->forward(feat);      // [B,N,d]
    auto Q = query.unsqueeze(1);         // [B,1,d]

    auto scores = (K * Q).sum(-1) / std::sqrt((double)d_model); // [B,N]
    scores = scores + (1.0f - mask) * (-1e9f);

    auto w = torch::softmax(scores, -1); // [B,N]
    return (w.unsqueeze(-1) * feat).sum(1); // [B,d]
}

/**
* @brief Runs the full mission actor-critic forward pass.
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticTrackAttention::forward(torch::Tensor flat_input)
{
    auto parsed = parse_input(flat_input);

    const auto B = parsed.tracks.size(0);

    // mask = dernière feature de chaque track
    auto mask = parsed.tracks.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        kTrackFeatureCount - 1
    }).clamp(0.0f, 1.0f);

    // Encodeur par track
    auto flat = parsed.tracks.contiguous().view({ B * kMaxTracks, kTrackFeatureCount });
    auto feat = torch::relu(enc2->forward(
                torch::relu(enc1->forward(flat))))
                .view({ B, kMaxTracks, d_model });

    // Pooling mean + max
    auto m = mask.unsqueeze(-1);
    auto mean = (feat * m).sum(1) / m.sum(1).clamp_min(1e-6f);
    auto maxv = std::get<0>(
        torch::where(m > 0.5f, feat, torch::full_like(feat, -1e9f)).max(1)
    );
    auto pooled = torch::cat({mean, maxv}, -1); // [B,2d]

    // Attention
    auto query = attn_q->forward(pooled); // [B,d]
    auto ctx = masked_attention_context(feat, query, mask); // [B,d]

    // Concat global
    auto global = torch::cat({pooled, ctx, parsed.ammo, parsed.impacts }, -1);

    // Actor
    auto mu = mu_head->forward(global);
    auto log_std = torch::clamp(logstd_head->forward(global), -20.0, 2.0);
    auto fire_logits = fire_head->forward(global);

    // Critic
    auto value = critic2->forward(
        torch::relu(critic1->forward(global))
    );

    auto action_logits = torch::cat({ mu, fire_logits }, -1);

    return {action_logits, log_std, value};
}

/**
* @brief Samples one mission action from the current policy.
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticTrackAttention::selectaction(
    const torch::Tensor& flat_input)
{
    torch::NoGradGuard no_grad;

    auto [action_out, log_std, value] = forward(flat_input);

    // ---- mu : [B,2]
    auto mu = action_out.slice(1, 0, 2);

    // ---- fire_logit : [B,1]
    auto fire_logits = action_out.slice(1, 2, 2 + 1);

    auto std = torch::exp(log_std);
    auto action_cont = mu + std * torch::randn_like(mu);

    auto logp_cont = gaussian_log_prob(action_cont, mu, log_std);

    auto fire_prob = torch::sigmoid(fire_logits);     // [B,1]
    auto fire = torch::bernoulli(fire_prob).to(torch::kFloat);
    auto logp_fire =
        -torch::nn::functional::binary_cross_entropy_with_logits(
            fire_logits,
            fire,
            torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
            .reduction(torch::kNone)
        ).squeeze(1);   // [B]

    return {
        torch::cat({action_cont, fire}, -1),
        logp_cont + logp_fire,
        value.view({-1})
    };
}

/**
* @brief Evaluates buffered actions for PPO loss computation.
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticTrackAttention::evaluate(
    const torch::Tensor& flat_input,
    const torch::Tensor& actions)   // [B,3]
{
    constexpr int64_t A = 2; // nb actions continues

    // ----- forward
    auto [action_out, log_std, value] = forward(flat_input);

    // ----- dissocier mu et fire_logit
    auto mu = action_out.slice(1, 0, A);           // [B,2]
    auto fire_logit = action_out.slice(1, A, A + 1); // [B,1]

    // ----- dissocier actions buffer
    auto action_cont = actions.slice(1, 0, A);     // [B,2]
    auto action_fire = actions.slice(1, A, A + 1);   // [B,1]

    // =====================================================
    // 1) Log-prob Gaussienne
    // =====================================================
    auto logp_cont =
        gaussian_log_prob(action_cont, mu, log_std); // [B]

    // =====================================================
    // 2) Log-prob Bernoulli (stable)
    // =====================================================
    auto logp_fire =
        -torch::nn::functional::binary_cross_entropy_with_logits(
            fire_logit,
            action_fire,
            torch::nn::functional::
            BinaryCrossEntropyWithLogitsFuncOptions()
            .reduction(torch::kNone)
        ).squeeze(1); // [B]

    // =====================================================
    // 3) Entropie
    // =====================================================

    // Gaussienne
    auto entropy_cont = gaussian_entropy(log_std); // [B]

    // Bernoulli
    auto p = torch::sigmoid(fire_logit);           // [B,1]

    auto entropy_fire =
        -(p * torch::log(p + 1e-8) +
            (1 - p) * torch::log(1 - p + 1e-8))
        .squeeze(1);                               // [B]

    auto entropy = entropy_cont + entropy_fire;    // [B]

    return {
        logp_cont + logp_fire,   // [B]
        value.view({-1}),        // [B]
        entropy                  // [B]
    };
}

/**
* @brief Returns only the actor output for production inference.
*/
torch::Tensor
ActorCriticTrackAttention::forward_mu(torch::Tensor flat_input)
{
    // Actor
    auto [actions, _, __] = forward(flat_input);
    
    return actions;
}

/**
* @brief Builds the production wrapper around the mission network.
*/
ActorCriticTrackAttentionProd::ActorCriticTrackAttentionProd(int64_t state_dim, int64_t action_dim)
{
    p_policy = std::make_shared<ActorCriticTrackAttention>(state_dim, action_dim);
}

/**
* @brief Loads the serialized production network from disk.
*/
void ActorCriticTrackAttentionProd::load(
    std::filesystem::path const& modelpath)
{
    torch::serialize::InputArchive archive;
    archive.load_from(modelpath.string());
    p_policy->load(archive);
    p_policy->eval();
}

/**
* @brief Runs production inference without gradient tracking.
*/
torch::Tensor
ActorCriticTrackAttentionProd::forward(torch::Tensor state)
{
    if (p_policy == nullptr)
        return {};

    torch::NoGradGuard no_grad;

    auto actions = p_policy->forward_mu(state);
    return actions;
}

} // namespace ppo::mission
