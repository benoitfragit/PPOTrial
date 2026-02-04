#include "actorcriticcategorical.hpp"

#include <torch/torch.h>
#include <tuple>
#include <string>
#include <stdexcept>
#include <cmath>

// =========================================================
// Utils Categorical
// =========================================================
static torch::Tensor categorical_log_prob(
    const torch::Tensor& actions,
    const torch::Tensor& logits)
{
    auto logp_all = torch::log_softmax(logits, -1);
    auto a = actions.to(torch::kLong);
    if (a.dim() == 1) a = a.unsqueeze(1);
    return logp_all.gather(1, a).squeeze(1);
}

static torch::Tensor categorical_entropy(const torch::Tensor& logits)
{
    auto p    = torch::softmax(logits, -1);
    auto logp = torch::log_softmax(logits, -1);
    return -(p * logp).sum(-1);
}

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

namespace ppo::mission {

ActorCriticCategoricalNet::ActorCriticCategoricalNet(
    int64_t N_,
    int64_t M_,
    int64_t actor_hidden_,
    int64_t critic_hidden_,
    int64_t d_model_) :
    N{ N_ },
    M{ M_ },
    d_model{ d_model_ },
    pooled_dim{ 2 * d_model_ },
    actor_in_dim{4 * d_model_},
    critic_in_dim{3 * d_model_},
    actor_hidden{actor_hidden_},
    critic_hidden{critic_hidden_}
{
    if (N <= 0) throw std::invalid_argument("N must be > 0");
    if (M <= 0) throw std::invalid_argument("M must be > 0");

    enc1 = register_module("enc1", torch::nn::Linear(M, 64));
    enc2 = register_module("enc2", torch::nn::Linear(64, d_model));

    attn_k = register_module("attn_k", torch::nn::Linear(d_model, d_model));
    attn_q = register_module("attn_q", torch::nn::Linear(d_model, d_model));
    attn_v = register_module("attn_v", torch::nn::Linear(d_model, d_model));
    attn_s = register_module("attn_s", torch::nn::Linear(d_model, 1));

    actor1 = register_module("actor1",
        torch::nn::Linear(actor_in_dim, actor_hidden));
    actor2 = register_module("actor2",
        torch::nn::Linear(actor_hidden, 1));

    critic1 = register_module("critic1",
        torch::nn::Linear(critic_in_dim, critic_hidden));
    critic2 = register_module("critic2",
        torch::nn::Linear(critic_hidden, 1));

    torch::nn::init::orthogonal_(enc1->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(enc2->weight, std::sqrt(2.0));

    torch::nn::init::orthogonal_(attn_k->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(attn_q->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(attn_v->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(attn_s->weight, 0.01);

    torch::nn::init::orthogonal_(actor1->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(actor2->weight, 0.01);

    torch::nn::init::orthogonal_(critic1->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(critic2->weight, 1.0);

    torch::nn::init::constant_(enc1->bias, 0.0);
    torch::nn::init::constant_(enc2->bias, 0.0);
    torch::nn::init::constant_(attn_k->bias, 0.0);
    torch::nn::init::constant_(attn_q->bias, 0.0);
    torch::nn::init::constant_(attn_v->bias, 0.0);
    torch::nn::init::constant_(attn_s->bias, 0.0);
    torch::nn::init::constant_(actor1->bias, 0.0);
    torch::nn::init::constant_(actor2->bias, 0.0);
    torch::nn::init::constant_(critic1->bias, 0.0);
    torch::nn::init::constant_(critic2->bias, 0.0);
}

torch::Tensor
ActorCriticCategoricalNet::masked_attention_context(
    const torch::Tensor& kv,
    const torch::Tensor& query,
    const torch::Tensor& mask)
{
    const auto B = kv.size(0);
    const auto N = kv.size(1);

    auto K = attn_k->forward(kv);
    auto V = attn_v->forward(kv);

    auto Q = attn_q->forward(query)
                 .unsqueeze(1)
                 .expand({ B, N, d_model });

    auto H = torch::tanh(K + Q);
    auto scores = attn_s->forward(H).squeeze(-1);

    scores = scores + (1.0f - mask) * (-1e9f);

    auto w = torch::softmax(scores, -1);
    return (V * w.unsqueeze(-1)).sum(1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticCategoricalNet::forward(torch::Tensor tracks)
{
    // tracks : [B, N*M]
    TORCH_CHECK(tracks.dim() == 2, "tracks must be [B, N*M]");
    TORCH_CHECK(tracks.size(1) == N * M, "tracks second dim must be N*M");

    const auto B = tracks.size(0);

    // reshape sûr vers [B,N,M]
    auto tracks3d = tracks.contiguous().view({ B, N, M });

    // ---- mask depuis la feature 'present' (dernière feature) ----
    auto present = tracks3d.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        M - 1
        }); // [B,N]

    auto mask = present.to(torch::kFloat32).clamp(0.0f, 1.0f);

    TORCH_CHECK(mask.sum(1).min().item<float>() > 0.5f,
        "At least one track must be present per batch sample");

    // ---- encoder per-track ----
    auto flat = tracks3d.contiguous().view({ B * N, M }); // [B*N,M]

    auto h1 = torch::relu(enc1->forward(flat)); // [B*N,64]
    auto feat = torch::relu(enc2->forward(h1))    // [B*N,32]
        .contiguous()
        .view({ B, N, d_model });     // [B,N,32]

    // ---- pooling masqué ----
    auto pooled = masked_mean_max_pool(feat, mask); // [B,64]

    // ---- query initiale ----
    auto query = pooled.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(0, d_model)
        }); // [B,32]

    // ---- attention ----
    auto ctx = masked_attention_context(feat, query, mask); // [B,32]

    // ---- actor ----
    auto pooled_expand = pooled.unsqueeze(1).expand({ B, N, pooled.size(1) }); // [B,N,64]
    auto ctx_expand = ctx.unsqueeze(1).expand({ B, N, ctx.size(1) });       // [B,N,32]

    auto actor_in = torch::cat(
        { feat, pooled_expand, ctx_expand }, -1); // [B,N,128]

    auto logits = actor2->forward(
        torch::relu(
            actor1->forward(
                actor_in.contiguous().view({ B * N, d_model * 4 })
            )
        )
    ).contiguous().view({ B, N }); // [B,N]

    // masque final
    logits = logits + (1.0f - mask) * (-1e9f);

    // ---- critic ----
    auto value = critic2->forward(
        torch::relu(
            critic1->forward(
                torch::cat({ pooled, ctx }, -1) // [B,96]
            )
        )
    ); // [B,1]

    auto dummy = torch::zeros({ B, 1 }, logits.options());

    return { logits, dummy, value };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticCategoricalNet::selectaction(const torch::Tensor& state)
{
    torch::NoGradGuard no_grad;
    auto [logits, dummy, value] = forward(state);
    auto probs = torch::softmax(logits, -1);
    auto action = torch::multinomial(probs, 1).squeeze(1);
    auto logp = categorical_log_prob(action, logits);
    return { action, logp, value.view({ -1 }) };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticCategoricalNet::evaluate(
    const torch::Tensor& states,
    const torch::Tensor& actions)
{
    auto [logits, dummy, value] = forward(states);
    return {
        categorical_log_prob(actions, logits),
        value.view({ -1 }),
        categorical_entropy(logits)
    };
}

ActorCriticCategoricalNetProd::ActorCriticCategoricalNetProd(
    int64_t N_,
    int64_t M_,
    int64_t actor_dim,
    int64_t critic_dim,
    int64_t d_model)
{
    p_policy = std::make_shared<ActorCriticCategoricalNet>(
        N_, M_, actor_dim, critic_dim, d_model);
}

void ActorCriticCategoricalNetProd::load(
    std::filesystem::path const& modelpath)
{
    torch::serialize::InputArchive archive;
    archive.load_from(modelpath.string());
    p_policy->load(archive);
    p_policy->eval();
}

torch::Tensor
ActorCriticCategoricalNetProd::forward(torch::Tensor state)
{
    if (p_policy == nullptr)
        return {};

    torch::NoGradGuard no_grad;

    auto [logits, dummy, value] = p_policy->forward(state);
    return logits;
}

} // namespace ppo::mission
