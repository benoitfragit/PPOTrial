#include "actorcriticcategorical.hpp"

#include <torch/torch.h>
#include <tuple>
#include <string>
#include <stdexcept>
#include <cmath>

// =========================================================
// Utils Categorical
// =========================================================
static torch::Tensor 
categorical_log_prob(
    const torch::Tensor& actions,   // [B] int64 or [B,1]
    const torch::Tensor& logits)    // [B,N]
{
    auto logp_all = torch::log_softmax(logits, -1); // [B,N]

    auto a = actions.to(torch::kLong);
    if (a.dim() == 1) a = a.unsqueeze(1);           // [B,1]

    return logp_all.gather(1, a).squeeze(1);        // [B]
}

static torch::Tensor 
categorical_entropy(const torch::Tensor& logits) // [B,N]
{
    auto p    = torch::softmax(logits, -1);
    auto logp = torch::log_softmax(logits, -1);
    return -(p * logp).sum(-1); // [B]
}

// Pooling masqué (mean + max) sur N
// x   : [B,N,32]
// mask: [B,N] float32 0/1
// -> [B,64]
static torch::Tensor 
masked_mean_max_pool(const torch::Tensor& x, const torch::Tensor& mask)
{
	auto m = mask.unsqueeze(-1); // [B,N,1]

	// mean masqué
	auto sum = (x * m).sum(1);                // [B,32]
	auto denom = m.sum(1).clamp_min(1e-6f);   // [B,1]
	auto mean = sum / denom;                  // [B,32]

	// max masqué : invalid -> très négatif
	auto neg_inf = torch::full_like(x, -1e9f); // [B,N,32]
	auto x_masked = torch::where(m > 0.5f, x, neg_inf);
	auto maxv = std::get<0>(x_masked.max(1)); // [B,32]

	return torch::cat({mean, maxv}, -1);      // [B,64]
}

namespace ppo::mission {

ActorCriticCategoricalNet::ActorCriticCategoricalNet(int64_t M_,
                                                  int64_t actor_hidden_,
                                                  int64_t critic_hidden_)
        : M(M_), actor_hidden(actor_hidden_), critic_hidden(critic_hidden_)
{
	if (M <= 0) throw std::invalid_argument("M must be > 0");
	if (actor_hidden_ <= 0 || critic_hidden_ <= 0) throw std::invalid_argument("hidden dims must be > 0");

	enc1 = register_module("enc1", torch::nn::Linear(M, 64));
	enc2 = register_module("enc2", torch::nn::Linear(64, d_model));

	attn_k = register_module("attn_k", torch::nn::Linear(d_model, d_model));
	attn_q = register_module("attn_q", torch::nn::Linear(d_model, d_model));
	attn_v = register_module("attn_v", torch::nn::Linear(d_model, d_model));
	attn_s = register_module("attn_s", torch::nn::Linear(d_model, 1));

	actor1 = register_module("actor1", torch::nn::Linear(d_model * 4, actor_hidden_)); // 128
	actor2 = register_module("actor2", torch::nn::Linear(actor_hidden_, 1));

	critic1 = register_module("critic1", torch::nn::Linear(d_model * 3, critic_hidden_)); // 96
	critic2 = register_module("critic2", torch::nn::Linear(critic_hidden_, 1));

	// Init orthogonal
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

	// Bias à zéro
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

// Attention masquée : ctx = sum softmax(score)*V
// kv   : [B,N,32]
// query: [B,32]
// mask : [B,N] float32 0/1
// -> [B,32]
torch::Tensor
ActorCriticCategoricalNet::masked_attention_context(const torch::Tensor& kv,
									   const torch::Tensor& query,
									   const torch::Tensor& mask)
{
	const auto B = kv.size(0);
	const auto N = kv.size(1);

	auto K = attn_k->forward(kv); // [B,N,32]
	auto V = attn_v->forward(kv); // [B,N,32]

	// expand OK: on veut broadcast query -> tous les tracks
	auto Q = attn_q->forward(query).unsqueeze(1).expand({B, N, d_model}); // [B,N,32]

	auto H = torch::tanh(K + Q); // [B,N,32]
	auto scores = attn_s->forward(H).squeeze(-1); // [B,N]

	// invalid -> très négatif
	scores = scores + (1.0f - mask) * (-1e9f);

	auto w = torch::softmax(scores, -1); // [B,N]
	auto ctx = (V * w.unsqueeze(-1)).sum(1); // [B,32]
	return ctx;
}

// ---------------------------------------------------------
// forward(tracks):
// tracks : [B,N,M] (dernière feature = present 0/1)
// -> logits [B,N], dummy [B,1], value [B,1]
// ---------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticCategoricalNet::forward(torch::Tensor tracks)
{
	TORCH_CHECK(tracks.dim() == 3, "tracks must be [B,N,M]");
	TORCH_CHECK(tracks.size(-1) == M, "tracks last dim must match M");

	const auto B = tracks.size(0);
	const auto N = tracks.size(1);

	// ---- mask depuis feature "present" (dernière) ----
	auto present = tracks.index({torch::indexing::Slice(),
								 torch::indexing::Slice(),
								 M - 1}); // [B,N]

	// mask float32 0/1
	auto mask = present.to(torch::kFloat32).clamp(0.0f, 1.0f); // [B,N]

	// IMPORTANT: éviter softmax sur "tout masqué"
	TORCH_CHECK(mask.sum(1).min().item<float>() > 0.5f,
				"At least one track must be present per batch sample");

	// ---- encoder per-track ----
	// contiguous().view => explicite + pas de copie implicite surprise
	auto flat = tracks.contiguous().view({B * N, M}); // [B*N,M]

	auto h1   = torch::relu(enc1->forward(flat));     // [B*N,64]
	auto feat = torch::relu(enc2->forward(h1));       // [B*N,32]
	feat      = feat.contiguous().view({B, N, d_model}); // [B,N,32]

	TORCH_CHECK(feat.size(0) == B && feat.size(1) == N && feat.size(2) == d_model,
				"feat shape mismatch");

	// ---- pooling masqué -> [B,64] ----
	auto pooled = masked_mean_max_pool(feat, mask);

	// ---- query initiale : premières 32 dims du pooled (mean part) ----
	auto query = pooled.index({torch::indexing::Slice(),
							   torch::indexing::Slice(0, d_model)}); // [B,32]

	// ---- attention ctx -> [B,32] ----
	auto ctx = masked_attention_context(feat, query, mask);

	// ---- actor logits par track ----
	auto pooled_expand = pooled.unsqueeze(1).expand({B, N, pooled.size(1)}); // [B,N,64]
	auto ctx_expand    = ctx.unsqueeze(1).expand({B, N, ctx.size(1)});       // [B,N,32]

	auto actor_in = torch::cat({feat, pooled_expand, ctx_expand}, -1); // [B,N,128]
	auto actor_in_flat = actor_in.contiguous().view({B * N, d_model * 4}); // [B*N,128]

	auto a1 = torch::relu(actor1->forward(actor_in_flat)); // [B*N,hidden]
	auto logit_1 = actor2->forward(a1);                     // [B*N,1]
	auto logits = logit_1.contiguous().view({B, N});         // [B,N]

	// interdit de choisir un track absent
	logits = logits + (1.0f - mask) * (-1e9f);

	TORCH_CHECK(logits.size(0) == B && logits.size(1) == N, "logits shape mismatch");

	// ---- critic ----
	auto critic_in = torch::cat({pooled, ctx}, -1); // [B,96]
	auto c1 = torch::relu(critic1->forward(critic_in));
	auto value = critic2->forward(c1); // [B,1]

	// Dummy safe
	auto dummy = torch::zeros({B, 1}, logits.options());

	return { logits, dummy, value };
}

// ---------------------------------------------------------
// selectaction: sample categorical sur N
// -> (action[B], logp[B], value[B])
// ---------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticCategoricalNet::selectaction(const torch::Tensor& state) 
{
	torch::NoGradGuard no_grad;

	auto [logits, dummy, value] = forward(state);

	auto probs = torch::softmax(logits, -1); // [B,N]
	auto action = torch::multinomial(probs, 1).squeeze(1).to(torch::kLong); // [B]

	auto logp = categorical_log_prob(action, logits); // [B]
	auto v = value.view({-1});                        // [B]

	return { action, logp, v };
}

// ---------------------------------------------------------
// evaluate: PPO update
// -> (logp[B], values[B], entropy[B])
// ---------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticCategoricalNet::evaluate(const torch::Tensor& states, const torch::Tensor& actions)
{
	auto [logits, dummy, value] = forward(states);

	auto logp = categorical_log_prob(actions, logits); // [B]
	auto ent  = categorical_entropy(logits);           // [B]
	auto v    = value.view({-1});                      // [B]

	return { logp, v, ent };
}

ActorCriticCategoricalNetProd::ActorCriticCategoricalNetProd(int64_t M_, int64_t actor_dim, int64_t critic_dim ) :
    p_policy{ std::make_shared<ActorCriticCategoricalNet>(M_, actor_dim, critic_dim)}
{

}

void 
ActorCriticCategoricalNetProd::load(std::filesystem::path const& modelpath)
{
    if (p_policy == nullptr)
        return;

    torch::serialize::InputArchive archive{};

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

} /* ppo::mission */