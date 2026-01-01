#include "ppo.hpp"

#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <filesystem>

#include "utils.hpp"


namespace ppo::model {

struct ActorCriticNet :
    torch::nn::Module
{
    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear mu_head{ nullptr };
    torch::nn::Linear logstd_head{ nullptr };
    torch::nn::Linear critic{ nullptr };

    ActorCriticNet(int64_t state_dim, int64_t action_dim, int64_t hidden_dim = 64);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        forward(torch::Tensor x);
};

ActorCriticNet::ActorCriticNet(int64_t state_dim, int64_t action_dim, int64_t hidden_dim)
{
    std::string fc1_name{ "fc1" };
    std::string fc2_name{ "fc2" };
    std::string  mu_name{ "mu_head" };
    std::string log_name{ "logstd_head" };
    std::string critic_name{ "critic" };

    fc1 = register_module(fc1_name, torch::nn::Linear(state_dim, hidden_dim));
    fc2 = register_module(fc2_name, torch::nn::Linear(hidden_dim, hidden_dim));

    mu_head     = register_module(mu_name,     torch::nn::Linear(hidden_dim, action_dim));
    logstd_head = register_module(log_name, torch::nn::Linear(hidden_dim, action_dim));

    critic = register_module(critic_name, torch::nn::Linear(hidden_dim, 1));

    // ===== ORTHOGONAL INIT =====
    torch::nn::init::orthogonal_(fc1->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(fc2->weight, std::sqrt(2.0));

    torch::nn::init::orthogonal_(mu_head->weight, 0.01);
    torch::nn::init::orthogonal_(critic->weight, 1.0);

    // Bias à zéro (important)
    torch::nn::init::constant_(fc1->bias, 0.0);
    torch::nn::init::constant_(fc2->bias, 0.0);
    torch::nn::init::constant_(mu_head->bias, 0.0);
    torch::nn::init::constant_(critic->bias, 0.0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticNet::forward(torch::Tensor state)
{
    if (!fc1
        || !fc2
        || !mu_head
        || !logstd_head
        || !critic)
        return { {}, {}, {} };


    torch::Tensor x = torch::relu(fc1->forward(state));
    x = torch::relu(fc2->forward(x));

    auto mu = mu_head->forward(x);
    auto log_std = torch::clamp(logstd_head->forward(x), -20.0, 2.0);
    auto value = critic->forward(x);

    return {mu, log_std, value};
}

static torch::Tensor 
gaussian_log_prob_pre_tanh(
    const torch::Tensor& x,
    const torch::Tensor& mu,
    const torch::Tensor& log_std) 
{
    auto var = torch::exp(2.0 * log_std);
    auto logp =
        -0.5 * (((x - mu).pow(2) / var) + 2.0 * log_std + std::log(2.0 * PI));
    return logp.sum(-1);
}

static torch::Tensor 
gaussian_entropy(const torch::Tensor& log_std) 
{
    auto c = 0.5 * (1.0 + std::log(2.0 * PI));
    return (log_std + c).sum(-1);
}

static torch::Tensor 
tanh_correction(const torch::Tensor& pre_tanh) {
    auto y = torch::tanh(pre_tanh);
    return torch::log(1.0 - y.pow(2) + EPS).sum(-1);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
EvaluateAction(
    std::shared_ptr<ActorCriticNet> net,
    const torch::Tensor& states,
    const torch::Tensor& actions)
{
    auto [mu, log_std, values] = net->forward(states);

    auto clipped = torch::clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6);
    auto pre_tanh = 0.5 * torch::log((1 + clipped) / (1 - clipped));

    auto logp = gaussian_log_prob_pre_tanh(pre_tanh, mu, log_std)
        - tanh_correction(pre_tanh);

    auto entropy = gaussian_entropy(log_std);
    return { logp, values.view({-1}), entropy };
}

struct PPO::Impl {
    Impl(int64_t state_dim, int64_t action_dim)
    {
        m_policy     = std::make_shared<ActorCriticNet>(state_dim, action_dim);

        m_optimizer  = std::make_shared<torch::optim::Adam>(
            m_policy->parameters(), torch::optim::AdamOptions(m_learningReate)
        );

        // load du checkpoint
        std::string ppopath{ m_ppodirectory };
        ppopath.append("\\ppoprogress.pt");
        if (std::filesystem::exists(ppopath))
        {
            torch::serialize::InputArchive archive{};
            archive.load_from(ppopath);
            m_policy->load(archive);
            m_optimizer->load(archive);
        }

        m_policy->train();
    };

    std::tuple<float, float, float, float>
    update(
            const torch::Tensor& states,
            const torch::Tensor& actions,
            const torch::Tensor& old_logp,
            const torch::Tensor& returns,
            const torch::Tensor& advantages) 
    {

        auto adv = (advantages - advantages.mean()) /
            (advantages.std(false) + 1e-8);

        auto [logp, values, entropy] = EvaluateAction(
            m_policy, states, actions);

        auto ratio = torch::exp(logp - old_logp.detach());

        auto surr1 = ratio * adv;
        auto surr2 = torch::clamp(
            ratio, 1.0 - m_clip_epsilon, 1.0 + m_clip_epsilon) * adv;

        auto actor_loss = -torch::min(surr1, surr2).mean();
        auto value_loss = torch::mse_loss(values, returns);
        auto entropy_loss = -entropy.mean();

        auto loss = actor_loss +
            m_value_coef * value_loss +
            m_entropy_coef * entropy_loss;

        m_optimizer->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(m_policy->parameters(), 0.5);
        m_optimizer->step();

        return {
            loss.item<float>(),
            actor_loss.item<float>(),
            value_loss.item<float>(),
            entropy_loss.item<float>()
        };
    };

    std::shared_ptr<ActorCriticNet>     m_policy;
    std::shared_ptr<torch::optim::Adam> m_optimizer;

    float m_gamma       {0.99F};
    float m_lambda_gae  {0.95F};
    float m_clip_epsilon{0.2F};
    float m_value_coef  {0.5F};
    float m_entropy_coef{ 0.02F };
    float m_learningReate{ 3e-4F };

    int m_K_epochs        { 4 };
    int m_batch_size      { 64 };
    unsigned int m_checkpoint      { 5 };
    unsigned int m_ppoversion      {0};
    std::string m_ppodirectory{ "C:\\Users\\33761\\Documents\\Projets\\C++\\PPOTrial\\progress" };

};

PPO::PPO(int64_t state_dim, int64_t action_dim) :
    p_impl{ std::make_unique<Impl>(state_dim, action_dim) }{}

PPO::~PPO() = default;

std::tuple<float, float, float, float>
PPO::UpdatePolicy(
    const torch::Tensor& states,
    const torch::Tensor& actions,
    const torch::Tensor& old_logp,
    const torch::Tensor& values,
    const torch::Tensor& rewards,
    const torch::Tensor& dones,
    const torch::Tensor& last_values) 
{
    if (p_impl == nullptr)
        return { 0.0F, 0.0F, 0.0F, 0.0F };

    int64_t N = states.size(0);
    float L = 0, A = 0, V = 0, E = 0;
    int n = 0;

    int64_t maxStep{ rewards.size(0) };
    torch::Tensor advantages = torch::zeros({ maxStep });
    torch::Tensor returns = torch::zeros({ maxStep });

    // use a context to be sure that GAE with no_grad is applied only in this context
    // otherwise we could use detach on values
    {
        torch::NoGradGuard no_grad;

        // --- Compute GAE and Returns ---
        float gae = 0.0f;
        for (int64_t t = maxStep - 1; t >= 0; --t)
        {
            float reward = rewards[t].item<float>();
            float value = values[t].item<float>();
            float done = dones[t].item<float>();
            float next_value{ last_values.item<float>()};
            if (t != maxStep - 1)
                next_value = values[t + 1].item<float>();

            float mask = 1.0F - done;
            float delta = reward + p_impl->m_gamma * next_value * mask - value;
            gae = delta + p_impl->m_gamma * p_impl->m_lambda_gae * mask * gae;
            advantages[t] = gae;
        }

        returns = advantages + values;
    }

    for (int e = 0; e < p_impl->m_K_epochs; ++e) 
    {
        auto idx = torch::randperm(N, states.options().dtype(torch::kLong));

        for (int64_t i = 0; i < N; i += p_impl->m_batch_size) 
        {
            auto j = std::min(i + p_impl->m_batch_size, N);
            auto b = idx.slice(0, i, j);

            auto r = p_impl->update(
                states.index_select(0, b),
                actions.index_select(0, b),
                old_logp.index_select(0, b),
                returns.index_select(0, b),
                advantages.index_select(0, b)
            );

            L += std::get<0>(r);
            A += std::get<1>(r);
            V += std::get<2>(r);
            E += std::get<3>(r);
            ++n;
        }
    }

    if (p_impl->m_ppoversion > 0 
     && p_impl->m_ppoversion % p_impl->m_checkpoint == 0)
    {
        if (!std::filesystem::exists(p_impl->m_ppodirectory))
        {
            std::filesystem::create_directories(p_impl->m_ppodirectory);
        }

        torch::serialize::OutputArchive archive{};
        p_impl->m_policy->save(archive);
        p_impl->m_optimizer->save(archive);

        std::string ppopath= p_impl->m_ppodirectory ;
        ppopath.append("\\ppoprogress.pt");
        archive.save_to(ppopath);
    }
    ++p_impl->m_ppoversion;

    return { L / n, A / n, V / n, E / n };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PPO::SelectAction(const torch::Tensor& state)
{
    torch::NoGradGuard no_grad;

    if (p_impl == nullptr)
        return { {}, {}, {} };

    auto [mu, log_std, value] = p_impl->m_policy->forward(state);
    auto std = torch::exp(log_std);

    auto eps = torch::randn_like(mu);
    auto pre = mu + std * eps;
    auto action = torch::tanh(pre);

    auto logp = gaussian_log_prob_pre_tanh(pre, mu, log_std)
        - tanh_correction(pre);

    return { action, logp, value.view({-1}) };
}


}   /* ppo::model */