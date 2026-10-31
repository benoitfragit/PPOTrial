/**
* @file PPO/src/ppo.cpp
* @brief Implementation of the PPO optimizer wrapper.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "ppo.hpp"

#include <torch/torch.h>

#include <iostream>
#include <math.h>
#include <vector>

namespace ppo::model {

/**
* @brief Private PPO implementation state.
*/
struct PPO::Impl {
    /**
    * @brief Builds the PPO backend and optionally restores previous weights.
    * @param policy Trainable policy network.
    * @param modelpath Base directory containing model checkpoints.
    * @param state_dim Observation size.
    * @param action_dim Action size.
    * @param parameters PPO hyper-parameters.
    * @param loadExistingModel True to restore the previous prod checkpoint.
    */
    Impl(std::shared_ptr<ITrainingNet> policy,
         std::string const& modelpath,
         int64_t state_dim,
         int64_t action_dim,
         PPOParamters const& parameters,
         bool loadExistingModel)
    {
        m_gamma = parameters.m_gamma;
        m_lambda_gae = parameters.m_lambda_gae;
        m_clip_epsilon = parameters.m_clip_epsilon;
        m_value_coef = parameters.m_value_coef;
        m_entropy_coef = parameters.m_entropy_coef;
        m_learningReate = parameters.m_learningReate;
        m_K_epochs = parameters.m_K_epochs;
        m_batch_size = parameters.m_batch_size;
        m_target_kl = parameters.m_target_kl;

        m_policy = policy;
        m_optimizer = std::make_shared<torch::optim::Adam>(
            m_policy->parameters(),
            torch::optim::AdamOptions(m_learningReate));

        // Restore the production checkpoint when warm-starting training.
        std::filesystem::path modelfile{ modelpath };
        modelfile = modelfile / "prod" / "model.pt";
        if (loadExistingModel && std::filesystem::exists(modelfile))
        {
            torch::serialize::InputArchive archive{};
            archive.load_from(modelfile.string());
            m_policy->load(archive);
            m_optimizer->load(archive);
        }

        m_policy->train();
    }

    /**
    * @brief Applies one PPO update on one mini-batch.
    * @param states Batched observations.
    * @param actions Batched actions.
    * @param old_logp Log-probabilities recorded during rollout collection.
    * @param returns Discounted returns.
    * @param advantages Normalized generalized advantages.
    * @return Tuple containing total, actor, value, and entropy losses.
    */
    std::tuple<float, float, float, float>
    update(
        const torch::Tensor& states,
        const torch::Tensor& actions,
        const torch::Tensor& old_logp,
        const torch::Tensor& returns,
        const torch::Tensor& advantages)
    {
        if (m_policy == nullptr)
            return { 0.F, 0.F, 0.F, 0.F };

        auto adv = (advantages - advantages.mean()) /
            (advantages.std(false) + 1e-8);

        auto [logp, values, entropy] = m_policy->evaluate(states, actions);

        // Stop early when the batch deviates too much from the reference policy.
        float kl_value = (old_logp - logp).mean().item<float>();
        if (kl_value >= 1.5F * m_target_kl)
            return { 0.F, 0.F, 0.F, 0.F };

        auto ratio = torch::exp(logp - old_logp.detach());
        auto surr1 = ratio * adv;
        auto surr2 = torch::clamp(
            ratio,
            1.0 - m_clip_epsilon,
            1.0 + m_clip_epsilon) * adv;

        auto actor_loss = -torch::min(surr1, surr2).mean();
        auto value_loss = torch::mse_loss(values, returns);
        auto entropy_loss = -entropy.mean();

        auto loss =
            actor_loss +
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
    }

    /** @brief Trainable policy network. */
    std::shared_ptr<ITrainingNet> m_policy;
    /** @brief Adam optimizer used to update the policy. */
    std::shared_ptr<torch::optim::Adam> m_optimizer;

    /** @brief Discount factor. */
    float m_gamma{ 0.99F };
    /** @brief GAE lambda factor. */
    float m_lambda_gae{ 0.95F };
    /** @brief PPO clip epsilon. */
    float m_clip_epsilon{ 0.2F };
    /** @brief Value-loss coefficient. */
    float m_value_coef{ 0.5F };
    /** @brief Entropy regularization coefficient. */
    float m_entropy_coef{ 0.02F };
    /** @brief Adam learning rate. */
    float m_learningReate{ 3e-4F };
    /** @brief KL divergence target used for early stopping. */
    float m_target_kl{ 0.01F };

    /** @brief Number of PPO epochs per rollout. */
    int m_K_epochs{ 4 };
    /** @brief Mini-batch size used during PPO updates. */
    int m_batch_size{ 64 };

    /** @brief Last aggregated PPO training outputs. */
    PPOOutputs m_outputs{};

    /** @brief Empty fallback output returned when no implementation exists. */
    inline static PPOOutputs const EMPTY_PPO_OUTPUTS{};
};

/**
* @brief Builds the PPO wrapper around one trainable policy.
* @param policy Trainable policy network.
* @param modelpath Base directory containing model checkpoints.
* @param state_dim Observation size.
* @param action_dim Action size.
* @param parameters PPO hyper-parameters.
* @param loadExistingModel True to restore the previous prod checkpoint.
*/
PPO::PPO(std::shared_ptr<ITrainingNet> policy,
         std::string const& modelpath,
         int64_t state_dim,
         int64_t action_dim,
         PPOParamters const& parameters,
         bool loadExistingModel) :
    p_impl{ std::make_unique<Impl>(
        policy,
        modelpath,
        state_dim,
        action_dim,
        parameters,
        loadExistingModel) }
{
}

/**
* @brief Destroys the PPO wrapper.
*/
PPO::~PPO() = default;

/**
* @brief Updates the policy from one collected rollout buffer.
*
* The method first computes generalized advantage estimates and returns, then
* iterates over shuffled mini-batches for several PPO epochs and averages the
* reported training losses.
*
* @param states Batched observations.
* @param actions Batched actions.
* @param old_logp Log-probabilities recorded during rollout collection.
* @param values Value estimates recorded during rollout collection.
* @param rewards Per-step rewards.
* @param dones Per-step done flags.
* @param last_values Bootstrap value for truncated rollouts.
*/
void
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
        return;

    int64_t N = states.size(0);
    float L = 0.F;
    float A = 0.F;
    float V = 0.F;
    float E = 0.F;
    int n = 0;
    int64_t maxStep{ rewards.size(0) };
    torch::Tensor advantages = torch::zeros({ maxStep });
    torch::Tensor returns = torch::zeros({ maxStep });

    // Compute GAE and returns without tracking gradients.
    {
        torch::NoGradGuard no_grad;

        float gae = 0.0F;
        for (int64_t t = maxStep - 1; t >= 0; --t)
        {
            float reward = rewards[t].item<float>();
            float value = values[t].item<float>();
            float done = dones[t].item<float>();
            float next_value{ last_values.item<float>() };
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
                advantages.index_select(0, b));

            L += std::get<0>(r);
            A += std::get<1>(r);
            V += std::get<2>(r);
            E += std::get<3>(r);
            ++n;
        }
    }

    p_impl->m_outputs.m_loss = L / n;
    p_impl->m_outputs.m_actorloss = A / n;
    p_impl->m_outputs.m_entropyloss = E / n;
    p_impl->m_outputs.m_valueloss = V / n;
}

/**
* @brief Samples one stochastic action from the current policy.
* @param state Input observation tensor.
* @return Tuple containing action, log-probability, and value.
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PPO::SelectAction(const torch::Tensor& state)
{
    torch::NoGradGuard no_grad;

    if (p_impl == nullptr)
        return { {}, {}, {} };

    return p_impl->m_policy->selectaction(state);
}

/**
* @brief Saves the current model to the eval directory and optionally prod.
* @param modelpath Base directory containing model checkpoints.
*/
void
PPO::Save(std::filesystem::path const& modelpath)
{
    if (p_impl == nullptr)
        return;

    if (!std::filesystem::exists(modelpath))
        std::filesystem::create_directories(modelpath);

    std::filesystem::path evaldirectory{ std::filesystem::path(modelpath) / "eval" };
    if (!std::filesystem::exists(evaldirectory))
        std::filesystem::create_directories(evaldirectory);

    bool prodCreation{ false };
    std::filesystem::path prodDirectory{ std::filesystem::path(modelpath) / "prod" };
    if (!std::filesystem::exists(prodDirectory))
    {
        std::filesystem::create_directories(prodDirectory);
        prodCreation = true;
    }

    torch::serialize::OutputArchive archive{};
    p_impl->m_policy->save(archive);
    p_impl->m_optimizer->save(archive);

    std::filesystem::path savepath{ evaldirectory };
    savepath = savepath / "model.pt";
    archive.save_to(savepath.string());

    if (prodCreation)
    {
        std::filesystem::path prodpath{ prodDirectory };
        prodpath = prodpath / "model.pt";
        archive.save_to(prodpath.string());
    }
}

/**
* @brief Returns the last aggregated PPO training outputs.
* @return Reference to the output structure.
*/
PPOOutputs const&
/**
 * @brief Retourne une vue ou une valeur calculée depuis l'état courant de PPO.
 */
PPO::GetOutputs(void) const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_PPO_OUTPUTS;

    return p_impl->m_outputs;
}

} /* ppo::model */
