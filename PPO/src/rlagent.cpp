/**
* @file PPO/src/rlagent.cpp
* @brief Implementation of the rollout-collection RL agent wrapper.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "rlagent.hpp"

#include "ppo.hpp"

#include <torch/torch.h>

namespace ppo::model {

/**
* @brief Private implementation for the rollout-collection agent.
*/
struct RLAgent::Impl
{
    /**
    * @brief Builds the internal PPO controller and rollout buffers.
    * @param policy Trainable policy network.
    * @param modelpath Base directory containing model checkpoints.
    * @param state_dim Observation size.
    * @param action_dim Action size.
    * @param buffer_dim Number of steps stored per rollout.
    * @param parameters PPO hyper-parameters.
    * @param stateVector External observation buffer.
    * @param actionVector External action buffer.
    * @param loadExistingModel True to restore the previous prod checkpoint.
    */
    Impl(std::shared_ptr<ITrainingNet> policy,
         std::string const& modelpath,
         long long state_dim,
         long long action_dim,
         long long buffer_dim,
         PPOParamters const& parameters,
         float* const stateVector,
         float* const actionVector,
         bool loadExistingModel) :
        p_stateVector{ stateVector },
        p_actionVector{ actionVector }
    {
        if (!m_randomInitTorch)
        {
            torch::manual_seed(42);
            m_randomInitTorch = true;
        }

        p_ppo = std::make_unique<PPO>(
            policy,
            modelpath,
            state_dim,
            action_dim,
            parameters,
            loadExistingModel);

        m_maxStep = buffer_dim;
        m_step = 0;
        m_stateDim = state_dim;
        m_actionDim = action_dim;
        m_bufferDim = buffer_dim;

        m_state_t = torch::from_blob(p_stateVector, { 1, m_stateDim }, torch::kFloat);
    }

    /**
    * @brief Allocates and resets rollout buffers.
    */
    void Init(void);

    /**
    * @brief Samples one action and stores rollout data for the current step.
    */
    void ChooseAction(void);

    /**
    * @brief Records reward and done information for the latest step.
    * @param reward Reward assigned to the latest action.
    * @param done Episode termination flag.
    */
    void SetNewState(float reward, bool done);

    /**
    * @brief Launches one PPO update when the rollout buffer is full.
    */
    void UpdatePolicy(void);

    /** @brief Pointer to the external state buffer owned by the agent wrapper. */
    float* p_stateVector{ nullptr };
    /** @brief Pointer to the external action buffer owned by the agent wrapper. */
    float* p_actionVector{ nullptr };

    /** @brief Torch view over the external state buffer. */
    torch::Tensor m_state_t;

    /** @brief Batched observations collected during one rollout. */
    torch::Tensor m_states;
    /** @brief Batched actions collected during one rollout. */
    torch::Tensor m_actions;
    /** @brief Batched rewards collected during one rollout. */
    torch::Tensor m_rewards;
    /** @brief Batched done flags collected during one rollout. */
    torch::Tensor m_dones;
    /** @brief Batched action log-probabilities collected during one rollout. */
    torch::Tensor m_old_logp;
    /** @brief Batched value estimates collected during one rollout. */
    torch::Tensor m_values;
    /** @brief Bootstrap value used when one rollout ends on truncation. */
    torch::Tensor m_last_values;

    /** @brief PPO optimizer wrapper used by the agent. */
    std::unique_ptr<PPO> p_ppo{ nullptr };

    /** @brief Observation size. */
    long long m_stateDim{ 0 };
    /** @brief Rollout buffer size. */
    long long m_bufferDim{ 0 };
    /** @brief Action size. */
    long long m_actionDim{ 0 };
    /** @brief Maximum number of steps stored in one rollout. */
    long long m_maxStep{ 0 };
    /** @brief Current step index inside the rollout. */
    long long m_step{ 0 };

    /** @brief Global guard used to seed Torch once. */
    inline static bool m_randomInitTorch{ false };
    /** @brief Empty fallback outputs returned when PPO is not available. */
    inline static PPOOutputs const EMPTY_PPO_OUTPUTS{};
};

/**
* @brief Allocates and resets rollout buffers.
*/
void
RLAgent::Impl::Init(void)
{
    m_step = 0;

    // Allocate every tensor used during one rollout sequence.
    m_states = torch::randn({ m_maxStep, m_stateDim });
    m_actions = torch::tanh(torch::randn({ m_maxStep, m_actionDim }));
    m_rewards = torch::randn({ m_maxStep });
    m_dones = torch::zeros({ m_maxStep });
    m_old_logp = torch::zeros({ m_maxStep });
    m_values = torch::zeros({ m_maxStep });
    m_last_values = torch::zeros({ 1 });
}

/**
* @brief Samples one action and stores rollout data for the current step.
*/
void
RLAgent::Impl::ChooseAction(void)
{
    if (p_stateVector == nullptr
        || m_step >= m_maxStep
        || p_ppo == nullptr)
    {
        return;
    }

    torch::NoGradGuard no_grad;

    auto [action, logpb, values] = p_ppo->SelectAction(m_state_t);

    // Persist the transition data required by PPO.
    m_states[m_step] = m_state_t.squeeze(0);
    m_actions[m_step] = action.squeeze(0);
    m_old_logp[m_step] = logpb.item<float>();
    m_values[m_step] = values.item<float>();

    // Copy the policy output back to the external action buffer.
    float const* srcActionPtr = action.data_ptr<float>();
    std::memcpy(p_actionVector, srcActionPtr, sizeof(float) * action.size(1));
}

/**
* @brief Launches one PPO update when the rollout buffer is full.
*/
void
RLAgent::Impl::UpdatePolicy(void)
{
    if (p_ppo == nullptr)
        return;

    p_ppo->UpdatePolicy(
        m_states,
        m_actions,
        m_old_logp,
        m_values,
        m_rewards,
        m_dones,
        m_last_values);
}

/**
* @brief Records reward and done information for the latest step.
*
* When the rollout buffer reaches its maximum size, the method computes the
* bootstrap value for truncated rollouts and immediately triggers a PPO update.
*
* @param reward Reward assigned to the latest action.
* @param done Episode termination flag.
*/
void
RLAgent::Impl::SetNewState(float reward, bool done)
{
    if (m_step >= m_maxStep)
        return;

    m_rewards[m_step] = reward;
    m_dones[m_step] = done ? 1.0F : 0.0F;

    if (m_step < m_maxStep)
        ++m_step;

    if (m_step == m_maxStep)
    {
        if (done)
        {
            m_last_values = torch::zeros({ 1 });
        }
        else
        {
            auto [_, __, last_val] = p_ppo->SelectAction(m_state_t);
            m_last_values = last_val;
        }

        UpdatePolicy();
    }
}

/**
* @brief Builds the public RL agent wrapper.
* @param policy Trainable policy network.
* @param modelpath Base directory containing model checkpoints.
* @param state_dim Observation size.
* @param action_dim Action size.
* @param buffer_dim Number of steps stored per rollout.
* @param parameters PPO hyper-parameters.
* @param stateVector External observation buffer.
* @param actionVector External action buffer.
* @param loadExistingModel True to restore the previous prod checkpoint.
*/
RLAgent::RLAgent(std::shared_ptr<ITrainingNet> policy,
    std::string const& modelpath,
    long long state_dim,
    long long action_dim,
    long long buffer_dim,
    PPOParamters const& parameters,
    float* const stateVector,
    float* const actionVector,
    bool loadExistingModel) :
    p_impl{ std::make_unique<Impl>(
        policy,
        modelpath,
        state_dim,
        action_dim,
        buffer_dim,
        parameters,
        stateVector,
        actionVector,
        loadExistingModel) }
{
}

/**
* @brief Destroys the RL agent wrapper.
*/
RLAgent::~RLAgent(void) = default;

/**
* @brief Resets the rollout buffers.
*/
void
RLAgent::Init(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->Init();
}

/**
* @brief Samples the next action from the current policy.
*/
void
RLAgent::ChooseAction(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->ChooseAction();
}

/**
* @brief Records the reward and terminal flag of the latest transition.
* @param reward Reward assigned to the latest action.
* @param done Episode termination flag.
*/
void
RLAgent::SetNewState(float reward, bool done)
{
    if (p_impl == nullptr)
        return;

    p_impl->SetNewState(reward, done);
}

/**
* @brief Returns the latest PPO outputs.
* @return Reference to the PPO output structure.
*/
PPOOutputs const&
RLAgent::GetOutputs(void) const noexcept
{
    if (p_impl == nullptr
        || p_impl->p_ppo == nullptr)
    {
        return Impl::EMPTY_PPO_OUTPUTS;
    }

    return p_impl->p_ppo->GetOutputs();
}

/**
* @brief Tells whether the agent is still collecting one rollout.
* @return True while the current rollout buffer is not full.
*/
bool
RLAgent::IsTraining(void) const noexcept
{
    return p_impl != nullptr
        && p_impl->p_ppo != nullptr
        && p_impl->m_step < p_impl->m_maxStep;
}

/**
* @brief Returns the current rollout step index.
* @return Current rollout step index.
*/
long long
RLAgent::GetCurrentStep(void) const noexcept
{
    if (p_impl == nullptr)
        return 0;
    return p_impl->m_step;
}

/**
* @brief Returns the maximum rollout length.
* @return Maximum number of steps stored in one rollout.
*/
long long
RLAgent::GetMaxStep(void) const noexcept
{
    if (p_impl == nullptr)
        return 0;
    return p_impl->m_maxStep;
}

/**
* @brief Saves the underlying PPO model to disk.
* @param modelpath Base directory containing model checkpoints.
*/
void
RLAgent::Save(std::filesystem::path const& modelpath)
{
    if (p_impl == nullptr
        || p_impl->p_ppo == nullptr)
    {
        return;
    }

    p_impl->p_ppo->Save(modelpath);
}

} /* ppo::model */
