/**
 * @file Examples/Airspace/Agents/Bindings/data/trainingagent.hpp
 * @brief Training agent of an Airspace model, driven from outside the world (the
 *        rlmission_environment DLL of ppokit-run, whose network DLL learns).
 * @details Each step the caller writes the action; the agent hands it to its model, runs its
 * reward function, and publishes the next observation, the clamped reward, the weighted
 * channels and the episode's end (IExternalTrainingAgent). Its settings are those of
 * settings/Training/<name>.json: rollout size (the episode's step limit), reward clamp and
 * channel weights.
 */
#pragma once

#include "external_training.hpp"

#include "ppo/core/agent_contract.hpp"
#include "ppo/runtime/iagent.hpp"
#include "serializer.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace ppo::model {

/**
 * @brief A training agent on a domain model (states in, actions out, a reward function).
 * @tparam Model Domain model with STATES, ACTIONS, REWARDS, SetNewState and SetNextAction.
 */
template<typename Model>
class TrainingAgent : public IAgent, public IExternalTrainingAgent
{
public:
    using StateVector = typename Model::STATES;
    using ActionVector = typename Model::ACTIONS;
    static constexpr std::size_t ObservationSize = std::tuple_size<StateVector>::value;
    static constexpr std::size_t ActionSize = std::tuple_size<ActionVector>::value;

    /** @brief Optional descriptor of the observation and action contract. */
    static constexpr ppo::core::AgentContract Contract() noexcept
    {
        return ppo::core::AgentContractOf<Model>();
    }

    /**
     * @param model Domain model, owned by its entity.
     * @param name Agent family: its settings are Training/<name>.json.
     */
    TrainingAgent(Model* model, std::string const& name);

    /** @brief Removes the settings callbacks, which capture this agent. */
    ~TrainingAgent() override;

    TrainingAgent(TrainingAgent const&) = delete;
    TrainingAgent& operator=(TrainingAgent const&) = delete;
    TrainingAgent(TrainingAgent&&) = delete;
    TrainingAgent& operator=(TrainingAgent&&) = delete;

    void Init(void) override;
    bool SetNewState(void) override;
    void ChooseAction(void) override;

    /**
     * @brief The model's reward function, run after each step (AddReward, SetDone).
     * @return True when the episode ends.
     */
    virtual bool RewardFunction(void) noexcept = 0;
    /** @brief Ends the episode on a terminal state. */
    void SetDone(bool done) noexcept { done_ |= done; }
    /** @brief Adds a reward to channel `type`, weighted by its weight. */
    void AddReward(unsigned char type, float reward) noexcept;
    /**
     * @brief Always false: the controllers' last-step mission failure is not paid.
     * @details In the former in-world training it was the last transition of the learner's
     * rollout buffer. Under ppokit-run there was no such buffer, so RLMission trained without
     * that penalty (RLSNA is only played and evaluated); it stays unpaid so that the reward
     * RLMission trains on does not change (AirspaceTrainingRewardTests).
     */
    bool IsLastStep(void) const noexcept { return false; }

    ExternalStep& External() noexcept override { return externalStep_; }
    std::size_t RolloutSteps() const noexcept override;
    void SetRewardWeights(double const* weights) noexcept override;
    void RewardWeights(double* weights) const noexcept override;

protected:
    Model* p_model{nullptr};

private:
    /** @brief Reads rollout size, reward clamp and channel weights (Training/<name>.json). */
    void LoadFromFile(ppo::serialization::json const& ar);
    /** @brief The step's results into the exchange. */
    void PublishStep(bool ended) noexcept;

    using RewardVector = std::array<float, Model::REWARDS.size()>;

    StateVector m_stateVector{};
    ActionVector m_actionVector{};
    /** @brief Training/<name>.json callbacks of this agent. */
    ppo::serialization::Serializer::Registration settingsRegistration_{0U};
    RewardVector m_rewardScalings{};
    /** @brief The step's weighted channels. */
    RewardVector stepChannels_{};
    /** @brief The step's reward: its weighted channels summed. */
    float reward_ = 0.F;
    float rewardClamp_ = 3.F;
    long long rolloutSteps_ = 0;
    bool done_ = false;
    ExternalStep externalStep_{};
    std::string m_name{};
};

} // namespace ppo::model

#include "trainingagent_def.hpp"
