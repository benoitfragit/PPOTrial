#pragma once
#include <array>
#include <vector>
#include "rlagent.hpp"
#include <functional>
#include <string>
#include "serializer.hpp"

namespace ppo::model {

/**
 * @brief Generic training-time agent implementation.
 *
 * @tparam Policy Policy network type used by RLAgent.
 * @tparam Model Domain model exposing state/action/reward definitions.
 */
template<typename Policy, typename Model>
class TrainingAgent:
    public IAgent
{
public:
    /** @brief Alias to the model-defined state vector type. */
    using StateVector  = typename Model::STATES;
    /** @brief Alias to the model-defined action vector type. */
    using ActionVector = typename Model::ACTIONS;

                             TrainingAgent           (Model* const model,
                                                      std::string const& name, 
                                                      std::string const& modelpath, 
                                                      IEvaluator* const evaluator = nullptr);
    
    /** @brief Initializes model bindings and internal RL components. */
    void                     Init                    (void) override;
    /** @brief Handles real-time control commands sent by the environment. */
    void                     ProcessCommand          (sAgentRealTimeCommand const& command) noexcept override;
    /** @brief Computes reward transition and forwards it to the RL backend. */
    bool                     SetNewState             (void) override;
    /** @brief Requests a new action from the RL backend and applies it to the model. */
    void                     ChooseAction            (void) override;

    /**
     * @brief Domain-specific reward computation hook.
     * @return True when the current episode should continue.
     */
    virtual bool             RewardFunction          (void) noexcept = 0;

    /** @brief Sets the terminal state flag for the current episode. */
    void                     SetDone                 (bool done)                noexcept;
    /**
     * @brief Adds one reward contribution in a reward channel.
     * @param type Reward channel index.
     * @param reward Reward contribution.
     */
    void                     AddReward               (unsigned char type, float reward)             noexcept;

    /** @brief Returns true when the configured training budget was reached. */
    bool                     IsTrained               (void)                  const noexcept;
    /** @brief Returns true while optimization is active. */
    bool                     IsTraining              (void)                  const noexcept;
    /** @brief Returns true on the final step of the current rollout window. */
    bool                     IsLastStep              (void)                  const noexcept;
    /** @brief Produces a serializable snapshot of the training state. */
    sAgentStateSnaphot       GetSnapshot             (void)                  const noexcept override;

protected:    
    /** @brief Domain model controlled by this agent. */
    Model*                      p_model{ nullptr };

private:
    void                        LoadFromFile(ppo::serialization::json& ar);
    void                        SaveToFile  (ppo::serialization::json& ar);

    std::shared_ptr<Policy>     p_policy        {nullptr};
    StateVector                 m_stateVector   {};
    ActionVector                m_actionVector  {};
    std::unique_ptr<RLAgent>    p_rlagent       {nullptr};
    sCommonAgentState           m_internal      {};
    using RewardVector          = std::array<float, Model::REWARDS.size()>;
    using RewardScaling         = std::array<float, Model::REWARDS.size()>;

    RewardVector                m_rewardVector{};
    RewardScaling               m_rewardScalings{};
    
    std::string                 m_name{ "" };
    std::string                 m_modelpath{ "" };
    IEvaluator*                 p_evaluator{nullptr};

    unsigned long long          m_id{ GetNewUniqId() };
};

} /* ppo::model */

#include "trainingagent_def.hpp"