#pragma once
#include <array>
#include <vector>
#include "rlagent.hpp"
#include <functional>
#include <string>
#include "serializer.hpp"

namespace ppo::model {

template<typename Policy, typename Model>
class TrainingAgent:
    public IAgent
{
public:
    using StateVector  = typename Model::STATES;
    using ActionVector = typename Model::ACTIONS;

                             TrainingAgent           (Model* const model,
                                                      std::string const& name, 
                                                      std::string const& modelpath, 
                                                      IEvaluator* const evaluator = nullptr);
    
    void                     Init                    (void) override;
    void                     ProcessCommand          (sAgentRealTimeCommand const& command) noexcept override;
    bool                     SetNewState             (void) override;
    void                     ChooseAction            (void) override;

    virtual bool             RewardFunction          (void) noexcept = 0;

    void                     SetDone                 (bool done)                noexcept;
    void                     AddReward               (unsigned char type, float reward)             noexcept;

    bool                     IsTrained               (void)                  const noexcept;
    bool                     IsTraining              (void)                  const noexcept;
    bool                     IsLastStep              (void)                  const noexcept;
    sAgentStateSnaphot       GetSnapshot             (void)                  const noexcept override;

protected:    
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