#pragma once
#include <array>
#include <vector>
#include "rlagent.hpp"
#include <functional>
#include <string>

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
    bool                     Update                  (void) noexcept override;
    bool                     SetNewState             (void) override;
    void                     ChooseAction            (void) override;

    virtual bool             RewardFunction          (void) noexcept = 0;

    void                     SetDone                 (bool done)                noexcept;
    void                     SetRewardClampValue     (float clampValue)         noexcept;
    void                     SetRewardScaling        (unsigned char type, float scaling)            noexcept;
    void                     AddReward               (unsigned char type, float reward)             noexcept;
    void                     SetRolloutBufferSize    (long long buffer_dim);
    void                     StopTraining            (void);
    bool                     IsTrained               (void)                  const noexcept;
    bool                     IsTraining              (void)                  const noexcept;
    bool                     IsLastStep              (void)                  const noexcept;
    std::string const&       GetName                 (void)                  const noexcept;

    std::vector<float>       GetRewardScalings       (void) const noexcept;
    std::vector<std::string> GetRewardNames          (void) const noexcept;

    sCommonAgentState        GetSnapshot             (std::vector<float>& rewards) const noexcept;

protected:    
    Model*                      p_model{ nullptr };

private:
    template<typename Ar>
    void                        serialize(Ar& ar);

    void                        UpdateDoubleBuffer(void);

    std::shared_ptr<Policy>     p_policy        {nullptr};
    StateVector                 m_stateVector   {};
    ActionVector                m_actionVector  {};
    std::unique_ptr<RLAgent>    p_rlagent       {nullptr};
    sCommonAgentState           m_internal      {};
    sCommonAgentState           m_buffers[2];
    using RewardVector          = std::array<float, Model::REWARDS.size()>;
    using RewardScaling         = std::array<float, Model::REWARDS.size()>;

    RewardVector                m_rewardVector{};
    RewardScaling               m_rewardScalings{};
    RewardVector                m_bufferRewardVector[2];
    RewardScaling               m_bufferRewardScaling[2];   

    std::atomic<long long>      m_maxRolloutBuffer{ 4096 };
    std::atomic<float>          m_rewardScaling[Model::REWARDS.size()];
    std::atomic<float>          m_rewardClamp{ 5.F };
    std::atomic<uint32_t>       m_index         { 0 };
    
    std::string                 m_name{ "" };
    std::string                 m_modelpath{ "" };
    IEvaluator*                 p_evaluator{nullptr};
};

} /* ppo::model */

#include "trainingagent_def.hpp"