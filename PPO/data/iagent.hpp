#pragma once

#include "interface.hpp"

#include <string>
#include <atomic>
#include <vector>

namespace ppo::model {

/// <summary>
/// this is a method to get a unique identifier
/// </summary>
/// <returns>a unique identifier</returns>
inline unsigned long long 
GetNewUniqId(void) 
{
    static std::atomic<unsigned long long> counter{ 0 };
    ++counter;
    return counter.load(std::memory_order_relaxed);
}

struct PPOParamters
{
    float           m_gamma{ 0.99F };
    float           m_lambda_gae{ 0.95F };
    float           m_clip_epsilon{ 0.2F };
    float           m_value_coef{ 0.5F };
    float           m_entropy_coef{ 0.02F };
    float           m_learningReate{ 3e-4F };
    int             m_K_epochs{ 4 };
    int             m_batch_size{ 64 };
    float           m_target_kl{ 0.01F };
};

struct sCommonAgentState  :
    public PPOParamters
{
    long long m_step{ 0 };
    long long m_maxStep{ 0 };
    float     m_reward{ 0.F };
    float     m_rewardClamp{ 3.F };
    float     m_loss{ 0.F };
    float     m_actorloss{ 0.F };
    float     m_entropyloss{ 0.F };
    float     m_valueloss{ 0.F };
    unsigned int m_checkpoint{ 0 };
    unsigned int m_version{ 0 };
    float     m_notation{ 0.F };
    float     m_evaluationprogress{ 0.F };
    bool      m_evaluationinprogress{ false };
    bool      m_done{ false };
    bool      m_training{ false };
};

struct sAgentStateSnaphot
{
    sCommonAgentState   m_common{};
    std::vector<float>  m_scalings{};
    std::vector<float>  m_rewards{};
    std::vector<std::string> m_rewardnames{};
    std::string         m_name{""};
    unsigned long long  m_id{ 0 };
};

struct PPOOutputs {
    float m_loss{ 0.F };
    float m_actorloss{ 0.F };
    float m_valueloss{ 0.F };
    float m_entropyloss{ 0.F };
};

/// <summary>
/// a comamnd to control an agent
/// 
/// targeted agent is the one with the m_id
/// </summary>
struct sAgentRealTimeCommand {
    /// <summary>
    /// type of the real time command
    /// </summary>
    enum class eTypeCommand {
        None,
        RolloutBuffer,
        RewardClamp,
        RewardScaling
    };
    /// <summary>
    /// current command type
    /// </summary>
    eTypeCommand m_type{ eTypeCommand::None };
    /// <summary>
    /// id of the agent
    /// 
    /// if the ID mismatch then an agent must ignore this command
    /// </summary>
    unsigned long long m_id{ 0 };
    /// <summary>
    /// new rollout buffer size
    /// </summary>
    long long    m_rolloutBuffer{ 0 };
    /// <summary>
    /// a value to clamp the max reward
    /// </summary>
    float        m_rewardClamp{ 0.F };
    /// <summary>
    /// a value of one reward scaling
    /// </summary>
    float        m_rewardScaling{ 0.F };
    /// <summary>
    /// anonymous reward type affected by the scaling
    /// </summary>
    unsigned char m_rewardScalingType{ 0 };
};

class IAgent {
public:
    virtual                     ~IAgent(void) = default;

    virtual void                Init(void) = 0;
    virtual void                ChooseAction(void) = 0;
    virtual bool                SetNewState(void) = 0;
    virtual void                ProcessCommand(sAgentRealTimeCommand const& command)  noexcept = 0;
    virtual sAgentStateSnaphot  GetSnapshot(void) const noexcept = 0;
};

class IEvaluator {
public:
    virtual ~IEvaluator(void) = default;
    virtual void Start(std::string const& name, std::filesystem::path const& modelpath) = 0;
    virtual void Stop(void) = 0;
    virtual sEvaluatorState GetSnapshot(void) const noexcept = 0;
};
} /* ppo::model */