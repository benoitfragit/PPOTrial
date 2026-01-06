#pragma once

#include <array>
#include <string>

namespace ppo::model {

    class Player;

using PlayerId = unsigned long long;

enum class eEnvironmentState : unsigned char {
    INIT = 0,
    RUNNING = 1,
    PAUSED = 2,
    STOPPED = 3
};

struct sEnvironmentState
{
    float           m_maxNorth  { 112500.F };
    float           m_maxEast   { 150000.F };
    float           m_dt        { 0.02F };
};

struct sPlayerState
{
    float           m_maxNoescapeZone{8000.F };
    PlayerId        m_target{ ULLONG_MAX };
    PlayerId        m_playerId{ ULLONG_MAX };
    float           m_dx{ 0.F };
    float           m_dy{ 0.F };
    float           m_x{ 0.F  };
    float           m_y{ 0.F  };
    float           m_target_x{ 0.F };
    float           m_target_y{ 0.F };
    float           m_target_dx{ 0.F };
    float           m_target_dy{ 0.F };
    float           m_maxSpeed{ 0.F };
    float           m_speed{ 0.F };
    float           m_rotation{ 0.F };
    float           m_rotationAngle{ 0.F };
    float           m_fuel{ 0.F };
    float           m_fuelconsumption{ 0.F };
    float           m_escapezone{ 0.F };
    float           m_noescapezone{ 0.F };
    float           m_maxfuel{ 0.F };
    float           m_maxShootAngle{ 0.F };
    float           m_pk{ 0.F };
    unsigned short  m_nbmissiles{ 0 };
    unsigned short  m_maxnbmissiles{ 0 };
    bool            m_destroyed{ false };
};

class IAgent {
public:
    virtual                     ~IAgent      (void) = default;

    virtual void                Init         (sEnvironmentState const& environment, Player* const player)             = 0;
    virtual void                ChooseAction (sEnvironmentState const& environment, Player* const player)             = 0;
    virtual void                AddReward    (unsigned char type, float reward)                                                  noexcept = 0;
    virtual void                SetDone      (bool done)                                                     noexcept = 0;
    virtual void                SetNewState  (sEnvironmentState const& environment, Player* const player)                                                                   = 0;
    virtual bool                SetupTraining(void)  noexcept = 0;
};

class IPlayerState
{
public:
    virtual                   ~IPlayerState(void) = default;
    virtual sPlayerState      GetSnapshot(void) const noexcept = 0;
};

struct sCommonAgentState
{
    long long m_step{ 0 };
    long long m_maxStep{ 0 };
    float     m_reward{ 0.F };
    float     m_rewardClamp{ 3.F };
    float     m_bestreward{ -FLT_MAX };
    float     m_loss{ 0.F };
    float     m_actorloss{ 0.F };
    float     m_entropyloss{ 0.F };
    float     m_valueloss{ 0.F };
    unsigned int m_version{ 0 };
    bool      m_done{ false };
    bool      m_training{ false };
};

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
    unsigned int    m_checkpoint{ 5 };
    float           m_bestreward{ -FLT_MAX };
    float           m_target_kl{ 0.01F };
};

struct PPOOutputs {
    float m_loss{ 0.F };
    float m_actorloss{ 0.F };
    float m_valueloss{ 0.F };
    float m_entropyloss{ 0.F };
    float m_bestreward{ -FLT_MAX };
    unsigned int m_version{ 0 };
};

struct sAgentState:
    public sCommonAgentState 
{
    enum RewardType
    {
        TargetDestruction   = 0,
        TargetInEZ          = 1,
        OwnshipDestruction  = 2,
        OwnshipInEZ         = 3,
        Closing             = 4,
        Interception        = 5
    };

    /// <summary>
    /// Define a list of possible rewards type at compile time
    /// </summary>                                        
    constexpr std::array<std::string_view, 6> const static REWARDS
    {
        "Target destruction",
        "Target in EZ",
        "Ownship destruction",
        "Ownship in EZ",
        "Closing",
        "Interception"
    };

    using RewardVector = std::array<float, REWARDS.size()>;
    using RewardScaling = std::array<float, REWARDS.size()>;

    RewardVector    m_rewardVector {};
    RewardScaling   m_rewardScaling{};
    PPOParamters    m_ppoparameters{};
};




} /* ppo::model */