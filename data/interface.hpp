#pragma once

#include <array>
#include <string>

namespace ppo::model {

    class Player;

using ppoArray2f = std::array<float, 2>;
using ppoArray4f = std::array<float, 4>;
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
    ppoArray2f      m_noescapeZoneRange{ 3000.F, 8000.F };
    PlayerId        m_target{ ULLONG_MAX };
    PlayerId        m_playerId{ ULLONG_MAX };
    float           m_dx{ 0.F };
    float           m_dy{ 0.F };
    float           m_x{ 0.F  };
    float           m_y{ 0.F  };
    float           m_target_x{ 0.F };
    float           m_target_y{ 0.F };
    float           m_maxSpeed{ 0.F };
    float           m_speed{ 0.F };
    float           m_rotation{ 0.F };
    float           m_rotationAngle{ 0.F };
    float           m_fuel{ 0.F };
    float           m_fuelconsumption{ 0.F };
    float           m_escapezone{ 0.F };
    float           m_noescapezone{ 0.F };
    float           m_maxfuel{ 0.F };
    unsigned short  m_nbmissiles{ 0 };
    unsigned short  m_maxnbmissiles{ 0 };
    bool            m_destroyed{ false };
};

class IAgent {
public:
    virtual                     ~IAgent                 (void) = default;

    virtual void                Init                    (sEnvironmentState const& environment, Player* const player)             = 0;
    virtual void                ChooseAction            (sEnvironmentState const& environment, Player* const player)             = 0;
    virtual void                SetRewardScaling        (unsigned char type, float scaling)                             noexcept = 0;
    virtual void                AddReward               (unsigned char type, float reward)                                                  noexcept = 0;
    virtual void                SetDone                 (bool done)                                                     noexcept = 0;
    virtual void                StartTraining           (long long buffer_dim)                                                   = 0;
    virtual void                StopTraining            (void)                                                          noexcept = 0;
    virtual bool                IsTraining              (void)                                                    const noexcept = 0;
    virtual long long           GetCurrentStep          (void)                                                    const noexcept = 0;
    virtual void                SetNewState             (sEnvironmentState const& environment, Player* const player)                                                                   = 0;
    virtual void                SetRewardClampValue     (float clampValue)                                              noexcept = 0;
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
    bool      m_done{ false };
    bool      m_training{ false };
    bool      m_backward{ false };
};

struct sAgentState:
    public sCommonAgentState 
{
    enum RewardType
    {
        FuelConsumption     = 0,
        SpeedRange          = 1,
        TargetDestruction   = 2,
        TargetInEZ          = 3,
        OwnshipDestruction  = 4,
        OwnshipInEZ         = 5,
        Range               = 6,
        Missile             = 7
    };

    /// <summary>
    /// Define a list of possible rewards type at compile time
    /// </summary>                                        
    constexpr std::array<std::string_view, 8> const static REWARDS
    {
        "Fuel consumption",
        "Speed range",
        "Target destruction",
        "Target in EZ",
        "Ownship destruction",
        "Ownship in EZ",
        "Range",
        "Missile"
    };

    using RewardVector = std::array<float, REWARDS.size()>;
    using RewardScaling = std::array<float, REWARDS.size()>;

    RewardVector    m_rewardVector {};
    RewardScaling   m_rewardScaling{};
};


} /* ppo::model */