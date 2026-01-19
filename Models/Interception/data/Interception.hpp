#pragma once
#include "environment.hpp"
#include "player.hpp"
#include "evaluator.hpp"
#include "actorcriticnet.hpp"
#include "trainingagent.hpp"
#include "prodagent.hpp"

#include <map>
#include <memory>
#include <string>
#include <array>
#include <type_traits>

namespace ppo::interception {

/// <summary>
/// This is missile model controlled by an AI agent
/// </summary>
class Missile :
    public ppo::model::Player
{
public:       
    
    /// <summary>
    ///  This is a template constructor
    /// to be able to use a environment or something else
    /// </summary>
    template <typename Ty>
    Missile                         (std::string const name, 
                                    ppo::model::ePlayerSide side,
                                    Ty& env, 
                                    std::shared_ptr<Evaluator> evaluator = nullptr) :
        ppo::model::Player("Missiles", name, side, ppo::model::ePlayerType::Missile, env),
        p_evaluator{ evaluator }
    {};

    void OnRun                      (void) override;
    void OnInit                     (void) override;

    void     SetTargetPosition      (float north, float east)                noexcept;
    void     SetTargetSpeed         (float north, float east)                noexcept;
    void     SetTarget              (ppo::model::PlayerId targetId)          noexcept;
    ppo::model::PlayerId GetTarget  (void)                             const noexcept;
    float    GetKillProbability     (void)                             const noexcept;
    float    GetCLosingSpeed        (void)                             const noexcept;
    float    GetTargetLOS           (void)                             const noexcept;
    float    GetTargetLOSRate       (void)                             const noexcept;

    /// <summary>
    /// Define the actions-state space
    /// </summary>
    float StateNorthRange           (void)          const noexcept;
    float StateEastRange            (void)          const noexcept;
    float StateRange                (void)          const noexcept;
    float StateClosingSpeed         (void)          const noexcept;
    float StateCosRotation          (void)          const noexcept;
    float StateSinRotation          (void)          const noexcept;
    void  ActionSetRotSpeed         (float value)         noexcept;
    void  ActionSetAcceleration     (float value)         noexcept;

    /// <summary>
    /// This is a state to be registered in an array
    /// </summary>
    using State = float(Missile::*)(void) const noexcept;

    constexpr std::array<State, 6> const static STATES
    {
        &StateNorthRange,
        &StateEastRange,
        &StateRange,
        &StateClosingSpeed,
        &StateCosRotation,
        &StateSinRotation
    };

    /// <summary>
    /// this is an action to be registered in an array
    /// </summary>
    using Action = void(Missile::*)(float);

    /// <summary>
    /// This is a list of action to be registered in an agent
    /// </summary>
    constexpr  std::array<Action, 2> const static ACTIONS
    {
        &ActionSetRotSpeed,
        &ActionSetAcceleration
    };

private:
    ppo::model::PlayerId m_target{ ULLONG_MAX };
    float                m_pk{ 0.F };
    float                m_target_x{ 0.F };
    float                m_target_y{ 0.F };
    float                m_target_dx{ 0.F };
    float                m_target_dy{ 0.F };
    float                m_closingspeed{ 0.F };
    float                m_target_los{ 0.F };
    float                m_target_los_rate{ 0.F };

    std::shared_ptr<Evaluator> p_evaluator{ nullptr };
};

void UpdateEnv(ppo::model::Environment& env, ppo::model::IPlayer* player);

/// <summary>
/// This is the Agent state used for the training
/// extend generic state with rewards
/// </summary>
struct sAgentState :
    public ppo::model::sCommonAgentState
{
    enum RewardType
    {
        TargetDestruction = 0,
        TargetInEZ = 1,
        OwnshipDestruction = 2,
        Closing = 3,
        NavPN = 4,
        NavAPN = 5,
        Fuel = 6,
        MissionFailure = 7
    };

    /// <summary>
    /// Define a list of possible rewards type at compile time
    /// </summary>                                        
    constexpr std::array<std::string_view, 8> const static REWARDS
    {
        "Target destruction",
        "Target in EZ",
        "Ownship destruction",
        "Closing",
        "NavPN",
        "NavAPN",
        "Fuel",
        "MissionFailure"
    };

    using RewardVector = std::array<float, REWARDS.size()>;
    using RewardScaling = std::array<float, REWARDS.size()>;

    RewardVector    m_rewardVector{};
    RewardScaling   m_rewardScaling{};
};

/// <summary>
/// This is the definitionn of the agent used for training
/// </summary>
using MissileTrainingAgent = ppo::model::TrainingAgent<ActorCriticNet, Missile, sAgentState>;

/// <summary>
/// This is the definition of the agent used for production
/// </summary>
using MissileProdAgent = ppo::model::ProdAgent<ActorCriticNetProd, Missile>;

/// <summary>
/// This is the function we used to reward our agent
/// </summary>
/// <param name="environment">this is the environment</param>
/// <param name="missile">this is the entity</param>
/// <param name="pAgent">this is the agent</param>
/// <returns>true when done</returns>
bool RewardingFunction(ppo::model::Environment& environment, Missile* const player, MissileTrainingAgent& pAgent);

template<typename Ty>
std::shared_ptr<Missile>
GetNewRandomMissile(std::string const& name, ppo::model::ePlayerSide side, Ty& env)
{
    std::shared_ptr<Missile> missile{ std::make_shared<Missile>(name, side, env) };

    if constexpr (std::is_same_v<Ty, ppo::model::Environment>)
    {
        env.AddAgent(missile, &UpdateEnv);
    }

    return missile;
}

template<typename Ty>
std::pair<std::shared_ptr<Missile>, std::shared_ptr<MissileTrainingAgent>>
GetNewTrainingMissile(std::string const& name, ppo::model::ePlayerSide side, Ty& env)
{
    // create the evaluation
    std::shared_ptr<Evaluator> evaluator{ std::make_shared<Evaluator>() };

    // create the missile
    std::shared_ptr<Missile> missile{ std::make_shared<Missile>(name, side, env, evaluator) };

    // Create the agent
    std::shared_ptr<MissileTrainingAgent> agent{ std::make_shared<MissileTrainingAgent>(
                        missile.get(),
                        "Missile",
                        RESOURCE_MISSILEMODEL_PATH,
                        [&](Missile* player, MissileTrainingAgent& agent)
                        {
                            return RewardingFunction(env, player, agent);
                        },
                        evaluator.get()) };

    if constexpr (std::is_same_v<Ty, ppo::model::Environment>)
    {
        env.AddAgent(missile,
            &UpdateEnv,
            agent);
    }

    return std::make_pair(missile, agent);
}

template<typename Ty>
std::pair<std::shared_ptr<Missile>, std::shared_ptr<MissileProdAgent>>
GetNewProductionMissile(std::string const& name, ppo::model::ePlayerSide side, Ty& env, std::string const& version = "prod")
{
    // create the missile
    std::shared_ptr<Missile> missile{ std::make_shared<Missile>(name, side, env) };

    std::filesystem::path modelpath{ RESOURCE_MISSILEMODEL_PATH };
    modelpath = modelpath / version / "model.pt";
    std::shared_ptr<MissileProdAgent> agent{ std::make_shared<MissileProdAgent>(missile.get(), "ProdAgent",modelpath) };

    if constexpr (std::is_same_v<Ty, ppo::model::Environment>)
    {
        env.AddAgent(missile,
            &UpdateEnv,
            agent);
    }

    return std::make_pair(missile, agent);
}

} /* ppo::interception */