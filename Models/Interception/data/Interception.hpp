#pragma once
#include "environment.hpp"
#include "aircraft.hpp"
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
/// This is SNA model controlled by an AI agent
/// </summary>
class RLSNA : public ppo::model::SNA
{
public:       
    /// <summary>
    ///  This is a template constructor
    /// to be able to use a environment or something else
    /// </summary>
    explicit RLSNA(std::shared_ptr<Evaluator> evaluator = nullptr) : 
        ppo::model::SNA(),
        p_evaluator{ evaluator }{};

    void                  SetAircraft(ppo::model::Aircraft* aircraft)         noexcept { p_aircraft = aircraft; };
    ppo::model::Aircraft* GetAircraft(void)                             const noexcept { return p_aircraft; };

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
    using State = float(RLSNA::*)(void) const noexcept;

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
    using Action = void(RLSNA::*)(float);

    /// <summary>
    /// This is a list of action to be registered in an agent
    /// </summary>
    constexpr  std::array<Action, 2> const static ACTIONS
    {
        &ActionSetRotSpeed,
        &ActionSetAcceleration
    };

private:
    ppo::model::Aircraft*       p_aircraft{ nullptr };
    std::shared_ptr<Evaluator> p_evaluator{ nullptr };
};

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
class SNATrainingAgent :
    public ppo::model::TrainingAgent<ActorCriticNet, RLSNA, sAgentState>
{
public:
    using Base = ppo::model::TrainingAgent<ActorCriticNet, RLSNA, sAgentState>;

    SNATrainingAgent(
        ppo::model::Environment* const pEnvironment,
        RLSNA* const model,
        std::string const& name,
        std::string const& modelpath,
        ppo::model::IEvaluator* const evaluator = nullptr) :
        Base(model, name, modelpath, evaluator) ,
        p_environment{pEnvironment}{}

private:
    /// <summary>
    /// this function is used to compute the reward at each step
    /// </summary>
    /// <param name=""></param>
    /// <returns></returns>
    bool RewardFunction(void) noexcept final;

    /// <summary>
    /// this is the environment pointer used to get the time and the other entities
    /// </summary>
    ppo::model::Environment* p_environment{ nullptr };
};

/// <summary>
/// This is the definition of the agent used for production
/// </summary>
using SNAProdAgent = ppo::model::ProdAgent<ActorCriticNetProd, RLSNA>;

template<typename Ty>
std::shared_ptr<ppo::model::Aircraft>
GetNewRandomAircraft(std::string const& name, ppo::model::eEntitySide side, Ty& env)
{
    // create the SNA
    std::shared_ptr<RLSNA> sna{ std::make_shared<RLSNA>() };

    std::shared_ptr<ppo::model::Aircraft> aircraft{ 
        std::make_shared<ppo::model::Aircraft>(name, side, env, sna) };

    sna->SetAircraft(aircraft.get());

    if constexpr (std::is_same_v<Ty, ppo::model::Environment>)
        env.AddEntity(aircraft);

    return aircraft;
}

template<typename Ty>
std::pair<std::shared_ptr<ppo::model::Aircraft>, std::shared_ptr<SNATrainingAgent>>
GetNewTrainingAircraft(std::string const& name, ppo::model::eEntitySide side, Ty& env)
{
    // create the evaluation
    std::shared_ptr<Evaluator> evaluator{ std::make_shared<Evaluator>() };

    std::shared_ptr<RLSNA> sna{ std::make_shared<RLSNA>(evaluator) };

    // create the aircraft
    std::shared_ptr<ppo::model::Aircraft> aircraft{ std::make_shared<ppo::model::Aircraft>(name, side, env, sna) };
                                  
    sna->SetAircraft(aircraft.get());

    // Create the agent
    std::shared_ptr<SNATrainingAgent> agent{ std::make_shared<SNATrainingAgent>(
                        &env,
                        sna.get(),
                        "RLSNA",
                        RESOURCE_RLSNAMODEL_PATH,
                        evaluator.get()) };

    if constexpr (std::is_same_v<Ty, ppo::model::Environment>)
    {
        env.AddEntity(aircraft);
        env.AddAgent(agent);
    }

    return std::make_pair(aircraft, agent);
}

template<typename Ty>
std::pair<std::shared_ptr<ppo::model::Aircraft>, std::shared_ptr<SNAProdAgent>>
GetNewProductionAircraft(std::string const& name, ppo::model::eEntitySide side, Ty& env, std::string const& version = "prod")
{
    std::shared_ptr<RLSNA> sna{ std::make_shared<RLSNA>() };

    // create the aircraft
    std::shared_ptr<ppo::model::Aircraft> aircraft{ std::make_shared<ppo::model::Aircraft>(name, side, env, sna) };

    sna->SetAircraft(aircraft.get());

    std::filesystem::path modelpath{ RESOURCE_RLSNAMODEL_PATH };
    modelpath = modelpath / version / "model.pt";
    std::shared_ptr<SNAProdAgent> agent{ std::make_shared<SNAProdAgent>(sna.get(), "ProdAgent",modelpath) };

    if constexpr (std::is_same_v<Ty, ppo::model::Environment>)
    {
        env.AddEntity(aircraft);
        env.AddAgent(agent);
    }

    return std::make_pair(aircraft, agent);
}

} /* ppo::interception */