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
    RLSNA(void) : ppo::model::SNA(){};

    using STATES = std::array<float, 6>;
    using ACTIONS = std::array<float, 2>;

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

    void SetNextAction  (ACTIONS const& action) noexcept;
    void SetNewState    (STATES& state) noexcept;
};

/// <summary>
/// This is the definitionn of the agent used for training
/// </summary>
class SNATrainingAgent :
    public ppo::model::TrainingAgent<ActorCriticNet, RLSNA>
{
public:
    using Base = ppo::model::TrainingAgent<ActorCriticNet, RLSNA>;

    SNATrainingAgent(
        ppo::model::Environment* const pEnvironment,
        RLSNA* const model,
        std::string const& name,
        std::string const& modelpath) :
        Base(model, name, modelpath, &m_evaluator) ,
        p_environment{pEnvironment}
    {}

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

    /// <summary>
    /// This is an evaluator periodically launch by the training agent
    /// </summary>
    Evaluator m_evaluator{};
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
        std::make_shared<ppo::model::Aircraft>(name, side, env) };
    
    aircraft->SetSNA(sna);

    env.AddEntity(aircraft);

    return aircraft;
}

template<typename Ty>
std::pair<std::shared_ptr<ppo::model::Aircraft>, std::shared_ptr<SNATrainingAgent>>
GetNewTrainingAircraft(std::string const& name, ppo::model::eEntitySide side, Ty& env)
{
    std::shared_ptr<RLSNA> sna{ std::make_shared<RLSNA>() };

    // create the aircraft
    std::shared_ptr<ppo::model::Aircraft> aircraft{ std::make_shared<ppo::model::Aircraft>(name, side, env) };
                                 
    aircraft->SetSNA(sna);

    // Create the agent
    std::shared_ptr<SNATrainingAgent> agent{ std::make_shared<SNATrainingAgent>(
                        &env,
                        sna.get(),
                        "RLSNA",
                        RESOURCE_RLSNAMODEL_PATH) };


    env.AddEntity(aircraft);
    env.AddAgent(agent);
    
    return std::make_pair(aircraft, agent);
}

template<typename Ty>
std::pair<std::shared_ptr<ppo::model::Aircraft>, std::shared_ptr<SNAProdAgent>>
GetNewProductionAircraft(std::string const& name, ppo::model::eEntitySide side, Ty& env, std::string const& version = "prod")
{
    std::shared_ptr<RLSNA> sna{ std::make_shared<RLSNA>() };

    // create the aircraft
    std::shared_ptr<ppo::model::Aircraft> aircraft{ std::make_shared<ppo::model::Aircraft>(name, side, env) };

    aircraft->SetSNA(sna);

    std::filesystem::path modelpath{ RESOURCE_RLSNAMODEL_PATH };
    modelpath = modelpath / version / "model.pt";
    std::shared_ptr<SNAProdAgent> agent{ std::make_shared<SNAProdAgent>(sna.get(), "ProdAgent",modelpath) };

    env.AddEntity(aircraft);
    env.AddAgent(agent);
    
    return std::make_pair(aircraft, agent);
}

} /* ppo::interception */