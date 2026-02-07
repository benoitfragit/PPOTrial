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

namespace rl {

struct StatesAndActions {
    explicit StatesAndActions(ppo::model::BaseEntity& entity )
    {
        m_entity = entity;
    };

    using STATES = std::array<float, 6>;
    using ACTIONS = std::array<float, 2>;

    STATES  m_states{};
    ACTIONS m_actions{};

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

    void SetNextAction(ACTIONS const& action) noexcept;
    void SetNewState(STATES& state) noexcept;

    ppo::model::BaseEntity m_entity{ nullptr };
};

/// <summary>
/// This is the definitionn of the agent used for training
/// </summary>
class SNATrainingAgent :
    public ppo::model::TrainingAgent<ActorCriticNet, StatesAndActions>
{
public:
    using Base = ppo::model::TrainingAgent<ActorCriticNet, StatesAndActions>;

    explicit SNATrainingAgent(ppo::model::BaseEntity& entity) :
        Base(&(entity.add<StatesAndActions>(entity)), "RLSNA", RESOURCE_RLSNAMODEL_PATH, &m_evaluator)
    {
        m_entity = entity;
    };

private:
    /// <summary>
    /// this function is used to compute the reward at each step
    /// </summary>
    /// <param name=""></param>
    /// <returns></returns>
    bool RewardFunction(void) noexcept final;

    /// <summary>
    /// this is the entity affected by this agent
    /// </summary>
    ppo::model::BaseEntity m_entity{nullptr};

    /// <summary>
    /// This is an evaluator periodically launch by the training agent
    /// </summary>
    Evaluator m_evaluator{};
};

/// <summary>
/// This is the definition of the agent used for production
/// </summary>
class SNAProdAgent :
    public ppo::model::ProdAgent<ActorCriticNetProd, StatesAndActions>
{
public:
    using Base = ppo::model::ProdAgent<ActorCriticNetProd, StatesAndActions>;

    SNAProdAgent(ppo::model::BaseEntity& entity, std::string const& version = "prod") :
        Base(&(entity.add<StatesAndActions>(entity)), "RLSNAProdAgent", std::filesystem::path(RESOURCE_RLSNAMODEL_PATH) / version / "model.pt"){};
};

} /* rl */