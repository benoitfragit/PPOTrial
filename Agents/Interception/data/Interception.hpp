/**
* @file Agents/Interception/data/Interception.hpp
* @brief Declarations for Interception.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include "aircraft.hpp"
#include "actorcriticnet.hpp"
#include "trainingagent.hpp"
#include "prodagent.hpp"
#include "evaluator.hpp"

#include <map>
#include <memory>
#include <string>
#include <array>
#include <type_traits>

namespace rl {

/**
* @brief Registers evaluation-time factories required by the interception agent.
*/
void EnsureInterceptionEvaluationFactoriesRegistered(void);

/**
* @brief State and action adapter used by interception agents.
*/
struct StatesAndActions
{
    /**
    * @brief Builds the adapter for one ECS entity.
    * @param entity Controlled entity wrapper.
    */
    explicit StatesAndActions(ppo::model::BaseEntity& entity)
    {
        m_entity = entity;
    }

    /**
    * @brief Flattened interception state vector type.
    */
    using STATES = std::array<float, 6>;
    /**
    * @brief Interception action vector type.
    */
    using ACTIONS = std::array<float, 2>;

    /**
    * @brief Current state buffer.
    */
    STATES m_states{};
    /**
    * @brief Current action buffer.
    */
    ACTIONS m_actions{};

    /**
    * @brief Reward channels used by the interception agent.
    */
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

    /**
    * @brief Human-readable reward channel names.
    */
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

    /**
    * @brief Applies the chosen action to the controlled entity.
    * @param action Action chosen by the policy.
    */
    void SetNextAction(ACTIONS const& action) noexcept;
    /**
    * @brief Builds the latest observation vector from the world state.
    * @param state Output state buffer.
    */
    void SetNewState(STATES& state) noexcept;

    /**
    * @brief Controlled entity wrapper.
    */
    ppo::model::BaseEntity m_entity{ nullptr };
};

/**
* @brief Training agent used for the interception task.
*/
class SNATrainingAgent :
    public ppo::model::TrainingAgent<ActorCriticNet, StatesAndActions>
{
public:
    /**
    * @brief Base training-agent alias.
    */
    using Base = ppo::model::TrainingAgent<ActorCriticNet, StatesAndActions>;

    /**
    * @brief Creates the interception training agent for one entity.
    * @param entity Controlled entity wrapper.
    */
    explicit SNATrainingAgent(ppo::model::BaseEntity& entity) :
        Base(&(entity.add<StatesAndActions>(entity)), "RLSNA", RESOURCE_RLSNAMODEL_PATH, nullptr)
    {
        EnsureInterceptionEvaluationFactoriesRegistered();
        SetEvaluator(&m_evaluator);
        m_entity = entity;
    }

private:
    /**
    * @brief Computes the interception reward decomposition for one step.
    * @return True when the episode should stop.
    */
    bool RewardFunction(void) noexcept final;

    /**
    * @brief Controlled entity wrapper.
    */
    ppo::model::BaseEntity m_entity{ nullptr };
    /**
    * @brief Evaluator launched periodically by the training flow.
    */
    ppo::evaluation::Evaluator m_evaluator{};
};

/**
* @brief Production agent used for interception inference and evaluation.
*/
class SNAProdAgent :
    public ppo::model::ProdAgent<ActorCriticNetProd, StatesAndActions>
{
public:
    /**
    * @brief Base production-agent alias.
    */
    using Base = ppo::model::ProdAgent<ActorCriticNetProd, StatesAndActions>;

    /**
    * @brief Creates the interception production agent for one entity.
    * @param entity Controlled entity wrapper.
    * @param version Model subfolder to load, usually prod or eval.
    */
    SNAProdAgent(ppo::model::BaseEntity& entity, std::string const& version = "prod") :
        Base(&(entity.add<StatesAndActions>(entity)),
             "RLSNAProdAgent",
             std::filesystem::path(RESOURCE_RLSNAMODEL_PATH) / version / "model.pt")
    {
    }
};

/**
* @brief Registers the production agent factory used by the evaluator.
*/
inline void
EnsureInterceptionEvaluationFactoriesRegistered(void)
{
    static bool const registered = []()
    {
        ppo::evaluation::RegisterAgentFactory(
            "SNAProdAgent",
            [](ppo::model::BaseEntity& entity, std::string_view version)
            {
                return std::make_shared<SNAProdAgent>(entity, std::string(version));
            });
        return true;
    }();

    (void)registered;
}

} /* rl */
