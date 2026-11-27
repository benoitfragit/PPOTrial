/**
* @file Examples/Airspace/Agents/Interception/interception.hpp
* @brief Declarations for Interception.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include "aircraft.hpp"
#include "graph_network.hpp"
#include "trainingagent.hpp"
#include "prodagent.hpp"
#include "ppo/core/agent_contract.hpp"
#include "evaluation_campaign.hpp"

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

    // Frozen semantic layouts for the existing PPO model.pt archives.
    // Changing either layout ID requires a new trained policy family.
    inline static constexpr ppo::core::AgentContract kAgentContract =
        ppo::core::MakeAgentContract<STATES, ACTIONS>(
            "RLSNA", "rlsna.observation.v1", "rlsna.action.v1");

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

/** @brief task.json and graph.json of the RLSNA networks (PPOKit training). */
ppo::model::GraphNetworkFiles InterceptionNetworkFiles();

/**
 * @brief Production policy of a model version ("prod" or "eval").
 * @details A policy trained with PPOKit (policy.ppokit) when there is one, else the policy
 *          converted from LibTorch (policy.txt); null, and an idle agent, when neither exists.
 */
std::unique_ptr<ppo::core::IPolicy> MakeInterceptionPolicy(std::string const& version);

/**
* @brief Training agent used for the interception task.
*/
class SNATrainingAgent :
    public ppo::model::TrainingAgent<StatesAndActions>
{
public:
    /**
    * @brief Base training-agent alias.
    */
    using Base = ppo::model::TrainingAgent<StatesAndActions>;

    /**
    * @brief Creates the interception training agent for one entity.
    * @param entity Controlled entity wrapper.
    */
    explicit SNATrainingAgent(ppo::model::BaseEntity& entity) :
        Base(&(entity.add<StatesAndActions>(entity)), "RLSNA")
    {
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
};

/**
* @brief Production agent used for interception inference and evaluation.
*/
class SNAProdAgent :
    public ppo::model::ProdAgent<StatesAndActions>
{
public:
    /**
    * @brief Base production-agent alias.
    */
    using Base = ppo::model::ProdAgent<StatesAndActions>;

    /**
    * @brief Creates the interception production agent for one entity.
    * @param entity Controlled entity wrapper.
    * @param version Model subfolder to load, usually prod or eval.
    */
    SNAProdAgent(ppo::model::BaseEntity& entity, std::string const& version = "prod") :
        Base(&(entity.add<StatesAndActions>(entity)), "RLSNAProdAgent", MakeInterceptionPolicy(version))
    {
    }

    /**
    * @brief The same agent playing a given policy (ppokit-run's evaluation through the environment DLL).
    * @param entity Controlled entity wrapper.
    * @param policy Policy to play.
    */
    SNAProdAgent(ppo::model::BaseEntity& entity, std::unique_ptr<ppo::core::IPolicy> policy) :
        Base(&(entity.add<StatesAndActions>(entity)), "RLSNAProdAgent", std::move(policy))
    {
    }
};

/**
* @brief Registers the production agent factory of the evaluation scenarios (their opponents).
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
