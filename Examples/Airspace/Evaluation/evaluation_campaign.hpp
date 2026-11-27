/**
 * @file Examples/Airspace/Evaluation/evaluation_campaign.hpp
 * @brief The cases of an Airspace evaluation, one step at a time: what the evaluator plays
 *        on its thread and what the environment DLLs play for ppokit-run's suites.
 * @details A campaign loads Scenarios/Evaluation/<agent>.json and its cases file into its
 * own EnTT world (not the Airspace world's rules: its own resolution of destructions). A case
 * resets every entity to its state in the cases file and re-initializes the agents; a step
 * lets every agent choose, advances the living entities, resolves the destructions and lets
 * every agent observe. A case ends when the evaluated entity is destroyed, when it destroys
 * its primary target, or after the step limit. Its result scores as in the evaluator:
 * 100 x (target destroyed + primary Pk - player destroyed + duration term
 * + 0.25 x secondary targets destroyed), the duration term being
 * (step limit + 1) / (last step index + 1).
 */
#pragma once

#include "entity.hpp"
#include "ppo/runtime/iagent.hpp"
#include "scenarioconfig.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "entt/entt.hpp"

namespace ppo::evaluation {

/** @brief Creates the agent of an evaluation scenario's entity (opponents), for a model version. */
using AgentFactory = std::function<std::shared_ptr<ppo::model::IAgent>(ppo::model::BaseEntity&, std::string_view)>;

/**
 * @brief Registers the factory of an agent name of the evaluation scenarios.
 * @param name Agent name in the scenario (e.g. TargetSelectorProdAgent).
 * @param factory Its factory.
 */
void RegisterAgentFactory(std::string_view name, AgentFactory factory);

/** @brief What one step of a case changed for the evaluated entity. */
struct CaseStep final
{
    /** @brief The evaluated entity is destroyed. */
    bool playerDestroyed = false;
    /** @brief The evaluated entity destroyed its primary target. */
    bool targetDestroyed = false;
    /** @brief Probability of kill of the evaluated SNA after the step (0 without one). */
    float primaryPk = 0.F;
    /** @brief Hostile secondary targets the evaluated entity destroyed at this step. */
    float secondaryDestroyed = 0.F;

    /** @brief The case ends at this step. */
    bool Ends() const noexcept { return playerDestroyed || targetDestroyed; }
};

/** @brief The result of one case, accumulated step by step. */
struct CaseResult final
{
    float playerDestroyed = 0.F;
    float targetDestroyed = 0.F;
    /** @brief Largest primary Pk of the case. */
    float primaryPk = 0.F;
    float secondaryDestroyed = 0.F;
    /** @brief (step limit + 1) / (last step index + 1): 1 for a case that reached its limit. */
    float duration = 0.F;
    /** @brief The case reached its step limit. */
    bool timeout = false;

    /** @brief Adds one step. */
    void Add(CaseStep const& step) noexcept;
    /**
     * @brief Closes the case.
     * @param lastStep Index of the step that ended it, or the step limit when none did.
     * @param stepLimit Steps of a case.
     */
    void Finish(int lastStep, int stepLimit) noexcept;
    /** @brief The case's score, the evaluator's campaign score for a single case. */
    float Score() const noexcept;
};

/** @brief Weight of a destroyed secondary target in the score. */
constexpr float SecondaryBonus = 0.25F;

/** @brief Loads an evaluation's scenario and cases, and plays its cases step by step. */
class EvaluationCampaign final
{
public:
    /** @brief Creates the agent of the evaluated entity (instead of the scenario's, with version "eval"). */
    using EvaluatedAgentFactory = std::function<std::shared_ptr<ppo::model::IAgent>(ppo::model::BaseEntity&)>;

    /**
     * @brief Loads Scenarios/Evaluation/<name>.json, its cases, its entities and agents.
     * @param name Agent family (RLMission, RLSNA).
     * @param evaluatedAgent Agent of the evaluated entity; when empty, its scenario agent
     *        with the "eval" policy (the other entities play "prod").
     * @return False, with a message, when the scenario or its cases are not usable.
     */
    bool Load(std::string const& name, EvaluatedAgentFactory const& evaluatedAgent = {});

    /** @brief The campaign's world (its context holds sEnvironmentState). */
    entt::registry& World() noexcept { return m_world; }
    /** @brief The campaign's world, read only. */
    entt::registry const& World() const noexcept { return m_world; }
    /** @brief Cases played by a campaign: the scenario's nbTests, at most the cases of the file. */
    int CaseCount() const noexcept { return m_caseCount; }
    /** @brief Steps of a case (nbSteps, at least 1). */
    int StepLimit() const noexcept { return m_stepLimit; }
    /** @brief The evaluated entity, after Load. */
    std::shared_ptr<ppo::model::Entity> const& Evaluated() const noexcept { return p_evaluated; }
    /** @brief The agents of the evaluated entity, in scenario order (after Load). */
    std::vector<std::shared_ptr<ppo::model::IAgent>> const& EvaluatedAgents() const noexcept
    {
        return m_evaluatedAgents;
    }

    /**
     * @brief Starts case `index`: entities to their state in the cases file, agents re-initialized.
     * @return False, with a message, for an unknown case or an entity without its state.
     */
    bool ResetCase(std::size_t index);
    /** @brief One step of every agent and entity; what it changed for the evaluated entity. */
    CaseStep Step();

private:
    bool ValidateScenario() const;
    bool ValidateCases() const;
    bool ResolveTargets();
    bool CreateAgents(EvaluatedAgentFactory const& evaluatedAgent);
    void ResolveDestructions();

    entt::registry m_world{};
    std::map<std::string, std::shared_ptr<ppo::model::Entity>> m_entities{};
    std::vector<std::shared_ptr<ppo::model::IAgent>> m_agents{};
    std::vector<std::shared_ptr<ppo::model::IAgent>> m_evaluatedAgents{};
    std::shared_ptr<ppo::model::Entity> p_evaluated{nullptr};
    ppo::config::ScenarioDefinition m_scenario{};
    ppo::config::EvaluationBufferDefinition m_cases{};
    std::string m_evaluatedName{};
    entt::entity m_primaryTargetId{entt::null};
    std::unordered_map<entt::entity, std::string> m_entityNames{};
    int m_caseCount = 0;
    int m_stepLimit = 1;
};

/** @brief Writes one evaluator log line to stderr. */
void LogEvaluatorEvent(std::string const& message);

} // namespace ppo::evaluation
