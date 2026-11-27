/**
 * @file Examples/Airspace/Evaluation/evaluation_campaign.cpp
 * @brief The cases of an Airspace evaluation, one step at a time (see the header).
 */
#include "evaluation_campaign.hpp"

#include "airspacecomponents.hpp"
#include "entityruntime.hpp"
#include "mobile.hpp"
#include "objective.hpp"
#include "ppo/core/multi_agent.hpp"
#include "restrictedzone.hpp"
#include "snasystem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string_view>

namespace ppo::evaluation {

namespace {

/**
* @brief Returns the mutable global factory registry.
*/
std::unordered_map<std::string, AgentFactory>&
AgentFactories(void)
{
    static std::unordered_map<std::string, AgentFactory> factories{};
    return factories;
}

/**
* @brief Returns the mutex protecting the factory registry.
*/
std::mutex&
AgentFactoriesMutex(void)
{
    static std::mutex mutex{};
    return mutex;
}

/**
* @brief Creates one registered evaluation agent by name.
*/
std::shared_ptr<ppo::model::IAgent>
CreateRegisteredAgent(std::string const& name,
                      ppo::model::BaseEntity& entity,
                      std::string_view version)
{
    std::lock_guard<std::mutex> lock{ AgentFactoriesMutex() };

    auto const iter = AgentFactories().find(name);
    if (iter == AgentFactories().end() || !iter->second)
        return nullptr;

    return iter->second(entity, version);
}

/**
* @brief Resolves the evaluation scenario path from the agent family name.
*/
std::filesystem::path
ScenarioPathOf(std::string const& name)
{
    return std::filesystem::path{ PPO_AIRSPACE_SETTINGS }
        / "Scenarios" / "Evaluation" / (name + ".json");
}


/**
* @brief Instantiates one entity from a parsed scenario definition.
*/
std::shared_ptr<ppo::model::Entity>
CreateEntityFromDefinition(entt::registry& world, ppo::config::ScenarioEntityDefinition const& definition)
{
    if (definition.m_type == "Mobile" || definition.m_type == "Aircraft")
        return std::make_shared<ppo::model::Mobile>(&world, definition.m_platform, definition.m_side);

    if (definition.m_type == "RestrictedZone" || definition.m_type == "SAM")
        return std::make_shared<ppo::model::RestrictedZone>(&world, definition.m_platform, definition.m_side);

    if (definition.m_type == "Objective")
        return std::make_shared<ppo::model::Objective>(&world, definition.m_side);

    return nullptr;
}

/**
* @brief Returns true when a test state defines planar coordinates.
*/
bool
RequiresNorthEast(ppo::config::EvaluationEntityState const& state)
{
    return state.m_north.has_value() && state.m_east.has_value();
}

/**
* @brief Returns true when a test state defines the minimum aircraft kinematics.
*/
bool
RequiresMobileState(ppo::config::EvaluationEntityState const& state)
{
    return RequiresNorthEast(state)
        && state.m_heading.has_value()
        && state.m_speed.has_value();
}

/**
* @brief Applies one parsed evaluation state to one runtime entity.
*/
void
ApplyInitialState(std::shared_ptr<ppo::model::Entity> const& entity,
                  ppo::config::EvaluationEntityState const& initialState)
{
    if (entity == nullptr)
        return;

    if (entity->has<ppo::model::sEntityState>())
    {
        auto [entityState] = entity->get<ppo::model::sEntityState>();

        if (initialState.m_north.has_value())
            entityState.m_x = *initialState.m_north;

        if (initialState.m_east.has_value())
            entityState.m_y = *initialState.m_east;

        if (initialState.m_heading.has_value())
            entityState.m_rotation = *initialState.m_heading;

        if (initialState.m_speed.has_value())
            entityState.m_speed = *initialState.m_speed;

        entityState.m_dx = entityState.m_speed * std::cos(entityState.m_rotation);
        entityState.m_dy = entityState.m_speed * std::sin(entityState.m_rotation);
        entityState.m_destroyed = false;
        entityState.m_prevDestroyed = false;
        entityState.m_destroyedById = entt::null;
    }

    if (entity->has<ppo::model::sFlightModelState>())
    {
        auto [flightModelState] = entity->get<ppo::model::sFlightModelState>();

        flightModelState.m_accpsi = initialState.m_accpsi.value_or(0.F);
        flightModelState.m_acc = initialState.m_acc.value_or(0.F);
    }
}

} // namespace

void
LogEvaluatorEvent(std::string const& message)
{
    std::cerr << "[EVALUATOR] " << message << std::endl;
}

/**
* @brief Registers a named factory used to create evaluator agents.
*/
void
RegisterAgentFactory(std::string_view name, AgentFactory factory)
{
    std::lock_guard<std::mutex> lock{ AgentFactoriesMutex() };
    AgentFactories()[std::string(name)] = std::move(factory);
}

void
CaseResult::Add(CaseStep const& step) noexcept
{
    if (step.playerDestroyed)
        playerDestroyed += 1.F;
    if (step.targetDestroyed)
        targetDestroyed += 1.F;
    primaryPk = std::max(primaryPk, step.primaryPk);
    secondaryDestroyed += step.secondaryDestroyed;
}

void
CaseResult::Finish(int lastStep, int stepLimit) noexcept
{
    duration = static_cast<float>(stepLimit + 1) / static_cast<float>(lastStep + 1);
    timeout = lastStep >= stepLimit;
}

float
CaseResult::Score() const noexcept
{
    return 100.F * (targetDestroyed + primaryPk - playerDestroyed + duration + SecondaryBonus * secondaryDestroyed);
}

bool
EvaluationCampaign::Load(std::string const& name, EvaluatedAgentFactory const& evaluatedAgent)
{
    m_scenario = ppo::config::LoadScenarioDefinition(ScenarioPathOf(name));
    if (!ValidateScenario())
        return false;

    m_cases = ppo::config::LoadEvaluationBufferDefinition(m_scenario.m_evaluation.m_casesFile);
    if (!ValidateCases())
        return false;

    m_agents.clear();
    m_evaluatedAgents.clear();
    m_entities.clear();
    m_entityNames.clear();
    p_evaluated.reset();

    m_world = entt::registry{};
    auto& environment = m_world.ctx().emplace<ppo::model::sEnvironmentState>();
    environment.m_dt = m_scenario.m_evaluation.m_timeStep;
    environment.m_maxNorth = m_scenario.m_evaluation.m_maxNorth;
    environment.m_maxEast = m_scenario.m_evaluation.m_maxEast;
    m_caseCount = m_scenario.m_evaluation.m_nbTests > 0
        ? std::min<int>(m_scenario.m_evaluation.m_nbTests, static_cast<int>(m_cases.m_tests.size()))
        : static_cast<int>(m_cases.m_tests.size());
    m_stepLimit = std::max(m_scenario.m_evaluation.m_nbSteps, 1);
    m_evaluatedName = m_scenario.m_evaluation.m_evaluatedEntity;

    for (auto const& definition : m_scenario.m_entities)
    {
        auto entity = CreateEntityFromDefinition(m_world, definition);
        if (entity == nullptr)
            continue;

        entity->add<ppo::model::sEntityMetadata>(ppo::model::sEntityMetadata{
            definition.m_name,
            definition.m_platform.empty() ? definition.m_type : definition.m_platform
        });

        m_entityNames[entity->uniqid()] = definition.m_name;
        m_entities.emplace(definition.m_name, entity);
        if (definition.m_name == m_evaluatedName)
            p_evaluated = entity;
    }

    if (p_evaluated == nullptr)
    {
        LogEvaluatorEvent("evaluated entity instance could not be created");
        return false;
    }

    return ResolveTargets() && CreateAgents(evaluatedAgent);
}

/** @brief The scenario names its evaluated entity, with an agent and a primary target, and a cases file. */
bool
EvaluationCampaign::ValidateScenario(void) const
{
    if (!m_scenario.m_evaluation.m_enabled)
    {
        LogEvaluatorEvent("scenario is missing Evaluation section");
        return false;
    }

    if (m_scenario.m_evaluation.m_evaluatedEntity.empty())
    {
        LogEvaluatorEvent("Evaluation.evaluatedEntity is empty");
        return false;
    }

    if (m_scenario.m_evaluation.m_casesFile.empty())
    {
        LogEvaluatorEvent("evaluation.casesFile is empty");
        return false;
    }

    for (auto const& entity : m_scenario.m_entities)
    {
        if (entity.m_name != m_scenario.m_evaluation.m_evaluatedEntity)
            continue;

        if (entity.m_agents.empty())
        {
            LogEvaluatorEvent("evaluated entity has no agent in scenario");
            return false;
        }

        if (entity.m_target.empty())
        {
            LogEvaluatorEvent("evaluated entity has no primary target");
            return false;
        }

        return true;
    }

    LogEvaluatorEvent("evaluated entity was not found in scenario");
    return false;
}

/** @brief Every case gives every entity of the scenario its mandatory state. */
bool
EvaluationCampaign::ValidateCases(void) const
{
    if (m_cases.m_tests.empty())
    {
        LogEvaluatorEvent("evaluation buffer contains no test");
        return false;
    }

    for (auto const& test : m_cases.m_tests)
    {
        for (auto const& entity : m_scenario.m_entities)
        {
            auto const iter = test.m_entities.find(entity.m_name);
            if (iter == test.m_entities.end())
            {
                LogEvaluatorEvent("one test is missing an entity state: " + entity.m_name);
                return false;
            }

            bool const isMobile = entity.m_type == "Aircraft" || entity.m_type == "Mobile";
            bool const hasMandatoryState = isMobile
                ? RequiresMobileState(iter->second)
                : RequiresNorthEast(iter->second);
            if (!hasMandatoryState)
            {
                LogEvaluatorEvent("one test has an incomplete state for entity: " + entity.m_name);
                return false;
            }
        }
    }

    return true;
}

/** @brief Gives every SNA its target; the evaluated entity's is the primary target. */
bool
EvaluationCampaign::ResolveTargets(void)
{
    std::map<std::string, entt::entity> entityIds{};
    for (auto const& [name, entity] : m_entities)
    {
        if (entity == nullptr)
            continue;

        entityIds[name] = entity->uniqid();
        m_entityNames[entity->uniqid()] = name;
    }

    m_primaryTargetId = entt::null;

    for (auto const& definition : m_scenario.m_entities)
    {
        if (definition.m_target.empty())
            continue;

        auto entityIter = m_entities.find(definition.m_name);
        auto targetIter = entityIds.find(definition.m_target);
        if (entityIter == m_entities.end()
        ||  entityIter->second == nullptr
        ||  targetIter == entityIds.end()
        ||  !entityIter->second->has<ppo::model::sSNAState>())
            continue;

        auto [snaState] = entityIter->second->get<ppo::model::sSNAState>();
        snaState.m_target = targetIter->second;

        if (definition.m_name == m_evaluatedName)
        {
            m_primaryTargetId = targetIter->second;
        }
    }

    if (m_primaryTargetId == entt::null)
    {
        LogEvaluatorEvent("could not resolve the evaluated entity primary target");
        return false;
    }

    return true;
}

/** @brief The scenario's agents: "eval" (or `evaluatedAgent`) for the evaluated entity, "prod" for the others. */
bool
EvaluationCampaign::CreateAgents(EvaluatedAgentFactory const& evaluatedAgent)
{
    for (auto const& definition : m_scenario.m_entities)
    {
        auto entityIter = m_entities.find(definition.m_name);
        if (entityIter == m_entities.end() || entityIter->second == nullptr)
            continue;

        bool const evaluated = definition.m_name == m_evaluatedName;
        for (auto const& agentName : definition.m_agents)
        {
            std::string_view version = evaluated ? "eval" : "prod";
            ppo::model::BaseEntity entityHandle{ entityIter->second->registry(), entityIter->second->uniqid() };
            auto agent = evaluated && evaluatedAgent
                ? evaluatedAgent(entityHandle)
                : CreateRegisteredAgent(agentName, entityHandle, version);
            if (agent == nullptr)
            {
                LogEvaluatorEvent("could not instantiate agent '" + agentName + "' for entity '" + definition.m_name + "'");
                return false;
            }

            if (evaluated)
                m_evaluatedAgents.push_back(agent);
            m_agents.push_back(std::move(agent));
        }
    }

    return true;
}

bool
EvaluationCampaign::ResetCase(std::size_t index)
{
    if (index >= m_cases.m_tests.size())
        return false;

    auto const& test = m_cases.m_tests[index];
    for (auto const& [name, entity] : m_entities)
    {
        if (entity == nullptr)
            continue;

        entity->Init();

        auto const stateIter = test.m_entities.find(name);
        if (stateIter == test.m_entities.end())
        {
            LogEvaluatorEvent("missing test state for entity: " + name);
            return false;
        }

        ApplyInitialState(entity, stateIter->second);
    }

    ppo::core::ResetAgents(
        m_agents, [](ppo::model::IAgent& agent) { agent.Init(); });

    return true;
}

/** @brief An SNA whose Pk on another entity exceeds 0.95 destroys it. */
void
EvaluationCampaign::ResolveDestructions(void)
{
    for (auto& [_, entity] : m_entities)
    {
        if (entity == nullptr
        ||  !entity->has<ppo::model::sEntityState, ppo::model::sSNAState>())
            continue;

        auto const [entityState, __] = entity->get<ppo::model::sEntityState, ppo::model::sSNAState>();
        if (entityState.m_destroyed)
            continue;

        for (auto& [__, otherEntity] : m_entities)
        {
            if (otherEntity == entity
            ||  otherEntity == nullptr
            || !otherEntity->has<ppo::model::sEntityState>())
                continue;

            auto [otherEntityState] = otherEntity->get<ppo::model::sEntityState>();

            if (otherEntityState.m_destroyed)
            {
                otherEntityState.m_prevDestroyed = true;
                continue;
            }

            if (float const pk = ppo::model::SNASystem::EvaluateDestruction(*entity.get(), *otherEntity.get());
                0.95F < pk)
            {
                otherEntityState.m_destroyed = true;
                otherEntityState.m_destroyedById = entity->uniqid();
            }
        }
    }
}

CaseStep
EvaluationCampaign::Step()
{
    // All agents act before any entity advances, then all observe.
    (void)ppo::core::StepAgents(
        m_agents,
        [](ppo::model::IAgent& agent) { agent.ChooseAction(); },
        [&]()
        {
            ppo::model::AdvanceLivingEntities(
                m_entities, [](auto& entry) -> auto& { return entry.second; });
        },
        [&]() { ResolveDestructions(); },
        [](ppo::model::IAgent& agent) { return agent.SetNewState(); });

    CaseStep step{};
    if (p_evaluated->has<ppo::model::sSNAState>())
    {
        auto [snaState] = p_evaluated->get<ppo::model::sSNAState>();
        step.primaryPk = snaState.m_pk;
    }

    if (p_evaluated->has<ppo::model::sEntityState>())
    {
        auto [entityState] = p_evaluated->get<ppo::model::sEntityState>();
        step.playerDestroyed = entityState.m_destroyed;
    }

    if (m_world.any_of<ppo::model::sEntityState>(m_primaryTargetId))
    {
        auto [targetState] = m_world.view<ppo::model::sEntityState>().get(m_primaryTargetId);
        step.targetDestroyed = targetState.m_destroyed
            && targetState.m_destroyedById == p_evaluated->uniqid();
    }

    auto view = m_world.view<ppo::model::sEntityState>();
    for (auto entityId : view)
    {
        auto [entityState] = view.get(entityId);
        if (!entityState.m_destroyed
        ||  entityState.m_prevDestroyed
        ||  entityState.m_destroyedById != p_evaluated->uniqid()
        ||  entityState.m_side != ppo::model::eEntitySide::Hostile
        ||  entityId == m_primaryTargetId)
            continue;

        step.secondaryDestroyed += 1.F;
    }
    return step;
}

} // namespace ppo::evaluation
