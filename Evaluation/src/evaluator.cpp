/**
* @file Evaluation/src/evaluator.cpp
* @brief Implementation for evaluator.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "evaluator.hpp"
#include "ppo/core/multi_agent.hpp"
#include "ppo/core/evaluation.hpp"
#include "entityruntime.hpp"

#include "aircraft.hpp"
#include "entity.hpp"
#include "objective.hpp"
#include "sam.hpp"
#include "scenarioconfig.hpp"
#include "snasystem.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <map>
#include <unordered_map>
#include <vector>

#include "entt/entt.hpp"

namespace ppo::evaluation {

using Entities = std::map<std::string, std::shared_ptr<ppo::model::Entity>>;
using Agents = std::vector<std::shared_ptr<ppo::model::IAgent>>;

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
ResolveEvaluationScenarioPath(std::string const& name)
{
    std::filesystem::path path{ RESOURCE_PATH };
    path /= "Scenarios";
    path /= name + "Evaluation.json";
    path.make_preferred();
    return path;
}

/**
* @brief Writes one evaluator log line to stderr.
*/
void
LogEvaluatorEvent(std::string const& message)
{
    std::cerr << "[EVALUATOR] " << message << std::endl;
}

/**
* @brief Instantiates one entity from a parsed scenario definition.
*/
std::shared_ptr<ppo::model::Entity>
CreateEntityFromDefinition(entt::registry& world, ppo::config::ScenarioEntityDefinition const& definition)
{
    if (definition.m_type == "Aircraft")
        return std::make_shared<ppo::model::Aircraft>(&world, definition.m_platform, definition.m_side);

    if (definition.m_type == "SAM")
        return std::make_shared<ppo::model::SAM>(&world, definition.m_platform, definition.m_side);

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
RequiresAircraftState(ppo::config::EvaluationEntityState const& state)
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

/**
* @brief Registers a named factory used to create evaluator agents.
*/
void
RegisterAgentFactory(std::string_view name, AgentFactory factory)
{
    std::lock_guard<std::mutex> lock{ AgentFactoriesMutex() };
    AgentFactories()[std::string(name)] = std::move(factory);
}

struct Evaluator::Impl
{
    entt::registry                           m_world{};
    std::thread                              m_thread;
    std::atomic<bool>                        m_running{ false };
    Entities                                 m_entities{};
    Agents                                   m_agents{};
    std::shared_ptr<ppo::model::Entity>      p_evaluated{ nullptr };
    ppo::config::ScenarioDefinition          m_scenario{};
    ppo::config::EvaluationBufferDefinition  m_buffer{};
    std::filesystem::path                    m_modelpath{};
    std::string                              m_evaluatedName{};
    std::string                              m_primaryTargetName{};
    entt::entity                             m_primaryTargetId{ entt::null };
    std::unordered_map<entt::entity, std::string> m_entityNames{};
    // UI snapshots are copies. A mutex prevents readers from racing with
    // evaluator publication while preserving the public state format.
    mutable std::mutex                        m_snapshotMutex{};
    ppo::model::sEvaluatorState              m_snapshot{};

    static inline const ppo::model::sEvaluatorState EMPTY_STATE{};
    static constexpr float SECONDARY_BONUS = 0.25F;

    Impl(void);

    ~Impl(void)
    {
        Stop();
    }

    void UpdateDoubleBuffer(void);
    void Stop(void);
    void Start(std::string const& name, std::filesystem::path const& modelpath);
    void Run(void);

    bool LoadScenario(std::string const& name);
    bool ValidateScenario(void) const;
    bool ValidateBuffer(void) const;
    bool ResolveTargets(void);
    bool CreateAgents(void);
    bool ResetEntitiesForTest(std::size_t testIndex);
    void ResolveDestructions(void);
};

/**
* @brief Builds the evaluator implementation and its shared context state.
*/
Evaluator::Impl::Impl(void)
{
    m_world.ctx().emplace<ppo::model::sEnvironmentState>();
    m_world.ctx().emplace<ppo::model::sEvaluatorState>();
    UpdateDoubleBuffer();
}

/**
* @brief Publishes a lock-free evaluator snapshot for UI readers.
*/
void
Evaluator::Impl::UpdateDoubleBuffer(void)
{
    auto const& state{ m_world.ctx().get<ppo::model::sEvaluatorState>() };
    std::lock_guard<std::mutex> lock{ m_snapshotMutex };
    m_snapshot = state;
}

bool
Evaluator::Impl::ValidateScenario(void) const
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

    if (m_scenario.m_evaluation.m_checkpointFile.empty())
    {
        LogEvaluatorEvent("Evaluation.checkpointFile is empty");
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

bool
Evaluator::Impl::ValidateBuffer(void) const
{
    if (m_buffer.m_tests.empty())
    {
        LogEvaluatorEvent("evaluation buffer contains no test");
        return false;
    }

    for (auto const& test : m_buffer.m_tests)
    {
        for (auto const& entity : m_scenario.m_entities)
        {
            auto const iter = test.m_entities.find(entity.m_name);
            if (iter == test.m_entities.end())
            {
                LogEvaluatorEvent("one test is missing an entity state: " + entity.m_name);
                return false;
            }

            bool const isAircraft = entity.m_type == "Aircraft";
            bool const hasMandatoryState = isAircraft
                ? RequiresAircraftState(iter->second)
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

bool
Evaluator::Impl::ResolveTargets(void)
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
    m_primaryTargetName.clear();

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
            m_primaryTargetName = definition.m_target;
        }
    }

    if (m_primaryTargetId == entt::null)
    {
        LogEvaluatorEvent("could not resolve the evaluated entity primary target");
        return false;
    }

    return true;
}

bool
Evaluator::Impl::CreateAgents(void)
{
    m_agents.clear();

    for (auto const& definition : m_scenario.m_entities)
    {
        auto entityIter = m_entities.find(definition.m_name);
        if (entityIter == m_entities.end() || entityIter->second == nullptr)
            continue;

        for (auto const& agentName : definition.m_agents)
        {
            std::string_view version = definition.m_name == m_evaluatedName ? "eval" : "prod";
            ppo::model::BaseEntity entityHandle{ entityIter->second->registry(), entityIter->second->uniqid() };
            auto agent = CreateRegisteredAgent(agentName, entityHandle, version);
            if (agent == nullptr)
            {
                LogEvaluatorEvent("could not instantiate agent '" + agentName + "' for entity '" + definition.m_name + "'");
                return false;
            }

            m_agents.push_back(std::move(agent));
        }
    }

    return true;
}

bool
Evaluator::Impl::LoadScenario(std::string const& name)
{
    auto const scenarioPath = ResolveEvaluationScenarioPath(name);
    m_scenario = ppo::config::LoadScenarioDefinition(scenarioPath);
    if (!ValidateScenario())
        return false;

    m_buffer = ppo::config::LoadEvaluationBufferDefinition(m_scenario.m_evaluation.m_checkpointFile);
    if (!ValidateBuffer())
        return false;

    ppo::model::sEvaluatorState persistedState{};
    {
        std::lock_guard<std::mutex> lock{ m_snapshotMutex };
        persistedState = m_snapshot;
    }

    m_agents.clear();
    m_entities.clear();
    m_entityNames.clear();
    p_evaluated.reset();

    m_world = entt::registry{};

    auto& envState = m_world.ctx().emplace<ppo::model::sEnvironmentState>();
    envState.m_dt = m_scenario.m_evaluation.m_timeStep;
    envState.m_maxNorth = m_scenario.m_evaluation.m_maxNorth;
    envState.m_maxEast = m_scenario.m_evaluation.m_maxEast;

    auto& evalState = m_world.ctx().emplace<ppo::model::sEvaluatorState>();
    evalState.m_notation = persistedState.m_notation;
    evalState.m_nbtests = m_scenario.m_evaluation.m_nbTests > 0
        ? std::min<int>(m_scenario.m_evaluation.m_nbTests, static_cast<int>(m_buffer.m_tests.size()))
        : static_cast<int>(m_buffer.m_tests.size());
    evalState.m_nbstep = std::max(m_scenario.m_evaluation.m_nbSteps, 1);

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

    if (!ResolveTargets())
        return false;

    if (!CreateAgents())
        return false;

    UpdateDoubleBuffer();
    return true;
}

bool
Evaluator::Impl::ResetEntitiesForTest(std::size_t testIndex)
{
    if (testIndex >= m_buffer.m_tests.size())
        return false;

    auto const& test = m_buffer.m_tests[testIndex];
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

void
Evaluator::Impl::ResolveDestructions(void)
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

void
Evaluator::Impl::Stop(void)
{
    m_running.store(false, std::memory_order_release);
    if (m_thread.joinable())
        m_thread.join();

    if (m_world.ctx().contains<ppo::model::sEvaluatorState>())
    {
        auto& state = m_world.ctx().get<ppo::model::sEvaluatorState>();
        state.m_inprogress = false;
        state.m_progress = 0.F;
        UpdateDoubleBuffer();
    }
}

void
Evaluator::Impl::Start(std::string const& name, std::filesystem::path const& modelpath)
{
    Stop();

    if (!LoadScenario(name))
        return;

    m_modelpath = modelpath;

    std::filesystem::path evalpath{ modelpath / "eval" / "model.pt" };
    if (!std::filesystem::exists(evalpath))
    {
        LogEvaluatorEvent("missing eval model: " + evalpath.string());
        return;
    }

    if (m_world.ctx().contains<ppo::model::sEvaluatorState>())
    {
        auto& state = m_world.ctx().get<ppo::model::sEvaluatorState>();
        state.m_inprogress = true;
        state.m_progress = 0.F;
        UpdateDoubleBuffer();
    }

    LogEvaluatorEvent(
        "starting evaluation '" + name
        + "' tests=" + std::to_string(m_world.ctx().get<ppo::model::sEvaluatorState>().m_nbtests)
        + " steps=" + std::to_string(m_world.ctx().get<ppo::model::sEvaluatorState>().m_nbstep));

    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&Evaluator::Impl::Run, this);
}

void
Evaluator::Impl::Run(void)
{
    if (p_evaluated == nullptr)
        return;

    auto& state{ m_world.ctx().get<ppo::model::sEvaluatorState>() };

    float playerDestroyed{ 0.F };
    float targetDestroyed{ 0.F };
    float primaryPkill{ 0.F };
    float secondaryDestroyed{ 0.F };
    float missionDuration{ 0.F };
    float notation{ -FLT_MAX };

    state.m_inprogress = true;
    state.m_progress = 0.F;
    UpdateDoubleBuffer();

    // Scheduling is generic. Scenario resets, entity dynamics and the exact
    // scoring calculations below remain owned by this evaluation adapter.
    float testPrimaryPk{ 0.F };
    float testSecondaryDestroyed{ 0.F };
    auto const evaluationResult = ppo::core::RunEvaluation(
        static_cast<int>(state.m_nbtests),
        static_cast<int>(state.m_nbstep),
        [&]() { return m_running.load(std::memory_order_acquire); },
        [&](int testIndex)
        {
            testPrimaryPk = 0.F;
            testSecondaryDestroyed = 0.F;
            return ResetEntitiesForTest(static_cast<std::size_t>(testIndex));
        },
        [&](int, int)
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

            if (p_evaluated->has<ppo::model::sSNAState>())
            {
                auto [snaState] = p_evaluated->get<ppo::model::sSNAState>();
                testPrimaryPk = std::max(testPrimaryPk, snaState.m_pk);
            }

            bool goToNextTest{ false };
            if (p_evaluated->has<ppo::model::sEntityState>())
            {
                auto [entityState] = p_evaluated->get<ppo::model::sEntityState>();
                if (entityState.m_destroyed)
                {
                    playerDestroyed += 1.F;
                    goToNextTest = true;
                }
            }

            if (m_world.any_of<ppo::model::sEntityState>(m_primaryTargetId))
            {
                auto [targetState] = m_world.view<ppo::model::sEntityState>().get(m_primaryTargetId);
                if (targetState.m_destroyed
                &&  targetState.m_destroyedById == p_evaluated->uniqid())
                {
                    targetDestroyed += 1.F;
                    goToNextTest = true;
                }
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

                testSecondaryDestroyed += 1.F;
            }
            return goToNextTest;
        },
        [&](int testIndex, int stepIndex)
        {
            primaryPkill += testPrimaryPk;
            secondaryDestroyed += testSecondaryDestroyed;
            missionDuration += static_cast<float>(state.m_nbstep + 1) / static_cast<float>(stepIndex + 1);
            state.m_inprogress = true;
            state.m_progress = static_cast<float>(testIndex + 1) / static_cast<float>(std::max(state.m_nbtests, 1U));
            UpdateDoubleBuffer();

            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        });

    // Never compare/promote a checkpoint on an aborted or incomplete campaign.
    // Cancellation and failed scenario resets are not evaluation results.
    if (!evaluationResult.completed
        || !m_running.load(std::memory_order_acquire))
    {
        LogEvaluatorEvent("evaluation stopped before all tests completed: " + m_evaluatedName);
        state.m_inprogress = false;
        state.m_progress = 0.F;
        UpdateDoubleBuffer();
        m_running.store(false, std::memory_order_release);
        return;
    }

    notation = 100.F * (
        targetDestroyed
        + primaryPkill
        - playerDestroyed
        + missionDuration
        + SECONDARY_BONUS * secondaryDestroyed) / static_cast<float>(std::max(state.m_nbtests, 1U));

    if (state.m_notation < notation)
    {
        std::filesystem::path proddirectory{ m_modelpath / "prod" };
        std::filesystem::path evalmodel{ m_modelpath / "eval" / "model.pt" };

        if (!std::filesystem::exists(proddirectory))
            std::filesystem::create_directories(proddirectory);

        std::filesystem::path prodmodel{ proddirectory / "model.pt" };
        std::filesystem::copy_file(evalmodel, prodmodel, std::filesystem::copy_options::overwrite_existing);

        state.m_notation = notation;
    }

    LogEvaluatorEvent(
        "completed evaluation '" + m_evaluatedName
        + "' notation=" + std::to_string(state.m_notation));

    state.m_inprogress = false;
    state.m_progress = 0.F;
    UpdateDoubleBuffer();
    m_running.store(false, std::memory_order_release);
}

Evaluator::Evaluator(void) :
    p_impl{ std::make_unique<Impl>() }
{
}

Evaluator::~Evaluator(void) = default;

ppo::model::sEvaluatorState
Evaluator::GetSnapshot(void) const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_STATE;

    std::lock_guard<std::mutex> lock{ p_impl->m_snapshotMutex };
    return p_impl->m_snapshot;
}

void
Evaluator::Start(std::string const& name, std::filesystem::path const& modelpath)
{
    if (p_impl == nullptr)
        return;

    p_impl->Start(name, modelpath);
}

void
Evaluator::Stop(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->Stop();
}

} /* ppo::evaluation */
