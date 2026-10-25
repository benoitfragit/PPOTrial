/**
* @file Env/src/environment.cpp
* @brief Implementation for environment.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "environment.hpp"
#include "ppo/core/multi_agent.hpp"
#include "entityruntime.hpp"
#include "utils.hpp"
#include "serializer.hpp"

#include <map>
#include <thread>
#include <atomic>
#include <chrono>
#include <array>
#include <algorithm>
#include <vector>
#include <mutex>

namespace ppo::model {

/**
* @brief Convenience alias for managed entity instances.
*/
using Entities = std::vector<std::shared_ptr<Entity>>;
/**
* @brief Convenience alias for managed agent instances.
*/
using Agents = std::vector<std::shared_ptr<IAgent>>;

/**
* @brief Internal execution states of the environment thread.
*/
enum class eEnvironmentState : unsigned char {
    INIT = 0,
    RUNNING = 1,
    PAUSED = 2,
    STOPPED = 3
};

/**
* @brief Internal command exchanged with the environment thread.
*/
struct sRealTimeCommand {
    enum class eTypeCommand {
        None,
        ChangeState,
        MaxRange,
        TimeStep
    };
    eTypeCommand           m_type{ eTypeCommand::None };
    eEnvironmentState       m_state{ eEnvironmentState::INIT };
    float                   m_timeStep{ 0.F };
    std::array<float, 2>    m_maxRanges{ 0.F, 0.F };
};

/**
* @brief Hidden implementation owning runtime state and the worker thread.
*/
struct Environment::Impl
{
    void                    Run           (void);
    void                    Stop          (void);
    void                    Start         (void);
    void                    CreateSnapshot(void);

    void                    ProcessCommand(sRealTimeCommand const& command);

                            Impl(void);

                            ~Impl(void);

    entt::registry                   m_world{};
    entt::dispatcher                 m_dispatcher{};
    Entities                         m_entities{};
    Agents                           m_agents{};
    std::array<entt::registry, 2>    m_buffers{};
    std::atomic<uint32_t>            m_index{ 0 };
    mutable std::mutex               m_snapshotMutex{};
    std::thread                     m_thread;
    std::unique_ptr<IWorldRules>     m_worldRules{};
    Entity::InitializerSystem       m_entityInitializer{ nullptr };

    std::atomic<bool>               m_running           { false };  
    eEnvironmentState               m_state             { eEnvironmentState::INIT };
    eEnvironmentState               m_prevState         { m_state };
};

Environment::Impl::Impl(void)
{
    // Create the shared environment-state context before any entity is added.
    m_world.ctx().emplace<sEnvironmentState>();

    // Route internal control commands through the dispatcher queue.
    m_dispatcher.sink<sRealTimeCommand>().connect<&Environment::Impl::ProcessCommand>(this);

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        "Environment.json",
        [this](ppo::serialization::json& ar)
        {
            auto& state = m_world.ctx().get<sEnvironmentState>();

            auto section = ar["environment"];

            state.m_maxNorth = section["maxNorth"];
            state.m_maxEast  = section["maxEast"];
            state.m_dt       = section["timeStep"];

            CreateSnapshot();
        },
        [this](ppo::serialization::json& ar)
        {
            auto const& state = m_world.ctx().get<sEnvironmentState>();

            auto& section = ar["environment"];

            section["maxNorth"] = state.m_maxNorth ;
            section["maxEast"]  = state.m_maxEast  ;
            section["timeStep"] = state.m_dt       ;
        }
    );
}

/**
* @brief Applies one queued real-time command to the environment state.
*/
void
Environment::Impl::ProcessCommand(sRealTimeCommand const& command)
{
    auto& environmentState{ m_world.ctx().get<sEnvironmentState>() };

    switch (command.m_type)
    {
    case sRealTimeCommand::eTypeCommand::TimeStep:
        environmentState.m_dt = command.m_timeStep;
        break;
    case sRealTimeCommand::eTypeCommand::MaxRange:
        environmentState.m_maxNorth = command.m_maxRanges[0];
        environmentState.m_maxEast = command.m_maxRanges[1];
        break;
    case sRealTimeCommand::eTypeCommand::ChangeState:
        m_state = command.m_state;
        break;
    default:
        break;
    }
}

/**
* @brief Publishes a double-buffered registry snapshot for renderers and tools.
*/
void 
Environment::Impl::CreateSnapshot(void)
{
    std::lock_guard<std::mutex> lock{ m_snapshotMutex };
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    // Rebuild the target snapshot from scratch to keep it fully self-consistent.
    m_buffers[next].clear();
    m_buffers[next].ctx().clear();

    for (auto entity : m_world.storage<entt::entity>())
        auto newentity = m_buffers[next].create(entity);

    m_world.view<sEntityState>().each([&](auto entity, const auto& comp)
    {
        m_buffers[next].emplace<sEntityState>(entity, comp);
    });

    m_world.view<sEntityMetadata>().each([&](auto entity, const auto& comp)
    {
        m_buffers[next].emplace<sEntityMetadata>(entity, comp);
    });

    m_buffers[next].ctx().emplace<sEnvironmentState>(m_world.ctx().get<sEnvironmentState>());
    if (m_worldRules != nullptr)
        m_worldRules->CopySnapshot(m_world, m_buffers[next]);

    for (auto agent : m_agents)
    {
        auto const agentSnapshot(agent->GetSnapshot());
        
        // Broadcast each agent snapshot through a synthetic snapshot entity.
        entt::entity entity_agent = m_buffers[next].create();
        m_buffers[next].emplace<sAgentStateSnaphot>(entity_agent, agentSnapshot);
    }

    m_index.store(next, std::memory_order_release);
}

/**
* @brief Stops the worker thread when the implementation is destroyed.
*/
Environment::Impl::~Impl(void)
{
    Stop();
}

/**
* @brief Requests a stop and joins the worker thread.
*/
void
Environment::Impl::Stop(void)
{
    sRealTimeCommand cmd{};
    cmd.m_type = sRealTimeCommand::eTypeCommand::ChangeState;
    cmd.m_state = eEnvironmentState::STOPPED;
    m_dispatcher.enqueue<sRealTimeCommand>(cmd);

    m_running.store(false, std::memory_order_release);
    if (m_thread.joinable())
        m_thread.join();
}

/**
* @brief Starts the worker thread.
*/
void
Environment::Impl::Start(void)
{
    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&Environment::Impl::Run, this);
}

/**
* @brief Runs the main simulation loop.
*/
void
Environment::Impl::Run(void)
{  
    bool reinitrequest{ false };

    while (m_running.load(std::memory_order_acquire))
    {
        // Apply every queued command before stepping the simulation.
        m_dispatcher.update();

        if (m_state == eEnvironmentState::RUNNING)
        {
            if (m_worldRules != nullptr)
                m_worldRules->BeforeStep(m_entities);

            // Reinitialize entities and agents after a stop or a completed rollout.
            if (m_prevState == eEnvironmentState::STOPPED
            ||  reinitrequest)
            {
                for (auto& entity : m_entities)
                {
                    if (entity == nullptr)
                        continue;
                        
                    entity->Init();
                }

                // Initialize agents only after every entity has rebuilt its state.
                ppo::core::ResetAgents(m_agents, [](IAgent& agent) { agent.Init(); });

                reinitrequest = false;
            }

            // Reuse the domain-neutral multi-agent scheduler. The ECS update,
            // outcome rules, and snapshots remain environment-owned.
            reinitrequest |= ppo::core::StepAgents(
                m_agents,
                [](IAgent& agent) { agent.ChooseAction(); },
                [&]()
                {
                    AdvanceLivingEntities(m_entities, [](auto& entity) -> auto& { return entity; });
                },
                [&]()
                {
                    if (m_worldRules != nullptr)
                        m_worldRules->Resolve(m_entities);
                },
                [](IAgent& agent) { return agent.SetNewState(); });

        }

        m_prevState = m_state;
        if (m_worldRules != nullptr)
            m_worldRules->AfterStep();

        CreateSnapshot();

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    m_prevState = eEnvironmentState::STOPPED;
}

/**
* @brief Creates the public environment wrapper.
*/
Environment::Environment(void):
    p_impl{std::make_unique<Impl>()}
{

}

/**
* @brief Destroys the public environment wrapper.
*/
Environment::~Environment(void) = default;

bool Environment::SetWorldRules(std::unique_ptr<IWorldRules> rules)
{
    if (p_impl == nullptr || p_impl->m_running.load(std::memory_order_acquire) || rules == nullptr)
        return false;

    p_impl->m_worldRules = std::move(rules);
    return true;
}

/**
* @brief Returns the live registry pointer used for creation.
*/
entt::registry* 
Environment::GetRegistry(void) noexcept
{
    if (p_impl == nullptr)
        return nullptr;

    return &p_impl->m_world;
}

Entity::InitializerSystem
Environment::GetEntityInitializer(void) const noexcept
{
    return p_impl != nullptr ? p_impl->m_entityInitializer : nullptr;
}

bool
Environment::SetEntityInitializer(Entity::InitializerSystem initializer) noexcept
{
    if (p_impl == nullptr || p_impl->m_running.load(std::memory_order_acquire))
        return false;

    p_impl->m_entityInitializer = initializer;
    return true;
}

/**
* @brief Stores one entity in the managed entity list.
*/
void              
Environment::AddEntity(std::shared_ptr<Entity> entity)
{
    if (p_impl == nullptr
    ||  entity == nullptr)
        return;

    p_impl->m_entities.push_back(entity);
}

/**
* @brief Stores one agent and connects it to the command dispatcher.
*/
void 
Environment::AddAgent(std::shared_ptr<IAgent> pAgent)
{
    if (p_impl == nullptr
    ||  pAgent == nullptr)
        return;

    auto Iter = std::find(
        p_impl->m_agents.begin(),
        p_impl->m_agents.end(),
        pAgent);

    if (Iter != p_impl->m_agents.end())
        return;

    // Every registered agent listens to the shared real-time command bus.
    p_impl->m_dispatcher.sink<sAgentRealTimeCommand>().connect<&IAgent::ProcessCommand>(pAgent.get());

    p_impl->m_agents.push_back(pAgent);
}

/**
* @brief Queues a new simulation time step value.
*/
void
Environment::SetTimeStep(float timestep)
{
    if (p_impl == nullptr)
        return;

    sRealTimeCommand cmd{};
    cmd.m_type = sRealTimeCommand::eTypeCommand::TimeStep;
    cmd.m_timeStep = timestep;
    p_impl->m_dispatcher.enqueue<sRealTimeCommand>(cmd);
}

/**
* @brief Queues new spatial limits for the environment.
*/
void 
Environment::SetMaxRange(float north, float east)
{
    if (p_impl == nullptr)
        return;

    sRealTimeCommand cmd{};
    cmd.m_type = sRealTimeCommand::eTypeCommand::MaxRange;
    cmd.m_maxRanges[0] = north;
    cmd.m_maxRanges[1] = east;
    p_impl->m_dispatcher.enqueue<sRealTimeCommand>(cmd);
}

/**
* @brief Reads the latest published snapshot under the snapshot mutex.
*/
void
Environment::ReadSnapshot(std::function<void(entt::registry const&)> const& reader) const
{
    static entt::registry empty{};
    if (reader == nullptr)
        return;

    if (p_impl == nullptr)
    {
        reader(empty);
        return;
    }

    std::lock_guard<std::mutex> lock{ p_impl->m_snapshotMutex };
    reader(p_impl->m_buffers[p_impl->m_index.load(std::memory_order_acquire)]);
}

/**
* @brief Starts the worker thread if it is not already running.
*/
void
Environment::Start(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_running.load())
        return;

    p_impl->Start();
}

/**
* @brief Ensures the worker exists and requests the running state.
*/
void
Environment::Play(void)
{
    if (p_impl == nullptr)
        return;
    
    // Lazily recreate the worker thread when the user resumes from a stopped state.
    if (!p_impl->m_running.load(std::memory_order_acquire))
        p_impl->Start();

    sRealTimeCommand cmd{};
    cmd.m_type = sRealTimeCommand::eTypeCommand::ChangeState;
    cmd.m_state = eEnvironmentState::RUNNING;
    p_impl->m_dispatcher.enqueue<sRealTimeCommand>(cmd);
}

/**
* @brief Requests the paused state.
*/
void
Environment::Pause(void)
{
    if (p_impl == nullptr)
        return;

    sRealTimeCommand cmd{};
    cmd.m_type = sRealTimeCommand::eTypeCommand::ChangeState;
    cmd.m_state = eEnvironmentState::PAUSED;
    p_impl->m_dispatcher.enqueue<sRealTimeCommand>(cmd);
}

/**
* @brief Stops the environment worker thread.
*/
void
Environment::Stop(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->Stop();
}

/**
* @brief Queues one agent command on the shared dispatcher.
*/
void
Environment::AddAgentCommand(sAgentRealTimeCommand const& command)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dispatcher.enqueue<sAgentRealTimeCommand>(command);
}

}

