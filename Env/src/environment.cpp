#include "environment.hpp"
#include "utils.hpp"
#include "serializer.hpp"

#include <map>
#include <thread>
#include <atomic>
#include <chrono>
#include <array>
#include <algorithm>
#include <vector>

namespace ppo::model {

using Entities = std::vector<std::shared_ptr<Entity>>;
using Agents = std::vector<std::shared_ptr<IAgent>>;

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

struct Environment::Impl
{
    void                    Run           (void);
    void                    Stop          (void);
    void                    Start         (void);
    void                    CreateSnapshot(void);

    void                    ProcessCommand(sRealTimeCommand const& command);

                            Impl(void);

                            ~Impl(void);

    /// <summary>
    /// This is the main entt::registry (world)
    /// 
    /// it is going to be filled and readen only by this environment
    /// </summary>
    entt::registry                  m_world{};

    /// <summary>
    /// Command dispatcher
    /// </summary>
    entt::dispatcher                m_dispatcher{};

    /// <summary>
    /// This is a list of all entities wrappers
    /// </summary>
    Entities                        m_entities          {};    

    /// <summary>
    /// this is a list of all agents
    /// </summary>
    Agents                          m_agents            {};

    /// <summary>
    /// Snasphot mechanism
    /// </summary>
    std::array<entt::registry,2>    m_buffers           {};
    std::atomic<uint32_t>           m_index             { 0 };

    /// <summary>
    /// this is the running thread of this environment
    /// </summary>
    std::thread                     m_thread;

    std::atomic<bool>               m_running           { false };  
    eEnvironmentState               m_state             { eEnvironmentState::INIT };
    eEnvironmentState               m_prevState         { m_state };
};

Environment::Impl::Impl(void)
{
    // add a buffer for the environment
    m_world.ctx().emplace<sEnvironmentState>();

    // subscribe to real time command through a thread safe queue
    m_dispatcher.sink<sRealTimeCommand>().connect<&Environment::Impl::ProcessCommand>(this);

    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        "Environment.json",
        [this](ppo::serialization::json& ar)
        {
            auto& state = m_world.ctx().get<sEnvironmentState>();

            auto section = ar["Env"];

            state.m_maxNorth = section["MaxNorth"];
            state.m_maxEast  = section["MaxEast"];
            state.m_dt       = section["TimeStep"];

            CreateSnapshot();
        },
        [this](ppo::serialization::json& ar)
        {
            auto const& state = m_world.ctx().get<sEnvironmentState>();

            auto& section = ar["Env"];

            section["MaxNorth"] = state.m_maxNorth ;
            section["MaxEast"]  = state.m_maxEast  ;
            section["TimeStep"] = state.m_dt       ;
        }
    );
}

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

void 
Environment::Impl::CreateSnapshot(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    // start by cleaning this registry
    m_buffers[next].clear();
    m_buffers[next].ctx().clear();

    for (auto entity : m_world.storage<entt::entity>())
        auto newentity = m_buffers[next].create(entity);

    m_world.view<sEntityState>().each([&](auto entity, const auto& comp)
    {
        m_buffers[next].emplace<sEntityState>(entity, comp);
    });

    m_world.view<sSNAState>().each([&](auto entity, const auto& comp)
    {
        m_buffers[next].emplace<sSNAState>(entity, comp);
    });

    m_world.view<sFlightModelState>().each([&](auto entity, const auto& comp)
    {
        m_buffers[next].emplace<sFlightModelState>(entity, comp);
    });

    m_buffers[next].ctx().emplace<sEnvironmentState>(m_world.ctx().get<sEnvironmentState>());

    for (auto agent : m_agents)
    {
        auto const agentSnapshot(agent->GetSnapshot());
        
        // add a special entity to broadcast agent state
        entt::entity entity_agent = m_buffers[next].create();
        m_buffers[next].emplace<sAgentStateSnaphot>(entity_agent, agentSnapshot);
    }

    m_index.store(next, std::memory_order_release);
}

Environment::Impl::~Impl(void)
{
    Stop();
}

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

void
Environment::Impl::Start(void)
{
    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&Environment::Impl::Run, this);
}

void
Environment::Impl::Run(void)
{  
    bool reinitrequest{ false };

    while (m_running.load(std::memory_order_acquire))
    {
        // proceed commands
        m_dispatcher.update();

        if (m_state == eEnvironmentState::RUNNING)
        {            
            // to initialize
            if (m_prevState == eEnvironmentState::STOPPED
            ||  reinitrequest)
            {
                for (auto& entity : m_entities)
                {
                    if (entity == nullptr)
                        continue;
                        
                    entity->Init();
                }

                // all players must be initialized before initliazing agents
                for (auto& agent : m_agents)
                {
                    if (agent == nullptr)
                        continue;
                 
                    agent->Init();
                }

                reinitrequest = false;
            }

            for (auto& agent : m_agents)
            {
                // according to the current state select an action
                if (agent == nullptr)
                    continue;
                    
                agent->ChooseAction();
            }

            for (auto& entity : m_entities)
            {
                if (entity == nullptr)
                    continue;

                // according to the action compute the new state
                entity->Run();
            }

            // update the whole environment after all player have been updated
            for (auto& entity: m_entities)
            {
                if (entity == nullptr
                ||  !entity->has<sSNAState>())
                    continue;

                auto const& [snaState] = entity->get<sSNAState>();

                if (m_world.any_of<sEntityState>(snaState.m_target))
                {
                    float const pkTarget{ snaState.m_pk };

                    auto [targetState] = m_world.view<sEntityState>().get(snaState.m_target);

                    if (0.95F < pkTarget)
                        targetState.m_destroyed = true;
                }
            }

            // here we know that all players have been processed for this step
            for (auto& agent : m_agents)
            {
                if (agent == nullptr)
                    continue;

                reinitrequest |= agent->SetNewState();
            }
        }

        m_prevState = m_state;

        CreateSnapshot();

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    m_prevState = eEnvironmentState::STOPPED;
}

Environment::Environment(void):
    p_impl{std::make_unique<Impl>()}
{

}

Environment::~Environment(void) = default;

entt::registry* 
Environment::GetRegistry(void) noexcept
{
    if (p_impl == nullptr)
        return nullptr;

    return &p_impl->m_world;
}

void              
Environment::AddEntity(std::shared_ptr<Entity> entity)
{
    if (p_impl == nullptr
    ||  entity == nullptr)
        return;

    p_impl->m_entities.push_back(entity);
}

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

    // all agent are going to receive this command
    p_impl->m_dispatcher.sink<sAgentRealTimeCommand>().connect<&IAgent::ProcessCommand>(pAgent.get());

    p_impl->m_agents.push_back(pAgent);
}

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

entt::registry& 
Environment::GetSnapshot(void) const noexcept
{
    if (p_impl == nullptr)
    {
        static entt::registry empty{};
        return empty;
    }
        
    return p_impl->m_buffers[p_impl->m_index.load(std::memory_order_acquire)];
}

void
Environment::Start(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_running.load())
        return;

    p_impl->Start();
}

void
Environment::Play(void)
{
    if (p_impl == nullptr)
        return;
    
    // start again if not launched
    if (!p_impl->m_running.load(std::memory_order_acquire))
        p_impl->Start();

    sRealTimeCommand cmd{};
    cmd.m_type = sRealTimeCommand::eTypeCommand::ChangeState;
    cmd.m_state = eEnvironmentState::RUNNING;
    p_impl->m_dispatcher.enqueue<sRealTimeCommand>(cmd);
}

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

void
Environment::Stop(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->Stop();
}

void
Environment::AddAgentCommand(sAgentRealTimeCommand const& command)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dispatcher.enqueue<sAgentRealTimeCommand>(command);
}

}

