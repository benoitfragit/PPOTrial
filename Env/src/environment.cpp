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

using Entities = std::vector<std::shared_ptr<IEntity>>;
using Agents = std::vector<std::shared_ptr<IAgent>>;

struct Environment::Impl :
    public sEnvironmentState
{
    void                    Run           (void);
    void                    Stop          (void);
    void                    Start         (void);
    IEntity*                GetEntityById (EntityId const id) const noexcept;
    void                    UpdateDoubbleBuffer(void);

    explicit                Impl(Environment& env):
                                m_env{env}
                            {
                                ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
                                serializer.Register(
                                    "Environment.toml",
                                    [this](ppo::serialization::TomlReadArchive& ar)
                                    {
                                        serialize< ppo::serialization::TomlReadArchive>(ar);

                                        UpdateDoubbleBuffer();

                                        m_timestep = m_dt;
                                        m_maxRangeEast = m_maxEast;
                                        m_maxRangeNorth = m_maxNorth;
                                    },
                                    [this](ppo::serialization::TomlWriteArchive& ar)
                                    {
                                        serialize<ppo::serialization::TomlWriteArchive>(ar);
                                    }
                                );
                            };

                            ~Impl         (void);

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section("Env");

        section("MaxNorth", m_maxNorth);
        section("MaxEast", m_maxEast);
        section("TimeStep", m_dt);
    };

    Entities                        m_entities          {};    
    Agents                          m_agents            {};
    sEnvironmentState               m_buffers           [2];
    std::atomic<uint32_t>           m_index             { 0 };
    std::thread                     m_thread;
    std::atomic<bool>               m_running           { false };  

    std::atomic<float>              m_timestep          { 0.02F };
    std::atomic<float>              m_maxRangeEast      { 150000.F };
    std::atomic<float>              m_maxRangeNorth     { 112500.F };

    std::atomic<eEnvironmentState>  m_state             { eEnvironmentState::STOPPED };

    eEnvironmentState               m_prevState         { m_state };

    Environment&                    m_env;

    static inline sEnvironmentState     EMPTY_ENV_STATE{};
};

void 
Environment::Impl::UpdateDoubbleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_maxEast = m_maxEast;
    m_buffers[next].m_maxNorth = m_maxNorth;
    m_buffers[next].m_dt = m_dt;

    m_index.store(next, std::memory_order_release);
}

IEntity*
Environment::Impl::GetEntityById(EntityId const id) const noexcept
{
    for (auto const& entity : m_entities)
    {
        if (entity->GetEntityId() == id)
            return entity.get();
    }
    return nullptr;
}

Environment::Impl::~Impl(void)
{
    Stop();
}

void
Environment::Impl::Stop(void)
{
    m_state.store(eEnvironmentState::STOPPED, std::memory_order_release);
    m_prevState = eEnvironmentState::STOPPED;
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
    eEnvironmentState state{ eEnvironmentState::INIT };
    bool reinitrequest{ false };

    while (m_running.load(std::memory_order_acquire))
    {
        m_maxNorth      = m_maxRangeNorth.load(std::memory_order_acquire);  
        m_maxEast       = m_maxRangeEast.load(std::memory_order_acquire);   
        m_dt            = m_timestep.load(std::memory_order_acquire);
        state           = m_state.load(std::memory_order_acquire);

        // ensure all agents are in training mode
        for (auto& agent : m_agents)
        {
            if (agent == nullptr)
                continue;

            reinitrequest |= agent->Update();
        }

        if (state == eEnvironmentState::RUNNING)
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
                if (entity == nullptr)
                    continue;

                entity->UpdateEnv();
            }

            // here we know that all players have been processed for this step
            for (auto& agent : m_agents)
            {
                if (agent == nullptr)
                    continue;

                reinitrequest |= agent->SetNewState();
            }
        }

        m_prevState = state;

        UpdateDoubbleBuffer();

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

Environment::Environment(void):
    p_impl{std::make_unique<Impl>(*this)}
{

}

Environment::~Environment(void) = default;

void              
Environment::AddEntity(std::shared_ptr<IEntity> entity)
{
    if (p_impl == nullptr
    ||  entity == nullptr)
        return;

    auto Iter = std::find(
        p_impl->m_entities.begin(),
        p_impl->m_entities.end(), 
        entity);    

    if (Iter != p_impl->m_entities.end())
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

    p_impl->m_agents.push_back(pAgent);
}

void
Environment::SetTimeStep(float timestep)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_timestep.store(timestep, std::memory_order_release);
}

void 
Environment::SetMaxRange(float north, float east)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_maxRangeNorth.store(north, std::memory_order_release);
    p_impl->m_maxRangeEast.store(east, std::memory_order_release);
}

sEnvironmentState 
Environment::GetSnapshot(void) const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_ENV_STATE;

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

    p_impl->m_state.store(eEnvironmentState::RUNNING, std::memory_order_release);
}

void
Environment::Pause(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_state.store(eEnvironmentState::PAUSED, std::memory_order_release);
}

void
Environment::Stop(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->Stop();
}

IEntity* 
Environment::GetIEntityById(EntityId playerId) noexcept
{
    if (p_impl == nullptr)
        return nullptr;

    return p_impl->GetEntityById(playerId);
}

float
Environment::GetTimeStep(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_dt;
}

float
Environment::GetMaxNorth(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_maxNorth;
}

float
Environment::GetMaxEast(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_maxEast;
}

EntityId 
Environment::GetNewEntityId(void)
{
    static std::atomic<EntityId> g_id{ 0 };
    return g_id.fetch_add(1, std::memory_order_relaxed);
}

unsigned int
Environment::GetNumberOfEntities(void) const noexcept
{ 
    if (p_impl == nullptr)
        return 0;

    return p_impl->m_entities.size();
}

IEntity* 
Environment::GetEntityAtIndex(unsigned int index) noexcept
{
    if (index >= GetNumberOfEntities())
        return nullptr;

    return p_impl->m_entities[index].get();
}

}

