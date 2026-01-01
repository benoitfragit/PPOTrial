#include "environment.hpp"
#include "utils.hpp"

#include <map>
#include <thread>
#include <atomic>
#include <chrono>
#include <array>
#include <algorithm>

namespace ppo::model {

using Agents = std::map<Player*, std::shared_ptr<IAgent>>;

struct Environment::Impl :
    public sEnvironmentState
{
    void                    Run           (void);
    void                    ComputeState  (Player* const player,float const dt);
    bool                    ComputeReward (Player* const player, std::shared_ptr<IAgent> pAgent);
    void                    Stop          (void);
    void                    Start         (void);
    Player*                 GetPlayerById (PlayerId const id) const noexcept;
    std::shared_ptr<IAgent> GetPlayerAgent(Player* player) const noexcept;
    float                   GetKillProbability(Player const* const player, 
                                               Player* const target,
                                               float minShootingRangeAngle,
                                               float maxShootingRangeAngle) const noexcept;

                            Impl          (void) = default;

                            ~Impl         (void);

    Agents                          m_agents            {};
    sEnvironmentState               m_buffers           [2];
    std::atomic<uint32_t>           m_index             { 0 };
    std::thread                     m_thread;
    std::atomic<bool>               m_running           { false };  
    std::atomic<bool>               m_training          { false };
    std::atomic<long long>          m_maxRolloutBuffer  { 4096 };
    std::atomic<float>              m_timestep          { 0.02F };
    std::atomic<float>              m_maxRangeEast      { 150000.F };
    std::atomic<float>              m_maxRangeNorth     { 112500.F };
    std::atomic<float>              m_rewardScaling     [sAgentState::REWARDS.size()];
    std::atomic<float>              m_rewardClamp       { 5.F  };
    std::atomic<eEnvironmentState>  m_state             { eEnvironmentState::STOPPED };

    eEnvironmentState               m_prevState         { m_state };

    static inline sEnvironmentState     EMPTY_ENV_STATE{};
};

Player*
Environment::Impl::GetPlayerById(PlayerId const id) const noexcept
{
    for (auto const& pair : m_agents)
    {
        if (pair.first->GetPlayerId() == id)
            return pair.first;
    }
    return nullptr;
}

std::shared_ptr<IAgent>
Environment::Impl::GetPlayerAgent(Player* player) const noexcept
{
    auto Iter = m_agents.find(player);
    if (Iter != m_agents.end())
        return Iter->second;

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
Environment::Impl::ComputeState(Player* const player, float const dt)
{
    if (player == nullptr)
        return;

    // build state vector
    if (Player* const target{ GetPlayerById(player->GetTarget())};
       (target != nullptr))
        player->SetTargetPosition(target->GetNorthPosition(), target->GetEastPosition());
    

    player->SetFightingArea( m_maxNorth / 2.F,
                            -m_maxEast / 2.F,
                            -m_maxNorth / 2.F,
                             m_maxEast / 2.F);

    // Run the model after the Agent run to apply the action
    player->Run(dt);
}

float                   
Environment::Impl::GetKillProbability(Player const* const player,
                                      Player* const target,
                                      float minShootingRangeAngle,
                                      float maxShootingRangeAngle) const noexcept
{
    float PK{ 0.F };

    if (player == nullptr
     || target == nullptr)
        return PK;

    float x       { player->GetNorthPosition() };
    float y       { player->GetEastPosition() };
    float heading { player->GetRotation() };
    float nez     { player->GetNoEscapeZone() };
    float ez      { player->GetEscapeZone() };
    float const tx{ target->GetNorthPosition() };
    float const ty{ target->GetEastPosition() };
    float const d { static_cast<float>(sqrt(pow(x - tx, 2.F) + pow(y - ty, 2.F))) };
    float const ux{ (tx - x) / d };
    float const uy{ (ty - y) / d };

    float az = atan2(uy, ux);
    az = wrapMinusPi2Pi(az);
    az = wrapMinusPi2Pi(az - heading);

    if ((minShootingRangeAngle <= az)
        && (az <= maxShootingRangeAngle))
    {
        if (d <= nez)
        {
            target->SetDestroyed(true);
            PK = 1.F;
        }
        else if (nez < d && d <= ez)
        {
            PK = std::clamp(1.F-(d - nez) / (ez - nez), 0.F, 1.F);
        }
    }

    return PK;
}

bool
Environment::Impl::ComputeReward(Player* const player, std::shared_ptr<IAgent> pAgent)
{
    bool ret{ false };

    if (player == nullptr)
        return ret;

    float                   fuelconsumption     { player->GetFuelConsumption() };
    float                   maxfuel             { player->GetMaxFuel() };
    float                   maxSpeed            { player->GetMaxSpeed() };
    PlayerId const          targetId            { player->GetTarget() };
    float                   x                   { player->GetNorthPosition() };
    float                   y                   { player->GetEastPosition() };
    float                   dx                  { player->GetNorthSpeed() };
    float                   dy                  { player->GetEastSpeed() };
    bool                    targetDestruction   { false };
    float                   pkTarget            { 0.F };
    float                   pkPlayer            { 0.F };
    bool                    targetDestroyed     { false };
    float                   R_close             (0.F);

    if (auto const target{ GetPlayerById(targetId) };
        target != nullptr)
    {
        float const tx{ target->GetNorthPosition() };
        float const ty{ target->GetEastPosition() };
        float const d { static_cast<float>(sqrt(pow(x - tx, 2.F) + pow(y - ty, 2.F))) };
        float const ux{ (tx - x)/d };
        float const uy{ (ty - y)/d };

        R_close = std::clamp((dx * ux + dy * uy) / maxSpeed,-1.F, 1.F);

        pkTarget = GetKillProbability(player, target, -HALFPI, HALFPI);
        pkPlayer = GetKillProbability(target, player, -HALFPI, HALFPI);

        targetDestruction = (1.F <= pkTarget);
    }

    bool done{ false };
    if (pAgent != nullptr)
    {
        if (std::numeric_limits<float>::epsilon() < maxfuel)
            pAgent->AddReward(sAgentState::RewardType::FuelConsumption, -(fuelconsumption / maxfuel));

        if (targetDestruction)
        {
            pAgent->AddReward(sAgentState::RewardType::TargetDestruction, 1.F);
        }
        else
        {
            // the more the target dives into our somain the more we receive reward
            pAgent->AddReward(sAgentState::RewardType::TargetInEZ, pkTarget);
        }

        if (player->GetDestroyed())
        {
            pAgent->AddReward(sAgentState::RewardType::OwnshipDestruction, -1.F);
        }
        else
        {
            // penalty when entering opponent escape zone
            pAgent->AddReward(sAgentState::RewardType::OwnshipInEZ, -pkPlayer);

            pAgent->AddReward(sAgentState::RewardType::Range, R_close);
        }

        done = player->GetDestroyed() || targetDestroyed;

        pAgent->SetDone(done);
    }

    return done;
}

void
Environment::Impl::Run(void)
{  
    eEnvironmentState state{ eEnvironmentState::INIT };
    long long maxRolloutBuffer{ 4096 };
    float clampReward{ 3.F };
    bool reinitrequest{ false };
    bool enableTraining{ false };

    std::array<float, sAgentState::REWARDS.size()> rewardScaling{};

    while (m_running.load(std::memory_order_acquire))
    {
        m_maxNorth      = m_maxRangeNorth.load(std::memory_order_acquire);  
        m_maxEast       = m_maxRangeEast.load(std::memory_order_acquire);   
        m_dt            = m_timestep.load(std::memory_order_acquire);
        clampReward     = m_rewardClamp.load(std::memory_order_acquire);
        enableTraining  = m_training.load(std::memory_order_acquire) ;
        maxRolloutBuffer= m_maxRolloutBuffer.load(std::memory_order_acquire);
        state           = m_state.load(std::memory_order_acquire);

        for (auto i = 0; i < sAgentState::REWARDS.size(); ++i)
            rewardScaling[i] = m_rewardScaling[i].load(std::memory_order_acquire);

        // ensure all agents are in training mode
        for (auto& [_, agent] : m_agents)
        {
            if (agent == nullptr)
                continue;

            for (auto i = 0; i < sAgentState::REWARDS.size(); ++i)
                agent->SetRewardScaling(i, rewardScaling[i]);
            
            agent->SetRewardClampValue(clampReward);

            if (enableTraining)
            {
                if (!agent->IsTraining())
                {
                    agent->StartTraining(maxRolloutBuffer);
                    reinitrequest = true;
                }
            }
            else if (agent->IsTraining())
            {
                agent->StopTraining();
            }
        }

        if (state == eEnvironmentState::RUNNING)
        {            
            // to initialize
            if (m_prevState == eEnvironmentState::STOPPED
            ||  reinitrequest)
            {
                for (auto& [player, agent] : m_agents)
                {
                    if (player != nullptr)
                        player->Init(*this);

                    if (agent != nullptr)
                        agent->Init(*this, player);
                }
                
                reinitrequest = false;
            }

            for (auto& [player, agent] : m_agents)
            {
                if (player == nullptr)
                    continue;

                // according to the current state select an action
                if (agent!= nullptr)
                    agent->ChooseAction(*this, player);

                // according to the action compute the new state
                ComputeState(player, m_dt);
            }

            // warning, reward would be valid after processing all players
            for (auto& [player, agent] : m_agents)
            {
                if (player == nullptr)
                    continue;

                reinitrequest |= ComputeReward(player, agent);
            }

            // here we know that all players have been processed for this step
            for (auto& [player, agent] : m_agents)
            {
                if (agent == nullptr)
                    continue;

                agent->SetNewState(*this, player);
            }
        }

        m_prevState = state;

        const uint32_t next         = 1u - m_index.load(std::memory_order_relaxed);

        m_buffers[next].m_maxEast   = m_maxEast;
        m_buffers[next].m_maxNorth  = m_maxNorth;
        m_buffers[next].m_dt        = m_dt;

        m_index.store(next, std::memory_order_release);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

Environment::Environment(void):
    p_impl{std::make_unique<Impl>()}
{

}

Environment::~Environment(void) = default;

void 
Environment::AddAgent(Player* const player, std::shared_ptr<IAgent> pAgent)
{
    if (p_impl == nullptr)
        return;

    PlayerId const playerId{ player->GetPlayerId() };

    auto const previous{ p_impl->GetPlayerById(playerId) };

    if (previous == nullptr)
    {
        p_impl->m_agents[player] = pAgent;
    }
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

void
Environment::SetRewardScaling(unsigned char type, float scaling)
{
    if (p_impl == nullptr
    || sAgentState::REWARDS.size() <= type)
        return;

    p_impl->m_rewardScaling[type].store(scaling, std::memory_order_relaxed);


}

void              
Environment::SetRewardClamp(float rewardClmp)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_rewardClamp.store(rewardClmp, std::memory_order_relaxed);
}


void
Environment::StartTraining(long long rolloutBuffSize)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_maxRolloutBuffer.store(rolloutBuffSize, std::memory_order_release);
    p_impl->m_training.store(true, std::memory_order_release);
}

void
Environment::StopTraining(void)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_training.store(false, std::memory_order_release);
}

}

