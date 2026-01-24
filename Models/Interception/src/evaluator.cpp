#include "evaluator.hpp"
#include "entity.hpp"
#include "serializer.hpp"
#include "prodagent.hpp"
#include "interception.hpp"

#include <thread>
#include <map>
#include <vector>
#include <chrono>

namespace ppo::interception {

using Agents = std::map<std::shared_ptr<ppo::model::Aircraft>, std::shared_ptr<SNAProdAgent>>;

struct sCheckPointGenerator {
    sCheckPointGenerator(Agents const& agents,
                         ppo::model::sEnvironmentState const& env) :
        m_agents{ agents },
        m_env{env}{};

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto checkpointsection = ar.begin_section("Data");

        for (auto const& [player, _] : m_agents)
        {
            if (player == nullptr)
                continue;

            auto section{ checkpointsection.begin_section(player->GetName().c_str()) };

            if (auto Iter = m_buffers.find(player->GetName());
                Iter != m_buffers.end())
            {
                for (auto i = 0; i < m_nbtests; ++i)
                    if (i < Iter->second.size())
                        section(std::to_string(i).c_str(), Iter->second[i]);
            }
            else
            {
                std::vector<ppo::model::CheckPointBuffer> buffers{};
                for (auto i = 0; i < m_nbtests; ++i)
                {
                    auto buf = player->GetCheckpoint();

                    section(std::to_string(i).c_str(), buf);

                    buffers.push_back(buf);
                }

                m_buffers[player->GetName()] = buffers;
            }
        }
    };

    std::map<std::string, std::vector<ppo::model::CheckPointBuffer>> m_buffers{};

    Agents const& m_agents;
    ppo::model::sEnvironmentState const& m_env;
    unsigned int m_nbtests{ 0 };
};

struct Evaluator::Impl : 
    public ppo::model::sEvaluatorState
{
         Impl(void);
         ~Impl(void)
         {
             if (m_thread.joinable())
                 m_thread.join();
         };

    template<typename Ar>
    void serialize          (Ar& ar);
    void UpdateDoubleBuffer (void);
    void Start              (std::string const& name, std::filesystem::path const& modelpath);
    void Stop               (void);
    void Run                (void);

    float             GetTimeStep(void) const noexcept { return m_dt; };
    
    ppo::model::Entity*
    GetIEntityById(ppo::model::EntityId const id) const noexcept
    {
        for (auto const& pair : m_agents)
        {
            if (pair.first->GetEntityId() == id)
                return pair.first.get();
        }
        return nullptr;
    };

    float             GetMaxNorth(void) const noexcept { return m_maxNorth; };
    float             GetMaxEast(void) const noexcept { return m_maxEast; };

    static ppo::model::EntityId
    GetNewEntityId(void)
    {
        static std::atomic<ppo::model::EntityId> g_id{ 0 };
        return g_id.fetch_add(1, std::memory_order_relaxed);
    }

    // this is the thread launch to perform the evaluation
    std::thread  m_thread;

    // a variable to detect stopping
    std::atomic<bool>  m_running{ false };

    std::shared_ptr<ppo::model::Aircraft> p_blue{nullptr};
    std::shared_ptr<ppo::model::Aircraft> p_red {nullptr};
    Agents m_agents{};

    std::filesystem::path m_modelpath{};

    sEvaluatorState       m_buffers[2];
    std::atomic<uint32_t> m_index{ 0 };

    // a tool to generate and store a checkpoint buffer
    sCheckPointGenerator m_checkpoint{ m_agents, *this };

    static inline const sEvaluatorState EMPTY_STATE{};
};

Evaluator::Impl::Impl(void)
{
    // use same env as training environment
    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        "Evaluation.toml",
        [this](ppo::serialization::TomlReadArchive& ar)
        {
            serialize< ppo::serialization::TomlReadArchive>(ar);
        },
        [this](ppo::serialization::TomlWriteArchive& ar)
        {
            serialize< ppo::serialization::TomlWriteArchive>(ar);
        }
    );
}

void
Evaluator::Impl::UpdateDoubleBuffer(void)
{
    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_maxEast = m_maxEast;
    m_buffers[next].m_maxNorth = m_maxNorth;
    m_buffers[next].m_dt = m_dt;
    m_buffers[next].m_nbstep = m_nbstep;
    m_buffers[next].m_nbtests = m_nbtests;
    m_buffers[next].m_notation = m_notation;
    m_buffers[next].m_inprogress = m_inprogress;
    m_buffers[next].m_progress = m_progress;

    m_index.store(next, std::memory_order_release);
}

template<typename Ar>
void 
Evaluator::Impl::serialize(Ar& ar)
{
    auto section = ar.begin_section("Eval");

    section("MaxNorth", m_maxNorth);
    section("MaxEast",  m_maxEast);
    section("TimeStep", m_dt);
    section("NbSteps",  m_nbstep);
    section("NbTests",  m_nbtests);
    section("Notation", m_notation);

    m_checkpoint.m_nbtests = m_nbtests;

    m_checkpoint.serialize(ar);

    UpdateDoubleBuffer();
}

void
Evaluator::Impl::Stop(void)
{
    m_running.store(false, std::memory_order_release);
    if (m_thread.joinable())
        m_thread.join();
}

void
Evaluator::Impl::Start(std::string const& name, std::filesystem::path const& modelpath)
{
    m_modelpath = modelpath;

    std::filesystem::path evalpath{ modelpath / "eval" / "model.pt" };
    if (!std::filesystem::exists(evalpath))
        return;

    p_red->SetTarget(p_blue->GetEntityId());
    p_blue->SetTarget(p_red->GetEntityId());

    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&Evaluator::Impl::Run, this);
}

void
Evaluator::Impl::Run(void)
{
    // for all test available
    float playerDestroyed{0.F};
    float targetDestroyed{0.F};
    float missionDuration{0.F};
    float pkill{ 0.F };
    float notation{ -FLT_MAX };

    m_inprogress = true;
    m_progress = 0.F;
    UpdateDoubleBuffer();

    for (int i = 0; i < m_nbtests; ++i)
    {
        // thread must terminate
        if (!m_running.load(std::memory_order_acquire))
            return;

        // initialize the player and the plicy
        for (auto& [player, agent] : m_agents)
        {
            if (player == nullptr)
                continue;

            std::string const& name{ player->GetName() };
            auto const Iter{ m_checkpoint.m_buffers.find(name) };
            if (Iter != m_checkpoint.m_buffers.end())
            {
                std::vector<ppo::model::CheckPointBuffer> const& buffers{ Iter->second };

                // set the ieme testcase 
                player->SetCheckpoint(buffers[i]);
            }

            if (agent != nullptr)
                agent->Init();
        }

        // perform all step until m_nbstep
        float pk{ 0.F };
        int j{ 0 };
        for (; j < m_nbstep; ++j)
        {
            // thread must terminate
            if (!m_running.load(std::memory_order_acquire))
                return;

            // choose action according to policy
            for (auto& [_, agent] : m_agents)
            {
                if (agent == nullptr)
                    continue;

                agent->ChooseAction();
            }

            // update model
            bool targetDestruction{ false };
            for (auto& [player, agent] : m_agents)
            {
                if (player == nullptr)
                    continue;

                // Run the model after the Agent run to apply the action
                player->Run();
            }

            for (auto& [player, agent] : m_agents)
            {
                if (agent == nullptr)
                    continue;

                player->UpdateEnv();

                agent->SetNewState();
            }

            bool goToNextTest(false);
            for (auto& [player, agent] : m_agents)
            {
                if (player == nullptr
                 || agent == nullptr)
                    continue;

                ppo::model::Entity* const target{ player.get() == p_red.get() ? p_blue.get() : p_red.get()};
                if (target == nullptr)
                    continue;

                ppo::model::SNA const& sna{ player->GetSNA() };

                float const newpk{ sna.GetKillProbability() };
                if (0.95F < newpk)
                    targetDestruction = true;
               
                pk = std::max(pk,newpk);

                if (player->GetDestroyed())
                {
                    playerDestroyed += 1.F;
                    goToNextTest = true;
                    break;
                }
                else if (targetDestruction)
                {
                    targetDestroyed += 1.F;
                    goToNextTest = true;
                    break;
                }
            }
                 
            if (goToNextTest)
                break;
        }

        pkill += pk;
        missionDuration += (m_nbstep+1)/(j+1);

        m_inprogress = true;
        m_progress = static_cast<float>(i) / static_cast<float>(m_nbtests);

        UpdateDoubleBuffer();

        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    notation = 100.F * (targetDestroyed + pkill - playerDestroyed + missionDuration) / static_cast<float>(m_nbtests);
    
    if (m_notation < notation)
    {
        // move the model into prod
        std::filesystem::path proddirectory{ m_modelpath / "prod" };
        std::filesystem::path evalmodel{ m_modelpath / "eval" / "model.pt" };

        if (!std::filesystem::exists(proddirectory))
            std::filesystem::create_directories(proddirectory);

        std::filesystem::path prodmodel{ proddirectory / "model.pt" };
        std::filesystem::copy_file(evalmodel, prodmodel, std::filesystem::copy_options::overwrite_existing);

        m_notation = notation;
    }

    m_inprogress = false;
    m_progress   = 0.F;
    UpdateDoubleBuffer();
}

Evaluator::Evaluator(void) :
    p_impl{ std::make_unique<Impl>() }
{
    auto [missileblue, prodblue] = GetNewProductionAircraft("blueeval", ppo::model::eEntitySide::Friend, *p_impl.get());
    auto missilered = GetNewRandomAircraft("redeval", ppo::model::eEntitySide::Hostile, *p_impl.get());
    p_impl->p_blue = missileblue;
    p_impl->p_red = missilered;

   p_impl->m_agents[p_impl->p_blue] = prodblue;
   p_impl->m_agents[p_impl->p_red] = nullptr;
}

Evaluator::~Evaluator(void) = default;

ppo::model::sEvaluatorState
Evaluator::GetSnapshot(void) const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_STATE;

    return p_impl->m_buffers[p_impl->m_index.load(std::memory_order_acquire)];
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

} /* ppo::model */