#include "evaluator.hpp"
#include "entity.hpp"
#include "serializer.hpp"
#include "prodagent.hpp"
#include "interception.hpp"
#include "utils.hpp"
#include "checkpointinitializersystem.hpp"

#include <thread>
#include <map>
#include <vector>
#include <chrono>

#include "entt/entt.hpp"

namespace rl {

using Entities = std::map<std::string, std::shared_ptr<ppo::model::Aircraft>>;

struct sCheckPointGenerator 
{
    void serialize(entt::registry& registry, Entities& entities, ppo::serialization::json& ar)
    {
        auto const& evalState = registry.ctx().get<ppo::model::sEvaluatorState>();

        auto& checkpointsection = ar["Data"];

        for (auto const& [name, entity] : entities)
        {
            auto const [checkpoint] = entity->get<ppo::model::sEntityCheckPoint>();

            checkpointsection[name] = checkpoint.m_buffers;
        }
    };
};

struct Evaluator::Impl
{
    entt::registry m_world{};

    Impl(void);

    ~Impl(void)
    {
        if (m_thread.joinable())
            m_thread.join();
    };

    void LoadFromFile       (ppo::serialization::json& ar);
    void SaveToFile         (ppo::serialization::json& ar);
    void UpdateDoubleBuffer (void);
    void Start              (std::string const& name, std::filesystem::path const& modelpath);
    void Stop               (void);
    void Run                (void);
    
    void AddEntity(std::shared_ptr<ppo::model::Entity> entity) {};
    void AddAgent(std::shared_ptr<ppo::model::IAgent> agent) {};

    template<typename T, typename... Args>
    std::shared_ptr<T>
    CreateEntity(Args&&... args)
    {
        if constexpr (!std::is_base_of_v<ppo::model::Entity, T>)
            return nullptr;

        auto entity{ std::make_shared<T>(&m_world, std::forward<Args>(args)...) };

        if (entity != nullptr)
        {
            entity->SetInitializerSystem(&ppo::model::CheckpointInitializerSystem::Init);
            AddEntity(entity);
        }

        return entity;
    };

    template<typename Ag, typename... Args>
    std::shared_ptr<Ag>
        CreateAgent(Args&&... args)
    {
        if constexpr (!std::is_base_of_v<ppo::model::IAgent, Ag>)
            return nullptr;

        auto agent{ std::make_shared<Ag>(std::forward<Args>(args)...) };
        if (agent != nullptr)
            AddAgent(agent);

        return agent;
    };

    std::thread                             m_thread;
    std::atomic<bool>                       m_running{ false };
    std::shared_ptr<ppo::model::Aircraft>   p_blue{nullptr};
    std::shared_ptr<ppo::model::Aircraft>   p_red {nullptr};
    Entities                                m_entities{};
    std::shared_ptr<SNAProdAgent>           m_agent{ nullptr };
    std::filesystem::path                   m_modelpath{};
    ppo::model::sEvaluatorState             m_buffers[2];
    std::atomic<uint32_t>                   m_index{ 0 };
    sCheckPointGenerator                    m_checkpoint{};

    static inline const ppo::model::sEvaluatorState EMPTY_STATE{};
};

Evaluator::Impl::Impl(void)
{
    m_world.ctx().emplace<ppo::model::sEnvironmentState>();
    m_world.ctx().emplace<ppo::model::sEvaluatorState>();
    auto& evalName = m_world.ctx().emplace<ppo::model::sEvaluatorName>();

    evalName.m_filename = "Evaluation.json";

    if (m_agent == nullptr)
    {
        p_blue  = CreateEntity<ppo::model::Aircraft>("evalaircraft", ppo::model::eEntitySide::Friend);
        p_red   = CreateEntity<ppo::model::Aircraft>("evalaircraft", ppo::model::eEntitySide::Hostile);
        m_agent = CreateAgent<SNAProdAgent>(*p_blue.get());

        auto& checkpointblue = p_blue->add<ppo::model::sEntityCheckPoint>();
        checkpointblue.m_name = "blueeval";
        ppo::model::CheckpointInitializerSystem::Load(*p_blue.get());

        auto& checkpointred = p_red->add<ppo::model::sEntityCheckPoint>();
        checkpointred.m_name = "redeval";
        ppo::model::CheckpointInitializerSystem::Load(*p_red.get());

        m_entities["blueeval"] = p_blue;
        m_entities["redeval"] = p_red;
    }

    // use same env as training environment
    ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
    serializer.Register(
        "Evaluation.json",
        [this](ppo::serialization::json& ar)
        {
            this->LoadFromFile(ar);
        },
        [this](ppo::serialization::json& ar)
        {
            this->SaveToFile(ar);
        }
    );
}

void
Evaluator::Impl::UpdateDoubleBuffer(void)
{
    auto const& state{ m_world.ctx().get<ppo::model::sEvaluatorState>() };

    const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    memcpy(&m_buffers[next], &state, sizeof(ppo::model::sEvaluatorState));

    m_index.store(next, std::memory_order_release);
}

void 
Evaluator::Impl::LoadFromFile(ppo::serialization::json& ar)
{
    auto& state   { m_world.ctx().get<ppo::model::sEvaluatorState>() };
    auto& envState{ m_world.ctx().get<ppo::model::sEnvironmentState>() };

    auto section = ar["Eval"];

    envState.m_maxNorth =  section["MaxNorth"];
    envState.m_maxEast  =  section["MaxEast" ];
    envState.m_dt       =  section["TimeStep"];
    state.m_nbstep      =  section["NbSteps" ];
    state.m_nbtests     =  section["NbTests" ];
    state.m_notation    =  section["Notation"];

    UpdateDoubleBuffer();
}

void
Evaluator::Impl::SaveToFile(ppo::serialization::json& ar)
{
    auto& state   { m_world.ctx().get<ppo::model::sEvaluatorState>() };
    auto& envState{ m_world.ctx().get<ppo::model::sEnvironmentState>() };

    auto& section = ar["Eval"];

    section["MaxNorth"] = envState.m_maxNorth;
    section["MaxEast" ] = envState.m_maxEast ;
    section["TimeStep"] = envState.m_dt      ;
    section["NbSteps" ] = state.m_nbstep     ;
    section["NbTests" ] = state.m_nbtests    ;
    section["Notation"] = state.m_notation   ;

    m_checkpoint.serialize(m_world, m_entities, ar);
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

    if (p_red != nullptr
     && p_red->has<ppo::model::sSNAState>()
     && p_blue != nullptr
     && p_blue->has<ppo::model::sSNAState>())
    {
        auto [blue_snaState] = p_blue->get<ppo::model::sSNAState>();
        auto [red_snaState] = p_red->get< ppo::model::sSNAState>();

        blue_snaState.m_target = p_red->uniqid();
        red_snaState.m_target = p_blue->uniqid();
    }

    m_running.store(true, std::memory_order_release);
    m_thread = std::thread(&Evaluator::Impl::Run, this);
}

void
Evaluator::Impl::Run(void)
{
    if (m_agent == nullptr
    || p_blue == nullptr
    || p_red == nullptr)
        return;

    auto& state{ m_world.ctx().get<ppo::model::sEvaluatorState>() };

    // for all test available
    float playerDestroyed{0.F};
    float targetDestroyed{0.F};
    float missionDuration{0.F};
    float pkill{ 0.F };
    float notation{ -FLT_MAX };

    state.m_inprogress = true;
    state.m_progress = 0.F;
    UpdateDoubleBuffer();

    for (int i = 0; i < state.m_nbtests; ++i)
    {
        // thread must terminate
        if (!m_running.load(std::memory_order_acquire))
            return;

        // initialize the player and the plicy
        for (auto& [name, entity] : m_entities)
        {
            if (entity == nullptr)
                continue;

            auto [checkpoint] = entity->get<ppo::model::sEntityCheckPoint>();
            checkpoint.m_index = i;
            entity->Init();
        }

        m_agent->Init();

        // perform all step until m_nbstep
        float pk{ 0.F };
        int j{ 0 };
        for (; j < state.m_nbstep; ++j)
        {
            // thread must terminate
            if (!m_running.load(std::memory_order_acquire))
                return;

            // choose action according to policy
            m_agent->ChooseAction();

            bool targetDestruction{ false };
            float newpk{ 0.F };

            for (auto& [_, entity] : m_entities)
            {
                if (entity == nullptr)
                    continue;

                entity->Run();
            }

            for (auto& [_, entity] : m_entities)
            {
                if (entity == nullptr
                ||  !entity->has<ppo::model::sSNAState>())
                    continue;

                auto const& [snaState] = entity->get<ppo::model::sSNAState>();

                if (m_world.any_of<ppo::model::sEntityState>(snaState.m_target))
                {
                    float const pkTarget{ snaState.m_pk };

                    auto [targetState] = m_world.view<ppo::model::sEntityState>().get(snaState.m_target);

                    if (0.95F < pkTarget)
                    {
                        targetState.m_destroyed = true;

                        if (entity == p_blue)
                            targetDestruction = true;
                    }

                    if (entity == p_blue)
                        newpk = pkTarget;
                }
            }

            m_agent->SetNewState();

            bool goToNextTest(false);

            if (p_blue->has<ppo::model::sEntityState>())
            {
                auto [entityState] = p_blue->get<ppo::model::sEntityState>();
               
                pk = std::max(pk,newpk);

                if (entityState.m_destroyed)
                {
                    playerDestroyed += 1.F;
                    goToNextTest = true;
                }
                else if (targetDestruction)
                {
                    targetDestroyed += 1.F;
                    goToNextTest = true;
                }
            }
                 
            if (goToNextTest)
                break;
        }

        pkill += pk;
        missionDuration += static_cast<float>(state.m_nbstep+1)/static_cast<float>(j+1);

        state.m_inprogress = true;
        state.m_progress = static_cast<float>(i) / static_cast<float>(state.m_nbtests);

        UpdateDoubleBuffer();

        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    notation = 100.F * (targetDestroyed + pkill - playerDestroyed + missionDuration) / static_cast<float>(state.m_nbtests);
    
    if (state.m_notation < notation)
    {
        // move the model into prod
        std::filesystem::path proddirectory{ m_modelpath / "prod" };
        std::filesystem::path evalmodel{ m_modelpath / "eval" / "model.pt" };

        if (!std::filesystem::exists(proddirectory))
            std::filesystem::create_directories(proddirectory);

        std::filesystem::path prodmodel{ proddirectory / "model.pt" };
        std::filesystem::copy_file(evalmodel, prodmodel, std::filesystem::copy_options::overwrite_existing);

        state.m_notation = notation;
    }

    state.m_inprogress = false;
    state.m_progress   = 0.F;
    UpdateDoubleBuffer();
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

} /* rl */