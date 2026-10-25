#include "patrol_training_runtime_agent.hpp"
#include "patrol_training_agent.hpp"
#include "patrol_policy.hpp"
#include "patrol_checkpoint.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppo::examples::twovstwo::defenders {
PatrolTrainingRuntimeAgent::PatrolTrainingRuntimeAgent(Parameters params,
    std::shared_ptr<PatrolPolicy> policy,std::filesystem::path checkpoint,
    std::uint32_t seed,std::uint64_t updates,std::uint64_t saveEvery)
    :params_(params),policy_(std::move(policy)),checkpoint_(std::move(checkpoint)),
     seed_(seed),maxUpdates_(updates),saveEvery_(saveEvery)
{
    if(!Valid(params_)||!policy_||checkpoint_.empty()
       ||maxUpdates_==0U||saveEvery_==0U)
        throw std::invalid_argument("Invalid embedded patrol learner");
}
PatrolTrainingRuntimeAgent::~PatrolTrainingRuntimeAgent()
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        stop_=true;
    }
    active_.store(false,std::memory_order_release);
    cv_.notify_all();
    if(worker_.joinable()) worker_.join();
}
void PatrolTrainingRuntimeAgent::Fail(char const* what) noexcept
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        try {error_=what;}catch(...){}
    }
    active_.store(false,std::memory_order_release);
    finished_.store(true,std::memory_order_release);
}
void PatrolTrainingRuntimeAgent::OnPlaybackState(ppo::model::PlaybackState state) noexcept
{
    bool const play=state==ppo::model::PlaybackState::Running;
    active_.store(play && !finished_.load(std::memory_order_acquire),
                  std::memory_order_release);
    if(play)
    {
        try
        {
            std::lock_guard<std::mutex> lock{mutex_};
            if(!started_ && !finished_.load(std::memory_order_acquire))
            {
                worker_=std::thread(&PatrolTrainingRuntimeAgent::Run,this);
                started_=true;
            }
        }
        catch(std::exception const& e)
        {
            std::cerr<<"[PATROL] Failed to start PPO thread: "<<e.what()<<'\n';
            Fail(e.what());
        }
    }
    cv_.notify_all();
}
void PatrolTrainingRuntimeAgent::Run()
{
    try
    {
        PatrolTrainingAgent learner{params_,seed_};
        std::uint64_t previous=0U;
        while(true)
        {
            {
                std::unique_lock<std::mutex> lock{mutex_};
                cv_.wait(lock,[&]{return stop_||active_.load(std::memory_order_acquire);});
                if(stop_) break;
            }
            bool const done=learner.Step();
            auto const count=learner.StepCount();
            completed_.store(count,std::memory_order_release);
            if(count>previous)
            {
                std::ostringstream buffer;
                learner.WithActor([&](auto const& actor) {
                    WritePatrolPolicy(buffer,actor);
                    if(count%saveEvery_==0U||done||count>=maxUpdates_)
                        SavePatrolPolicy(checkpoint_,actor);
                });
                {
                    std::lock_guard<std::mutex> lock{mutex_};
                    publishedActor_=buffer.str();
                    publishedVersion_=count;
                }
                previous=count;
            }
            if(done||count>=maxUpdates_)
            {
                finished_.store(true,std::memory_order_release);
                active_.store(false,std::memory_order_release);
                std::cout<<"[PATROL] PPO completed after "<<count<<" updates\n";
                break;
            }
        }
        if(learner.StepCount()>0)
            learner.WithActor([&](auto const& actor){
                SavePatrolPolicy(checkpoint_,actor);
            });
    }
    catch(std::exception const& e)
    {
        std::cerr<<"[PATROL] PPO training failed: "<<e.what()<<'\n';
        Fail(e.what());
    }
}
void PatrolTrainingRuntimeAgent::ChooseAction()
{
    // Called before PatrolProdAgent in the ECS loop, never on the trainer thread.
    std::string text;
    std::uint64_t version=0U;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if(publishedVersion_<=loadedVersion_) return;
        text=publishedActor_;
        version=publishedVersion_;
    }
    try {policy_->ReloadFromText(text);loadedVersion_=version;}
    catch(std::exception const& e)
    {
        std::cerr<<"[PATROL] PPO actor publication failed: "<<e.what()<<'\n';
        Fail(e.what());
    }
}
ppo::model::sAgentStateSnaphot PatrolTrainingRuntimeAgent::GetSnapshot() const noexcept
{
    ppo::model::sAgentStateSnaphot snap{};
    snap.m_name="Patrol PPO learner (RLtools)";
    snap.m_id=id_;
    snap.m_common.m_step=static_cast<long long>(completed_.load(std::memory_order_acquire));
    snap.m_common.m_maxStep=static_cast<long long>(maxUpdates_);
    snap.m_common.m_training=active_.load(std::memory_order_acquire)
        && !finished_.load(std::memory_order_acquire);
    snap.m_common.m_done=finished_.load(std::memory_order_acquire);
    return snap;
}
} // namespace ppo::examples::twovstwo::defenders
