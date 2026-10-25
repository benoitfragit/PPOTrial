#pragma once
// Embedded RLtools learner, controlled by PPO.exe Play/Pause/Stop.
#include "patrol_core.hpp"
#include "iagent.hpp"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ppo::examples::twovstwo::defenders {
class PatrolPolicy;
class PatrolTrainingRuntimeAgent final : public ppo::model::IAgent,
                                         public ppo::model::IPlaybackObserver
{
public:
    PatrolTrainingRuntimeAgent(Parameters params,
        std::shared_ptr<PatrolPolicy> policy, std::filesystem::path checkpoint,
        std::uint32_t seed=42U,std::uint64_t updates=300U,
        std::uint64_t saveEvery=10U);
    ~PatrolTrainingRuntimeAgent() override;
    PatrolTrainingRuntimeAgent(PatrolTrainingRuntimeAgent const&)=delete;
    PatrolTrainingRuntimeAgent& operator=(PatrolTrainingRuntimeAgent const&)=delete;
    void Init() override {} // No optimizer reset on ECS episode reset.
    void ChooseAction() override;
    bool SetNewState() override {return false;}
    void ProcessCommand(ppo::model::sAgentRealTimeCommand const&) noexcept override {}
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;
    void OnPlaybackState(ppo::model::PlaybackState s) noexcept override;
private:
    void Run();
    void Fail(char const* what) noexcept;
    Parameters params_;
    std::shared_ptr<PatrolPolicy> policy_;
    std::filesystem::path checkpoint_;
    std::uint32_t seed_;
    std::uint64_t maxUpdates_;
    std::uint64_t saveEvery_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::string publishedActor_;
    std::string error_;
    std::uint64_t publishedVersion_=0U;
    std::uint64_t loadedVersion_=0U; // accessed by ECS thread only
    std::atomic<std::uint64_t> completed_{0U};
    std::atomic<bool> active_{false},finished_{false};
    bool stop_=false,started_=false;
    unsigned long long id_=ppo::model::GetNewUniqId();
};
} // namespace ppo::examples::twovstwo::defenders
