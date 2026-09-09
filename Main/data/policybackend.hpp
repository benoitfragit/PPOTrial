#pragma once
#include "ppo/rl/runtime.hpp"
#include <memory>

namespace navigation {
struct BackendTelemetry
{
    std::size_t rollout_step = 0;
    float loss = 0;
    float actor_loss = 0;
    float critic_loss = 0;
    float entropy = 0;
};
/// Application-side policy services. Implementations and referenced learners must agree.
/// Keeps desktop orchestration testable without creating or loading a neural model.
class IPolicyBackend
{
public:
    virtual ~IPolicyBackend() = default;
    virtual std::unique_ptr<ppo::rl::ILearner> create_learner(
        std::size_t observations, std::size_t actions, std::size_t rollout,
        const std::filesystem::path& root) = 0;
    virtual std::unique_ptr<ppo::rl::IPolicy> load_policy(
        std::size_t observations, std::size_t actions, const std::filesystem::path& checkpoint) = 0;
    virtual BackendTelemetry telemetry(const ppo::rl::ILearner& learner) const = 0;
    virtual void save(ppo::rl::ILearner& learner, const std::filesystem::path& candidate) = 0;
};
class TorchBackend final : public IPolicyBackend
{
public:
    std::unique_ptr<ppo::rl::ILearner> create_learner(
        std::size_t observations, std::size_t actions, std::size_t rollout,
        const std::filesystem::path& root) override;
    std::unique_ptr<ppo::rl::IPolicy> load_policy(
        std::size_t observations, std::size_t actions, const std::filesystem::path& checkpoint) override;
    BackendTelemetry telemetry(const ppo::rl::ILearner& learner) const override;
    void save(ppo::rl::ILearner& learner, const std::filesystem::path& candidate) override;
};
}
