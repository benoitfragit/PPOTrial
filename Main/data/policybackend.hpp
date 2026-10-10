#pragma once
#include "ppo/rl/runtime.hpp"
#include <memory>

namespace navigation {
/// Backend-agnostic PPO settings for newly constructed learners.
struct TrainingParameters
{
    float gamma = 0.99F;
    float lambda_gae = 0.95F;
    float clip_epsilon = 0.2F;
    float value_coefficient = 0.5F;
    float entropy_coefficient = 0.02F;
    float learning_rate = 3e-4F;
    int epochs = 4;
    int batch_size = 64;
    float target_kl = 0.01F;
};
void validate_training_parameters(const TrainingParameters& parameters);

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
    /// Backends supporting PPO configuration should override this overload.
    /// Existing test/mock implementations can retain the legacy factory.
    virtual std::unique_ptr<ppo::rl::ILearner> create_learner_configured(
        std::size_t observations, std::size_t actions, std::size_t rollout,
        const std::filesystem::path& root, const TrainingParameters& settings)
    {
        validate_training_parameters(settings);
        return create_learner(observations, actions, rollout, root);
    }
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
    std::unique_ptr<ppo::rl::ILearner> create_learner_configured(
        std::size_t observations, std::size_t actions, std::size_t rollout,
        const std::filesystem::path& root, const TrainingParameters& settings) override;
    std::unique_ptr<ppo::rl::IPolicy> load_policy(
        std::size_t observations, std::size_t actions, const std::filesystem::path& checkpoint) override;
    BackendTelemetry telemetry(const ppo::rl::ILearner& learner) const override;
    void save(ppo::rl::ILearner& learner, const std::filesystem::path& candidate) override;
};
}
