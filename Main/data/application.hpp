#pragma once
#include "navigation.hpp"
#include "policybackend.hpp"
#include <memory>
#include <string>

namespace navigation
{
enum class ControlMode
{
    Manual,
    Training,
    Policy
};
struct Telemetry
{
    std::size_t episodes = 0;
    std::size_t successes = 0;
    std::size_t rollout_step = 0;
    double reward = 0;
    std::size_t length = 0;
    float loss = 0, actor_loss = 0, critic_loss = 0, entropy = 0;
    bool terminated = false, truncated = false;
    std::vector<float> rewards;
    std::vector<float> losses;
};
/// Desktop orchestration only. The environment and RL core never reference this class.
/// All calls occur on the UI thread; policies are created only by an explicit command.
class Application
{
public:
    /// backend must outlive this application. Construction starts paused in manual mode.
    Application(Scenario scenario, std::filesystem::path metrics, std::filesystem::path checkpoints,
                IPolicyBackend& backend);
    ~Application();
    void play(ControlMode mode, std::size_t rollout, const std::filesystem::path& checkpoint);
    void pause();
    void stop();
    void tick(double elapsed, const std::vector<float>& manual_action, double time_scale,
              EntityId manual_mobile = no_entity);
    void evaluate_candidate(std::size_t episodes, EntityId evaluated_mobile = no_entity);
    /// Fixed, seed-randomized or explicitly replayed complete-world starts.
    /// Change only while stopped, before an episode starts or after it finishes.
    void configure_episode_initialization(InitializationMode mode, std::vector<Scenario> cases = {});
    void set_reward_weights(EntityId mobile, RewardWeights weights);
    /// PPO settings used only for subsequently created training controllers.
    void set_training_parameters(TrainingParameters parameters);
    const TrainingParameters& training_parameters() const noexcept;
    /// Configure any additional mobile independently, at a stopped episode boundary.
    void configure_mobile(EntityId mobile, ControlMode mode, std::size_t rollout,
                          const std::filesystem::path& checkpoint = {});
    ControlMode mobile_mode(EntityId mobile) const;
    const std::filesystem::path& mobile_checkpoint(EntityId mobile) const;
    const Telemetry& telemetry(EntityId mobile) const;
    /// Rebuild a stopped, model-free world from a complete validated scenario.
    void replace_scenario(Scenario scenario);
    /// Select an explicit immutable Prod/N-1 model.pt, or an empty path for linear motion.
    /// Stop the current episode first. A private snapshot is pinned until changed explicitly.
    void set_goal_checkpoint(const std::filesystem::path& checkpoint);
    const std::filesystem::path& goal_checkpoint() const noexcept;
    const Environment& environment() const noexcept;
    const ppo::rl::Value& observation() const noexcept;
    const Telemetry& telemetry() const noexcept;
    bool running() const noexcept;
    bool evaluating() const noexcept;
    ControlMode mode() const noexcept;
    const std::string& status() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace navigation
