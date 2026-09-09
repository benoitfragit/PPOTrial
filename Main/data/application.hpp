#pragma once
#include "navigation.hpp"
#include "policybackend.hpp"
#include <memory>
#include <string>

namespace navigation {
enum class ControlMode { Manual, Training, Policy };
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
    void tick(double elapsed, const std::vector<float>& manual_action, double time_scale);
    void evaluate_candidate(std::size_t episodes);
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
}
