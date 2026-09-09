#pragma once
#include "ppo/rl/environment.hpp"
#include <filesystem>
#include <string>

namespace navigation {
struct Point { double x = 0; double y = 0; };
struct Obstacle { Point center; double radius = 1; };
struct Scenario
{
    Point start{0, 0};
    Point goal{10, 0}; // initial position of the second mobile
    Point goal_velocity{0, 0}; // constant world-space velocity
    double heading = 0;
    double speed = 1;
    double half_extent = 100;
    double arrival_radius = 0.5;
    double dt = 0.2;
    double max_speed = 5;
    double max_acceleration = 2;
    double max_turn_rate = 1;
    std::size_t max_steps = 500;
    std::vector<Obstacle> obstacles;
};
/// Versioned, strict text format; rejects unknown keys, trailing fields and invalid values.
Scenario load_scenario(const std::filesystem::path& path);
/// Validated canonical scenario serialization used for persistence and evaluation identity.
std::string scenario_text(const Scenario& scenario);
void save_scenario(const Scenario& scenario, const std::filesystem::path& path);
struct State { Point position; double heading; double speed; };
/// Local example with a moving goal mobile and circular forbidden regions.
/// Its linear trajectory must remain inside the world for the episode horizon.
/// Kinematic update reuses the original heading/speed Euler integration principle.
class Environment final : public ppo::rl::IEnvironment
{
public:
    explicit Environment(Scenario scenario);
    ppo::rl::ResetResult reset(const ppo::rl::ResetOptions& options = {}) override;
    ppo::rl::StepResult step(const ppo::rl::Value& action) override;
    const ppo::rl::Space& observation_space() const noexcept override;
    const ppo::rl::Space& action_space() const noexcept override;
    Point position() const noexcept;
    State state() const noexcept;
    State goal_state() const noexcept;
    const Scenario& scenario() const noexcept;
private:
    ppo::rl::Value observe() const;
    Scenario scenario_;
    ppo::rl::Space observations_;
    ppo::rl::Space actions_ = ppo::rl::Box({2}, {-1.F,-1.F}, {1.F,1.F});
    Point position_;
    Point goal_position_;
    double heading_ = 0;
    double speed_ = 0;
    std::size_t steps_ = 0;
    bool active_ = false;
};
} // namespace navigation
