#pragma once
#include "components.hpp"
#include "lidar.hpp"
#include "ppo/rl/runtime.hpp"
#include <entt/entity/registry.hpp>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <vector>

namespace navigation
{
struct MobileDefinition
{
    EntityId id;
    std::string name;
    State initial;
    EntityId destination = no_entity;
    double arrival_radius = 0.5;
};
struct ZoneDefinition
{
    EntityId id;
    std::string name;
    Point center;
    double radius = 1;
};
struct Scenario
{
    EntityId controlled = 1;
    std::vector<MobileDefinition> mobiles{{1, "Mobile", {{0, 0}, 0, 1}, 2, 0.5}, {2, "Goal", {{10, 0}, 0, 0}, 1, 0.5}};
    std::vector<ZoneDefinition> zones;
    double half_extent = 100;
    double dt = 0.2;
    double max_speed = 5;
    double max_acceleration = 2;
    double max_turn_rate = 1;
    std::size_t max_steps = 500;
};
enum class InitializationMode { Fixed, Randomized, PresetCases };

/// Strict v3 text format. Entity IDs/names must be unique; destinations reference mobiles.
Scenario load_scenario(const std::filesystem::path& path);
std::string scenario_text(const Scenario& scenario);
void save_scenario(const Scenario& scenario, const std::filesystem::path& path);
/// Manifest: one scenario filename per line, relative to the manifest directory.
/// Cases must contain the same mobile and zone IDs and controller relationships.
std::vector<Scenario> load_episode_cases(const std::filesystem::path& manifest);

/// One ECS world and a Gymnasium view for scenario.controlled. All entities advance once
/// per step, after every controller has observed the same pre-step world. Single-thread use.
/// Other mobiles may coast, accept manual commands or own an independent generic policy.
/// Only explicitly attached learners receive feedback. PPO/LibTorch are never referenced.
class Environment final : public ppo::rl::IEnvironment
{
public:
    explicit Environment(Scenario scenario);
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    Environment(Environment&&) = delete;
    Environment& operator=(Environment&&) = delete;
    ppo::rl::ResetResult reset(const ppo::rl::ResetOptions& options = {}) override;
    /// Configure varied episode starts without invalidating attached mobile controllers.
    /// Only accepted at an episode boundary; seed reproducibility is preserved.
    void configure_initialization(InitializationMode mode, std::vector<Scenario> cases = {});
    InitializationMode initialization_mode() const noexcept;
    const std::vector<Scenario>& episode_cases() const noexcept;
    ppo::rl::StepResult step(const ppo::rl::Value& action) override;
    const ppo::rl::Space& observation_space() const noexcept override;
    const ppo::rl::Space& action_space() const noexcept override;
    const entt::registry& registry() const noexcept;
    const std::vector<EntityId>& entities() const noexcept;
    entt::entity handle(EntityId id) const; // throws for unknown IDs, including removed entities
    State state(EntityId id) const;
    State state() const;
    State goal_state() const;
    Point position() const;
    ppo::rl::Value observation(EntityId mobile) const;
    /// Independent 360-degree scan for any mobile, including non-primary agents.
    LidarScan lidar(EntityId mobile) const;
    const ppo::rl::StepResult& transition(EntityId mobile) const;
    /// Changes are allowed only before the first step or after an episode boundary.
    /// The externally driven primary cannot also have an internal controller.
    void set_policy(EntityId mobile, std::unique_ptr<ppo::rl::IPolicy> policy, bool learning = false);
    void set_manual_action(EntityId mobile, Commands commands);
    void set_goal_policy(std::unique_ptr<ppo::rl::IPolicy> policy);
    /// Topology edits validate a complete scenario; reject while active or policies attached.
    /// They require reset and preserve remaining stable IDs, not EnTT handles.
    void create_mobile(MobileDefinition mobile);
    void create_zone(ZoneDefinition zone);
    void remove_entity(EntityId id);
    void request_stop() noexcept;
    const Scenario& scenario() const noexcept;
    /// The initial scenario stays fixed even while episode presets are active.
    const Scenario& baseline_scenario() const noexcept;

private:
    struct Controller
    {
        std::unique_ptr<ppo::rl::IPolicy> policy;
        ppo::rl::ILearner* learner = nullptr; // owned by policy
    };
    void rebuild(Scenario scenario);
    void require_editable() const;
    Scenario scenario_;
    Scenario baseline_scenario_;
    InitializationMode initialization_mode_ = InitializationMode::Fixed;
    std::vector<Scenario> episode_cases_;
    std::mt19937_64 random_engine_{5489};
    std::size_t episode_index_ = 0;
    entt::registry world_;
    std::map<EntityId, entt::entity> handles_;
    std::vector<EntityId> entities_;
    std::map<EntityId, Controller> controllers_;
    std::map<EntityId, ppo::rl::StepResult> transitions_;
    ppo::rl::Space observations_;
    ppo::rl::Space actions_ = ppo::rl::Box({2}, {-1.F, -1.F}, {1.F, 1.F});
    std::size_t steps_ = 0;
    bool active_ = false;
    bool stop_requested_ = false;
};
} // namespace navigation
