#pragma once
#include <cstdint>
#include <string>

namespace navigation
{
using EntityId = std::uint64_t;
constexpr EntityId no_entity = 0;
struct Point
{
    double x = 0;
    double y = 0;
};
struct State
{
    Point position;
    double heading = 0;
    double speed = 0;
};
/// Stable scenario identity; independent of EnTT's recyclable runtime handle.
struct Identity
{
    EntityId id;
    std::string name;
};
struct Mobile
{
};
struct ForbiddenZone
{
    double radius;
};
/// Optional 2D mobility endurance model. Capacity 0 disables fuel accounting.
/// Burn per second: idle + speed_cubic * v^3 + turning * omega^2.
struct EnergyModel
{
    double min_speed = 0;
    double capacity = 0;
    double idle = 0;
    double speed_cubic = 0;
    double turning = 0;
};
/// Editable reward-channel weights for an individual mobile.
struct RewardWeights
{
    double progress = 1.0;
    double step_cost = 0.001;
    double arrival = 1.0;
    double collision = 1.0;
    double safety = 0.05;
    double boundary = 0.05;
    double fuel = 0.01;
    double failure = 1.0;
};
struct EnergyState
{
    double remaining = 0;
    double last_burn = 0;
    bool exhausted = false;
    bool stalled = false;
};
struct Destination
{
    EntityId entity;
    double radius;
};
struct InitialState
{
    State value;
};
struct Commands
{
    float turn = 0;
    float acceleration = 0;
};
} // namespace navigation
