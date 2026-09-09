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
