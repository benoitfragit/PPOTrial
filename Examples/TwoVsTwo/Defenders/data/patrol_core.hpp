#pragma once

// Deterministic 2D patrol curriculum. No enemy is spawned in this first stage.
// Coordinates: north x / east y, metres; heading 0 towards north.
// No dependencies on Torch, EnTT, a renderer, or RLtools in this domain layer.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace ppo::examples::twovstwo::defenders {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 2.F * kPi;

struct ObservationLayout final
{
    static constexpr std::size_t Own = 7;
    static constexpr std::size_t Lidar = 61;
    static constexpr std::size_t RedTracks = 2;
    static constexpr std::size_t RedFeatures = 13; // 9 historical + normalized x/y + inside + depth
    static constexpr std::size_t FriendlyTracks = 2; // other guard + fixed objective
    static constexpr std::size_t FriendlyFeatures = 9; // typed, masked tracks
    static constexpr std::size_t Patrol = 10;
    static constexpr std::size_t Global = 2;
    static constexpr std::size_t RedOffset = Own + Lidar;
    static constexpr std::size_t FriendOffset = RedOffset + RedTracks * RedFeatures;
    static constexpr std::size_t PatrolOffset = FriendOffset + FriendlyTracks * FriendlyFeatures;
    static constexpr std::size_t GlobalOffset = PatrolOffset + Patrol;
    static constexpr std::size_t Size = GlobalOffset + Global;
    static constexpr std::size_t ActionSize = 2;
    static constexpr std::uint32_t Version = 2;
};

using Observation = std::array<float, ObservationLayout::Size>;
using Action = std::array<float, ObservationLayout::ActionSize>;

struct Parameters
{
    float halfNorth = 22329.375F;
    float halfEast = 35727.F;
    float dt = 0.2F;
    float cruiseSpeed = 160.F;
    float maxSpeed = 400.F;
    float maxAcceleration = 100.F;
    float maxRotationRate = 0.4F;
    std::size_t episodeSteps = 4096;
    float collisionDistance = 50.F;
    float cautionDistance = 250.F;
};

inline bool Valid(Parameters const& p) noexcept
{
    return std::isfinite(p.halfNorth) && std::isfinite(p.halfEast)
        && std::isfinite(p.dt) && std::isfinite(p.cruiseSpeed)
        && std::isfinite(p.maxSpeed) && std::isfinite(p.maxAcceleration)
        && std::isfinite(p.maxRotationRate)
        && p.halfNorth > 1000.F && p.halfEast > 1000.F
        && p.dt > 0.F && p.dt <= 1.F && p.cruiseSpeed > 0.F
        && p.maxSpeed >= p.cruiseSpeed && p.maxAcceleration > 0.F
        && p.maxRotationRate > 0.F && p.episodeSteps > 0U
        && std::isfinite(p.collisionDistance) && p.collisionDistance > 0.F
        && std::isfinite(p.cautionDistance)
        && p.cautionDistance > p.collisionDistance;
}

inline float Clamp(float x, float a, float b) noexcept
{
    return std::max(a, std::min(x, b));
}

inline float PositiveModulo(float value, float period) noexcept
{
    float const result = std::fmod(value, period);
    return result < 0.F ? result + period : result;
}

struct Mobile
{
    float north, east;
    float heading, speed;
    float turnRate;
    bool available;
};

struct Reference
{
    float north = 0.F, east = 0.F;
    float heading = 0.F, phase = 0.F;
    float distanceToTurn = 0.F;
};

inline float Radius(Parameters const& p) noexcept
{
    return std::min(p.halfEast * 0.09F, p.halfNorth * 0.13F);
}

inline float StraightHalfLength(Parameters const& p) noexcept
{
    return 0.50F * p.halfNorth;
}

inline float CircuitLength(Parameters const& p) noexcept
{
    return 4.F * StraightHalfLength(p) + kTwoPi * Radius(p);
}

// A genuine racetrack path: two parallel straight segments joined by round
// U-turns. The second guard is slightly inward and behind the first, not
// on exactly the same track. A phase offset staggers both turn events.
inline Reference PatrolReference(Parameters const& p, int side, std::size_t role,
                                 float distance) noexcept
{
    float const half = StraightHalfLength(p);
    float const radius = Radius(p);
    float const straight = 2.F * half;
    float const arc = kPi * radius;
    float const length = 2.F * (straight + arc);
    float const offset = role == 0U ? 0.13F * straight : 0.F;
    float const s = PositiveModulo(distance + offset, length);
    float const centre = (side >= 0 ? 1.F : -1.F)
        * (0.65F - 0.06F * static_cast<float>(role)) * p.halfEast;

    Reference result{};
    result.phase = s / length;
    if (s < straight)
    {
        result.north = -half + s;
        result.east = centre - radius;
        result.heading = 0.F;
        result.distanceToTurn = straight - s;
    }
    else if (s < straight + arc)
    {
        float const angle = -0.5F * kPi + (s - straight) / radius;
        result.north = half + radius * std::cos(angle);
        result.east = centre + radius * std::sin(angle);
        result.heading = angle + 0.5F * kPi;
    }
    else if (s < 2.F * straight + arc)
    {
        float const onLeg = s - straight - arc;
        result.north = half - onLeg;
        result.east = centre + radius;
        result.heading = kPi;
        result.distanceToTurn = straight - onLeg;
    }
    else
    {
        float const angle = 0.5F * kPi + (s - 2.F * straight - arc) / radius;
        result.north = -half + radius * std::cos(angle);
        result.east = centre + radius * std::sin(angle);
        result.heading = angle + 0.5F * kPi;
    }
    return result;
}

inline Mobile OnReference(Parameters const& p, Reference const& ref) noexcept
{
    return {ref.north, ref.east, ref.heading, p.cruiseSpeed, 0.F, true};
}

// Mirror the shared SNA control semantics: heading-rate and longitudinal
// acceleration are scaled by the platform's maxima, then position is
// integrated using the NEW heading and speed. Revisit as a shared kernel
// before expanding beyond this no-threat curriculum.
inline Mobile Advance(Parameters const& p, Mobile current, Action const& action) noexcept
{
    float const turn = std::isfinite(action[0]) ? Clamp(action[0], -1.F, 1.F) : 0.F;
    float const accel = std::isfinite(action[1]) ? Clamp(action[1], -1.F, 1.F) : 0.F;
    current.turnRate = p.maxRotationRate * turn;
    current.heading += current.turnRate * p.dt;
    current.speed = Clamp(current.speed + p.maxAcceleration * accel * p.dt,
                          0.F, p.maxSpeed);
    current.north += std::cos(current.heading) * current.speed * p.dt;
    current.east += std::sin(current.heading) * current.speed * p.dt;
    return current;
}

struct State
{
    std::array<Mobile, 2> guards;
    std::size_t controlledRole;
    int side;
    float arclength;
    std::size_t step;
    bool failed;
    bool collided;
};

// Separation is evaluated for the team, independently of the learning role.
inline float GuardSeparation(State const& state) noexcept
{
    return std::hypot(state.guards[0].north - state.guards[1].north,
                      state.guards[0].east - state.guards[1].east);
}

// Check the relative path over the entire step: two fast cars can cross
// between discrete observations without ending inside the collision radius.
inline float MinimumGuardSeparation(State const& before,
                                    State const& after) noexcept
{
    float const rx = before.guards[0].north - before.guards[1].north;
    float const ry = before.guards[0].east - before.guards[1].east;
    float const dx = (after.guards[0].north - after.guards[1].north) - rx;
    float const dy = (after.guards[0].east - after.guards[1].east) - ry;
    float const travelled2 = dx * dx + dy * dy;
    float const t = travelled2 > 1.e-8F
        ? Clamp(-(rx * dx + ry * dy) / travelled2, 0.F, 1.F)
        : 0.F;
    return std::hypot(rx + t * dx, ry + t * dy);
}

inline bool GuardsCollided(Parameters const& p, State const& before,
                           State const& after) noexcept
{
    return MinimumGuardSeparation(before, after) <= p.collisionDistance;
}

static_assert(std::is_trivially_copyable_v<State>,
    "RLtools stores patrol state in a matrix of trivially copyable values");

// Model one controlled defender and one reference-following teammate.
// Both roles share the same actor weights across RLtools training episodes.
// This curriculum is intentionally not yet a coupled two-actor rollout.
inline State InitialState(Parameters const& p, int side,
                          std::size_t controlledRole, float phase) noexcept
{
    State state{};
    state.side = side < 0 ? -1 : 1;
    state.controlledRole = controlledRole % 2U;
    state.arclength = PositiveModulo(phase, CircuitLength(p));
    for (std::size_t i = 0; i < state.guards.size(); ++i)
        state.guards[i] = OnReference(p, PatrolReference(p, state.side, i, state.arclength));
    return state;
}

inline State Advance(Parameters const& p, State const& prev,
                     Action const& action) noexcept
{
    State next = prev;
    std::size_t const me = next.controlledRole;
    std::size_t const other = 1U - me;
    next.guards[me] = Advance(p, next.guards[me], action);
    next.arclength += p.cruiseSpeed * p.dt;
    next.guards[other] = OnReference(p, PatrolReference(
        p, next.side, other, next.arclength));
    ++next.step;
    next.collided = GuardsCollided(p, prev, next);
    next.failed |= next.collided;
    for (auto const& guard : next.guards)
        if (!std::isfinite(guard.north) || !std::isfinite(guard.east)
            || std::abs(guard.north) >= p.halfNorth
            || std::abs(guard.east) >= p.halfEast)
            next.failed = true;
    return next;
}

// Advance two independent PPO-controlled defenders with a shared world clock.
// Kept separate from the single-learner curriculum's Advance(State, Action).
inline State AdvanceBoth(Parameters const& p, State const& previous,
                         std::array<Action, 2> const& actions) noexcept
{
    State next = previous;
    for (std::size_t role = 0; role < next.guards.size(); ++role)
        next.guards[role] = Advance(p, previous.guards[role], actions[role]);
    next.arclength += p.cruiseSpeed * p.dt;
    ++next.step;
    next.collided = GuardsCollided(p, previous, next);
    next.failed |= next.collided;
    for (auto const& guard : next.guards)
        if (!guard.available || !std::isfinite(guard.north)
            || !std::isfinite(guard.east) || !std::isfinite(guard.heading)
            || !std::isfinite(guard.speed) || !std::isfinite(guard.turnRate)
            || std::abs(guard.north) >= p.halfNorth
            || std::abs(guard.east) >= p.halfEast)
            next.failed = true;
    return next;
}

inline float RayDistanceToBounds(Parameters const& p, Mobile const& car,
                                float angle) noexcept
{
    float const nx = std::cos(angle);
    float const ey = std::sin(angle);
    float best = std::numeric_limits<float>::infinity();
    auto hit = [&](float d, float north, float east) {
        if (d >= 0.F && std::abs(north) <= p.halfNorth + 1.e-3F
            && std::abs(east) <= p.halfEast + 1.e-3F)
            best = std::min(best, d);
    };
    if (std::abs(nx) > 1.e-5F)
    {
        float const a = (p.halfNorth - car.north) / nx;
        float const b = (-p.halfNorth - car.north) / nx;
        hit(a, p.halfNorth, car.east + a * ey);
        hit(b, -p.halfNorth, car.east + b * ey);
    }
    if (std::abs(ey) > 1.e-5F)
    {
        float const a = (p.halfEast - car.east) / ey;
        float const b = (-p.halfEast - car.east) / ey;
        hit(a, car.north + a * nx, p.halfEast);
        hit(b, car.north + b * nx, -p.halfEast);
    }
    return std::isfinite(best) ? best : -1.F;
}

inline Observation Observe(Parameters const& p, State const& state) noexcept
{
    Observation obs{};  // Red-track slots remain exactly zero: absent/masked.
    Mobile const& own = state.guards[state.controlledRole];
    Mobile const& mate = state.guards[1U - state.controlledRole];
    Reference const ref = PatrolReference(
        p, state.side, state.controlledRole, state.arclength);
    std::size_t k = 0;
    obs[k++] = own.north / p.halfNorth;
    obs[k++] = own.east / p.halfEast;
    obs[k++] = own.speed / p.maxSpeed;
    obs[k++] = std::cos(own.heading);
    obs[k++] = std::sin(own.heading);
    obs[k++] = own.turnRate / p.maxRotationRate;
    obs[k++] = own.available ? 1.F : 0.F;
    float const diag = std::hypot(p.halfNorth, p.halfEast);
    for (std::size_t i = 0; i < ObservationLayout::Lidar; ++i)
    {
        float const angle = own.heading - kPi +
            static_cast<float>(i) * (kTwoPi / 60.F);
        float const distance = RayDistanceToBounds(p, own, angle);
        obs[k++] = distance < 0.F ? 1.F : distance / (distance + diag);
    }
    k = ObservationLayout::FriendOffset;
    auto writeFriend = [&](Mobile const& target, float type) {
        float const dx = target.north - own.north;
        float const dy = target.east - own.east;
        float const rel = std::atan2(dy, dx) - own.heading;
        float const distance = std::hypot(dx, dy);
        obs[k++] = dx / (2.F * p.halfNorth);
        obs[k++] = dy / (2.F * p.halfEast);
        obs[k++] = (target.speed * std::cos(target.heading) -
                    own.speed * std::cos(own.heading)) / p.maxSpeed;
        obs[k++] = (target.speed * std::sin(target.heading) -
                    own.speed * std::sin(own.heading)) / p.maxSpeed;
        obs[k++] = distance / (2.F * diag);
        obs[k++] = std::cos(rel);
        obs[k++] = std::sin(rel);
        obs[k++] = type;
        obs[k++] = target.available ? 1.F : 0.F;
    };
    writeFriend(mate, 0.F);
    Mobile const objective{0.F, (state.side >= 0 ? 1.F : -1.F) *
                                   0.92F * p.halfEast, 0.F, 0.F, 0.F, true};
    writeFriend(objective, 1.F);
    k = ObservationLayout::PatrolOffset;
    float const dx = ref.north - own.north;
    float const dy = ref.east - own.east;
    float const along = dx * std::cos(ref.heading) + dy * std::sin(ref.heading);
    float const across = -dx * std::sin(ref.heading) + dy * std::cos(ref.heading);
    obs[k++] = dx / p.halfNorth;
    obs[k++] = dy / p.halfEast;
    obs[k++] = std::cos(ref.heading - own.heading);
    obs[k++] = std::sin(ref.heading - own.heading);
    obs[k++] = along / p.halfNorth;
    obs[k++] = across / p.halfEast;
    obs[k++] = ref.phase;
    obs[k++] = static_cast<float>(state.controlledRole);
    obs[k++] = p.cruiseSpeed / p.maxSpeed;
    obs[k++] = ref.distanceToTurn / CircuitLength(p);
    obs[ObservationLayout::GlobalOffset] = 1.F; // Initial ammunition ratio.
    obs[ObservationLayout::GlobalOffset + 1] = 0.F; // No threat.
    return obs;
}

inline float TrackingReward(Parameters const& p, State const& state,
                            std::size_t role) noexcept
{
    Mobile const& own = state.guards[role];
    Reference const ref = PatrolReference(p, state.side, role, state.arclength);
    float const error = std::hypot(
        (own.north - ref.north) / p.halfNorth,
        (own.east - ref.east) / p.halfEast);
    float const direction = std::cos(own.heading - ref.heading);
    float const progress = (own.speed / p.cruiseSpeed) * direction;
    float const speedError = std::abs(own.speed - p.cruiseSpeed) / p.cruiseSpeed;
    return 0.35F * Clamp(progress, -2.F, 2.F)
         + 0.20F * direction
         - 1.20F * Clamp(error, 0.F, 2.F)
         - 0.20F * Clamp(speedError, 0.F, 2.F);
}

inline float TeamReward(Parameters const& p, State const& before,
                        Action const& action, State const& after) noexcept
{
    if (after.failed)
        return -100.F;
    float const shared = 0.5F * (TrackingReward(p, after, 0U)
                               + TrackingReward(p, after, 1U));
    // Shape close approaches before the actual -100 collision termination.
    // The reference teammate is included, exactly as during PPO rollout.
    float const distance = GuardSeparation(after);
    float const proximity = Clamp(
        (p.cautionDistance - distance) /
        (p.cautionDistance - p.collisionDistance), 0.F, 1.F);
    float const control = 0.002F * (std::abs(action[0]) + std::abs(action[1]));
    float const timeout = after.step >= p.episodeSteps ? -15.F : 0.F;
    (void)before;
    return shared - control - 0.005F - 0.5F * proximity + timeout;
}

} // namespace ppo::examples::twovstwo::defenders
