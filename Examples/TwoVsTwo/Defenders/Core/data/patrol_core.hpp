#pragma once

// Deterministic 2D patrol and protection domain used by the blue MAPPO team.
// Coordinates: north x / east y, metres; heading 0 towards north.
// No dependencies on Torch, EnTT, a renderer, or RLtools in this domain layer.
#include "patrol_geometry.hpp"
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
    // Kinematics/type, normalized world position, defended-zone membership,
    // then the alive/availability mask in the final slot (as in RLMission).
    static constexpr std::size_t RedFeatures = 13;
    static constexpr std::size_t FriendlyTracks = 2; // other guard + fixed objective
    static constexpr std::size_t FriendlyFeatures = 9; // typed, masked tracks
    static constexpr std::size_t Patrol = 10;
    static constexpr std::size_t Global = 2;
    static constexpr std::size_t RedOffset = Own + Lidar;
    static constexpr std::size_t FriendOffset = RedOffset + RedTracks * RedFeatures;
    static constexpr std::size_t PatrolOffset = FriendOffset + FriendlyTracks * FriendlyFeatures;
    static constexpr std::size_t GlobalOffset = PatrolOffset + Patrol;
    static constexpr std::size_t LocalSize = GlobalOffset + Global;
    static constexpr std::size_t CentralEntityFeatures = 7;
    static constexpr std::size_t CentralEntities = 4; // Two guards and two attackers.
    static constexpr std::size_t CentralObjective = 3;
    static constexpr std::size_t CentralPatrol = 1;
    static constexpr std::size_t CentralOffset = LocalSize;
    static constexpr std::size_t CentralSize =
        CentralEntities * CentralEntityFeatures + CentralObjective + CentralPatrol;
    static constexpr std::size_t Size = LocalSize + CentralSize;
    static constexpr std::size_t ActionSize = 2;
    static constexpr std::uint32_t Version = 5;
};

using Observation = std::array<float, ObservationLayout::Size>;
using Action = std::array<float, ObservationLayout::ActionSize>;

struct Parameters
{
    float halfNorth = 22329.375F;
    float halfEast = 35727.F;
    float dt = 0.2F;
    float cruiseSpeed = 300.F;
    float maxSpeed = 700.F;
    float maxAcceleration = 600.F;
    float maxRotationRate = 0.7000006F;
    std::size_t episodeSteps = 4096;
    float collisionDistance = 1000.F;
    float cautionDistance = 2000.F;
    float hardBoundaryScale = 2.F;
    float boundaryPenaltyDepth = 0.25F;
    float minimumThreatBoundaryScale = 0.20F;
    float patrolProgressWeight = 0.50F;
    float patrolCrossTrackWeight = 0.75F;
    float patrolCrossTrackScale = 4000.F;
    float formationErrorWeight = 0.40F;
    float formationTolerance = 0.10F;
    float separationPenaltyWeight = 0.50F;
    float controlEffortWeight = 0.002F;
    PatrolGeometry geometry{};
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
        && p.cautionDistance > p.collisionDistance
        && std::isfinite(p.hardBoundaryScale) && p.hardBoundaryScale > 1.F
        && std::isfinite(p.boundaryPenaltyDepth) && p.boundaryPenaltyDepth > 0.F
        && std::isfinite(p.minimumThreatBoundaryScale)
        && p.minimumThreatBoundaryScale >= 0.F
        && p.minimumThreatBoundaryScale <= 1.F
        && std::isfinite(p.patrolProgressWeight)
        && p.patrolProgressWeight >= 0.F
        && std::isfinite(p.patrolCrossTrackWeight)
        && p.patrolCrossTrackWeight >= 0.F
        && std::isfinite(p.patrolCrossTrackScale)
        && p.patrolCrossTrackScale > 0.F
        && std::isfinite(p.formationErrorWeight)
        && p.formationErrorWeight >= 0.F
        && std::isfinite(p.formationTolerance)
        && p.formationTolerance >= 0.F
        && std::isfinite(p.separationPenaltyWeight)
        && p.separationPenaltyWeight >= 0.F
        && std::isfinite(p.controlEffortWeight)
        && p.controlEffortWeight >= 0.F
        && p.geometry.length > p.geometry.width
        && p.geometry.width > 0.F
        && p.geometry.separationDistance > p.collisionDistance
        && p.geometry.separationAngle > 0.F
        && p.geometry.separationAngle < 0.5F * kPi
        && std::abs(p.geometry.direction) == 1;
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
    return 0.5F * p.geometry.width;
}

inline float StraightHalfLength(Parameters const& p) noexcept
{
    return 0.5F * (p.geometry.length - p.geometry.width);
}

inline float CircuitLength(Parameters const& p) noexcept
{
    return CircuitLength(p.geometry);
}

/**
 * @brief Returns the nominal leader-to-follower arclength gap in metres.
 * @param p Valid patrol geometry.
 * @return Gap scaled to the long straight and kept above the caution radius.
 */
inline float FormationGap(Parameters const& p) noexcept
{
    return p.geometry.separationDistance
        * std::cos(p.geometry.separationAngle);
}

/**
 * @brief Returns the long-axis racetrack reference for one defender.
 * @param p Valid patrol and world parameters.
 * @param side Objective side; it mirrors the initial travel direction.
 * @param role Defender role in the shared policy, in the range [0, 1].
 * @param distance Team arclength travelled on the circuit in metres.
 * @return Reference pose and progress information.
 *
 * The straight legs follow the east (long) axis. The second guard is behind
 * the first on the same patrol envelope so their U-turns remain staggered.
 */
inline Reference PatrolReference(Parameters const& p, int side, std::size_t role,
                                 float distance) noexcept
{
    (void)side;
    PatrolCurvePose const pose = EvaluatePatrolCurve(
        p.geometry, role % 2U, distance);
    return {pose.north, pose.east, pose.heading, pose.phase, 0.F};
}

inline Mobile OnReference(Parameters const& p, Reference const& ref) noexcept
{
    return {ref.north, ref.east, ref.heading, p.cruiseSpeed, 0.F, true};
}

// Mirror the shared SNA control semantics: heading-rate and longitudinal
// acceleration are scaled by the platform's maxima, then position is
// integrated using the NEW heading and speed. Revisit as a shared kernel
// before adding another vehicle model.
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
    std::array<Mobile, 2> attackers{};
    std::size_t controlledRole;
    int side;
    /** @brief Nearest continuous team arclength for each guard. */
    std::array<float, 2> patrolProgress{};
    /** @brief Per-guard blend from patrol (0) to defence (1). */
    std::array<float, 2> defenseWeight{};
    /** @brief Initial team phase retained for deterministic evaluation cases. */
    float arclength;
    std::size_t step;
    bool failed;
    bool collided;
    bool objectiveAvailable{true};
};

/**
 * @brief Signed longitudinal and inward formation errors in leader coordinates.
 */
struct FormationErrors final
{
    float longitudinal{};
    float inward{};
};

/**
 * @brief Computes the current wedge error without advancing either mobile.
 * @param p Episode patrol parameters.
 * @param state Current team state.
 * @return Errors normalized by the configured separation distance.
 */
inline FormationErrors MeasureFormationErrors(Parameters const& p,
                                              State const& state) noexcept
{
    PatrolProjection const leaderProjection = ProjectOnPatrolCurve(
        p.geometry, 0U, state.guards[0].north, state.guards[0].east);
    float const tangentNorth = std::cos(leaderProjection.pose.heading);
    float const tangentEast = std::sin(leaderProjection.pose.heading);
    float const inwardNorth = static_cast<float>(p.geometry.direction)
        * -tangentEast;
    float const inwardEast = static_cast<float>(p.geometry.direction)
        * tangentNorth;
    float const backward = p.geometry.separationDistance
        * std::cos(p.geometry.separationAngle);
    float const inward = p.geometry.separationDistance
        * std::sin(p.geometry.separationAngle);
    float const actualNorth = state.guards[1].north - state.guards[0].north;
    float const actualEast = state.guards[1].east - state.guards[0].east;
    float const scale = std::max(p.geometry.separationDistance, 1.F);
    return {
        (actualNorth * tangentNorth + actualEast * tangentEast + backward)
            / scale,
        (actualNorth * inwardNorth + actualEast * inwardEast - inward)
            / scale
    };
}

/**
 * @brief Refreshes both continuous curve projections from physical positions.
 * @param p Episode patrol parameters.
 * @param state Mutable state receiving per-guard team progress.
 */
inline void UpdatePatrolProgress(Parameters const& p, State& state) noexcept
{
    for (std::size_t role = 0U; role < state.guards.size(); ++role)
        state.patrolProgress[role] = ProjectOnPatrolCurve(
            p.geometry, role, state.guards[role].north,
            state.guards[role].east).pose.arclength;
}

/**
 * @brief Returns physically plausible signed progress between two states.
 * @param p Episode patrol parameters.
 * @param before Pre-step state.
 * @param after Post-step state.
 * @param role Defender role in [0, 1].
 * @return Signed metres on the loop, or zero for a projection discontinuity.
 */
inline float PatrolProgressDelta(Parameters const& p,
                                 State const& before,
                                 State const& after,
                                 std::size_t role) noexcept
{
    float const circuit = CircuitLength(p);
    float const delta = std::remainder(
        after.patrolProgress[role] - before.patrolProgress[role], circuit);
    float const maximumPhysicalProgress =
        2.F * p.maxSpeed * p.dt + 1.F;
    return std::abs(delta) <= maximumPhysicalProgress ? delta : 0.F;
}

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

// Smooth dense shaping: an agent learns to maintain separation before
// collision, without penalizing nominal staggered patrol formations.
inline float SeparationPenalty(Parameters const& p,
                               State const& state) noexcept
{
    float const fraction = Clamp(
        (p.cautionDistance - GuardSeparation(state)) /
        (p.cautionDistance - p.collisionDistance), 0.F, 1.F);
    return -p.separationPenaltyWeight * fraction;
}

/**
 * @brief Rewards an ordered leader/follower formation when no threat is active.
 * @param p Valid patrol geometry and safety distances.
 * @param state Current team state.
 * @param maintainFormation False while one or both cars are intercepting.
 * @return Dense relative-pose shaping plus the close-range safety penalty.
 */
inline float FormationReward(Parameters const& p, State const& state,
                             bool maintainFormation) noexcept
{
    float reward = SeparationPenalty(p, state);
    if (!maintainFormation)
        return reward;

    FormationErrors const errors = MeasureFormationErrors(p, state);
    float const longitudinal = std::max(
        std::abs(errors.longitudinal) - p.formationTolerance, 0.F);
    float const inward = std::max(
        std::abs(errors.inward) - p.formationTolerance, 0.F);
    float const relativeError = std::hypot(longitudinal, inward);
    reward -= p.formationErrorWeight * Clamp(relativeError, 0.F, 1.F);
    return reward;
}

static_assert(std::is_trivially_copyable_v<State>,
    "RLtools stores patrol state in a matrix of trivially copyable values");

// Build a deterministic initial team state. The embedded MAPPO controller
// advances both defenders; controlledRole remains useful to the single-role
// headless environment and reference-controller tests.
inline State InitialState(Parameters const& p, int side,
                          std::size_t controlledRole, float phase) noexcept
{
    State state{};
    state.side = side < 0 ? -1 : 1;
    state.controlledRole = controlledRole % 2U;
    state.arclength = PositiveModulo(phase, CircuitLength(p));
    for (std::size_t i = 0; i < state.guards.size(); ++i)
        state.guards[i] = OnReference(p, PatrolReference(p, state.side, i, state.arclength));
    state.patrolProgress.fill(state.arclength);
    return state;
}

inline State Advance(Parameters const& p, State const& prev,
                     Action const& action) noexcept
{
    State next = prev;
    std::size_t const me = next.controlledRole;
    next.guards[me] = Advance(p, next.guards[me], action);
    UpdatePatrolProgress(p, next);
    ++next.step;
    next.collided = GuardsCollided(p, prev, next);
    next.failed |= next.collided;
    for (auto const& guard : next.guards)
        if (!std::isfinite(guard.north) || !std::isfinite(guard.east)
            || std::abs(guard.north) >= p.hardBoundaryScale * p.halfNorth
            || std::abs(guard.east) >= p.hardBoundaryScale * p.halfEast)
            next.failed = true;
    return next;
}

// Advance two MAPPO-controlled defenders with a shared world clock. The
// single-action overload remains the adapter for the optional headless loop.
inline State AdvanceBoth(Parameters const& p, State const& previous,
                         std::array<Action, 2> const& actions) noexcept
{
    State next = previous;
    for (std::size_t role = 0; role < next.guards.size(); ++role)
        next.guards[role] = Advance(p, previous.guards[role], actions[role]);
    UpdatePatrolProgress(p, next);
    ++next.step;
    next.collided = GuardsCollided(p, previous, next);
    next.failed |= next.collided;
    for (auto const& guard : next.guards)
        if (!guard.available || !std::isfinite(guard.north)
            || !std::isfinite(guard.east) || !std::isfinite(guard.heading)
            || !std::isfinite(guard.speed) || !std::isfinite(guard.turnRate)
            || std::abs(guard.north) >= p.hardBoundaryScale * p.halfNorth
            || std::abs(guard.east) >= p.hardBoundaryScale * p.halfEast)
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
    PatrolProjection const projection = ProjectOnPatrolCurve(
        p.geometry, state.controlledRole, own.north, own.east);
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
    k = ObservationLayout::RedOffset;
    for (Mobile const& target : state.attackers)
    {
        if (!target.available)
        {
            k += ObservationLayout::RedFeatures;
            continue;
        }
        float const dx = target.north - own.north;
        float const dy = target.east - own.east;
        float const rel = std::atan2(dy, dx) - own.heading;
        float const distance = std::hypot(dx, dy);
        obs[k++] = dx / (2.F * p.halfNorth);
        obs[k++] = dy / (2.F * p.halfEast);
        obs[k++] = (target.speed * std::cos(target.heading)
                  - own.speed * std::cos(own.heading)) / p.maxSpeed;
        obs[k++] = (target.speed * std::sin(target.heading)
                  - own.speed * std::sin(own.heading)) / p.maxSpeed;
        obs[k++] = distance / (2.F * diag);
        obs[k++] = std::cos(rel);
        obs[k++] = std::sin(rel);
        obs[k++] = target.speed / p.maxSpeed;
        obs[k++] = 1.F; // Hostile mobile track type.
        obs[k++] = target.north / p.halfNorth;
        obs[k++] = target.east / p.halfEast;
        obs[k++] = (std::abs(target.north) < p.halfNorth
                 && std::abs(target.east) < p.halfEast) ? 1.F : 0.F;
        obs[k++] = target.available ? 1.F : 0.F; // Attention mask.
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
                                   0.92F * p.halfEast, 0.F, 0.F, 0.F,
                           state.objectiveAvailable};
    writeFriend(objective, 1.F);
    k = ObservationLayout::PatrolOffset;
    float const relativeTangent = projection.pose.heading - own.heading;
    float const phaseAngle = kTwoPi * projection.pose.phase;
    float const lookahead = std::max(
        3.F * own.speed, 0.25F * p.geometry.width);
    PatrolCurvePose const preview = EvaluatePatrolCurve(
        p.geometry, state.controlledRole,
        projection.pose.arclength + lookahead);
    FormationErrors const formation = MeasureFormationErrors(p, state);
    float const minimumTurnRadius = p.cruiseSpeed / p.maxRotationRate;
    obs[k++] = std::cos(relativeTangent);
    obs[k++] = std::sin(relativeTangent);
    obs[k++] = Clamp(projection.signedCrossTrackError
        / std::max(0.5F * p.geometry.width, 1.F), -2.F, 2.F);
    obs[k++] = std::sin(phaseAngle);
    obs[k++] = std::cos(phaseAngle);
    obs[k++] = Clamp(preview.curvature * minimumTurnRadius, -1.F, 1.F);
    obs[k++] = Clamp(formation.longitudinal, -2.F, 2.F);
    obs[k++] = Clamp(formation.inward, -2.F, 2.F);
    obs[k++] = state.controlledRole == 0U ? -1.F : 1.F;
    obs[k++] = Clamp(state.defenseWeight[state.controlledRole], 0.F, 1.F);
    obs[ObservationLayout::GlobalOffset] = 1.F; // Initial ammunition ratio.
    obs[ObservationLayout::GlobalOffset + 1] =
        std::any_of(state.attackers.begin(), state.attackers.end(),
                    [](Mobile const& attacker) { return attacker.available; })
        ? 1.F : 0.F;

    // CTDE contract: the actor masks this suffix in AttentionActor, while the
    // critic receives the complete team state during MAPPO optimization.
    k = ObservationLayout::CentralOffset;
    auto writeCentralMobile = [&](Mobile const& mobile) {
        obs[k++] = mobile.north / p.halfNorth;
        obs[k++] = mobile.east / p.halfEast;
        obs[k++] = (mobile.speed * std::cos(mobile.heading)) / p.maxSpeed;
        obs[k++] = (mobile.speed * std::sin(mobile.heading)) / p.maxSpeed;
        obs[k++] = std::cos(mobile.heading);
        obs[k++] = std::sin(mobile.heading);
        obs[k++] = mobile.available ? 1.F : 0.F;
    };
    for (Mobile const& guard : state.guards)
        writeCentralMobile(guard);
    for (Mobile const& attacker : state.attackers)
        writeCentralMobile(attacker);
    obs[k++] = 0.F;
    obs[k++] = (state.side >= 0 ? 1.F : -1.F) * 0.92F;
    obs[k++] = state.objectiveAvailable ? 1.F : 0.F;
    obs[k++] = projection.pose.phase;
    return obs;
}

/**
 * @brief Scores physically realized signed progress and cross-track accuracy.
 * @param p Valid patrol and shaping parameters.
 * @param before State before vehicle integration.
 * @param after State after vehicle integration and projection.
 * @param role Defender role in [0, 1].
 * @return Bounded dense tracking reward without time-based advancement.
 */
inline float TrackingReward(Parameters const& p,
                            State const& before,
                            State const& after,
                            std::size_t role) noexcept
{
    Mobile const& own = after.guards[role];
    PatrolProjection const projection = ProjectOnPatrolCurve(
        p.geometry, role, own.north, own.east);
    float const progress = Clamp(
        PatrolProgressDelta(p, before, after, role)
            / std::max(p.cruiseSpeed * p.dt, 1.F),
        -1.F, 1.F);
    float const crossTrackError = Clamp(
        std::abs(projection.signedCrossTrackError)
            / p.patrolCrossTrackScale,
        0.F, 1.F);
    return p.patrolProgressWeight * progress
         - p.patrolCrossTrackWeight * crossTrackError;
}

/**
 * @brief Reward channels published by the blue MAPPO team.
 */
enum class RewardChannel : std::size_t
{
    PatrolTracking,
    FormationSafety,
    ControlEffort,
    ZoneProtection,
    Interception,
    MissionOutcome,
    Timeout,
    Count
};

constexpr std::size_t kRewardChannelCount =
    static_cast<std::size_t>(RewardChannel::Count);

constexpr std::array<char const*, kRewardChannelCount> kRewardChannelNames{
    "Patrol tracking",
    "Formation safety",
    "Control effort",
    "Zone protection",
    "Interception",
    "Mission outcome",
    "Timeout"
};

using RewardValues = std::array<float, kRewardChannelCount>;

/**
 * @brief Stores the unweighted team reward decomposition for one ECS step.
 */
struct RewardBreakdown final
{
    RewardValues values{};

    /**
     * @brief Computes the weighted and clamped team reward.
     * @param weights Per-channel configurable multipliers.
     * @param clampMagnitude Symmetric absolute reward limit.
     * @return Scalar reward supplied to both MAPPO trajectories.
     */
    float Total(RewardValues const& weights, float clampMagnitude) const noexcept
    {
        float total = 0.F;
        for (std::size_t i = 0; i < values.size(); ++i)
            total += values[i] * weights[i];
        return Clamp(total, -clampMagnitude, clampMagnitude);
    }
};

inline bool HasActiveThreat(State const& state) noexcept
{
    return std::any_of(state.attackers.begin(), state.attackers.end(),
                       [](Mobile const& attacker) { return attacker.available; });
}

inline float DistanceToObjective(Parameters const& p, State const& state,
                                 Mobile const& mobile) noexcept
{
    float const objectiveEast = (state.side >= 0 ? 1.F : -1.F) * 0.92F * p.halfEast;
    return std::hypot(mobile.north, mobile.east - objectiveEast);
}

/**
 * @brief Returns the planar distance between two mobile entities.
 * @param first First mobile state.
 * @param second Second mobile state.
 * @return Centre-to-centre distance in metres.
 */
inline float DistanceBetween(Mobile const& first, Mobile const& second) noexcept
{
    return std::hypot(first.north - second.north,
                      first.east - second.east);
}

/**
 * @brief One-to-one defensive assignment computed from the pre-step state.
 *
 * `ThreatAssignment::Unassigned` denotes a reserve guard. With one threat,
 * the closest guard intercepts and its teammate keeps patrolling. With two
 * threats, the minimum-total-distance one-to-one assignment is used.
 */
struct ThreatAssignment final
{
    /** @brief Sentinel stored for a guard kept in patrol reserve. */
    static constexpr std::size_t Unassigned =
        std::numeric_limits<std::size_t>::max();

    /** @brief Attacker index for each guard, or Unassigned for a reserve. */
    std::array<std::size_t, 2> attackerForGuard{Unassigned, Unassigned};
    /** @brief Number of available hostile tracks in the assignment state. */
    std::size_t activeThreats{};
};

/**
 * @brief Assigns active attackers to guards without allocating memory.
 * @param state Team state used to select the stable pre-step assignment.
 * @return Assignment and number of active attackers.
 */
inline ThreatAssignment AssignThreats(State const& state) noexcept
{
    ThreatAssignment result{};
    std::array<std::size_t, 2> active{ThreatAssignment::Unassigned,
                                      ThreatAssignment::Unassigned};
    for (std::size_t attacker = 0U; attacker < state.attackers.size();
         ++attacker)
        if (state.attackers[attacker].available)
            active[result.activeThreats++] = attacker;

    if (result.activeThreats == 1U)
    {
        std::size_t const attacker = active[0];
        std::size_t const guard =
            DistanceBetween(state.guards[0], state.attackers[attacker])
                <= DistanceBetween(state.guards[1], state.attackers[attacker])
            ? 0U : 1U;
        result.attackerForGuard[guard] = attacker;
    }
    else if (result.activeThreats == 2U)
    {
        float const direct =
            DistanceBetween(state.guards[0], state.attackers[active[0]])
            + DistanceBetween(state.guards[1], state.attackers[active[1]]);
        float const crossed =
            DistanceBetween(state.guards[0], state.attackers[active[1]])
            + DistanceBetween(state.guards[1], state.attackers[active[0]]);
        bool const useDirect = direct <= crossed;
        result.attackerForGuard[0] = active[useDirect ? 0U : 1U];
        result.attackerForGuard[1] = active[useDirect ? 1U : 0U];
    }
    return result;
}

/**
 * @brief Computes mission-aware patrol shaping for the blue team.
 * @param p Valid patrol parameters.
 * @param before Pre-step team state used for signed progress.
 * @param after Post-step team state used for cross-track error.
 * @param assignment Threat assignment selected before the step.
 * @return Mean tracking reward of guards currently held in patrol reserve.
 *
 * Both guards patrol when there is no threat. With one threat, only the
 * unassigned reserve guard is evaluated against the circuit. With two threats,
 * both guards are released from the patrol reference to intercept.
 */
inline float MissionPatrolReward(Parameters const& p,
                                 State const& before,
                                 State const& after,
                                 ThreatAssignment const& assignment) noexcept
{
    float reward = 0.F;
    std::size_t patrollingGuards = 0U;
    for (std::size_t guard = 0U; guard < after.guards.size(); ++guard)
    {
        if (assignment.attackerForGuard[guard] < after.attackers.size())
            continue;
        reward += TrackingReward(p, before, after, guard);
        ++patrollingGuards;
    }
    return patrollingGuards > 0U
        ? reward / static_cast<float>(patrollingGuards) : 0.F;
}

/**
 * @brief Computes dense progress towards assigned active threats.
 * @param p Valid patrol parameters.
 * @param before Team state before ECS physics.
 * @param after Team state after ECS physics and destruction resolution.
 * @param assignment Stable assignment selected from `before`.
 * @return Engagement shaping; larger when interceptors close their targets.
 *
 * The range-rate term gives immediate credit to blue actions. The small
 * distance cost prevents an agent from being indifferent while matching a
 * threat's velocity far from weapon range. Destruction itself is scored by the
 * separate Interception channel.
 */
inline float EngagementReward(Parameters const& p, State const& before,
                              State const& after,
                              ThreatAssignment const& assignment) noexcept
{
    float reward = 0.F;
    float const diagonal = std::hypot(p.halfNorth, p.halfEast);
    float const distancePerStep = std::max(p.maxSpeed * p.dt, 1.F);
    for (std::size_t guard = 0U; guard < before.guards.size(); ++guard)
    {
        std::size_t const attacker = assignment.attackerForGuard[guard];
        if (attacker >= before.attackers.size()
            || !before.guards[guard].available || !after.guards[guard].available
            || !before.attackers[attacker].available
            || !after.attackers[attacker].available)
            continue;

        float const beforeRange = DistanceBetween(
            before.guards[guard], before.attackers[attacker]);
        float const afterRange = DistanceBetween(
            after.guards[guard], after.attackers[attacker]);
        float const closingProgress = Clamp(
            (beforeRange - afterRange) / distancePerStep, -1.F, 1.F);
        float const normalizedRange = Clamp(afterRange / diagonal, 0.F, 1.F);
        reward += 0.35F * closingProgress - 0.10F * normalizedRange;
    }
    return reward;
}

/**
 * @brief Rewards distinct tactical roles for assigned hostile tracks.
 * @param p Valid world parameters.
 * @param state Post-step team state.
 * @param assignment Stable assignment selected before the step.
 * @return Dense specialization and, for two threats, team-spread shaping.
 *
 * Each assigned guard is rewarded for being closer to its own target than its
 * teammate. With two simultaneous targets, an additional bounded term favors
 * enough physical spread to cover both approaches instead of tailing the same
 * hostile.
 */
inline float ThreatCoverageReward(Parameters const& p, State const& state,
                                  ThreatAssignment const& assignment) noexcept
{
    if (assignment.activeThreats == 0U)
        return 0.F;

    float const diagonal = std::hypot(p.halfNorth, p.halfEast);
    float specialization = 0.F;
    std::size_t assignedGuards = 0U;
    for (std::size_t guard = 0U; guard < state.guards.size(); ++guard)
    {
        std::size_t const attacker = assignment.attackerForGuard[guard];
        if (attacker >= state.attackers.size()
            || !state.attackers[attacker].available)
            continue;
        float const assignedRange = DistanceBetween(
            state.guards[guard], state.attackers[attacker]);
        float const teammateRange = DistanceBetween(
            state.guards[1U - guard], state.attackers[attacker]);
        specialization += Clamp(
            (teammateRange - assignedRange) / diagonal, -1.F, 1.F);
        ++assignedGuards;
    }

    float reward = assignedGuards > 0U
        ? 0.25F * specialization / static_cast<float>(assignedGuards) : 0.F;
    if (assignment.activeThreats == 2U)
    {
        float const desiredSpread = std::max(
            2.F * p.cautionDistance, FormationGap(p));
        reward += 0.15F * Clamp(
            GuardSeparation(state) / desiredSpread, 0.F, 1.F);
    }
    return reward;
}

/**
 * @brief Computes dense shaping for excursions beyond the visible world.
 * @param p Valid patrol and soft-boundary parameters.
 * @param state Current team and threat state.
 * @return Non-positive team penalty, reduced for remote active threats.
 *
 * Crossing the regular world rectangle is legal. Without a threat the full
 * penalty encourages patrol containment. A remote attacker reduces it so a
 * defender may pursue; a threat near the objective restores the full cost.
 */
inline float SoftBoundaryPenalty(Parameters const& p,
                                 State const& state) noexcept
{
    float excursion = 0.F;
    for (Mobile const& guard : state.guards)
    {
        float const normalized = std::max(
            std::abs(guard.north) / p.halfNorth,
            std::abs(guard.east) / p.halfEast);
        float const outside = Clamp(
            (normalized - 1.F) / p.boundaryPenaltyDepth, 0.F, 1.F);
        excursion += outside * outside;
    }
    excursion *= 0.5F;
    if (excursion <= 0.F)
        return 0.F;

    float scale = 1.F;
    float nearestThreat = std::numeric_limits<float>::infinity();
    for (Mobile const& attacker : state.attackers)
        if (attacker.available)
            nearestThreat = std::min(
                nearestThreat, DistanceToObjective(p, state, attacker));
    if (std::isfinite(nearestThreat))
    {
        float const diagonal = std::hypot(p.halfNorth, p.halfEast);
        float const defendedRadius = 0.25F * diagonal;
        float const remoteness = Clamp(
            (nearestThreat - defendedRadius)
                / std::max(diagonal - defendedRadius, 1.F),
            0.F, 1.F);
        scale = 1.F - (1.F - p.minimumThreatBoundaryScale) * remoteness;
    }
    return -2.F * scale * excursion;
}

/**
 * @brief Computes every explainable MAPPO reward channel for one team step.
 * @param p Valid patrol parameters.
 * @param before Team state before ECS physics.
 * @param actions Commands selected independently by both decentralized actors.
 * @param after Team state after ECS physics and world resolution.
 * @param pendingThreats True when a delayed hostile is still scheduled.
 * @return Unweighted per-channel rewards for telemetry and training.
 */
inline RewardBreakdown TeamRewardBreakdown(
    Parameters const& p, State const& before,
    std::array<Action, 2> const& actions, State const& after,
    bool pendingThreats = false) noexcept
{
    RewardBreakdown result{};
    auto& values = result.values;
    ThreatAssignment const assignment = AssignThreats(before);
    values[static_cast<std::size_t>(RewardChannel::PatrolTracking)] =
        MissionPatrolReward(p, before, after, assignment);
    values[static_cast<std::size_t>(RewardChannel::FormationSafety)] =
        FormationReward(p, after, assignment.activeThreats == 0U);
    values[static_cast<std::size_t>(RewardChannel::ControlEffort)] =
        -p.controlEffortWeight
        * (std::abs(actions[0][0]) + std::abs(actions[0][1])
           + std::abs(actions[1][0]) + std::abs(actions[1][1]));

    float protection = EngagementReward(p, before, after, assignment);
    float interception = 0.F;
    bool hadThreat = false;
    for (std::size_t i = 0; i < after.attackers.size(); ++i)
    {
        Mobile const& previous = before.attackers[i];
        Mobile const& current = after.attackers[i];
        hadThreat |= previous.available;
        if (previous.available && !current.available)
        {
            interception += 10.F;
            continue;
        }
        if (!previous.available || !current.available)
            continue;
        float const beforeRange = DistanceToObjective(p, before, previous);
        float const afterRange = DistanceToObjective(p, after, current);
        float const normalizedProgress = Clamp(
            (afterRange - beforeRange) / std::max(p.cruiseSpeed * p.dt, 1.F),
            -1.F, 1.F);
        float const defendedRadius = 0.25F * std::hypot(p.halfNorth, p.halfEast);
        float const penetration = Clamp(
            (defendedRadius - afterRange) / defendedRadius, 0.F, 1.F);
        protection += 0.15F * normalizedProgress - 0.5F * penetration;
    }
    values[static_cast<std::size_t>(RewardChannel::ZoneProtection)] =
        protection + ThreatCoverageReward(p, after, assignment)
        + SoftBoundaryPenalty(p, after);
    values[static_cast<std::size_t>(RewardChannel::Interception)] = interception;

    float outcome = 0.F;
    if (after.failed || !after.objectiveAvailable)
        outcome = -100.F;
    else if (hadThreat && !HasActiveThreat(after) && !pendingThreats)
        outcome = 25.F;
    values[static_cast<std::size_t>(RewardChannel::MissionOutcome)] = outcome;
    values[static_cast<std::size_t>(RewardChannel::Timeout)] =
        after.step >= p.episodeSteps
            && (HasActiveThreat(after) || pendingThreats) ? -15.F : 0.F;
    return result;
}

inline float TeamReward(Parameters const& p, State const& before,
                        Action const& action, State const& after) noexcept
{
    RewardValues weights{};
    weights.fill(1.F);
    std::array<Action, 2> actions{};
    actions[after.controlledRole % 2U] = action;
    return TeamRewardBreakdown(p, before, actions, after).Total(weights, 100.F);
}

} // namespace ppo::examples::twovstwo::defenders
