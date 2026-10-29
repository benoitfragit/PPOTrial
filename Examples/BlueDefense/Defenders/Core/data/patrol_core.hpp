#pragma once

// Deterministic 2D patrol and protection domain used by the blue MAPPO team.
// Coordinates: north x / east y, metres; heading 0 towards north.
// No dependencies on Torch, EnTT, a renderer, or RLtools in this domain layer.
#include "patrol_geometry.hpp"
#include "patrol_defense.hpp"
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
    static constexpr std::uint32_t Version = 7;
};

using Observation = std::array<float, ObservationLayout::Size>;
using Action = std::array<float, ObservationLayout::ActionSize>;

struct Parameters
{
    float halfNorth = 22329.375F;
    float halfEast = 35727.F;
    float dt = 0.2F;
    float cruiseSpeed = 700.F;
    float maxSpeed = 700.F;
    float maxAcceleration = 600.F;
    float maxRotationRate = 0.70000059604645F;
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
    float controlEffortWeight = 0.002F;
    DefenseSettings defense{};
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
        && std::isfinite(p.controlEffortWeight)
        && p.controlEffortWeight >= 0.F
        && Valid(p.defense)
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
 * @brief Returns the leader's long-axis racetrack reference.
 * @param p Valid patrol and world parameters.
 * @param side Objective side; it mirrors the initial travel direction.
 * @param distance Team arclength travelled on the circuit in metres.
 * @return Reference pose and progress information.
 */
inline Reference PatrolReference(Parameters const& p, int side,
                                 float distance) noexcept
{
    (void)side;
    PatrolCurvePose const pose = EvaluatePatrolCurve(
        p.geometry, distance);
    return {pose.north, pose.east, pose.heading, pose.phase, 0.F};
}

/**
 * @brief Returns the desired wingman pose relative to the leader.
 * @param p Valid wedge geometry.
 * @param leader Current or predicted leader reference.
 * @return Rear/inward formation pose, independent of the patrol curve.
 */
inline Reference WingmanReference(Parameters const& p,
                                  Reference const& leader) noexcept
{
    PatrolCurvePose const leaderPose{
        leader.north, leader.east, leader.heading, 0.F, leader.phase, 0.F};
    PatrolCurvePose const pose = EvaluateWedgePose(p.geometry, leaderPose);
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
    /** @brief ECS pseudo-LiDAR ranges for each semantic blue role. */
    std::array<std::array<float, ObservationLayout::Lidar>, 2> lidarRanges{};
    /** @brief True when the matching range array came from the ECS sensor. */
    std::array<bool, 2> lidarAvailable{};
    std::size_t controlledRole;
    int side;
    /** @brief Nearest continuous leader arclength on the unique racetrack. */
    float patrolProgress{};

    /** @brief Initial team phase retained for deterministic evaluation cases. */
    float arclength;
    std::size_t step;
    bool failed;
    bool collided;
    bool objectiveAvailable{true};
};

/**
 * @brief Counts active hostile tracks independently of their slot order.
 * @param state Current team state.
 * @return Number of living reds visible to both blue Actors.
 */
inline std::size_t ActiveThreatCount(State const& state) noexcept
{
    return static_cast<std::size_t>(std::count_if(
        state.attackers.begin(), state.attackers.end(),
        [](Mobile const& attacker) { return attacker.available; }));
}

/**
 * @brief Measures shared threat pressure without assigning a target or role.
 * @param p Valid threat assessment parameters.
 * @param state Physical team state.
 * @return Pressure in [0, 1]; every active red contributes immediately.
 */
inline float DefensePressure(Parameters const& p, State const& state) noexcept
{
    std::array<DefenseTrack, 2> guards{};
    for (std::size_t i = 0U; i < guards.size(); ++i)
    {
        Mobile const& guard = state.guards[i];
        guards[i] = {guard.north, guard.east, guard.heading,
                     guard.speed, guard.available};
    }
    DefenseTrack const objective{
        0.F, (state.side >= 0 ? 1.F : -1.F) * 0.92F * p.halfEast,
        0.F, 0.F, state.objectiveAvailable};
    float pressure = 0.F;
    for (Mobile const& attacker : state.attackers)
    {
        if (!attacker.available)
            continue;
        DefenseTrack const track{
            attacker.north, attacker.east, attacker.heading,
            attacker.speed, true};
        pressure = std::max(pressure, std::max(0.25F,
            ThreatRisk(p.defense, track, objective, guards)
                / p.defense.activationRisk));
    }
    return Clamp(pressure, 0.F, 1.F);
}

/**
 * @brief Releases mutual support only under multiple simultaneous threats.
 * @param p Valid threat assessment parameters.
 * @param state Current physical state.
 * @return Wedge weight; separation safety remains active independently.
 */
inline float MutualSupportWeight(Parameters const& p,
                                 State const& state) noexcept
{
    return ActiveThreatCount(state) > 1U ? 1.F - DefensePressure(p, state)
                                        : 1.F;
}

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
    float const tangentNorth = std::cos(state.guards[0].heading);
    float const tangentEast = std::sin(state.guards[0].heading);
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
 * @brief Refreshes the leader's continuous curve projection.
 * @param p Episode patrol parameters.
 * @param state Mutable state receiving leader progress.
 */
inline void UpdatePatrolProgress(Parameters const& p, State& state) noexcept
{
    state.patrolProgress = ProjectOnPatrolCurve(
        p.geometry, state.guards[0].north,
        state.guards[0].east).pose.arclength;
}

/**
 * @brief Returns physically plausible signed progress between two states.
 * @param p Episode patrol parameters.
 * @param before Pre-step state.
 * @param after Post-step state.
 * @return Signed metres on the loop, or zero for a projection discontinuity.
 */
inline float PatrolProgressDelta(Parameters const& p,
                                 State const& before,
                                 State const& after) noexcept
{
    float const circuit = CircuitLength(p);
    float const delta = std::remainder(
        after.patrolProgress - before.patrolProgress, circuit);
    float const maximumPhysicalProgress =
        2.F * p.maxSpeed * p.dt + 1.F;
    return std::abs(delta) <= maximumPhysicalProgress ? delta : 0.F;
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

/**
 * @brief Rewards the wingman's ordered pose relative to the leader.
 * @param p Valid patrol geometry and safety distances.
 * @param state Current team state.
 * @param maintainFormation False when simultaneous threats release the wedge.
 * @return Dense relative-pose shaping; collisions remain physical failures.
 */
inline float FormationReward(Parameters const& p, State const& state,
                             bool maintainFormation) noexcept
{
    if (!maintainFormation)
        return 0.F;
    float reward = 0.F;

    FormationErrors const errors = MeasureFormationErrors(p, state);
    float const longitudinal = std::max(
        std::abs(errors.longitudinal) - p.formationTolerance, 0.F);
    float const inward = std::max(
        std::abs(errors.inward) - p.formationTolerance, 0.F);
    float const relativeError = std::hypot(longitudinal, inward);
    reward -= p.formationErrorWeight * relativeError / (1.F + relativeError);
    return reward;
}

/**
 * @brief Blends wingman wedge shaping during defence and recovery.
 * @param p Valid patrol geometry and safety distances.
 * @param state Current team state.
 * @param formationWeight Wingman formation contribution in [0, 1].
 * @return Blended relative-pose shaping; collision is resolved by world rules.
 */
inline float FormationReward(Parameters const& p, State const& state,
                             float formationWeight) noexcept
{
    FormationErrors const errors = MeasureFormationErrors(p, state);
    float const longitudinal = std::max(
        std::abs(errors.longitudinal) - p.formationTolerance, 0.F);
    float const inward = std::max(
        std::abs(errors.inward) - p.formationTolerance, 0.F);
    float const relativeError = std::hypot(longitudinal, inward);
    return -Clamp(formationWeight, 0.F, 1.F)
        * p.formationErrorWeight * relativeError / (1.F + relativeError);
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
    Reference const leader = PatrolReference(p, state.side, state.arclength);
    state.guards[0] = OnReference(p, leader);
    state.guards[1] = OnReference(p, WingmanReference(p, leader));
    state.patrolProgress = state.arclength;
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
    PatrolProjection const leaderProjection = ProjectOnPatrolCurve(
        p.geometry, state.guards[0].north, state.guards[0].east);
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
        if (state.lidarAvailable[state.controlledRole])
            obs[k++] = Clamp(
                state.lidarRanges[state.controlledRole][i], 0.F, 1.F);
        else
        {
            float const angle = own.heading - kPi
                + static_cast<float>(i) * (kTwoPi / 60.F);
            float const distance = RayDistanceToBounds(p, own, angle);
            obs[k++] = distance < 0.F ? 1.F
                                      : distance / (distance + diag);
        }
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
    if (state.controlledRole == 0U)
    {
        float const relativeTangent =
            leaderProjection.pose.heading - own.heading;
        float const phaseAngle = kTwoPi * leaderProjection.pose.phase;
        float const lookahead = std::max(
            3.F * own.speed, 0.25F * p.geometry.width);
        PatrolCurvePose const preview = EvaluatePatrolCurve(
            p.geometry, leaderProjection.pose.arclength + lookahead);
        float const minimumTurnRadius = p.cruiseSpeed / p.maxRotationRate;
        obs[k++] = std::cos(relativeTangent);
        obs[k++] = std::sin(relativeTangent);
        obs[k++] = Clamp(leaderProjection.signedCrossTrackError
            / std::max(0.5F * p.geometry.width, 1.F), -2.F, 2.F);
        obs[k++] = std::sin(phaseAngle);
        obs[k++] = std::cos(phaseAngle);
        obs[k++] = Clamp(preview.curvature * minimumTurnRadius, -1.F, 1.F);
        k += 2U; // Wedge errors are reserved for the wingman Actor.
    }
    else
    {
        FormationErrors const formation = MeasureFormationErrors(p, state);
        k += 6U; // The wingman never receives racetrack-relative inputs.
        obs[k++] = Clamp(formation.longitudinal, -2.F, 2.F);
        obs[k++] = Clamp(formation.inward, -2.F, 2.F);
    }
    obs[k++] = state.controlledRole == 0U ? -1.F : 1.F;
    obs[k++] = DefensePressure(p, state);
    obs[ObservationLayout::GlobalOffset] = 1.F; // Initial ammunition ratio.
    // Reserved: a target-slot index is incompatible with invariant pooling.
    obs[ObservationLayout::GlobalOffset + 1] = 0.F;

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
    obs[k++] = leaderProjection.pose.phase;
    return obs;
}

/**
 * @brief Scores physically realized signed progress and cross-track accuracy.
 * @param p Valid patrol and shaping parameters.
 * @param before State before vehicle integration.
 * @param after State after vehicle integration and projection.
 * @return Bounded dense tracking reward without time-based advancement.
 */
inline float TrackingReward(Parameters const& p,
                            State const& before,
                            State const& after) noexcept
{
    Mobile const& own = after.guards[0];
    PatrolProjection const projection = ProjectOnPatrolCurve(
        p.geometry, own.north, own.east);
    float const progress = Clamp(
        PatrolProgressDelta(p, before, after)
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
using RoleRewards = std::array<float, 2>;

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

using RoleRewardBreakdowns = std::array<RewardBreakdown, 2>;

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
 * @brief Scores the leader's racetrack while patrol remains relevant.
 * @param p Valid patrol parameters.
 * @param before Pre-step team state.
 * @param after Post-step team state.
 * @return Leader-only tracking reward faded by shared hostile pressure.
 */
inline float MissionPatrolReward(Parameters const& p,
                                 State const& before,
                                 State const& after) noexcept
{
    return (1.F - DefensePressure(p, before))
        * TrackingReward(p, before, after);
}

namespace patrol_reward_detail {

/**
 * @brief Measures interception distance including immediate heading effort.
 * @param p Valid world and vehicle parameters.
 * @param guard Candidate blue physical state.
 * @param attacker Candidate red physical state.
 * @return Finite pursuit cost; no target selection is exposed to an Actor.
 */
inline float PursuitCost(Parameters const& p, Mobile const& guard,
                         Mobile const& attacker) noexcept
{
    if (!guard.available)
        return 4.F * std::hypot(p.halfNorth, p.halfEast);
    float const bearing = std::atan2(attacker.east - guard.east,
                                     attacker.north - guard.north);
    float const headingEffort = 1.F - std::cos(bearing - guard.heading);
    return DistanceBetween(guard, attacker)
        + 4.F * p.maxSpeed * p.dt * headingEffort;
}

/**
 * @brief Evaluates joint coverage without publishing an assignment.
 * @param p Valid world and vehicle parameters.
 * @param state Physical state being scored.
 * @param threats Identical red indices for both ends of a transition.
 * @param count Number of surviving threats, at most two.
 * @return Minimum joint coverage cost over interchangeable defenders.
 *
 * Both pairings are scored only inside the cooperative reward. Neither Actor
 * receives a target index or a prescribed vehicle command.
 */
inline float CoverageCost(Parameters const& p, State const& state,
                          std::array<std::size_t, 2> const& threats,
                          std::size_t count) noexcept
{
    if (count == 0U)
        return 0.F;
    auto cost = [&](std::size_t guard, std::size_t threat) {
        return PursuitCost(p, state.guards[guard],
                           state.attackers[threats[threat]]);
    };
    if (count == 1U)
        return std::min(cost(0U, 0U), cost(1U, 0U));
    return std::min(cost(0U, 0U) + cost(1U, 1U),
                    cost(0U, 1U) + cost(1U, 0U));
}

} // namespace patrol_reward_detail

/**
 * @brief Rewards joint progress towards all living reds from their first step.
 * @param p Valid world and vehicle parameters.
 * @param before Team state before physics.
 * @param after Team state after physics and destruction resolution.
 * @return Shared pursuit reward used by both blue trajectories.
 *
 * The same surviving threats are scored before/after. Destruction is rewarded
 * by the sparse Interception channel. Stationary heading alignment alone
 * cannot generate a permanent positive pursuit reward.
 */
inline float EngagementReward(Parameters const& p, State const& before,
                              State const& after) noexcept
{
    std::array<std::size_t, 2> threats{};
    std::size_t count = 0U;
    for (std::size_t i = 0U; i < before.attackers.size(); ++i)
        if (before.attackers[i].available && after.attackers[i].available)
            threats[count++] = i;
    if (count == 0U)
        return 0.F;
    float const beforeCost = patrol_reward_detail::CoverageCost(
        p, before, threats, count);
    float const afterCost = patrol_reward_detail::CoverageCost(
        p, after, threats, count);
    float const scale = static_cast<float>(count);
    float const progress = Clamp((beforeCost - afterCost)
        / std::max(scale * p.maxSpeed * p.dt, 1.F), -1.F, 1.F);
    float const remaining = Clamp(afterCost
        / (scale * std::hypot(p.halfNorth, p.halfEast)), 0.F, 2.F);
    return 0.8F * progress - 0.1F * remaining;
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
 * @brief Scores the shared evolution of hostile ranges to the objective.
 * @param p Valid world and vehicle parameters.
 * @param before State before the physical step.
 * @param after State after the physical step.
 * @return Shared dense protected-zone reward independent of target allocation.
 */
inline float ObjectiveProtectionReward(Parameters const& p,
                                       State const& before,
                                       State const& after) noexcept
{
    float reward = 0.F;
    for (std::size_t i = 0; i < after.attackers.size(); ++i)
    {
        Mobile const& previous = before.attackers[i];
        Mobile const& current = after.attackers[i];
        if (!previous.available || !current.available)
            continue;
        float const beforeRange = DistanceToObjective(p, before, previous);
        float const afterRange = DistanceToObjective(p, after, current);
        float const normalizedProgress = Clamp(
            (afterRange - beforeRange) / std::max(p.cruiseSpeed * p.dt, 1.F),
            -1.F, 1.F);
        float const defendedRadius =
            0.25F * std::hypot(p.halfNorth, p.halfEast);
        float const penetration = Clamp(
            (defendedRadius - afterRange) / defendedRadius, 0.F, 1.F);
        reward += 0.15F * normalizedProgress - 0.5F * penetration;
    }
    return reward;
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
    values[static_cast<std::size_t>(RewardChannel::PatrolTracking)] =
        MissionPatrolReward(p, before, after);
    float const wingmanFormationWeight = MutualSupportWeight(p, before);
    values[static_cast<std::size_t>(RewardChannel::FormationSafety)] =
        FormationReward(p, after, wingmanFormationWeight);
    values[static_cast<std::size_t>(RewardChannel::ControlEffort)] =
        -0.5F * p.controlEffortWeight
        * (std::abs(actions[0][0]) + std::abs(actions[0][1])
           + std::abs(actions[1][0]) + std::abs(actions[1][1]));

    float const protection = EngagementReward(p, before, after)
        + ObjectiveProtectionReward(p, before, after);
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
    }
    values[static_cast<std::size_t>(RewardChannel::ZoneProtection)] =
        protection + SoftBoundaryPenalty(p, after);
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

/**
 * @brief Combines role-specific patrol tasks with shared defensive rewards.
 * @param p Valid patrol and vehicle parameters.
 * @param before Team state before physics.
 * @param actions Commands selected by leader and wingman.
 * @param after Team state after physics and destruction resolution.
 * @param pendingThreats True while a delayed hostile remains scheduled.
 * @return Per-role channel breakdowns consumed by MAPPO.
 *
 * Racetrack credit belongs only to the leader, wedge credit to the wingman.
 * Pursuit, interception and mission outcomes are cooperative. The centralized
 * Critic trains coordination without assigning targets to either Actor.
 */
inline RoleRewardBreakdowns TrainingRoleRewardBreakdowns(
    Parameters const& p, State const& before,
    std::array<Action, 2> const& actions, State const& after,
    bool pendingThreats = false) noexcept
{
    RewardBreakdown const team = TeamRewardBreakdown(
        p, before, actions, after, pendingThreats);
    RoleRewardBreakdowns result{team, team};
    result[1].values[static_cast<std::size_t>(
        RewardChannel::PatrolTracking)] = 0.F;
    result[0].values[static_cast<std::size_t>(
        RewardChannel::FormationSafety)] = 0.F;
    for (std::size_t role = 0U; role < result.size(); ++role)
        result[role].values[static_cast<std::size_t>(
            RewardChannel::ControlEffort)] = -p.controlEffortWeight
            * (std::abs(actions[role][0]) + std::abs(actions[role][1]));
    return result;
}

/**
 * @brief Applies configured channel weights to per-role training rewards.
 * @param breakdowns Per-role channel values.
 * @param weights Runtime reward multipliers.
 * @param clampMagnitude Symmetric reward clamp.
 * @return Scalar reward for each blue trajectory.
 */
inline RoleRewards TotalRoleRewards(
    RoleRewardBreakdowns const& breakdowns, RewardValues const& weights,
    float clampMagnitude) noexcept
{
    RoleRewards result{};
    for (std::size_t role = 0U; role < result.size(); ++role)
        result[role] = breakdowns[role].Total(weights, clampMagnitude);
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
