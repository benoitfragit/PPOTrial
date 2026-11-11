/**
 * @file generalized_pvs0_corrections_tests.cpp
 * @brief Deterministic unified physical-slot, feasibility and reward tests.
 */
#include "generalized_patrol_waypoint_sequencer.hpp"
#include "generalized_reward_shaping.hpp"

#include <cmath>
#include <iostream>

namespace blue = ppo::examples::bluedefense::defenders;
namespace {

bool Near(float const x, float const y, float const eps = 0.08F)
{
    return std::isfinite(x) && std::isfinite(y)
        && std::abs(x - y) <= eps;
}

blue::Parameters Config(int const direction = 1)
{
    blue::Parameters p{};
    p.geometry.length = 40000.F;
    p.geometry.width = 16000.F;
    p.geometry.separationDistance = 2800.F;
    p.geometry.separationAngle = 0.55F;
    p.geometry.direction = direction;
    p.cruiseSpeed = 300.F;
    p.maxSpeed = 750.F;
    p.maxRotationRate = 0.8F;
    p.dt = 0.2F;
    return p;
}

blue::GeneralizedRuntimeState Sample(
    blue::Parameters const& p, blue::BlueTeamRoles const& roles,
    float const arc = 2000.F)
{
    blue::GeneralizedRuntimeState state{};
    state.counts = {roles.ActiveBlueCount(), 0U};
    auto const curve = blue::EvaluatePatrolCurve(p.geometry, arc);
    for (std::size_t i = 0U; i < state.counts.blueMobiles; ++i)
        state.blues[i] = {curve.north, curve.east, curve.heading,
                          300.F, 0.F, true};
    return state;
}

bool PhysicalSlots()
{
    for (std::size_t count : {1U, 2U, 3U, 4U, 9U})
        for (int direction : {-1, 1})
            for (float omega : {0.F, 0.12F, -0.12F})
            {
                auto const p = Config(direction);
                auto const roles = *blue::BlueTeamRoles::Create(count, count - 1U);
                auto state = Sample(p, roles);
                auto& leader = state.blues[roles.LeaderPhysicalIndex()];
                leader.heading = 0.65F;
                leader.north = 10000.F;
                leader.east = 5000.F;
                leader.speed = 300.F;
                leader.turnRate = omega;
                blue::GeneralizedPatrolWaypointSequencer seq{};
                blue::PatrolNoThreatSettings settings{};
                auto const reference = seq.PhysicalTargets(p, state, roles, settings);
                auto const original = seq.Targets(p, state, roles, settings);
                if (!Near(reference[roles.LeaderPhysicalIndex()].pose.north,
                            original[roles.LeaderPhysicalIndex()].pose.north)
                    || !Near(reference[roles.LeaderPhysicalIndex()].guidancePose.east,
                            original[roles.LeaderPhysicalIndex()].guidancePose.east))
                    return false;
                std::array<blue::GeneralizedObservation, 9U> observations{};
                if (!blue::BuildGeneralizedObservations(
                        p, state, roles, observations, &reference))
                    return false;
                for (std::size_t i = 0U; i < count; ++i)
                {
                    if (i == roles.LeaderPhysicalIndex()) continue;
                    auto const& t = reference[i];
                    auto const slot = blue::EvaluateFormationOffset(
                        roles.Assignment(i)->formationSlot,
                        p.geometry.separationDistance, p.geometry.separationAngle);
                    float const c = std::cos(leader.heading);
                    float const s = std::sin(leader.heading);
                    float const inward = direction * slot.inwardMeters;
                    float const rNorth = slot.longitudinalMeters * c - inward * s;
                    float const rEast = slot.longitudinalMeters * s + inward * c;
                    float const vNorth = leader.speed * c - omega * rEast;
                    float const vEast = leader.speed * s + omega * rNorth;
                    if (!t.valid
                        || !Near(t.pose.north, leader.north + rNorth)
                        || !Near(t.pose.east, leader.east + rEast)
                        || !Near(t.speed, std::hypot(vNorth, vEast))
                        || !Near(t.guidanceSpeed, t.speed, 0.F)
                        || !Near(t.pose.north, t.guidancePose.north, 0.F)
                        || !Near(t.pose.east, t.guidancePose.east, 0.F)
                        || !Near(t.pose.heading, t.guidancePose.heading, 0.F)
                        || !Near(t.speed * std::cos(t.pose.heading), vNorth)
                        || !Near(t.speed * std::sin(t.pose.heading), vEast))
                        return false;
                    if (omega == 0.F && !Near(t.speed, leader.speed))
                        return false;
                    auto const offset = blue::GeneralizedObservationLayout::PatrolOffset;
                    if (!Near(observations[i].values[offset + 7U], 1.F))
                        return false;
                }
                for (std::size_t i = count; i < 9U; ++i)
                    if (reference[i].valid) return false;
                // A destroyed participant has its old availability mask.
                if (count > 1U)
                {
                    state.blues[0].available = false;
                    auto const absent = seq.PhysicalTargets(p, state, roles, settings);
                    if (0U != roles.LeaderPhysicalIndex() && absent[0].valid)
                        return false;
                }
                state.blues[roles.LeaderPhysicalIndex()].available = false;
                auto const leaderLost = seq.PhysicalTargets(p, state, roles, settings);
                for (std::size_t i = 0U; i < count; ++i)
                    if (leaderLost[i].valid) return false;
            }
    auto p = Config();
    auto const roles = *blue::BlueTeamRoles::Create(2U, 1U);
    auto stationary = Sample(p, roles);
    stationary.blues[1].speed = 0.F;
    stationary.blues[1].turnRate = 0.F;
    auto const zero = blue::GeneralizedPhysicalSlot(
        p, stationary.blues[1], blue::EvaluateFormationOffset(
            0U, p.geometry.separationDistance, p.geometry.separationAngle));
    return zero.valid && zero.speed == 0.F
        && std::isfinite(zero.pose.heading);
}

bool FeasibilityAndSequencing()
{
    auto p = Config();
    auto const roles = *blue::BlueTeamRoles::Create(9U, 8U);
    auto previous = Sample(p, roles);
    previous.blues[8].speed = 750.F;
    previous.blues[8].turnRate = 0.F;
    blue::GeneralizedPatrolWaypointSequencer seq{};
    blue::PatrolNoThreatSettings settings{};
    if (seq.Observe(p, previous, previous, roles, settings)
            != blue::GeneralizedWaypointUpdate::None
        || seq.Observe(p, previous, previous, roles, settings)
            != blue::GeneralizedWaypointUpdate::EnteredFollow)
        return (std::cerr << "feasibility-check-1\n", false);
    auto before = seq.PhysicalTargets(p, previous, roles, settings);
    if (!seq.Following() || !seq.ActiveWaypoint()) return (std::cerr << "feasibility-check-2\n", false);
    auto curved = previous;
    curved.blues[8].turnRate = 0.6F;
    auto after = seq.PhysicalTargets(p, curved, roles, settings);
    bool anyOverSpeed = false;
    for (std::size_t i = 0U; i < 8U; ++i)
    {
        if (!after[i].valid) return (std::cerr << "feasibility-check-3\n", false);
        if (after[i].speed > p.maxSpeed)
        {
            anyOverSpeed = true;
            if (after[i].physicallyFeasible
                || blue::GeneralizedPvs0WedgeHeld(curved, after[i], i, settings))
                return (std::cerr << "feasibility-check-4\n", false);
        }
    }
    if (!anyOverSpeed) return (std::cerr << "feasibility-check-5\n", false);
    // The same attainable-speed slot can demand an impossible instantaneous
    // slot-heading rotation across consecutive fixed-dt snapshots.
    curved.blues[8].speed = 300.F;
    curved.blues[8].turnRate = 0.F;
    before = seq.PhysicalTargets(p, curved, roles, settings);
    curved.blues[8].heading += 0.35F;
    after = seq.PhysicalTargets(p, curved, roles, settings);
    if (blue::GeneralizedPhysicalSlotRotationFeasible(p, before[0], after[0]))
        return (std::cerr << "feasibility-check-6\n", false);
    // Changing leader WPT must never move the follower's physical slot.
    auto const wpt1 = blue::ComputePatrolWaypointTargets(
        p, curved, roles, seq.Waypoints()[0].pose);
    auto const wpt10 = blue::ComputePatrolWaypointTargets(
        p, curved, roles, seq.Waypoints()[9].pose);
    auto const physical1 = blue::ComputePhysicalWedgeTargets(
        p, curved, roles, wpt1);
    auto const physical10 = blue::ComputePhysicalWedgeTargets(
        p, curved, roles, wpt10);
    return Near(physical1[0].pose.north, physical10[0].pose.north, 0.F)
        && Near(physical1[0].guidancePose.east,
                physical10[0].guidancePose.east, 0.F)
        && std::hypot(
            physical1[8].guidancePose.north - physical10[8].guidancePose.north,
            physical1[8].guidancePose.east - physical10[8].guidancePose.east)
                > 100.F;
}

bool RewardSpeedAndFrozen()
{
    auto p = Config();
    auto const roles = *blue::BlueTeamRoles::Create(2U, 0U);
    blue::PatrolNoThreatSettings settings{};
    settings.stepRewardScale = 0.004F;
    auto before = Sample(p, roles, 2600.F);
    auto const beforeTargets = blue::ComputePhysicalWedgeTargets(
        p, before, roles,
        blue::ComputePatrolTargets(p, before, roles));
    float const heading = before.blues[0].heading;
    auto const fixed = beforeTargets[0].guidancePose;
    // Fixed target well ahead, common initial geometry for the speed sweep.
    before.blues[0].north = fixed.north - 12000.F * std::cos(heading);
    before.blues[0].east = fixed.east - 12000.F * std::sin(heading);
    float previous = 2.F;
    float at300 = 0.F;
    float at750 = 0.F;
    for (int speed = 0; speed <= 750; ++speed)
    {
        auto after = before;
        after.blues[0].speed = static_cast<float>(speed);
        after.blues[0].north += speed * p.dt * std::cos(heading);
        after.blues[0].east += speed * p.dt * std::sin(heading);
        // Keep before reference frozen: only after measurement is recomputed.
        auto afterTargets = blue::ComputePhysicalWedgeTargets(
            p, after, roles, blue::ComputePatrolTargets(p, after, roles));
        float const value = blue::GeneralizedUnifiedPatrolConvergenceShaping(
            p, before, after, beforeTargets, afterTargets, roles, 0U, settings);
        float const weighted = blue::Clamp(2.F * value, -1.F, 1.F);
        if (!std::isfinite(value) || !std::isfinite(weighted)
            || value < -1.F || value > 1.F)
            return false;
        if (speed == 300) at300 = weighted;
        if (speed == 750) at750 = weighted;
        if (speed > 300 && weighted > previous + 0.001F)
            return false;
        if (speed >= 300) previous = weighted;
        if (speed == 360 || speed == 299 || speed == 301
            || speed == 359 || speed == 361)
        {
            if (!std::isfinite(weighted)) return false;
        }
    }
    if (!(at750 < at300)) return false;
    auto stationary = before;
    auto movedLeader = before;
    movedLeader.blues[0].north += 50.F;
    movedLeader.blues[0].east += 10.F;
    auto noMotionTargets = blue::ComputePhysicalWedgeTargets(
        p, movedLeader, roles,
        blue::ComputePatrolTargets(p, movedLeader, roles));
    float const idle = blue::GeneralizedUnifiedPatrolConvergenceShaping(
        p, before, movedLeader, beforeTargets, noMotionTargets,
        roles, 1U, settings);
    // A frozen wingman with a moving reference cannot collect positive bonus.
    if (idle > 0.F) return false;
    auto backward = before;
    backward.blues[0].speed = 750.F;
    backward.blues[0].north -= 60.F * std::cos(heading);
    backward.blues[0].east -= 60.F * std::sin(heading);
    auto const backwardTargets = blue::ComputePhysicalWedgeTargets(
        p, backward, roles,
        blue::ComputePatrolTargets(p, backward, roles));
    float const retreat = blue::GeneralizedUnifiedPatrolConvergenceShaping(
        p, before, backward, beforeTargets, backwardTargets,
        roles, 0U, settings);
    if (!(retreat < 0.F)) return false;
    // Delayed/dead PvsN must keep historical, unmodified reference/reward path.
    before.counts.redMobiles = 1U;
    stationary.counts.redMobiles = 1U;
    before.reds[0].available = false;
    stationary.reds[0].available = false;
    auto const previousTargets = blue::ComputePatrolTargets(p, before, roles);
    auto const currentTargets = blue::ComputePatrolTargets(p, stationary, roles);
    blue::GeneralizedPatrolWaypointSequencer seq{};
    auto const legacyTargets = seq.Targets(p, before, roles, settings);
    if (!Near(legacyTargets[0].guidancePose.north,
              previousTargets[0].guidancePose.north, 0.F))
        return false;
    float const legacy = blue::GeneralizedPatrolConvergenceShaping(
        p, before, stationary, previousTargets, currentTargets,
        roles, 0U, settings);
    return std::isfinite(legacy);
}


/** The same physical guidance is observable with 0, 1 or multiple Reds.
 * No scenario-specific target mask or choice of actor actions exists here.
 */
bool SharedPhysicalGuidance()
{
    auto const p = Config();
    blue::PatrolNoThreatSettings settings{};
    auto const roles = *blue::BlueTeamRoles::Create(3U, 1U);
    blue::GeneralizedPatrolWaypointSequencer seq{};
    for (std::size_t n : {0U, 1U, 2U, 4U})
        for (int activeMode = 0; activeMode < 3; ++activeMode)
        {
            auto state = Sample(p, roles);
            state.counts.redMobiles = n;
            state.blues[roles.LeaderPhysicalIndex()].speed = 500.F;
            for (std::size_t red = 0U; red < n; ++red)
                state.reds[red] = {10000.F, 9000.F, 0.F, 300.F, 0.F,
                    activeMode == 0 || (activeMode == 1 && red == 0U)};
            auto const baseline = seq.Targets(p, state, roles, settings);
            auto const expected = blue::ComputePhysicalWedgeTargets(
                p, state, roles, baseline);
            auto const selected = seq.PhysicalTargets(p, state, roles, settings);
            if (!selected[roles.LeaderPhysicalIndex()].valid
                || selected[roles.LeaderPhysicalIndex()].guidanceSpeed
                       != baseline[roles.LeaderPhysicalIndex()].guidanceSpeed)
                return false;
            for (std::size_t i = 0U; i < state.counts.blueMobiles; ++i)
            {
                if (selected[i].valid != expected[i].valid
                    || selected[i].guidanceSpeed != expected[i].guidanceSpeed
                    || selected[i].guidancePose.north != expected[i].guidancePose.north
                    || selected[i].guidancePose.east != expected[i].guidancePose.east)
                    return false;
            }
            std::array<blue::GeneralizedObservation, 9U> observation{};
            if (!blue::BuildGeneralizedObservations(
                p, state, roles, observation, &selected))
                return false;
            for (std::size_t red = 0U; red < n; ++red)
            {
                std::size_t const offset =
                    blue::GeneralizedObservationLayout::RedOffset
                    + red * blue::GeneralizedObservationLayout::RedFeatures;
                // Last feature in the Red record is its actual availability mask.
                float const mask = observation[0].values[
                    offset + blue::GeneralizedObservationLayout::RedFeatures - 1U];
                if (mask != (state.reds[red].available ? 1.F : 0.F))
                    return false;
            }
            auto after = state;
            for (std::size_t i = 0U; i < state.counts.blueMobiles; ++i)
            {
                after.blues[i].north += 60.F;
                after.blues[i].east += 20.F;
            }
            auto const afterTargets = seq.PhysicalTargets(p, after, roles, settings);
            float const guidance = blue::GeneralizedUnifiedPatrolConvergenceShaping(
                p, state, after, selected, afterTargets,
                roles, roles.LeaderPhysicalIndex(), settings);
            if (!std::isfinite(guidance))
                return false;
        }
    return true;
}

/** Regression: real Red presence, not configured P/N or their proximity,
 * determines whether patrol shaping is relevant to a transition.
 */
/** Hostiles must provide a full interception gradient at arbitrary range,
 * for one or several visible Reds. No proximity selector may suppress it.
 */
bool DistantHostilePursuit()
{
    auto const p = Config();
    for (std::size_t count : {1U, 2U, 4U})
    {
        blue::GeneralizedRuntimeState before{};
        before.counts = {1U, count};
        before.blues[0] = {0.F, 0.F, 0.F, 300.F, 0.F, true};
        before.objectiveNorth = 0.F;
        before.objectiveEast = 0.F;
        before.objectiveAvailable = true;
        for (std::size_t i = 0U; i < count; ++i)
            before.reds[i] = {200000.F + 1000.F * static_cast<float>(i),
                              0.F, 0.F, 300.F, 0.F, true};
        auto after = before;
        after.blues[0].north = 60.F;
        float const forward = blue::GeneralizedPursuitProgress(
            p, before, after);
        if (!Near(forward, 60.F / (p.maxSpeed * p.dt), 0.01F))
            return false;
        after.blues[0].north = -60.F;
        float const retreat = blue::GeneralizedPursuitProgress(
            p, before, after);
        if (!Near(retreat, -60.F / (p.maxSpeed * p.dt), 0.01F))
            return false;
    }
    return true;
}

bool PatrolRewardRelevance()
{
    blue::GeneralizedRuntimeState before{};
    before.counts = {1U, 4U};
    before.blues[0].available = true;
    auto after = before;
    if (!blue::GeneralizedPatrolRewardRelevant(before, after))
        return false; // Delayed or destroyed Reds: navigation remains useful.
    before.reds[3].available = true;
    if (blue::GeneralizedPatrolRewardRelevant(before, after))
        return false; // Red destroyed during the step.
    after.reds[3].available = true;
    if (blue::GeneralizedPatrolRewardRelevant(before, after))
        return false;
    before.reds[3].available = false;
    if (blue::GeneralizedPatrolRewardRelevant(before, after))
        return false; // Red spawned during the step.
    after.reds[3].available = false;
    return blue::GeneralizedPatrolRewardRelevant(before, after);
}

/** @brief Quantifies virtual slot motion independently of vehicle motion. */
bool HeadingSensitiveReferences()
{
    auto const parameters = Config();
    blue::Mobile leader{10000.F, 5000.F, 0.F, 300.F, 0.F, true};
    auto const offset = blue::EvaluateFormationOffset(0U, 8000.F, 0.55F);
    auto const first = blue::GeneralizedPhysicalSlot(parameters, leader, offset);
    for (float const delta : {0.08726646F, -0.08726646F, 0.5F, -0.5F})
    {
        leader.heading = delta;
        auto const next = blue::GeneralizedPhysicalSlot(parameters, leader, offset);
        float const displacement = std::hypot(next.pose.north - first.pose.north,
            next.pose.east - first.pose.east);
        if (!first.valid || !next.valid
            || !Near(displacement, 16000.F * std::abs(std::sin(0.5F * delta)))
            || leader.north != 10000.F || leader.east != 5000.F)
            return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!PhysicalSlots()) { std::cerr << "physical slots\n"; return 2; }
    if (!FeasibilityAndSequencing()) { std::cerr << "feasibility\n"; return 3; }
    if (!RewardSpeedAndFrozen()) { std::cerr << "reward\n"; return 4; }
    if (!SharedPhysicalGuidance()) { std::cerr << "shared physical guidance\n"; return 5; }
    if (!PatrolRewardRelevance()) { std::cerr << "patrol reward priority\n"; return 6; }
    if (!DistantHostilePursuit()) { std::cerr << "distant red pursuit\n"; return 7; }
    if (!HeadingSensitiveReferences()) { std::cerr << "virtual slot motion\n"; return 8; }
    return 0;
}
