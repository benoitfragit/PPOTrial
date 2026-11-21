/**
 * @file flight_mission.cpp
 * @brief Implementation of the substep state machine and of the per-substep shaping.
 *
 * Order of one substep (it matters for the rewards):
 * 1. potentials Phi(s) of the state before the substep;
 * 2. guidance and contact detection on the state after the substep, first contact recorded;
 * 3. crash rules, else envelope timeout, else the running Direction cost, the touchdown,
 *    the rollout and the time limit;
 * 4. potentials Phi(s') after the outcome update (zero if absorbing), then the shaping
 *    gamma_sub * Phi(s') - Phi(s) is added to each channel.
 *
 * Read next: flight_landing.cpp (contact rules), flight_reward.cpp (formulas).
 */
#include "flight_mission.hpp"
#include "flight_landing.hpp"
#include "flight_reward.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
/** @brief Unweighted value of the Crash event; its weight is non-positive. */
constexpr double CrashEventValue = 1.0;

void SetReward(StepResult& result, Config const& config, Reward channel, double value) noexcept
{
    result.rewards[static_cast<std::size_t>(channel)] = Weight(config.reward, channel) * value;
}

/** @brief Records where and how the first ground contact happened, once per episode. */
void RecordFirstContact(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, Guidance const& guidance) noexcept
{
    if (mission.contactAttempted || !IsFinite(after))
    {
        return;
    }
    mission.contactAttempted = true;
    mission.firstWheelContact = after.weightOnWheels;
    // The contact point is the first wheel on the ground, or the CG without wheel data.
    auto contactPoint = after.position;
    if (after.wheelDataValid)
    {
        for (auto const& wheel : after.wheels)
        {
            if (wheel.onGround)
            {
                contactPoint = wheel.point;
                break;
            }
        }
    }
    auto const local = DeckCoordinates(config.runway, contactPoint);
    mission.touchdownFromThreshold = local.north;
    mission.touchdownCrossMeters = local.east;
    auto const velocityBefore = RunwayAxes(config.runway, before.velocity);
    mission.touchdownSinkMps = std::max(-velocityBefore.up, -guidance.relativeVelocity.up);
    mission.touchdownSpeedMps = std::hypot(guidance.relativeVelocity.north, guidance.relativeVelocity.east);
    mission.touchdownGearPosition = after.gearPosition;
}

/** @brief Crash rules: invalid state, unsafe or off-surface contact, or leaving the runway while rolling. */
bool Crashed(Config const& config, MissionState const& mission, AircraftState const& before,
    AircraftState const& after, Guidance const& guidance, bool contact) noexcept
{
    if (!IsFinite(before) || !IsFinite(after))
    {
        return true;
    }
    bool const onSurface = OnLandingSurface(config.runway, guidance.along, guidance.cross)
        && WheelsInside(config, after);
    if (contact && (!onSurface || !LandingSafe(config, before, after) || !after.weightOnWheels))
    {
        return true;
    }
    return mission.phase == Phase::Rollout && !onSurface;
}

/** @brief Converts the first safe contact into the rollout phase and pays the touchdown reward. */
void Touchdown(Config const& config, MissionState& mission, StepResult& result) noexcept
{
    mission.touchedDown = true;
    mission.touchdownQuality = TouchdownQuality(config, mission);
    mission.landingAreaFactor = LandingRewardFactor(config.runway, mission.touchdownFromThreshold,
        mission.touchdownCrossMeters);
    mission.phase = Phase::Rollout;
    SetReward(result, config, Reward::Touchdown, mission.touchdownQuality);
}

/** @brief Tracks the ground roll and ends the episode after a confirmed stop. */
void Rollout(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, Guidance const& guidance, StepResult& result) noexcept
{
    double const dt = config.mission.physicsDt;
    auto const previous = DeckCoordinates(config.runway, before.position);
    mission.groundRoll += std::hypot(guidance.along - previous.north, guidance.cross - previous.east);
    double const groundSpeed = std::hypot(guidance.relativeVelocity.north, guidance.relativeVelocity.east);
    // The stop timer accumulates only while every wheel is down below the
    // stopped speed; any movement or bounce restarts it.
    bool const stopped = AllWheelsDown(after) && groundSpeed <= config.mission.stoppedSpeed;
    if (stopped)
    {
        mission.stoppedSeconds = mission.stoppedSeconds + dt;
    }
    else
    {
        mission.stoppedSeconds = 0;
    }
    if (mission.stoppedSeconds < config.mission.stoppedHoldSeconds)
    {
        return;
    }
    mission.stopFromThreshold = guidance.along;
    mission.outcome = Outcome::Stopped;
    SetReward(result, config, Reward::Stop, mission.landingAreaFactor);
}

/**
 * @brief Running cost, touchdown, rollout and time limit of a substep that neither crashed nor
 *        left the envelope.
 */
void AdvanceNominal(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, Guidance const& guidance, bool contact, StepResult& result) noexcept
{
    double const dt = config.mission.physicsDt;
    if (mission.phase == Phase::Approach)
    {
        SetReward(result, config, Reward::Direction, DirectionCost(guidance, dt));
    }
    if (contact && !mission.touchedDown)
    {
        Touchdown(config, mission, result);
    }
    if (mission.phase == Phase::Rollout)
    {
        Rollout(config, mission, before, after, guidance, result);
    }
    bool const timeElapsed = static_cast<double>(mission.steps) * dt >= config.mission.maxSeconds;
    if (mission.outcome == Outcome::Flying && timeElapsed)
    {
        mission.outcome = Outcome::Timeout;
    }
}

/**
 * @brief Adds the potential-based shaping of one substep to every channel.
 *
 * F = gamma_sub * Phi(s') - Phi(s), with gamma_sub = gamma^(1 / actionSubsteps) so that
 * the substep terms of one action telescope to gamma * Phi(s_next) - Phi(s_now) once the
 * environment discounts them inside the action. A terminal state has zero potential; a
 * truncated Timeout keeps its potential for the critic bootstrap.
 */
void AddShaping(Config const& config, MissionState const& mission, AircraftState const& after,
    std::array<double, RewardCount> const& previousPotentials, StepResult& result) noexcept
{
    double const gammaSub = std::pow(config.training.gamma, 1.0
        / static_cast<double>(config.mission.actionSubsteps));
    auto const potentials = ShapingPotentials(config, mission, after);
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        result.rewards[channel] += gammaSub * potentials[channel] - previousPotentials[channel];
    }
}
} // namespace

StepResult AdvanceMission(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, CrashDiagnostic* crashDiagnostic) noexcept
{
    StepResult result{};
    // After a boundary the mission is frozen: only the flags are reported.
    if (mission.outcome != Outcome::Flying)
    {
        result.terminated = mission.outcome != Outcome::Timeout;
        result.truncated = mission.outcome == Outcome::Timeout;
        return result;
    }
    auto const previousPotentials = ShapingPotentials(config, mission, before);
    ++mission.steps;
    auto const guidance = Guide(config, mission, after);
    bool const contact = ContactDetected(config, after);
    if (contact)
    {
        RecordFirstContact(config, mission, before, after, guidance);
    }
    bool const outsideLimits = guidance.distance > config.mission.maxDistanceMeters
        || guidance.height > config.mission.maxAltitudeMeters;
    if (Crashed(config, mission, before, after, guidance, contact))
    {
        if (crashDiagnostic)
        {
            *crashDiagnostic = DiagnoseCrash(config, mission.phase, before, after, contact);
        }
        mission.outcome = Outcome::Crash;
        SetReward(result, config, Reward::Crash, CrashEventValue);
    }
    else if (outsideLimits)
    {
        mission.outcome = Outcome::Timeout;
    }
    else
    {
        AdvanceNominal(config, mission, before, after, guidance, contact, result);
    }
    AddShaping(config, mission, after, previousPotentials, result);
    for (double const reward : result.rewards)
    {
        result.reward += reward;
    }
    result.terminated = mission.outcome == Outcome::Crash || mission.outcome == Outcome::Stopped;
    result.truncated = mission.outcome == Outcome::Timeout;
    return result;
}
} // namespace ppo::examples::flight
