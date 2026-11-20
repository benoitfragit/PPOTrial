#include "flight_mission.hpp"
#include "flight_landing.hpp"
#include "flight_reward.hpp"
#include <algorithm>

namespace ppo::examples::flight {
namespace {
void SetReward(StepResult& result, Config const& config, Reward channel, double value) noexcept
{
    result.rewards[static_cast<std::size_t>(channel)] = Weight(config.reward, channel) * value;
}

/** @brief Records where and how the first ground contact happened, once per episode. */
void RecordFirstContact(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, Guidance const& guidance) noexcept
{
    if (mission.contactAttempted || !IsFinite(after)) return;
    mission.contactAttempted = true;
    mission.firstWheelContact = after.weightOnWheels;
    auto point = after.position;
    if (after.wheelDataValid)
        for (auto const& wheel : after.wheels)
            if (wheel.onGround) { point = wheel.point; break; }
    auto const local = DeckCoordinates(config.runway, point);
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
    if (!IsFinite(before) || !IsFinite(after)) return true;
    bool const onSurface = OnLandingSurface(config.runway, guidance.along, guidance.cross)
        && WheelsInside(config, after);
    if (contact && (!onSurface || !LandingSafe(config, before, after) || !after.weightOnWheels)) return true;
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
    bool const stopped = AllWheelsDown(after) && groundSpeed <= config.mission.stoppedSpeed;
    mission.stoppedSeconds = stopped ? mission.stoppedSeconds + dt : 0;
    if (mission.stoppedSeconds < config.mission.stoppedHoldSeconds) return;
    mission.stopFromThreshold = guidance.along;
    mission.outcome = Outcome::Stopped;
    SetReward(result, config, Reward::Stop, mission.landingAreaFactor);
}
} // namespace

StepResult AdvanceMission(Config const& config, MissionState& mission, AircraftState const& before,
    AircraftState const& after, CrashDiagnostic* crashDiagnostic) noexcept
{
    StepResult result{};
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
    if (contact) RecordFirstContact(config, mission, before, after, guidance);
    bool const outsideLimits = guidance.distance > config.mission.maxDistanceMeters
        || guidance.height > config.mission.maxAltitudeMeters;
    if (Crashed(config, mission, before, after, guidance, contact))
    {
        if (crashDiagnostic) *crashDiagnostic = DiagnoseCrash(config, mission.phase, before, after, contact);
        mission.outcome = Outcome::Crash;
        SetReward(result, config, Reward::Crash, 1.0);
    }
    else if (outsideLimits)
        mission.outcome = Outcome::Timeout;
    else
    {
        double const dt = config.mission.physicsDt;
        if (mission.phase == Phase::Approach)
            SetReward(result, config, Reward::Direction, DirectionCost(guidance, dt));
        if (contact && !mission.touchedDown) Touchdown(config, mission, result);
        if (mission.phase == Phase::Rollout) Rollout(config, mission, before, after, guidance, result);
        if (mission.outcome == Outcome::Flying && static_cast<double>(mission.steps) * dt >= config.mission.maxSeconds)
            mission.outcome = Outcome::Timeout;
    }
    // Each physics substep contributes F = gamma_sub * Phi(s') - Phi(s) per
    // shaping channel. A terminal state has zero potential; a truncated
    // Timeout keeps its potential for the critic bootstrap. The environment
    // discounts these substep terms before presenting the action to PPO.
    double const gammaSub = std::pow(config.training.gamma, 1.0 / static_cast<double>(config.mission.actionSubsteps));
    auto const potentials = ShapingPotentials(config, mission, after);
    for (std::size_t i = 0; i < RewardCount; ++i)
        result.rewards[i] += gammaSub * potentials[i] - previousPotentials[i];
    for (double const reward : result.rewards) result.reward += reward;
    result.terminated = mission.outcome == Outcome::Crash || mission.outcome == Outcome::Stopped;
    result.truncated = mission.outcome == Outcome::Timeout;
    return result;
}
} // namespace ppo::examples::flight
