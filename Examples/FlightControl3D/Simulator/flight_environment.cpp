#include "flight_environment.hpp"
#include <stdexcept>
#include <utility>

namespace ppo::examples::flight {
namespace {
Config Checked(Config config) { Validate(config); return config; }
}
FlightEnvironment::FlightEnvironment(Config config, std::filesystem::path const& data, std::optional<std::uint32_t> randomSeed)
    : config_(Checked(std::move(config))),episodeConfig_(config_),fdm_(config_,data),rng_(randomSeed.value_or(config_.training.seed))
{
}

Observation FlightEnvironment::Reset()
{
    mission_ = {};
    mission_.phase = config_.initial.approach ? Phase::Approach : Phase::Patrol;
    motion_=SampleRunway(config_,rng_);
    episodeConfig_.runway=RunwayAt(motion_,0);
    plan_=MakeFlightPlan(episodeConfig_);
    fdm_.SetRunwayMotion(motion_);
    auto const placement=spawnPlacements_.Next(config_.initial.runwayAlignedProbability,rng_);
    auto const initial=SampleAircraft(config_,episodeConfig_.runway,placement,rng_);
    mission_.waypoint=ClosestPatrolWaypoint(plan_,initial.position);
    aircraft_ = fdm_.Reset(initial);
    return ObservationNow();
}

StepResult FlightEnvironment::Step(Controls const& action)
{
    if (mission_.outcome != Outcome::Flying) throw std::logic_error("FlightControl3D: reset required after episode boundary");
    StepResult total{};
    auto controls=action;
    controls.flap=mission_.phase==Phase::Patrol ? config_.initial.flap : mission_.phase==Phase::Approach ? config_.initial.approachFlap : config_.initial.rolloutFlap;
    for (std::size_t substep=0; substep<config_.mission.actionSubsteps; ++substep)
    {
        auto const previous = aircraft_;
        auto const previousRunway=episodeConfig_.runway;
        aircraft_ = fdm_.Advance(controls);
        episodeConfig_.runway=RunwayAt(motion_,static_cast<double>(mission_.steps+1)*config_.mission.physicsDt);
        plan_=MakeFlightPlan(episodeConfig_);
        auto const step = AdvanceMission(episodeConfig_,plan_,mission_,previous,aircraft_,&previousRunway);
        total.reward += step.reward;
        for (std::size_t i=0; i<RewardCount; ++i) total.rewards[i] += step.rewards[i];
        total.terminated = step.terminated; total.truncated = step.truncated;
        if (total.terminated || total.truncated) break;
    }
    return total;
}
} // namespace ppo::examples::flight
