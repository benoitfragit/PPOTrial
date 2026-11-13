#include "flight_environment.hpp"
#include <stdexcept>
#include <utility>

namespace ppo::examples::flight {
namespace {
Config Checked(Config config) { Validate(config); return config; }
}
Controls ApplyPhaseControls(Controls controls, Phase phase, InitialSettings const& initial) noexcept
{
    controls.flap=phase==Phase::Patrol ? initial.flap
        : phase==Phase::Approach ? initial.approachFlap : initial.rolloutFlap;
    if(phase!=Phase::Rollout) controls.brake=0.0;
    return controls;
}
FlightEnvironment::FlightEnvironment(Config config, std::filesystem::path const& data, std::optional<std::uint32_t> randomSeed)
    : config_(Checked(std::move(config))),episodeConfig_(config_),fdm_(config_,data),rng_(randomSeed.value_or(config_.training.seed))
{
}

Observation FlightEnvironment::Reset()
{
    mission_ = {};
    episodeConfig_=config_;
    auto const stage=config_.training.curriculumStage;
    // Gate 2 rehearses LOC episodes without constructing a second policy.
    bool const reminder=stage==CurriculumStage::Glide
        && std::bernoulli_distribution(config_.training.locReminderProbability)(rng_);
    episodeConfig_.training.curriculumStage=reminder ? CurriculumStage::Loc : stage;
    mission_.phase = config_.initial.approach ? Phase::Approach : Phase::Patrol;
    motion_=SampleRunway(config_,rng_);
    episodeConfig_.runway=RunwayAt(motion_,0);
    plan_=MakeFlightPlan(episodeConfig_);
    fdm_.SetRunwayMotion(motion_);
    auto const active=episodeConfig_.training.curriculumStage;
    auto const placement=active==CurriculumStage::Loc
        ? spawnPlacements_.Next(0.0,rng_)
        : active==CurriculumStage::Glide ? SpawnPlacement::LocalizerGlide
        : spawnPlacements_.Next(config_.initial.runwayAlignedProbability,rng_);
    auto initial=SampleAircraft(config_,episodeConfig_.runway,placement,rng_);
    if(active==CurriculumStage::Glide)
    {
        // Start above/below the 3-degree plane instead of on it.
        double const magnitude=std::uniform_real_distribution<double>(
            config_.training.glideOffsetMeters*0.25,config_.training.glideOffsetMeters)(rng_);
        double const sign=std::bernoulli_distribution(0.5)(rng_) ? 1.0 : -1.0;
        initial.position=Add(initial.position,Scale(RunwayNormal(episodeConfig_.runway),sign*magnitude));
    }
    mission_.waypoint=ClosestPatrolWaypoint(plan_,initial.position);
    aircraft_ = fdm_.Reset(initial);
    return ObservationNow();
}

StepResult FlightEnvironment::Step(Controls const& action)
{
    if (mission_.outcome != Outcome::Flying) throw std::logic_error("FlightControl3D: reset required after episode boundary");
    StepResult total{};
    auto const controls=ApplyPhaseControls(action,mission_.phase,config_.initial);
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
