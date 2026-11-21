/**
 * @file flight_environment.cpp
 * @brief Episode resets, the control filter and the action loop with per-substep discounting.
 *
 * Flow of one action (0.1 s = 10 substeps of 0.01 s in the scenario):
 * 1. reject a call after a boundary and non-finite controls;
 * 2. resolve the flap and brake for the phase, clamp, scale the airborne rudder;
 * 3. for each substep: filter the controls (first order, alpha = 1 - exp(-dt / tau)),
 *    advance JSBSim, AdvanceMission, accumulate rewards (potential channels multiplied by
 *    gamma_sub^k, k = substep index), stop at a boundary;
 * 4. if a timeout cut the action short, correct the end potential so PPO's gamma bootstrap
 *    stays consistent.
 *
 * Read next: Core/flight_mission.cpp (substep rules), Training/flight_learner.cpp (GAE).
 */
#include "flight_environment.hpp"
#include "flight_landing.hpp"
#include "flight_mission.hpp"
#include "flight_reward.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ppo::examples::flight {
namespace {
/** @brief Native strut compression of a rolling start; avoids the ambiguous height == 0 contact boundary. */
constexpr double StrutCompressionMeters = 0.001;
/** @brief Probability of the right side for a seeded airborne start. */
constexpr double SeededRightSideProbability = 0.5;

Config Checked(Config config)
{
    Validate(config);
    return config;
}

/** @brief Advance one bounded command through a first-order filter: y += alpha * (u - y). */
double FilteredValue(double previous, double target, double alpha) noexcept
{
    return previous + alpha * (target - previous);
}

/**
 * @brief Apply the first-order actuator command filter for one physics step.
 * @param previous Controls applied during the previous physics step.
 * @param target Finite, physically bounded requested controls.
 * @param alpha Discrete filter coefficient in [0,1].
 * @return Filtered continuous controls with an immediate flap selection.
 */
Controls FilterControls(Controls const& previous, Controls const& target, double alpha) noexcept
{
    return {FilteredValue(previous.elevator, target.elevator, alpha),
        FilteredValue(previous.aileron, target.aileron, alpha),
        FilteredValue(previous.rudder, target.rudder, alpha),
        FilteredValue(previous.throttle, target.throttle, alpha),
        FilteredValue(previous.brake, target.brake, alpha), target.flap};
}

/** @brief Clamp requested controls to their physical ranges; airborne rudder authority is reduced. */
Controls BoundedControls(Controls controls, Phase phase, double airborneRudderAuthority) noexcept
{
    controls.elevator = std::clamp(controls.elevator, -1.0, 1.0);
    controls.aileron = std::clamp(controls.aileron, -1.0, 1.0);
    double rudderAuthority = airborneRudderAuthority;
    if (phase == Phase::Rollout)
    {
        rudderAuthority = 1.0;
    }
    controls.rudder = std::clamp(controls.rudder, -1.0, 1.0) * rudderAuthority;
    controls.throttle = std::clamp(controls.throttle, 0.0, 1.0);
    controls.brake = std::clamp(controls.brake, 0.0, 1.0);
    return controls;
}

/** @brief Per-substep discount gamma^(1 / actionSubsteps). */
double SubstepDiscount(Config const& config) noexcept
{
    return std::pow(config.training.gamma, 1.0 / static_cast<double>(config.mission.actionSubsteps));
}
} // namespace

Controls ApplyPhaseControls(Controls controls, Phase phase, InitialSettings const& initial) noexcept
{
    if (phase == Phase::Approach)
    {
        controls.flap = initial.approachFlap;
    }
    else
    {
        controls.flap = initial.rolloutFlap;
    }
    if (phase != Phase::Rollout)
    {
        controls.brake = 0.0;
    }
    return controls;
}

FlightEnvironment::FlightEnvironment(Config config, std::filesystem::path const& data,
    std::optional<std::uint32_t> randomSeed)
    : config_(Checked(std::move(config))), episodeConfig_(config_), fdm_(config_, data),
      rng_(randomSeed.value_or(config_.training.seed))
{
}

void FlightEnvironment::SetDifficulty(double difficulty) noexcept
{
    difficulty_ = std::clamp(difficulty, 0.0, 1.0);
}

ApproachSide FlightEnvironment::NextTrainingSide() noexcept
{
    auto const side = trainingSide_;
    if (side == ApproachSide::Left)
    {
        trainingSide_ = ApproachSide::Right;
    }
    else
    {
        trainingSide_ = ApproachSide::Left;
    }
    return side;
}

AircraftState FlightEnvironment::ResetOnWheels(AircraftState initial)
{
    auto const& runway = episodeConfig_.runway;
    // First reset: measure where the native wheels sit relative to the deck.
    auto const measured = fdm_.Reset(initial, Phase::Rollout);
    if (!measured.wheelDataValid)
    {
        throw std::runtime_error("FlightControl3D: rollout requires native wheel geometry");
    }
    double highestWheel = DeckCoordinates(runway, measured.wheels.front().point).up;
    for (auto const& wheel : measured.wheels)
    {
        highestWheel = std::max(highestWheel, DeckCoordinates(runway, wheel.point).up);
    }
    // Second reset: lower the aircraft so that every wheel is just compressed on the deck.
    auto const lowering = Scale(RunwayNormal(runway), -highestWheel - StrutCompressionMeters);
    initial.position = Add(initial.position, lowering);
    auto const grounded = fdm_.Reset(initial, Phase::Rollout);
    if (!IsFinite(grounded) || !grounded.wheelDataValid || !AllWheelsDown(grounded)
        || grounded.structuralContact)
    {
        throw std::runtime_error(
            "FlightControl3D: native rollout reset did not establish safe wheel contact");
    }
    return grounded;
}

Observation FlightEnvironment::Reset()
{
    // Training mixes rolling starts, to practise braking, with airborne starts
    // whose difficulty is drawn up to the curriculum level. The density grows
    // linearly toward that level (two thirds on average), so the frontier is
    // practised most while easier cases keep being rehearsed: with u uniform
    // in [0, 1], difficulty = level * sqrt(u) has density 2x / level^2.
    double const rollingShare = config_.training.curriculum.rolloutStartShare;
    bool const rolling = std::bernoulli_distribution(rollingShare)(rng_);
    PrepareEpisode(rolling);
    if (rolling)
    {
        return StartRolling();
    }
    auto const side = NextTrainingSide();
    double const difficulty = difficulty_ * std::sqrt(std::uniform_real_distribution<double>(0.0, 1.0)(rng_));
    return StartAirborne(side, difficulty);
}

Observation FlightEnvironment::Reset(std::uint32_t episodeSeed, EpisodeStart start)
{
    // A seeded episode is fully determined by its seed and its explicit start,
    // whatever the training stream or curriculum did before.
    rng_.seed(episodeSeed);
    PrepareEpisode(start.rolling);
    if (start.rolling)
    {
        return StartRolling();
    }
    bool const rightSide = std::bernoulli_distribution(SeededRightSideProbability)(rng_);
    auto side = ApproachSide::Left;
    if (rightSide)
    {
        side = ApproachSide::Right;
    }
    return StartAirborne(side, start.difficulty);
}

void FlightEnvironment::PrepareEpisode(bool rolling)
{
    crashDiagnostic_ = {};
    startedAirborne_ = !rolling;
    episodeConfig_ = config_;
    episodeConfig_.runway = SampleRunway(config_, rng_);
    fdm_.SetRunway(episodeConfig_.runway);
    mission_ = {};
    // The command filter starts from the controls the native reset applies:
    // idle on the ground, the trimmed approach in flight.
    if (rolling)
    {
        mission_.phase = Phase::Rollout;
        mission_.filteredControls.throttle = 0.0;
        mission_.filteredControls.elevator = 0.0;
    }
    else
    {
        mission_.phase = Phase::Approach;
        mission_.filteredControls.throttle = config_.initial.throttle;
        mission_.filteredControls.elevator = config_.initial.elevatorTrim;
    }
}

Observation FlightEnvironment::StartRolling()
{
    // An already-landed start practises braking without a free touchdown reward.
    mission_.touchedDown = true;
    mission_.contactAttempted = true;
    mission_.firstWheelContact = true;
    aircraft_ = ResetOnWheels(SampleRolloutStart(config_, episodeConfig_.runway, rng_));
    return ObservationNow();
}

Observation FlightEnvironment::StartAirborne(ApproachSide side, double difficulty)
{
    auto const initial = SampleApproachStart(config_, episodeConfig_.runway, side, difficulty, rng_);
    aircraft_ = fdm_.Reset(initial, Phase::Approach);
    return ObservationNow();
}

std::size_t FlightEnvironment::RunSubsteps(Controls const& controls, StepResult& total)
{
    auto const& mission = config_.mission;
    // Discrete first-order filter: exact for a constant target over dt.
    double const filterAlpha = -std::expm1(-mission.physicsDt / mission.controlFilterTimeConstantSeconds);
    double const gammaSub = SubstepDiscount(config_);
    double discount = 1.0;
    std::size_t executed = 0;
    for (std::size_t substep = 0; substep < mission.actionSubsteps; ++substep)
    {
        auto const previous = aircraft_;
        mission_.filteredControls = FilterControls(mission_.filteredControls, controls, filterAlpha);
        aircraft_ = fdm_.Advance(mission_.filteredControls);
        auto const step = AdvanceMission(episodeConfig_, mission_, previous, aircraft_, &crashDiagnostic_);
        // Potential terms are discounted inside the action so that they
        // telescope to gamma * Phi(s_next) - Phi(s_now) for PPO.
        for (std::size_t channel = 0; channel < RewardCount; ++channel)
        {
            bool const potential = IsPotentialChannel(static_cast<Reward>(channel));
            if (potential)
            {
                total.rewards[channel] += discount * step.rewards[channel];
            }
            else
            {
                total.rewards[channel] += step.rewards[channel];
            }
        }
        total.terminated = step.terminated;
        total.truncated = step.truncated;
        ++executed;
        if (total.terminated || total.truncated)
        {
            break;
        }
        discount *= gammaSub;
    }
    return executed;
}

void FlightEnvironment::CorrectPartialTimeout(std::size_t executed, StepResult& total) const noexcept
{
    // A timeout can truncate an action after fewer than actionSubsteps. PPO
    // still bootstraps with gamma for that transition, so match gamma rather
    // than gamma_sub^executed on its non-absorbing end potential.
    if (!total.truncated || executed >= config_.mission.actionSubsteps)
    {
        return;
    }
    double const gammaSub = SubstepDiscount(config_);
    auto const finalPotentials = ShapingPotentials(episodeConfig_, mission_, aircraft_);
    double const correction = config_.training.gamma - std::pow(gammaSub, static_cast<double>(executed));
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        total.rewards[channel] += correction * finalPotentials[channel];
    }
}

StepResult FlightEnvironment::Step(Controls const& action)
{
    if (mission_.outcome != Outcome::Flying)
    {
        throw std::logic_error("FlightControl3D: reset required after episode boundary");
    }
    for (double const value : {action.elevator, action.aileron, action.rudder, action.throttle, action.brake})
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("FlightControl3D: invalid flight control");
        }
    }
    auto const controls = BoundedControls(ApplyPhaseControls(action, mission_.phase, config_.initial),
        mission_.phase, config_.mission.airborneRudderAuthority);
    StepResult total{};
    std::size_t const executed = RunSubsteps(controls, total);
    CorrectPartialTimeout(executed, total);
    for (double const reward : total.rewards)
    {
        total.reward += reward;
    }
    return total;
}
} // namespace ppo::examples::flight
