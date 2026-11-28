/**
 * @file flight_environment.hpp
 * @brief Episode loop shared by training, validation, inference, the desktop and the tests.
 *
 * Role in the chain: JSBSim -> mission -> rewards -> observation, packaged as one
 * reinforcement-learning environment. One Step() is one policy action of
 * MissionSettings::actionSubsteps physics substeps (10 x 0.01 s = 0.1 s in the scenario).
 * Inside an action the requested controls go through a first-order filter at every
 * substep, AdvanceMission scores each substep, and the potential channels are discounted
 * by gamma_sub^k so they telescope to gamma * Phi(s_next) - Phi(s_now) for PPO.
 *
 * Two reset paths:
 * - Reset(): continuous training stream (curriculum difficulty, rolling share, alternating
 *   sides), drawn from the environment RNG;
 * - Reset(seed, start): self-contained, reproducible episode used by validations.
 *
 * Not here: the policy, PPO and statistics (Training/, Runtime/).
 *
 * Read next: flight_environment.cpp, Core/flight_mission.hpp, Core/flight_spawn.hpp.
 */
#pragma once

#include "jsbsim_aircraft.hpp"
#include "flight_guidance.hpp"
#include "flight_observation.hpp"
#include "flight_spawn.hpp"
#include <optional>
#include <random>

namespace ppo::examples::flight {

/**
 * @brief Resolve phase-dependent controls: flap detent and brakes only on the ground.
 * @param controls Requested physical controls.
 * @param phase Current mission phase.
 * @param initial Configured flap detents.
 * @return Controls applied to the simulator.
 */
Controls ApplyPhaseControls(Controls controls, Phase phase, InitialSettings const& initial) noexcept;

/** @brief Same transition loop for desktop, headless training and deterministic tests. */
class FlightEnvironment final
{
public:
    /**
     * @brief Loads JSBSim and seeds the episode RNG.
     * @param config Configuration; validated again here.
     * @param data JSBSim data directory.
     * @param randomSeed Seed of the training stream; defaults to TrainingSettings::seed.
     */
    explicit FlightEnvironment(Config config, std::filesystem::path const& data = DefaultDataRoot(),
        std::optional<std::uint32_t> randomSeed = std::nullopt);
    /**
     * @brief Starts a training episode from the continuous training stream.
     *
     * Training mixes rolling starts with airborne starts drawn up to the
     * curriculum difficulty set by SetDifficulty.
     * @return Initial observation.
     */
    Observation Reset();
    /**
     * @brief Starts a self-contained episode, reproducible from its seed.
     * @param episodeSeed Seed of the episode random generator.
     * @param start Rolling or airborne start and its difficulty; the default is the full airborne task.
     * @return Initial observation.
     */
    Observation Reset(std::uint32_t episodeSeed, EpisodeStart start = {});
    /**
     * @brief Applies one policy action for actionSubsteps physics substeps.
     * @param action Requested physical controls (from ToControls).
     * @return Rewards of the action, with discounted potential terms, and the boundary flags.
     * @throws std::logic_error after an episode boundary; std::invalid_argument for non-finite controls.
     */
    StepResult Step(Controls const& action);
    /** @brief Observation of the current state. */
    Observation ObservationNow() const noexcept
    {
        return Observe(episodeConfig_, mission_, aircraft_);
    }
    /** @brief Configuration of the current episode, including its sampled runway pose. */
    Config const& Settings() const noexcept
    {
        return episodeConfig_;
    }
    AircraftState const& Aircraft() const noexcept
    {
        return aircraft_;
    }
    MissionState const& Mission() const noexcept
    {
        return mission_;
    }
    /** @brief Terminal crash conditions for the current episode. */
    CrashDiagnostic const& CrashDetails() const noexcept
    {
        return crashDiagnostic_;
    }
    /** @brief Deck motion of the current episode and its present heave, for the renderer snapshot. */
    DeckState CurrentDeck() const noexcept;
    /** @brief Guidance of the current state, for the renderer snapshot. */
    Guidance CurrentGuidance() const noexcept
    {
        return Guide(episodeConfig_, mission_, aircraft_);
    }
    /**
     * @brief Sets the maximum start difficulty of later training episodes.
     * @param difficulty Fraction in [0, 1]; seeded episodes use their own EpisodeStart.
     */
    void SetDifficulty(double difficulty) noexcept;
    /** @brief Whether the current episode started airborne rather than already rolling. */
    bool StartedAirborne() const noexcept
    {
        return startedAirborne_;
    }
    /** @brief Replace only existing reward settings for this and later episodes. */
    void SetRewardSettings(RewardSettings const& settings) noexcept
    {
        config_.reward = settings;
        episodeConfig_.reward = settings;
    }
    /** @brief The training stream: what draws the next training episodes, and their next approach side. */
    struct TrainingStream final
    {
        std::mt19937 random;
        ApproachSide nextSide = ApproachSide::Left;
    };
    /** @brief The training stream, to resume it later exactly where it is. */
    TrainingStream SavedTrainingStream() const
    {
        return {rng_, trainingSide_};
    }
    /** @brief Continues a training stream saved by SavedTrainingStream. */
    void ResumeTrainingStream(TrainingStream const& stream)
    {
        rng_ = stream.random;
        trainingSide_ = stream.nextSide;
    }

private:
    /** @brief Alternates the training approach side so both intercepts are practised equally. */
    ApproachSide NextTrainingSide() noexcept;
    /** @brief Clears the previous episode and draws the runway pose of the next one. */
    void PrepareEpisode(bool rolling);
    /** @brief Starts an already-landed episode without a free touchdown reward. */
    Observation StartRolling();
    /** @brief Starts an airborne approach on the given side and difficulty, gear down or retracted. */
    Observation StartAirborne(ApproachSide side, double difficulty, bool gearUp);
    /**
     * @brief Initial elevator and throttle of an airborne start: JSBSim's trim of its speed,
     *        flight path and gear, or the action reference when the trim fails.
     */
    InitialCommands StartCommands(AircraftState const& initial);
    /** @brief Native reset with all wheels settled on the deck, for rolling starts. */
    AircraftState ResetOnWheels(AircraftState initial);
    /**
     * @brief Runs the physics substeps of one action and accumulates their rewards.
     * @param controls Bounded, phase-resolved target controls.
     * @param total Action result to fill.
     * @return Number of substeps executed before the end of the action or of the episode.
     */
    std::size_t RunSubsteps(Controls const& controls, StepResult& total);
    /** @brief Adds (gamma - gamma_sub^executed) * Phi(s) when a timeout cut the action short. */
    void CorrectPartialTimeout(std::size_t executed, StepResult& total) const noexcept;
    /**
     * @brief Places a moving deck at its pose after a number of physics substeps, in the episode
     *        configuration and in the JSBSim ground; does nothing for a static deck.
     */
    void MoveDeck(std::size_t steps);

    /** @brief Base configuration (reward weights may be replaced at a rollout boundary). */
    Config config_;
    /** @brief Configuration of the current episode, with its sampled runway. */
    Config episodeConfig_;
    JsbsimAircraft fdm_;
    /** @brief Episode RNG: training stream, or reseeded by Reset(seed, start). */
    std::mt19937 rng_;
    /** @brief Maximum difficulty of airborne training starts. */
    double difficulty_{1.0};
    bool startedAirborne_{true};
    ApproachSide trainingSide_{ApproachSide::Left};
    /** @brief Mean deck pose at the episode start, origin of the deck motion. */
    Runway deckStart_{};
    /** @brief Deck motion of the current episode; static outside the moving-deck stages. */
    DeckMotionSettings deckMotion_{};
    /** @brief Swell phase of the current episode, in radians. */
    double swellPhase_{};
    AircraftState aircraft_{};
    MissionState mission_{};
    CrashDiagnostic crashDiagnostic_{};
};

} // namespace ppo::examples::flight
