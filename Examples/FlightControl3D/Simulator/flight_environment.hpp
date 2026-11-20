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

/** Same transition loop for desktop, headless training and deterministic tests. */
class FlightEnvironment final
{
public:
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
    StepResult Step(Controls const& action);
    Observation ObservationNow() const noexcept { return Observe(episodeConfig_, mission_, aircraft_); }
    Config const& Settings() const noexcept { return episodeConfig_; }
    AircraftState const& Aircraft() const noexcept { return aircraft_; }
    MissionState const& Mission() const noexcept { return mission_; }
    /** @brief Terminal crash conditions for the current episode. */
    CrashDiagnostic const& CrashDetails() const noexcept { return crashDiagnostic_; }
    Guidance CurrentGuidance() const noexcept { return Guide(episodeConfig_, mission_, aircraft_); }
    /**
     * @brief Sets the maximum start difficulty of later training episodes.
     * @param difficulty Fraction in [0, 1]; seeded episodes use their own EpisodeStart.
     */
    void SetDifficulty(double difficulty) noexcept;
    /** @brief Whether the current episode started airborne rather than already rolling. */
    bool StartedAirborne() const noexcept { return startedAirborne_; }
    /** @brief Replace only existing reward settings for this and later episodes. */
    void SetRewardSettings(RewardSettings const& settings) noexcept
    { config_.reward = settings; episodeConfig_.reward = settings; }
private:
    /** @brief Alternates the training approach side so both intercepts are practised equally. */
    ApproachSide NextTrainingSide() noexcept;
    /** @brief Clears the previous episode and draws the runway pose of the next one. */
    void PrepareEpisode(bool rolling);
    /** @brief Starts an already-landed episode without a free touchdown reward. */
    Observation StartRolling();
    /** @brief Starts an airborne approach on the given side and difficulty. */
    Observation StartAirborne(ApproachSide side, double difficulty);
    /** @brief Native reset with all wheels settled on the deck, for rolling starts. */
    AircraftState ResetOnWheels(AircraftState initial);

    Config config_;
    Config episodeConfig_;
    JsbsimAircraft fdm_;
    std::mt19937 rng_;
    /** @brief Maximum difficulty of airborne training starts. */
    double difficulty_{1.0};
    bool startedAirborne_{true};
    ApproachSide trainingSide_{ApproachSide::Left};
    AircraftState aircraft_{};
    MissionState mission_{};
    CrashDiagnostic crashDiagnostic_{};
};
} // namespace ppo::examples::flight
