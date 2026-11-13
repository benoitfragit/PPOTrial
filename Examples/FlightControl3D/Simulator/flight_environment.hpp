#pragma once
#include "jsbsim_aircraft.hpp"
#include <random>
#include <optional>

namespace ppo::examples::flight {

/** Resolve phase-controlled flaps and disable brakes before rollout. */
Controls ApplyPhaseControls(Controls controls, Phase phase, InitialSettings const& initial) noexcept;

/** Same transition loop for desktop, headless training and deterministic tests. */
class FlightEnvironment final
{
public:
    explicit FlightEnvironment(Config config, std::filesystem::path const& data = DefaultDataRoot(),
        std::optional<std::uint32_t> randomSeed = std::nullopt);
    Observation Reset();
    StepResult Step(Controls const& action);
    Observation ObservationNow() const noexcept { return Observe(episodeConfig_,plan_,mission_,aircraft_); }
    Config const& Settings() const noexcept { return episodeConfig_; }
    AircraftState const& Aircraft() const noexcept { return aircraft_; }
    MissionState const& Mission() const noexcept { return mission_; }
    FlightPlan const& Plan() const noexcept { return plan_; }
    Guidance CurrentGuidance() const noexcept { return Guide(episodeConfig_,plan_,mission_,aircraft_); }
private:
    Config config_;
    Config episodeConfig_;
    RunwayMotion motion_{};
    FlightPlan plan_;
    JsbsimAircraft fdm_;
    std::mt19937 rng_;
    SpawnPlacementSampler spawnPlacements_{};
    AircraftState aircraft_{};
    MissionState mission_{};
};
} // namespace ppo::examples::flight
