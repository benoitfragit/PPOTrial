/**
 * @file jsbsim_aircraft.hpp
 * @brief Adapter that owns the native JSBSim T-6 flight dynamics model.
 *
 * Role in the chain: **JSBSim** -> mission. This is the only translation unit that talks
 * to JSBSim. It converts between the SI local tangent frame of FlightControl3D
 * (north, east, up, metres) and JSBSim (feet, NED, geodetic locations), applies the
 * physical controls and reads back an AircraftState.
 *
 * The ground seen by JSBSim's native wheel and structure solver is a finite level deck (the
 * landable surface of the episode runway) surrounded by the sea, provided by a ground
 * callback; JSBSim therefore computes real gear forces and contacts.
 *
 * Not here: no policy, no mission outcome, no reward, no command filter (see
 * flight_environment.cpp).
 *
 * Read next: jsbsim_aircraft.cpp (conversions and conventions), flight_environment.hpp.
 */
#pragma once

#include "runway_geometry.hpp"
#include <filesystem>
#include <memory>

namespace ppo::examples::flight {

/** @brief Ground query result, used by the simulator tests to check the deck and the sea. */
struct GroundSample final
{
    /** @brief Height of the queried point above the surface, in metres. */
    double height{};
    /** @brief Surface point below the query, in world metres. */
    Vec3 contact{};
    /** @brief Unit surface normal, in world axes. */
    Vec3 normal{};
    /** @brief Surface velocity, in world metres per second: the deck velocity, zero on the sea. */
    Vec3 velocity{};
};

/** @brief Steady, wings-level flight condition solved by JsbsimAircraft::Trim. */
struct TrimCondition final
{
    /** @brief Calibrated airspeed, in metres per second. */
    double calibratedSpeed{};
    /** @brief Flight-path angle, in radians (negative when descending). */
    double flightPathRadians{};
    /** @brief Flap command, in [0, 1]. */
    double flap{};
    /** @brief Gear command, in [0, 1] (1 = down). */
    double gear{1.0};
    /** @brief Height above the local frame origin, in metres. */
    double heightMeters{};
};

/** @brief Controls and attitude of a steady flight condition. */
struct TrimResult final
{
    /** @brief True when JSBSim's longitudinal trim converged. */
    bool converged{};
    /** @brief Elevator command equivalent to the trimmed pitch-trim command, in [-1, 1]. */
    double elevator{};
    /** @brief Throttle lever, in [0, 1]. */
    double throttle{};
    /** @brief Trimmed pitch attitude, in radians. */
    double pitch{};
};

/** @brief Elevator and throttle commands applied at a reset. */
struct InitialCommands final
{
    /** @brief Elevator command, in [-1, 1]. */
    double elevator{};
    /** @brief Throttle lever, in [0, 1]. */
    double throttle{};
};

/** @brief Owns the real FDM. No neural policy, mission outcome or reward lives here. */
class JsbsimAircraft final
{
public:
    /**
     * @brief Loads the aircraft model and installs the deck ground callback.
     * @param config Validated configuration (aircraft name, time step, runway, sea height).
     * @param dataRoot Directory holding aircraft/, engine/ and systems/ of the pinned JSBSim data.
     * @throws std::runtime_error when the model or a required property cannot be loaded.
     */
    JsbsimAircraft(Config const& config, std::filesystem::path const& dataRoot);
    ~JsbsimAircraft();
    JsbsimAircraft(JsbsimAircraft const&) = delete;
    JsbsimAircraft& operator=(JsbsimAircraft const&) = delete;
    /**
     * @brief Re-initializes the native simulator at a requested state.
     * @param initial Finite initial position, velocity, attitude and gear position (1 down, 0 up).
     * @param phase Phase selecting the flap detent and the initial elevator and throttle.
     * @return State measured by JSBSim after initialization.
     * @throws std::invalid_argument for a non-finite state or a climb faster than the true airspeed.
     */
    AircraftState Reset(AircraftState const& initial, Phase phase = Phase::Approach);
    /**
     * @brief Re-initializes the native simulator with explicit initial elevator and throttle.
     * @param initial Finite initial position, velocity, attitude and gear position (1 down, 0 up).
     * @param phase Phase selecting the flap detent.
     * @param commands Initial elevator and throttle, for example the trim of the start condition.
     * @return State measured by JSBSim after initialization.
     * @throws std::invalid_argument for a non-finite state or a climb faster than the true airspeed.
     */
    AircraftState Reset(AircraftState const& initial, Phase phase, InitialCommands commands);
    /**
     * @brief Applies the controls and integrates one JSBSim time step (MissionSettings::physicsDt).
     * @param controls Finite physical controls; they are clamped to their ranges.
     * @return Measured state; finite is false when JSBSim failed to advance.
     */
    AircraftState Advance(Controls const& controls);
    /**
     * @brief Solves JSBSim's longitudinal trim (throttle, alpha, pitch trim) at a steady condition.
     *
     * The aircraft model sums the elevator and pitch-trim commands, so the trimmed pitch-trim
     * command is the elevator command that holds the condition. The simulator state is left
     * at the trimmed condition: call Reset before the next episode.
     * @param condition Steady wings-level condition to hold.
     * @return Trimmed controls and pitch; converged is false when the trim failed.
     */
    TrimResult Trim(TrimCondition const& condition);
    /** @brief Places the finite contact deck used by the native wheel and structure solver. */
    void SetRunway(Runway const& runway);
    /**
     * @brief Queries the ground callback at a world point.
     * @param point World position, in metres.
     * @return Height above the deck or the sea, contact point and normal.
     */
    GroundSample ProbeGround(Vec3 point) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** @brief JSBSim data directory copied next to the build by CMake (PPO_JSBSIM_DATA_ROOT). */
std::filesystem::path DefaultDataRoot();

} // namespace ppo::examples::flight
