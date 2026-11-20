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
     * @param initial Finite initial position, velocity and attitude; the gear is always extended.
     * @param phase Phase selecting the flap detent and the initial elevator and throttle.
     * @return State measured by JSBSim after initialization.
     * @throws std::invalid_argument for a non-finite state or a climb faster than the true airspeed.
     */
    AircraftState Reset(AircraftState const& initial, Phase phase = Phase::Approach);
    /**
     * @brief Applies the controls and integrates one JSBSim time step (MissionSettings::physicsDt).
     * @param controls Finite physical controls; they are clamped to their ranges.
     * @return Measured state; finite is false when JSBSim failed to advance.
     */
    AircraftState Advance(Controls const& controls);
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
