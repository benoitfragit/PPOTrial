#pragma once

#include "runway_geometry.hpp"
#include <filesystem>
#include <memory>

namespace ppo::examples::flight {
struct GroundSample final { double height{}; Vec3 contact{}, normal{}; };

/** Owns the real FDM. No neural policy, mission outcome or reward lives here. */
class JsbsimAircraft final
{
public:
    JsbsimAircraft(Config const& config, std::filesystem::path const& dataRoot);
    ~JsbsimAircraft();
    JsbsimAircraft(JsbsimAircraft const&) = delete;
    JsbsimAircraft& operator=(JsbsimAircraft const&) = delete;
    /**
     * @brief Re-initializes the native simulator at a requested state.
     * @param initial Finite initial position, velocity and attitude; the gear is always extended.
     * @param phase Phase selecting the flap detent and the initial elevator and throttle.
     * @return State measured by JSBSim after initialization.
     */
    AircraftState Reset(AircraftState const& initial, Phase phase = Phase::Approach);
    AircraftState Advance(Controls const& controls);
    /** @brief Places the finite contact deck used by the native wheel and structure solver. */
    void SetRunway(Runway const& runway);
    GroundSample ProbeGround(Vec3 point) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::filesystem::path DefaultDataRoot();

} // namespace ppo::examples::flight
