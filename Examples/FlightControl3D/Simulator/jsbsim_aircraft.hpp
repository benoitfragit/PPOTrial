#pragma once

#include "flight_mission.hpp"
#include "runway_motion.hpp"
#include <filesystem>
#include <memory>

namespace ppo::examples::flight {
struct GroundSample final { double height{}; Vec3 contact{}, normal{}, velocity{}; };

/** Owns the real FDM. No neural policy, mission outcome or reward lives here. */
class JsbsimAircraft final
{
public:
    JsbsimAircraft(Config const& config, std::filesystem::path const& dataRoot);
    ~JsbsimAircraft();
    JsbsimAircraft(JsbsimAircraft const&) = delete;
    JsbsimAircraft& operator=(JsbsimAircraft const&) = delete;
    AircraftState Reset(AircraftState const& initial);
    AircraftState Advance(Controls const& controls);
    void SetRunwayMotion(RunwayMotion const& motion);
    GroundSample ProbeGround(Vec3 point) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::filesystem::path DefaultDataRoot();

} // namespace ppo::examples::flight
