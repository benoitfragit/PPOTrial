#pragma once

#include <algorithm>
#include <cmath>

namespace ppo::examples::flight {
/**
 * @brief Conservative 1 g stall CAS from the pinned JSBSim t6texan2 lift table.
 * Full initial mass: 5000 lb empty + 500 lb fuel; S=176 ft2; clean CLmax=1.47.
 * Flap travel is 30 deg, with delta CL={0,.2,.3,.35} at {0,10,20,30} deg.
 * CAS uses sea-level density; fuel burn only reduces this conservative bound.
 * This is a model-derived envelope, not a certified real-aircraft V-speed.
 */
inline double T6StallCalibratedSpeed(double flapPosition) noexcept
{
    double const flapDegrees=30.0*std::clamp(flapPosition,0.0,1.0);
    double const flapLift=flapDegrees<=10 ? 0.02*flapDegrees
        : flapDegrees<=20 ? 0.2+0.01*(flapDegrees-10) : 0.3+0.005*(flapDegrees-20);
    constexpr double weightNewtons=5500.0*0.45359237*9.80665;
    constexpr double wingAreaSquareMeters=176.0*0.09290304;
    return std::sqrt(2.0*weightNewtons/(1.225*wingAreaSquareMeters*(1.47+flapLift)));
}

/** @brief Approach reserve against the current physical flap position. */
inline double T6MinimumCalibratedSpeed(double flapPosition) noexcept
{
    return 1.3*T6StallCalibratedSpeed(flapPosition);
}
} // namespace ppo::examples::flight
