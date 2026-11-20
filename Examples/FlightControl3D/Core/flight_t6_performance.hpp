/**
 * @file flight_t6_performance.hpp
 * @brief Stall-speed envelope of the pinned JSBSim T-6 Texan II model.
 *
 * Role in the chain: used by the configuration validation (reference speeds must keep a
 * stall reserve) and by the guidance (the reference airspeed never drops below it).
 *
 * The values are derived from the pinned JSBSim t6texan2 aerodynamic tables, not from a
 * certified flight manual: full initial mass 5000 lb empty + 500 lb fuel, wing area
 * S = 176 ft^2, clean CLmax = 1.47, 30 degrees of flap travel with delta CL =
 * {0, 0.2, 0.3, 0.35} at {0, 10, 20, 30} degrees. CAS uses sea-level density; fuel burn
 * only reduces this conservative bound.
 *
 * Read next: flight_guidance.cpp (TargetSpeed), flight_config.cpp (ValidateMission).
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ppo::examples::flight {

/** @brief Full flap travel of the pinned T-6, in degrees (flap position 1). */
inline constexpr double T6FlapTravelDegrees = 30.0;
/** @brief Initial mass, 5000 lb empty plus 500 lb fuel, in pounds. */
inline constexpr double T6InitialMassPounds = 5500.0;
/** @brief Exact avoirdupois pound, in kilograms. */
inline constexpr double PoundsToKilograms = 0.45359237;
/** @brief Standard gravity, in metres per second squared. */
inline constexpr double StandardGravity = 9.80665;
/** @brief Reference wing area of the model, in square feet. */
inline constexpr double T6WingAreaSquareFeet = 176.0;
/** @brief Exact square foot, in square metres. */
inline constexpr double SquareFeetToSquareMeters = 0.09290304;
/** @brief ISA sea-level air density, in kilograms per cubic metre. */
inline constexpr double SeaLevelDensity = 1.225;
/** @brief Clean maximum lift coefficient of the model. */
inline constexpr double T6CleanMaximumLiftCoefficient = 1.47;
/** @brief Stall-speed multiple kept as approach reserve (1.3 Vs). */
inline constexpr double T6ApproachSpeedStallFactor = 1.3;

/**
 * @brief Extra lift coefficient of the flaps, linear between the model breakpoints.
 *
 * 0.02 per degree up to 10 degrees, then 0.01 per degree to 20 degrees, then 0.005 per
 * degree to 30 degrees.
 * @param flapDegrees Flap deflection, in degrees, in [0, 30].
 * @return Lift coefficient increment.
 */
inline double T6FlapLiftIncrement(double flapDegrees) noexcept
{
    constexpr double FirstBreakpointDegrees = 10.0;
    constexpr double SecondBreakpointDegrees = 20.0;
    constexpr double FirstSlopePerDegree = 0.02;
    constexpr double SecondSlopePerDegree = 0.01;
    constexpr double ThirdSlopePerDegree = 0.005;
    constexpr double LiftAtFirstBreakpoint = 0.2;
    constexpr double LiftAtSecondBreakpoint = 0.3;
    if (flapDegrees <= FirstBreakpointDegrees)
    {
        return FirstSlopePerDegree * flapDegrees;
    }
    if (flapDegrees <= SecondBreakpointDegrees)
    {
        return LiftAtFirstBreakpoint + SecondSlopePerDegree * (flapDegrees - FirstBreakpointDegrees);
    }
    return LiftAtSecondBreakpoint + ThirdSlopePerDegree * (flapDegrees - SecondBreakpointDegrees);
}

/**
 * @brief Conservative 1 g stall CAS from the pinned JSBSim t6texan2 lift table.
 *
 * Vs = sqrt(2 W / (rho0 S (CLmax + delta CL flap))).
 * This is a model-derived envelope, not a certified real-aircraft V-speed.
 * @param flapPosition Normalized flap position, clamped to [0, 1].
 * @return Stall calibrated airspeed, in metres per second.
 */
inline double T6StallCalibratedSpeed(double flapPosition) noexcept
{
    double const flapDegrees = T6FlapTravelDegrees * std::clamp(flapPosition, 0.0, 1.0);
    double const flapLift = T6FlapLiftIncrement(flapDegrees);
    constexpr double weightNewtons = T6InitialMassPounds * PoundsToKilograms * StandardGravity;
    constexpr double wingAreaSquareMeters = T6WingAreaSquareFeet * SquareFeetToSquareMeters;
    return std::sqrt(2.0 * weightNewtons
        / (SeaLevelDensity * wingAreaSquareMeters * (T6CleanMaximumLiftCoefficient + flapLift)));
}

/**
 * @brief Approach reserve against the current physical flap position.
 * @param flapPosition Normalized flap position, in [0, 1].
 * @return 1.3 times the stall calibrated airspeed, in metres per second.
 */
inline double T6MinimumCalibratedSpeed(double flapPosition) noexcept
{
    return T6ApproachSpeedStallFactor * T6StallCalibratedSpeed(flapPosition);
}

} // namespace ppo::examples::flight
