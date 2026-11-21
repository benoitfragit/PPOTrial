/**
 * @file flight_aircraft_performance.hpp
 * @brief Stall-speed envelopes of the pinned JSBSim aircraft models.
 *
 * Role in the chain: used by the configuration validation (reference speeds must keep a
 * stall reserve), by the guidance (the reference airspeed never drops below it) and by
 * the simulator adapter (flap travel).
 *
 * The values are derived from the pinned JSBSim aerodynamic tables, not from certified
 * flight manuals. CAS uses sea-level density; fuel burn only reduces these conservative
 * bounds.
 * - t6texan2: 5000 lb empty + 500 lb fuel, S = 176 ft^2, clean CLmax = 1.47, 30 degrees of
 *   flap with delta CL = {0, 0.2, 0.3, 0.35} at {0, 10, 20, 30} degrees.
 * - pc7: 2932.65 lb empty + 1147.26 lb payload + 363.16 lb fuel, S = 178.68 ft^2, clean
 *   CLmax = 1.7471 (peak of the alpha lift table), 30 degrees of flap with delta CL =
 *   0.0092 per degree.
 *
 * Read next: flight_guidance.cpp (TargetSpeed), flight_config.cpp (ValidateMission).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>

namespace ppo::examples::flight {

/** @brief Exact avoirdupois pound, in kilograms. */
inline constexpr double PoundsToKilograms = 0.45359237;
/** @brief Standard gravity, in metres per second squared. */
inline constexpr double StandardGravity = 9.80665;
/** @brief Exact square foot, in square metres. */
inline constexpr double SquareFeetToSquareMeters = 0.09290304;
/** @brief ISA sea-level air density, in kilograms per cubic metre. */
inline constexpr double SeaLevelDensity = 1.225;
/** @brief Stall-speed multiple kept as approach reserve (1.3 Vs). */
inline constexpr double ApproachSpeedStallFactor = 1.3;
/** @brief Largest number of breakpoints of a flap lift table. */
inline constexpr std::size_t MaximumFlapLiftPoints = 4;

/** @brief One breakpoint of a flap lift table: deflection and extra lift coefficient. */
struct FlapLiftPoint final
{
    double degrees{};
    double liftIncrement{};
};

/** @brief Model-derived mass, wing and lift data of one JSBSim aircraft. */
struct AircraftPerformance final
{
    /** @brief JSBSim model name, as in the scenario "aircraft" field. */
    std::string_view name;
    /** @brief Initial mass, in pounds. */
    double massPounds{};
    /** @brief Reference wing area, in square feet. */
    double wingAreaSquareFeet{};
    /** @brief Clean maximum lift coefficient. */
    double cleanMaximumLiftCoefficient{};
    /** @brief Full flap travel, in degrees (flap position 1). */
    double flapTravelDegrees{};
    /** @brief Flap lift breakpoints, increasing in degrees, interpolated linearly. */
    std::array<FlapLiftPoint, MaximumFlapLiftPoints> flapLift{};
    /** @brief Breakpoints in use, at least two. */
    std::size_t flapLiftPoints{};
    /**
     * @brief Whether JSBSim's FGTrim can trim the aircraft.
     *
     * FGTrim fails on the PC-7 turboprop and leaves a non-finite throttle behind; such an
     * aircraft is trimmed by simulated flight (FlyTrim) and starts on its action reference.
     */
    bool jsbsimTrim{};
};

/** @brief Pinned JSBSim T-6 Texan II. */
inline constexpr AircraftPerformance T6Performance{"t6texan2", 5500.0, 176.0, 1.47, 30.0,
    {{{0.0, 0.0}, {10.0, 0.2}, {20.0, 0.3}, {30.0, 0.35}}}, 4, true};
/** @brief Pinned JSBSim Pilatus PC-7. */
inline constexpr AircraftPerformance Pc7Performance{"pc7", 2932.65 + 1147.26 + 2.0 * 181.58, 178.68, 1.7471,
    30.0, {{{0.0, 0.0}, {30.0, 0.0092 * 30.0}}}, 2, false};
/** @brief Every supported aircraft. */
inline constexpr std::array<AircraftPerformance, 2> SupportedAircraft{T6Performance, Pc7Performance};

/**
 * @brief Extra lift coefficient of the flaps, linear between the model breakpoints.
 * @param performance Aircraft data.
 * @param flapDegrees Flap deflection, in degrees, clamped to the table.
 * @return Lift coefficient increment.
 */
inline double FlapLiftIncrement(AircraftPerformance const& performance, double flapDegrees) noexcept
{
    auto const& table = performance.flapLift;
    std::size_t const last = performance.flapLiftPoints - 1;
    double const degrees = std::clamp(flapDegrees, table[0].degrees, table[last].degrees);
    for (std::size_t point = 1; point <= last; ++point)
    {
        if (degrees <= table[point].degrees)
        {
            auto const& from = table[point - 1];
            auto const& to = table[point];
            double const fraction = (degrees - from.degrees) / (to.degrees - from.degrees);
            return from.liftIncrement + fraction * (to.liftIncrement - from.liftIncrement);
        }
    }
    return table[last].liftIncrement;
}

/**
 * @brief Conservative 1 g stall CAS from the pinned JSBSim lift table.
 *
 * Vs = sqrt(2 W / (rho0 S (CLmax + delta CL flap))).
 * This is a model-derived envelope, not a certified real-aircraft V-speed.
 * @param performance Aircraft data.
 * @param flapPosition Normalized flap position, clamped to [0, 1].
 * @return Stall calibrated airspeed, in metres per second.
 */
inline double StallCalibratedSpeed(AircraftPerformance const& performance, double flapPosition) noexcept
{
    double const flapDegrees = performance.flapTravelDegrees * std::clamp(flapPosition, 0.0, 1.0);
    double const flapLift = FlapLiftIncrement(performance, flapDegrees);
    double const weightNewtons = performance.massPounds * PoundsToKilograms * StandardGravity;
    double const wingAreaSquareMeters = performance.wingAreaSquareFeet * SquareFeetToSquareMeters;
    return std::sqrt(2.0 * weightNewtons
        / (SeaLevelDensity * wingAreaSquareMeters * (performance.cleanMaximumLiftCoefficient + flapLift)));
}

/**
 * @brief Approach reserve against the current physical flap position.
 * @param performance Aircraft data.
 * @param flapPosition Normalized flap position, in [0, 1].
 * @return 1.3 times the stall calibrated airspeed, in metres per second.
 */
inline double MinimumCalibratedSpeed(AircraftPerformance const& performance, double flapPosition) noexcept
{
    return ApproachSpeedStallFactor * StallCalibratedSpeed(performance, flapPosition);
}

/**
 * @brief Finds the performance data of a supported aircraft.
 * @param name JSBSim model name.
 * @return The aircraft data, or nullptr for an unsupported aircraft.
 */
inline AircraftPerformance const* FindAircraftPerformance(std::string_view name) noexcept
{
    for (auto const& performance : SupportedAircraft)
    {
        if (performance.name == name)
        {
            return &performance;
        }
    }
    return nullptr;
}

} // namespace ppo::examples::flight
