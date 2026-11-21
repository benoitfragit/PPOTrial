/**
 * @file flight_trim.cpp
 * @brief Closed-loop flight of a steady condition, averaged once settled.
 *
 * Loops (JSBSim sign conventions: a positive elevator command pitches the nose down):
 * - elevator = reference + Kg (gamma - gamma_target) + Ig integral + Kq q: a path above the
 *   target pitches the nose down, a nose-up pitch rate is damped;
 * - throttle = reference - Kv (V - Vtarget) - Iv integral: too slow adds power;
 * - aileron = -Kphi phi - Kp p: wings level against the propeller torque.
 * Integrals freeze while their command saturates. The gains are deliberately soft: the loops
 * only need to settle slowly, the average then removes the residual phugoid.
 *
 * Read next: flight_trim.hpp.
 */
#include "flight_trim.hpp"
#include <algorithm>
#include <cmath>

namespace ppo::examples::flight {
namespace {
/** @brief Simulated time before averaging, in seconds (several phugoid periods). */
constexpr double SettleSeconds = 150.0;
/** @brief Averaging window after settling, in seconds. */
constexpr double AverageSeconds = 40.0;
/** @brief Throttle per m/s of airspeed error. */
constexpr double SpeedGain = 0.05;
/** @brief Throttle per m/s of integrated airspeed error per second. */
constexpr double SpeedIntegralGain = 0.01;
/** @brief Elevator per rad/s of pitch rate. */
constexpr double PitchRateGain = 0.5;
/** @brief Aileron per radian of bank (a positive aileron command rolls right). */
constexpr double BankGain = 1.0;
/** @brief Aileron per rad/s of roll rate. */
constexpr double RollRateGain = 0.3;
/** @brief Elevator per radian of flight-path error. */
constexpr double PathGain = 1.5;
/** @brief Elevator per radian of integrated flight-path error per second. */
constexpr double PathIntegralGain = 0.3;
/** @brief Largest mean airspeed error of a converged trim, in m/s. */
constexpr double SpeedTolerance = 0.3;
/** @brief Largest mean flight-path error of a converged trim, in degrees. */
constexpr double PathToleranceDegrees = 0.15;
/** @brief Largest standard deviation of a settled command. */
constexpr double CommandSpreadTolerance = 0.03;

/** @brief Running mean and variance of a settled quantity. */
struct Average final
{
    double sum{};
    double squares{};
    double samples{};

    void Add(double value) noexcept
    {
        sum += value;
        squares += value * value;
        samples += 1;
    }
    double Mean() const noexcept
    {
        return sum / samples;
    }
    double Spread() const noexcept
    {
        return std::sqrt(std::max(0.0, squares / samples - Mean() * Mean()));
    }
};

/** @brief Flight-path angle of an inertial velocity, in radians. */
double PathAngle(Vec3 velocity) noexcept
{
    return std::atan2(velocity.up, std::hypot(velocity.north, velocity.east));
}
} // namespace

TrimResult FlyTrim(JsbsimAircraft& aircraft, Config const& config, TrimCondition const& condition)
{
    // Wings-level start on the condition, nose along the flight path. A descent starts
    // high enough to stay airborne for the whole flight.
    double const totalSeconds = SettleSeconds + AverageSeconds;
    double const descentMeters = std::max(0.0, -std::sin(condition.flightPathRadians) * condition.calibratedSpeed
        * totalSeconds);
    AircraftState initial{};
    initial.position = {0.0, 0.0, condition.heightMeters + descentMeters};
    initial.airspeed = condition.calibratedSpeed;
    initial.pitch = condition.flightPathRadians;
    initial.gearPosition = condition.gear;
    double const horizontal = condition.calibratedSpeed * std::cos(condition.flightPathRadians);
    initial.velocity = {horizontal, 0.0, condition.calibratedSpeed * std::sin(condition.flightPathRadians)};
    InitialCommands const reference{config.initial.elevatorTrim, config.initial.throttle};
    auto state = aircraft.Reset(initial, Phase::Approach, reference);

    double const dt = config.mission.physicsDt;
    auto const settleSteps = static_cast<int>(SettleSeconds / dt);
    auto const totalSteps = static_cast<int>(totalSeconds / dt);
    double speedIntegral = 0;
    double pathIntegral = 0;
    Controls controls{};
    controls.elevator = reference.elevator;
    controls.throttle = reference.throttle;
    controls.flap = condition.flap;
    controls.gear = condition.gear;
    Average elevator;
    Average throttle;
    Average pitch;
    Average speedError;
    Average pathError;
    for (int step = 0; step < totalSteps && IsFinite(state); ++step)
    {
        double const speed = state.airspeed - condition.calibratedSpeed;
        double const path = PathAngle(state.velocity) - condition.flightPathRadians;
        double const elevatorCommand = reference.elevator + PathGain * path + PathIntegralGain * pathIntegral
            + PitchRateGain * state.pitchRate;
        double const throttleCommand = reference.throttle - SpeedGain * speed - SpeedIntegralGain * speedIntegral;
        controls.elevator = std::clamp(elevatorCommand, -1.0, 1.0);
        controls.throttle = std::clamp(throttleCommand, 0.0, 1.0);
        // Wings level: the propeller torque would otherwise roll the aircraft into a spiral.
        controls.aileron = std::clamp(-BankGain * state.roll - RollRateGain * state.rollRate, -1.0, 1.0);
        // Anti-windup: an integral only grows while its command is free.
        if (controls.elevator == elevatorCommand)
        {
            pathIntegral += path * dt;
        }
        if (controls.throttle == throttleCommand)
        {
            speedIntegral += speed * dt;
        }
        if (step >= settleSteps)
        {
            elevator.Add(controls.elevator);
            throttle.Add(controls.throttle);
            pitch.Add(state.pitch);
            speedError.Add(speed);
            pathError.Add(path);
        }
        state = aircraft.Advance(controls);
    }
    TrimResult result{};
    if (!IsFinite(state) || elevator.samples == 0)
    {
        return result;
    }
    result.elevator = elevator.Mean();
    result.throttle = throttle.Mean();
    result.pitch = pitch.Mean();
    result.converged = std::abs(speedError.Mean()) <= SpeedTolerance
        && std::abs(pathError.Mean()) <= PathToleranceDegrees * DegreesToRadians
        && elevator.Spread() <= CommandSpreadTolerance && throttle.Spread() <= CommandSpreadTolerance;
    return result;
}
} // namespace ppo::examples::flight
