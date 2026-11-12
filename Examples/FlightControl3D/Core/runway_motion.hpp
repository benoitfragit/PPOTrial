#pragma once
#include "flight_config.hpp"
#include <random>

namespace ppo::examples::flight {
Vec3 Add(Vec3 a, Vec3 b) noexcept;
Vec3 Subtract(Vec3 a, Vec3 b) noexcept;
Vec3 Scale(Vec3 a, double scale) noexcept;
double Dot(Vec3 a, Vec3 b) noexcept;
Vec3 Cross(Vec3 a, Vec3 b) noexcept;
Vec3 RunwayForward(Runway const& runway) noexcept;
Vec3 RunwayRight(Runway const& runway) noexcept;
Vec3 RunwayNormal(Runway const& runway) noexcept;
Vec3 DeckCoordinates(Runway const& runway, Vec3 point) noexcept;
Vec3 DeckVelocity(Runway const& runway, Vec3 point) noexcept;

struct RunwayMotion final
{
    Runway base{};
    double forwardSpeed{}, heaveAmplitude{}, period{}, pitchAmplitude{}, rollAmplitude{};
    double heavePhase{}, pitchPhase{}, rollPhase{}, seaHeight{};
};
Runway RunwayAt(RunwayMotion const& motion, double seconds) noexcept;
RunwayMotion SampleRunway(Config const& config, std::mt19937& rng);
AircraftState SampleAircraft(Config const& config, Runway const& runway, std::mt19937& rng);
} // namespace ppo::examples::flight
