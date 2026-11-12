#pragma once
#include "flight_config.hpp"
#include <array>
#include <cstdint>
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

/** @brief Mutually exclusive aircraft placements relative to the runway. */
enum class SpawnPlacement : std::uint8_t { Left, LocalizerGlide, Right };

/**
 * @brief Produces a prefix-balanced sequence matching the configured LOC share.
 *
 * The scheduler prevents avoidable streaks while retaining deterministic
 * pseudo-random tie breaking. It uses fixed-capacity counters and allocates no
 * memory per episode.
 */
class SpawnPlacementSampler final
{
public:
    /**
     * @brief Selects the placement with the largest target-count deficit.
     * @param alignedProbability Long-run target share for LOC/glide placements.
     * @param rng Episode generator used only to break equal deficits.
     * @return Left, exact LOC/glide or right placement for the next episode.
     */
    SpawnPlacement Next(double alignedProbability,std::mt19937& rng);
private:
    std::array<std::size_t,3> counts_{};
    std::size_t samples_{};
};

/**
 * @brief Samples an inbound LOC/glide, left-intercept or right-intercept start.
 * @param config Validated episode configuration.
 * @param runway Instantaneous runway pose used as the local frame.
 * @param placement Placement selected by the episode scheduler.
 * @param rng Episode random generator advanced deterministically.
 * @return Finite airborne state expressed in the world tangent frame.
 */
AircraftState SampleAircraft(Config const& config, Runway const& runway,
    SpawnPlacement placement,std::mt19937& rng);
} // namespace ppo::examples::flight
