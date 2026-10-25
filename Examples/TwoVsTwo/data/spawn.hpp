#pragma once
#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <string_view>

namespace ppo::examples::twovstwo {

struct WorldBounds
{
    float halfNorth{};
    float halfEast{};
};

struct Spawn
{
    float north{}, east{}, heading{}, speed{};
    bool destroyed{};
};

// Both teams' spawns are relative to the world used by the LiDAR.
inline std::optional<Spawn> MakeSpawn(std::string_view name, int side,
                                     float probability, WorldBounds const& world,
                                     std::mt19937& rng)
{
    if (!std::isfinite(world.halfNorth) || !std::isfinite(world.halfEast)
        || world.halfNorth <= 0.F || world.halfEast <= 0.F
        || !std::isfinite(probability) || probability < 0.F || probability > 1.F)
        return std::nullopt;

    constexpr float halfPi = 1.5707963267948966F;
    auto offset = [&](float radius) {
        return std::uniform_real_distribution<float>{-radius, radius}(rng);
    };
    float const s = side > 0 ? 1.F : -1.F;
    float const guardNorth = 0.18F * world.halfNorth;
    float const redNorth = 0.15F * world.halfNorth;
    float const guardEast = 0.70F * world.halfEast;
    float const guardEastJitter = 0.02F * world.halfEast;

    if (name == "blue_objective")
        return Spawn{0.F, s * (0.92F * world.halfEast), 0.F, 0.F, false};
    if (name == "blue_guard_1")
        return Spawn{-guardNorth + offset(std::min(500.F, 0.025F * world.halfNorth)),
                     s * (guardEast + offset(guardEastJitter)),
                     -s * halfPi, 0.F, false};
    if (name == "blue_guard_2")
        return Spawn{guardNorth + offset(std::min(500.F, 0.025F * world.halfNorth)),
                     s * (guardEast + offset(guardEastJitter)),
                     -s * halfPi, 0.F, false};
    if (name == "blue_restricted_zone")
        return Spawn{0.F, 0.F, 0.F, 0.F, false};

    // Red attackers start just outside the defended/LiDAR rectangle.
    auto attacker = [&](float north, bool destroyed) {
        float const standoff = std::uniform_real_distribution<float>{1000.F, 3000.F}(rng);
        return Spawn{north, -s * (world.halfEast + standoff),
                     s * halfPi + offset(0.12F), 220.F + offset(40.F), destroyed};
    };
    if (name == "red_1")
        return attacker(-redNorth + offset(std::min(1200.F, 0.05F * world.halfNorth)), false);
    if (name == "red_2")
        return attacker(redNorth + offset(std::min(1200.F, 0.05F * world.halfNorth)),
                        std::bernoulli_distribution{static_cast<double>(probability)}(rng));
    return std::nullopt;
}
} // namespace ppo::examples::twovstwo
