#pragma once
#include <optional>
#include <random>
#include <string_view>

namespace ppo::examples::twovstwo {
struct Spawn {
    float north{}, east{}, heading{}, speed{};
    bool destroyed{};
};

inline std::optional<Spawn> MakeSpawn(std::string_view name, int side,
                                     float probability, std::mt19937& rng)
{
    constexpr float halfPi = 1.5707963267948966F;
    auto offset = [&](float radius) {
        return std::uniform_real_distribution<float>{-radius, radius}(rng);
    };
    float const s = side > 0 ? 1.F : -1.F;
    if (name == "blue_objective")
        return Spawn{0.F, s * 17800.F, 0.F, 0.F, false};
    if (name == "blue_guard_1")
        return Spawn{-4000.F + offset(500.F), s * (13000.F + offset(800.F)),
                     -s * halfPi, 0.F, false};
    if (name == "blue_guard_2")
        return Spawn{4000.F + offset(500.F), s * (13000.F + offset(800.F)),
                     -s * halfPi, 0.F, false};
    if (name == "blue_restricted_zone")
        return Spawn{0.F, 0.F, 0.F, 0.F, false};
    if (name == "red_1")
        return Spawn{-3300.F + offset(1200.F), -s * (26500.F + offset(1500.F)),
                     s * halfPi + offset(0.12F), 220.F + offset(40.F), false};
    if (name == "red_2")
        return Spawn{3300.F + offset(1200.F), -s * (26500.F + offset(1500.F)),
                     s * halfPi + offset(0.12F), 220.F + offset(40.F),
                     std::bernoulli_distribution{static_cast<double>(probability)}(rng)};
    return std::nullopt;
}
} // namespace ppo::examples::twovstwo
