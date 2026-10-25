#include "spawn.hpp"
#include "defenderbackend.hpp"

#include <cmath>
#include <random>

namespace {
using ppo::examples::twovstwo::WorldBounds;
using ppo::examples::twovstwo::Spawn;

bool Finite(Spawn const& spawn) noexcept
{
    return std::isfinite(spawn.north) && std::isfinite(spawn.east)
        && std::isfinite(spawn.heading) && std::isfinite(spawn.speed);
}

bool InWorld(Spawn const& spawn, WorldBounds const& world) noexcept
{
    return Finite(spawn)
        && std::fabs(spawn.north) < world.halfNorth
        && std::fabs(spawn.east) < world.halfEast;
}
} // namespace

int main()
{
    static_assert(ppo::examples::twovstwo::defenders::kBackendAvailable);
    using ppo::examples::twovstwo::MakeSpawn;

    // No hidden defended rectangle: test multiple world dimensions.
    for (WorldBounds const world : {WorldBounds{22329.375F, 35727.F},
                                    WorldBounds{15000.F, 22000.F},
                                    WorldBounds{30000.F, 45000.F}})
    {
        for (int side : {-1, 1})
        {
            bool sawDifferentRedPosition = false;
            float firstRedNorth = 0.F;
            float firstRedEast = 0.F;

            for (unsigned seed = 0; seed < 128U; ++seed)
            {
                std::mt19937 rng{seed};
                auto const objective = MakeSpawn("blue_objective", side, 0.F, world, rng);
                auto const blue1 = MakeSpawn("blue_guard_1", side, 0.F, world, rng);
                auto const blue2 = MakeSpawn("blue_guard_2", side, 0.F, world, rng);
                auto const zone = MakeSpawn("blue_restricted_zone", side, 0.F, world, rng);
                auto const red1 = MakeSpawn("red_1", side, 0.F, world, rng);
                auto const red2 = MakeSpawn("red_2", side, 0.F, world, rng);

                if (!objective || !blue1 || !blue2 || !zone || !red1 || !red2)
                    return 1;
                if (!InWorld(*objective, world) || !InWorld(*blue1, world)
                    || !InWorld(*blue2, world) || !InWorld(*zone, world)
                    || !Finite(*red1) || !Finite(*red2)
                    || std::fabs(red1->north) >= world.halfNorth
                    || std::fabs(red2->north) >= world.halfNorth)
                    return 2;
                if (objective->east * side <= blue1->east * side
                    || objective->east * side <= blue2->east * side)
                    return 3;
                if (blue1->east * side <= 0.F || blue2->east * side <= 0.F
                    || std::fabs(blue1->east) >= world.halfEast
                    || std::fabs(blue2->east) >= world.halfEast
                    || std::fabs(objective->east) >= world.halfEast)
                    return 4;
                if (blue1->north >= blue2->north || zone->north != 0.F
                    || zone->east != 0.F)
                    return 5;

                // Attackers start 1-3 km outside the defended / LiDAR bounds.
                for (auto const& red : {*red1, *red2})
                {
                    float const standoff = -red.east * side - world.halfEast;
                    if (standoff < 1000.F || standoff > 3000.F
                        || InWorld(red, world))
                        return 6;
                    if (std::sin(red.heading) * side < 0.9F)
                        return 7;
                }
                if (blue1->speed != 0.F || blue2->speed != 0.F
                    || objective->speed != 0.F || red1->speed <= 0.F || red2->speed <= 0.F)
                    return 8;
                if (red1->destroyed || red2->destroyed)
                    return 9;

                if (seed == 0U)
                {
                    firstRedNorth = red1->north;
                    firstRedEast = red1->east;
                }
                else if (red1->north != firstRedNorth || red1->east != firstRedEast)
                    sawDifferentRedPosition = true;

                std::mt19937 forcedRng{seed};
                auto const destroyedRed2 = MakeSpawn("red_2", side, 1.F, world, forcedRng);
                if (!destroyedRed2 || !destroyedRed2->destroyed)
                    return 10;
            }
            if (!sawDifferentRedPosition)
                return 11;
        }
    }
    std::mt19937 rng{42U};
    WorldBounds const world{22329.375F, 35727.F};
    if (MakeSpawn("unknown", 1, 0.F, world, rng)
        || MakeSpawn("red_1", 1, 0.F, WorldBounds{}, rng))
        return 12;
    return 0;
}
