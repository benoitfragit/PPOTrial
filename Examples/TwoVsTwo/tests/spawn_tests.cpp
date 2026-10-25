#include "spawn.hpp"
#include "defenderbackend.hpp"

#include <cmath>
#include <random>

namespace {
bool InWorld(ppo::examples::twovstwo::Spawn const& spawn) noexcept
{
    return std::isfinite(spawn.north) && std::isfinite(spawn.east)
        && std::isfinite(spawn.heading) && std::isfinite(spawn.speed)
        && std::fabs(spawn.north) < 22329.F
        && std::fabs(spawn.east) < 35727.F;
}
} // namespace

int main()
{
    static_assert(ppo::examples::twovstwo::defenders::kBackendAvailable);
    using ppo::examples::twovstwo::MakeSpawn;

    for (int side : {-1, 1})
    {
        bool sawDifferentRedPosition = false;
        float firstRedNorth = 0.F;
        float firstRedEast = 0.F;

        for (unsigned seed = 0; seed < 128U; ++seed)
        {
            std::mt19937 rng{seed};
            auto const objective = MakeSpawn("blue_objective", side, 0.F, rng);
            auto const blue1 = MakeSpawn("blue_guard_1", side, 0.F, rng);
            auto const blue2 = MakeSpawn("blue_guard_2", side, 0.F, rng);
            auto const zone = MakeSpawn("blue_restricted_zone", side, 0.F, rng);
            auto const red1 = MakeSpawn("red_1", side, 0.F, rng);
            auto const red2 = MakeSpawn("red_2", side, 0.F, rng);

            if (!objective || !blue1 || !blue2 || !zone || !red1 || !red2)
                return 1;
            if (!InWorld(*objective) || !InWorld(*blue1) || !InWorld(*blue2)
                || !InWorld(*zone) || !InWorld(*red1) || !InWorld(*red2))
                return 2;

            // Both guards must be between the objective and attackers;
            // the objective and blue guards remain inside the defended area.
            if (objective->east * side <= blue1->east * side
                || objective->east * side <= blue2->east * side)
                return 3;
            if (blue1->east * side <= 0.F || blue2->east * side <= 0.F
                || std::fabs(blue1->east) >= 20000.F
                || std::fabs(blue2->east) >= 20000.F
                || std::fabs(objective->east) >= 20000.F)
                return 4;
            if (blue1->north >= blue2->north || zone->north != 0.F
                || zone->east != 0.F)
                return 5;

            // Attackers start outside the defended area, but inside the world,
            // pointing approximately toward the blue side of the map.
            if (red1->east * side >= -20000.F || red2->east * side >= -20000.F)
                return 6;
            if (std::sin(red1->heading) * side < 0.9F
                || std::sin(red2->heading) * side < 0.9F)
                return 7;
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
            auto const destroyedRed2 = MakeSpawn("red_2", side, 1.F, forcedRng);
            if (!destroyedRed2 || !destroyedRed2->destroyed)
                return 10;
        }

        if (!sawDifferentRedPosition)
            return 11;

        std::mt19937 rng{42U};
        if (MakeSpawn("unknown", side, 0.F, rng))
            return 12;
    }
    return 0;
}
