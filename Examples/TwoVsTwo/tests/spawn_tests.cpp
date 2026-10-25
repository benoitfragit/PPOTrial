#include "spawn.hpp"
#include "defenderbackend.hpp"
#include <cmath>
#include <random>

int main()
{
    static_assert(ppo::examples::twovstwo::defenders::kBackendAvailable);
    using ppo::examples::twovstwo::MakeSpawn;
    for (int side : {-1, 1})
    {
        std::mt19937 rng{42U};
        auto const objective = MakeSpawn("blue_objective", side, 0.15F, rng);
        auto const blue1 = MakeSpawn("blue_guard_1", side, 0.15F, rng);
        auto const blue2 = MakeSpawn("blue_guard_2", side, 0.15F, rng);
        auto const red1 = MakeSpawn("red_1", side, 0.15F, rng);
        auto const red2 = MakeSpawn("red_2", side, 1.F, rng);
        if (!objective || !blue1 || !blue2 || !red1 || !red2) return 1;
        if (objective->east * side <= blue1->east * side) return 2;
        if (objective->east * side <= blue2->east * side) return 3;
        if (blue1->east * side <= 0.F || blue2->east * side <= 0.F) return 4;
        if (red1->east * side >= -20000.F || red2->east * side >= -20000.F) return 5;
        if (std::fabs(red1->east) >= 35727.F || std::fabs(red2->east) >= 35727.F) return 6;
        if (!red2->destroyed || red1->destroyed) return 7;
        if (blue1->speed != 0.F || blue2->speed != 0.F) return 8;
        if (MakeSpawn("unknown", side, 0.15F, rng)) return 9;
    }
}
