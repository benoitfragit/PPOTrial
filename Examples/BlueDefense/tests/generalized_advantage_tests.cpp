#include "generalized_advantage.hpp"

#include <cmath>

namespace blue = ppo::examples::bluedefense::defenders;
namespace {
bool Near(float const value, float const expected) noexcept
{
    return std::abs(value - expected) < 1.e-5F;
}
}

int main()
{
    // The agent dies on step zero. Mission loss two steps later must still
    // enter that action's advantage through the two dead participant rows.
    constexpr float gamma = 0.9F;
    constexpr float lambda = 0.8F;
    float const terminal = blue::GeneralizedMissionAdvantage(
        -10.F, 0.F, 500.F, 999.F, gamma, lambda, true, false);
    float const dead = blue::GeneralizedMissionAdvantage(
        0.F, 0.F, 0.F, terminal, gamma, lambda, false, false);
    float const deathAction = blue::GeneralizedMissionAdvantage(
        -24.F, 0.F, 0.F, dead, gamma, lambda, false, false);
    if (!Near(terminal, -10.F) || !Near(dead, -7.2F)
        || !Near(deathAction, -29.184F))
        return 1;
    // Truncations bootstrap, but must not include a new episode's trace.
    if (!Near(blue::GeneralizedMissionAdvantage(
            2.F, 3.F, 4.F, 100.F, gamma, lambda, false, true), 2.6F))
        return 2;
    // A real terminal, including terminal+timeout, has no bootstrap or trace.
    if (!Near(blue::GeneralizedMissionAdvantage(
            2.F, 3.F, 4.F, 100.F, gamma, lambda, true, true), -1.F))
        return 3;
    // At a rollout cut the following trace is zero, V(s') is retained.
    if (!Near(blue::GeneralizedMissionAdvantage(
            2.F, 3.F, 4.F, 0.F, gamma, lambda, false, false), 2.6F))
        return 4;
    // A next-episode advantage cannot cross a previous mission terminal.
    if (!Near(blue::GeneralizedMissionAdvantage(
            -1.F, 0.F, 4.F, 100.F, gamma, lambda, true, false), -1.F))
        return 5;
    return 0;
}
