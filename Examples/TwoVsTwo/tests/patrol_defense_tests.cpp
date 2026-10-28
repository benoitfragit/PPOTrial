#include "patrol_defense.hpp"

#include <array>

namespace blue = ppo::examples::twovstwo::defenders;

int main()
{
    blue::DefenseSettings const settings{};
    if (!blue::Valid(settings))
        return 1;
    blue::DefenseTrack const objective{0.F, 0.F, 0.F, 0.F, true};
    std::array<blue::DefenseTrack, 2> const guards{
        blue::DefenseTrack{-4000.F, 0.F, 0.F, 300.F, true},
        blue::DefenseTrack{4000.F, 0.F, 0.F, 300.F, true}};
    blue::DefenseTrack attacker{40000.F, 0.F, 0.F, 300.F, false};
    if (blue::ThreatRisk(settings, attacker, objective, guards) != 0.F)
        return 2;

    // Living reds remain visible; risk only shapes mission priorities.
    attacker.available = true;
    float const receding = blue::ThreatRisk(
        settings, attacker, objective, guards);
    attacker.heading = 3.14159265358979323846F;
    float const approaching = blue::ThreatRisk(
        settings, attacker, objective, guards);
    if (!(approaching > receding && approaching >= settings.activationRisk))
        return 3;

    attacker.north = 5000.F;
    if (blue::ThreatRisk(settings, attacker, objective, guards) != 1.F)
        return 4;

    // A close pursuer remains dangerous far from the defended objective.
    blue::DefenseTrack const farObjective{100000.F, 0.F, 0.F, 0.F, true};
    std::array<blue::DefenseTrack, 2> const pursued{
        blue::DefenseTrack{1000.F, 0.F, 0.F, 0.F, true},
        blue::DefenseTrack{50000.F, 0.F, 0.F, 0.F, true}};
    attacker = {0.F, 0.F, 0.F, 300.F, true};
    if (blue::ThreatRisk(settings, attacker, farObjective, pursued)
            < settings.activationRisk)
        return 5;
    return 0;
}
