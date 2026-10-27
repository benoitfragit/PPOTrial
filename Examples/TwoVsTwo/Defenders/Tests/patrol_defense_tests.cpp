#include "patrol_defense.hpp"

#include <array>
#include <cmath>

namespace blue = ppo::examples::twovstwo::defenders;

namespace {

blue::DefenseTrack Track(float north, float east, float heading,
                         float speed = 300.F, bool available = true)
{
    return {north, east, heading, speed, available};
}

} // namespace

int main()
{
    blue::DefenseSettings settings{};
    settings.deactivationHoldSeconds = 0.6F;
    settings.minimumAssignmentSeconds = 1.F;
    settings.recoverySeconds = 1.F;
    if (!blue::Valid(settings))
        return 1;

    constexpr float dt = 0.2F;
    blue::DefenseTrack const objective = Track(0.F, 0.F, 0.F, 0.F);
    std::array<blue::DefenseTrack, 2> guards{
        Track(-4000.F, 0.F, 0.F), Track(4000.F, 0.F, 0.F)};
    std::array<blue::DefenseTrack, 2> attackers{
        Track(0.F, 0.F, 0.F, 0.F, false),
        Track(0.F, 0.F, 0.F, 0.F, false)};

    blue::DefenseCoordinator coordinator;
    auto decision = coordinator.Update(
        settings, dt, guards, attackers, objective);
    if (decision.credibleThreats != 0U
        || decision.attackerForGuard[0] != blue::DefenseDecision::Unassigned
        || decision.attackerForGuard[1] != blue::DefenseDecision::Unassigned)
        return 2;

    // A distant hostile moving away is visible but not credible.
    attackers[0] = Track(40000.F, 0.F, 0.F);
    decision = coordinator.Update(settings, dt, guards, attackers, objective);
    if (decision.threatRisk[0] >= settings.activationRisk
        || decision.credibleThreats != 0U)
        return 3;

    // The defensive profile now commits while a closing hostile is still
    // outside the former 25 km activation envelope.
    coordinator.Reset();
    attackers[0] = Track(30000.F, 0.F, 3.14159265358979323846F);
    decision = coordinator.Update(settings, dt, guards, attackers, objective);
    if (decision.threatRisk[0] < settings.activationRisk
        || decision.credibleThreats != 1U)
        return 13;

    // A hostile closing on the defended objective activates one interceptor.
    coordinator.Reset();
    attackers[0] = Track(20000.F, 0.F, 3.14159265358979323846F);
    decision = coordinator.Update(settings, dt, guards, attackers, objective);
    if (decision.threatRisk[0] < settings.activationRisk
        || decision.credibleThreats != 1U
        || (decision.attackerForGuard[0] != 0U
            && decision.attackerForGuard[1] != 0U))
        return 4;

    std::size_t const initialGuard = decision.attackerForGuard[0] == 0U
        ? 0U : 1U;
    // Small geometric reversals cannot churn an assignment during its hold.
    guards[0] = Track(10000.F, 0.F, 0.F);
    guards[1] = Track(19000.F, 0.F, 0.F);
    for (unsigned step = 0U; step < 3U; ++step)
    {
        decision = coordinator.Update(
            settings, dt, guards, attackers, objective);
        if (decision.attackerForGuard[initialGuard] != 0U)
            return 5;
    }

    // A risk between the two thresholds remains latched.
    blue::DefenseTrack farObjective = Track(100000.F, 0.F, 0.F, 0.F);
    guards[0] = Track(6000.F, 0.F, 0.F, 0.F);
    guards[1] = Track(50000.F, 0.F, 0.F, 0.F);
    attackers[0] = Track(0.F, 0.F, 0.F);
    decision = coordinator.Update(
        settings, dt, guards, attackers, farObjective);
    if (!(decision.threatRisk[0] > settings.deactivationRisk
          && decision.threatRisk[0] < settings.activationRisk)
        || decision.credibleThreats != 1U)
        return 6;

    // Once below the lower threshold, release occurs only after the hold.
    attackers[0] = Track(-50000.F, 0.F, 3.14159265358979323846F);
    for (unsigned step = 0U; step < 2U; ++step)
    {
        decision = coordinator.Update(
            settings, dt, guards, attackers, farObjective);
        if (decision.credibleThreats != 1U)
            return 7;
    }
    decision = coordinator.Update(
        settings, dt, guards, attackers, farObjective);
    if (decision.credibleThreats != 0U)
        return 8;

    // Two credible tracks always receive distinct one-to-one assignments.
    coordinator.Reset();
    guards = {Track(-5000.F, 0.F, 0.F), Track(5000.F, 0.F, 0.F)};
    attackers = {
        Track(-15000.F, 0.F, 0.F),
        Track(15000.F, 0.F, 3.14159265358979323846F)};
    decision = coordinator.Update(settings, dt, guards, attackers, objective);
    if (decision.credibleThreats != 2U
        || decision.attackerForGuard[0] >= attackers.size()
        || decision.attackerForGuard[1] >= attackers.size()
        || decision.attackerForGuard[0] == decision.attackerForGuard[1])
        return 9;

    // A close pursuer is credible even when it is far from the objective.
    coordinator.Reset();
    guards = {Track(1000.F, 0.F, 0.F, 0.F),
              Track(50000.F, 0.F, 0.F, 0.F)};
    attackers = {Track(0.F, 0.F, 0.F),
                 Track(0.F, 0.F, 0.F, 0.F, false)};
    decision = coordinator.Update(
        settings, dt, guards, attackers, farObjective);
    if (decision.threatRisk[0] < settings.activationRisk
        || decision.credibleThreats != 1U)
        return 10;

    // On disappearance, the tactical role fades instead of snapping to patrol.
    attackers[0].available = false;
    decision = coordinator.Update(
        settings, dt, guards, attackers, farObjective);
    float const recoveryWeight = std::max(
        decision.defenseWeight[0], decision.defenseWeight[1]);
    if (!(recoveryWeight > 0.F && recoveryWeight < 1.F)
        || decision.credibleThreats != 0U)
        return 11;
    for (unsigned step = 0U; step < 4U; ++step)
        decision = coordinator.Update(
            settings, dt, guards, attackers, farObjective);
    if (decision.defenseWeight[0] != 0.F
        || decision.defenseWeight[1] != 0.F)
        return 12;

    return 0;
}
