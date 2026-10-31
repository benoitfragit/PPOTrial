/**
 * @file Examples/BlueDefense/tests/patrol_defense_tests.cpp
 * @brief Tests et assertions déterministes pour patrol defense tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "patrol_defense.hpp"
#include "patrol_core.hpp"
#include "patrol_rltools.hpp"

#include <array>
#include <cstddef>

namespace blue = ppo::examples::bluedefense::defenders;

namespace {

/**
 * @brief Verifies the MAPPO outcome produced after a hostile zone destroys a guard.
 * @return True when the failure, terminal flag, and role rewards stay consistent.
 *
 * The ECS bridge maps a destroyed blue entity to an unavailable guard. Zone
 * attribution itself is covered by the BlueDefense collision-rule tests; this
 * test owns the downstream MAPPO contract without duplicating world rules.
 */
bool RestrictedZoneDestructionTerminatesMappo()
{
    blue::Parameters const parameters{};
    blue::State const before = blue::InitialState(parameters, 1, 0U, 0.F);
    blue::State destroyedByRestrictedZone = before;
    destroyedByRestrictedZone.guards[0].available = false;

    std::array<blue::Action, 2> const actions{};
    blue::State const after = blue::AdvanceBoth(
        parameters, destroyedByRestrictedZone, actions);
    if (!after.failed || after.collided)
        return false;

    std::size_t constexpr outcome = static_cast<std::size_t>(
        blue::RewardChannel::MissionOutcome);
    blue::RoleRewardBreakdowns const breakdowns =
        blue::TrainingRoleRewardBreakdowns(
            parameters, before, actions, after);
    if (breakdowns[0].values[outcome] != -100.F
        || breakdowns[1].values[outcome] != -100.F)
        return false;

    blue::RewardValues weights{};
    weights[outcome] = 1.F;
    blue::RoleRewards const rewards = blue::TotalRoleRewards(
        breakdowns, weights, 100.F);
    if (rewards[0] != -100.F || rewards[1] != -100.F)
        return false;

/** @brief Déclare le type TestContext, utilisé par Tests et assertions déterministes pour patrol defense tests. */
    struct TestContext final {};
    TestContext device{};
    TestContext rng{};
    rl_tools::rl::environments::BluePatrol<float, std::size_t> environment{};
    environment.parameters = parameters;
    return !rl_tools::terminated(device, environment, before, rng)
        && rl_tools::terminated(device, environment, after, rng);
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
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
    if (!RestrictedZoneDestructionTerminatesMappo())
        return 6;
    return 0;
}
