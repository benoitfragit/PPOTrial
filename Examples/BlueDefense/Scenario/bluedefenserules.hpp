/**
 * @file Examples/BlueDefense/Scenario/bluedefenserules.hpp
 * @brief Règles de monde ECS : collisions, zones, destruction et activation des menaces.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once
#include "airspaceworldrules.hpp"

namespace ppo::examples::bluedefense {

/**
 * @brief BlueDefense outcome, collision, sensor, and snapshot rules.
 */
class BlueDefenseWorldRules final : public ppo::model::IWorldRules {
public:
/**
 * @brief Réalise l'opération BeforeStep; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
    void BeforeStep(ppo::model::WorldEntities& entities) override;
/**
 * @brief Traite les entrées et applique les effets associés à Resolve.
 * @details Consulte ou modifie des composants ECS.
 */
    void Resolve(ppo::model::WorldEntities& entities) override;
/**
 * @brief Réalise l'opération AfterStep; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
    void AfterStep() override;
/**
 * @brief Réalise l'opération CopySnapshot; voir les préconditions et effets dans son corps C++.
 */
    void CopySnapshot(entt::registry& source, entt::registry& target) const override;
private:
    /**
     * @brief Activates red entities whose training delay has elapsed.
     * @param entities Scenario entities before agent inference and physics.
     */
    static void ActivateScheduledThreats(
        ppo::model::WorldEntities& entities);

    /**
     * @brief Destroys friendly mobiles whose swept trajectories collide.
     * @param entities Scenario entities after the current physics step.
     */
    static void ResolveFriendlyMobileCollisions(
        ppo::model::WorldEntities& entities);

    /**
     * @brief Applies deterministic swept hostile-zone penetration rules.
     * @param entities Scenario entities after the current physics step.
     */
    static void ResolveRestrictedZonePenetrations(
        ppo::model::WorldEntities& entities);

    /** @brief Legacy rules with probabilistic zone weapons disabled here. */
    ppo::examples::airspace::AirspaceWorldRules original_{false};
};
}
