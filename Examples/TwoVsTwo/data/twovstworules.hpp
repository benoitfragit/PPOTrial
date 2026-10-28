#pragma once
#include "airspaceworldrules.hpp"

namespace ppo::examples::twovstwo {

/**
 * @brief TwoVsTwo outcome, collision, sensor, and snapshot rules.
 */
class TwoVsTwoWorldRules final : public ppo::model::IWorldRules {
public:
    void BeforeStep(ppo::model::WorldEntities& entities) override;
    void Resolve(ppo::model::WorldEntities& entities) override;
    void AfterStep() override;
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
