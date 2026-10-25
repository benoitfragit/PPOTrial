#include "worldrules.hpp"
#include "airspacecomponents.hpp"
#include "baseentity.hpp"
#include "components.hpp"
#include "restrictedzonelidar.hpp"
#include "sharedworldcomponents.hpp"

namespace ppo::examples::twovstwo {
void TwoVsTwoWorldRules::BeforeStep(ppo::model::WorldEntities& entities)
{
    original_.BeforeStep(entities);
}

void TwoVsTwoWorldRules::Resolve(ppo::model::WorldEntities& entities)
{
    original_.Resolve(entities);
    // RLMission already computes the same sensor for red during observation
    // construction. This also publishes readings for agent-free blue cars.
    for (auto const& entity : entities)
    {
        if (!entity || !entity->has<ppo::model::sEntityState, ppo::model::sPseudoLidarState>())
            continue;
        auto const& [state] = entity->get<ppo::model::sEntityState>();
        if (state.m_type != ppo::model::eEntityType::Mobile || state.m_destroyed)
            continue;
        auto* registry = entity->registry();
        if (!registry || !registry->ctx().contains<ppo::model::sEnvironmentState>())
            continue;
        auto [sensor] = entity->get<ppo::model::sPseudoLidarState>();
        ppo::model::BaseEntity own{registry, entity->uniqid()};
        ppo::model::BuildRestrictedZonePseudoLidar(
            *registry, own, state, sensor.m_nearestCollisionRanges,
            sensor.m_nearestCollisionDistances);
        ppo::model::BuildZoneBoundaryPseudoLidar(
            state, registry->ctx().get<ppo::model::sEnvironmentState>(),
            sensor.m_nearestCollisionRanges, sensor.m_nearestCollisionDistances);
    }
}

void TwoVsTwoWorldRules::AfterStep()
{
    original_.AfterStep();
}

void TwoVsTwoWorldRules::CopySnapshot(entt::registry& source, entt::registry& target) const
{
    original_.CopySnapshot(source, target);
    if (source.ctx().contains<ppo::model::sDefendedArea>())
        target.ctx().emplace<ppo::model::sDefendedArea>(
            source.ctx().get<ppo::model::sDefendedArea>());
}
}
