#include "twovstworules.hpp"
#include "initializer.hpp"
#include "airspacecomponents.hpp"
#include "baseentity.hpp"
#include "entity.hpp"
#include "components.hpp"
#include "restrictedzonelidar.hpp"
#include "sharedworldcomponents.hpp"
#include "patrol_core.hpp"

#include <algorithm>
#include <cmath>

namespace {

/**
 * @brief Computes the closest relative distance during the latest ECS step.
 * @param first First mobile after integration.
 * @param second Second mobile after integration.
 * @param timeStep Duration of the integration step in seconds.
 * @return Minimum centre-to-centre distance along both swept segments.
 */
float MinimumStepSeparation(ppo::model::sEntityState const& first,
                            ppo::model::sEntityState const& second,
                            float timeStep) noexcept
{
    float const currentNorth = first.m_x - second.m_x;
    float const currentEast = first.m_y - second.m_y;
    float const previousNorth = currentNorth
        - (first.m_dx - second.m_dx) * timeStep;
    float const previousEast = currentEast
        - (first.m_dy - second.m_dy) * timeStep;
    float const deltaNorth = currentNorth - previousNorth;
    float const deltaEast = currentEast - previousEast;
    float const travelledSquared = deltaNorth * deltaNorth
                                 + deltaEast * deltaEast;
    float const closestTime = travelledSquared > 1.e-8F
        ? std::clamp(-(previousNorth * deltaNorth
                     + previousEast * deltaEast) / travelledSquared, 0.F, 1.F)
        : 0.F;
    return std::hypot(previousNorth + closestTime * deltaNorth,
                      previousEast + closestTime * deltaEast);
}

} // namespace

namespace ppo::examples::twovstwo {
void TwoVsTwoWorldRules::BeforeStep(ppo::model::WorldEntities& entities)
{
    ActivateScheduledThreats(entities);
    original_.BeforeStep(entities);
}

void TwoVsTwoWorldRules::Resolve(ppo::model::WorldEntities& entities)
{
    // Resolve physical impacts before the legacy SNA rules record destruction
    // events. This makes both wrecks and explosions visible in the same ECS
    // snapshot consumed by the MAPPO feedback.
    ResolveFriendlyMobileCollisions(entities);
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

    if (!entities.empty() && entities.front()
        && entities.front()->registry() != nullptr)
    {
        auto* registry = entities.front()->registry();
        if (registry->ctx().contains<InitializerContext>())
            ++registry->ctx().get<InitializerContext>().elapsedSteps;
    }
}

void TwoVsTwoWorldRules::ActivateScheduledThreats(
    ppo::model::WorldEntities& entities)
{
    entt::registry* registry = nullptr;
    for (auto const& entity : entities)
        if (entity && entity->registry() != nullptr)
        {
            registry = entity->registry();
            break;
        }
    if (registry == nullptr || !registry->ctx().contains<InitializerContext>())
        return;

    auto const& context = registry->ctx().get<InitializerContext>();
    if (!context.episodeLayout)
        return;

    for (auto const& entity : entities)
    {
        if (!entity || !entity->has<ppo::model::sEntityState,
                                    ppo::model::sEntityMetadata>())
            continue;
        auto const& [metadata] = entity->get<ppo::model::sEntityMetadata>();
        std::size_t red = metadata.m_name == "red_1" ? 0U
            : metadata.m_name == "red_2" ? 1U
            : context.episodeLayout->redActivationSteps.size();
        if (red >= context.episodeLayout->redActivationSteps.size()
            || context.episodeLayout->redActivationSteps[red]
                != context.elapsedSteps)
            continue;

        auto [state] = entity->get<ppo::model::sEntityState>();
        state.m_destroyed = false;
        state.m_prevDestroyed = false;
        state.m_destroyedById = entt::null;
    }
}

void TwoVsTwoWorldRules::ResolveFriendlyMobileCollisions(
    ppo::model::WorldEntities& entities)
{
    entt::registry* registry = nullptr;
    for (auto const& entity : entities)
        if (entity && entity->registry() != nullptr)
        {
            registry = entity->registry();
            break;
        }
    if (registry == nullptr
        || !registry->ctx().contains<ppo::model::sEnvironmentState>())
        return;
    float const timeStep = registry->ctx().get<
        ppo::model::sEnvironmentState>().m_dt;
    if (!std::isfinite(timeStep) || timeStep <= 0.F)
        return;

    float const collisionDistance =
        defenders::Parameters{}.collisionDistance;
    for (std::size_t i = 0; i < entities.size(); ++i)
    {
        auto& first = entities[i];
        if (!first || !first->has<ppo::model::sEntityState>())
            continue;
        auto [firstState] = first->get<ppo::model::sEntityState>();
        if (firstState.m_destroyed
            || firstState.m_type != ppo::model::eEntityType::Mobile
            || firstState.m_side != ppo::model::eEntitySide::Friend)
            continue;

        for (std::size_t j = i + 1U; j < entities.size(); ++j)
        {
            auto& second = entities[j];
            if (!second || !second->has<ppo::model::sEntityState>())
                continue;
            auto [secondState] = second->get<ppo::model::sEntityState>();
            if (secondState.m_destroyed
                || secondState.m_type != ppo::model::eEntityType::Mobile
                || secondState.m_side != ppo::model::eEntitySide::Friend)
                continue;
            if (MinimumStepSeparation(firstState, secondState, timeStep)
                > collisionDistance)
                continue;

            firstState.m_destroyed = true;
            firstState.m_destroyedById = second->uniqid();
            secondState.m_destroyed = true;
            secondState.m_destroyedById = first->uniqid();
            break;
        }
    }
}

void TwoVsTwoWorldRules::AfterStep()
{
    original_.AfterStep();
}

void TwoVsTwoWorldRules::CopySnapshot(entt::registry& source, entt::registry& target) const
{
    original_.CopySnapshot(source, target);
    // The defended area and the shared LiDAR use exactly the same world bounds.
    if (target.ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& world = target.ctx().get<ppo::model::sEnvironmentState>();
        target.ctx().emplace<ppo::model::sDefendedArea>(
            ppo::model::sDefendedArea{0.F, 0.F,
                                      0.5F * world.m_maxNorth,
                                      0.5F * world.m_maxEast});
    }
}
}
