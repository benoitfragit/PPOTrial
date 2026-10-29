#include "twovstworules.hpp"
#include "initializer.hpp"
#include "airspacecomponents.hpp"
#include "baseentity.hpp"
#include "entity.hpp"
#include "components.hpp"
#include "restrictedzonelidar.hpp"
#include "restrictedzonegeometry.hpp"
#include "sharedworldcomponents.hpp"
#include "patrol_core.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <optional>
#include <string_view>

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

/**
 * @brief Decodes the stable one-based `red_<index>` runtime metadata.
 * @param name Entity metadata name.
 * @return Zero-based red-team index, or no value for malformed metadata.
 */
std::optional<std::size_t> RedEntityIndex(std::string_view name) noexcept
{
    constexpr std::string_view prefix{"red_"};
    if (name.size() <= prefix.size()
        || name.substr(0U, prefix.size()) != prefix)
        return std::nullopt;

    std::string_view const suffix = name.substr(prefix.size());
    if (suffix.front() == '0')
        return std::nullopt;
    std::size_t oneBasedIndex = 0U;
    auto const result = std::from_chars(
        suffix.data(), suffix.data() + suffix.size(), oneBasedIndex);
    if (result.ec != std::errc{}
        || result.ptr != suffix.data() + suffix.size()
        || oneBasedIndex == 0U)
        return std::nullopt;
    return oneBasedIndex - 1U;
}

} // namespace

namespace ppo::examples::twovstwo {
void TwoVsTwoWorldRules::BeforeStep(ppo::model::WorldEntities& entities)
{
    ActivateScheduledThreats(entities);
    original_.BeforeStep(entities);
    for (auto const& entity : entities)
        if (entity && entity->registry() != nullptr)
        {
            ppo::model::UpdateRestrictedZonePseudoLidars(
                *entity->registry(), ppo::model::eEntitySide::Friend,
                defenders::Parameters{}.cautionDistance);
            break;
        }
}

void TwoVsTwoWorldRules::Resolve(ppo::model::WorldEntities& entities)
{
    // Resolve physical impacts before the legacy SNA rules record destruction
    // events. This makes both wrecks and explosions visible in the same ECS
    // snapshot consumed by the MAPPO feedback.
    ResolveFriendlyMobileCollisions(entities);
    ResolveRestrictedZonePenetrations(entities);
    original_.Resolve(entities);
    for (auto const& entity : entities)
        if (entity && entity->registry() != nullptr)
        {
            ppo::model::UpdateRestrictedZonePseudoLidars(
                *entity->registry(), ppo::model::eEntitySide::Friend,
                defenders::Parameters{}.cautionDistance);
            break;
        }

    if (!entities.empty() && entities.front()
        && entities.front()->registry() != nullptr)
    {
        auto* registry = entities.front()->registry();
        if (registry->ctx().contains<InitializerContext>())
            ++registry->ctx().get<InitializerContext>().elapsedSteps;
    }
}

void TwoVsTwoWorldRules::ResolveRestrictedZonePenetrations(
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

    for (auto& mobile : entities)
    {
        if (!mobile || !mobile->has<ppo::model::sEntityState>())
            continue;
        auto [mobileState] = mobile->get<ppo::model::sEntityState>();
        if (mobileState.m_destroyed
            || mobileState.m_type != ppo::model::eEntityType::Mobile
            || mobileState.m_side != ppo::model::eEntitySide::Friend)
            continue;
        ppo::model::PlanarPosition const current{
            mobileState.m_x, mobileState.m_y};
        ppo::model::PlanarPosition const previous{
            mobileState.m_x - mobileState.m_dx * timeStep,
            mobileState.m_y - mobileState.m_dy * timeStep};
        for (auto const& zone : entities)
        {
            if (!zone || !zone->has<ppo::model::sEntityState,
                                    ppo::model::sSNAState>())
                continue;
            auto const [zoneState, zoneSna] =
                zone->get<ppo::model::sEntityState, ppo::model::sSNAState>();
            if (zoneState.m_destroyed
                || zoneState.m_type
                    != ppo::model::eEntityType::RestrictedZone
                || zoneState.m_side == mobileState.m_side)
                continue;
            if (!ppo::model::CrossesRestrictedZone(
                    previous, current,
                    ppo::model::PlanarPosition{
                        zoneState.m_x, zoneState.m_y},
                    zoneSna.m_noescapezone))
                continue;
            mobileState.m_destroyed = true;
            mobileState.m_destroyedById = zone->uniqid();
            break;
        }
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
        auto [state] = entity->get<ppo::model::sEntityState>();
        if (state.m_type != ppo::model::eEntityType::Mobile
            || state.m_side != ppo::model::eEntitySide::Hostile)
            continue;
        auto const red = RedEntityIndex(metadata.m_name);
        if (!red
            || *red >= context.episodeLayout->redActivationSteps.size()
            || context.episodeLayout->redActivationSteps[*red]
                != context.elapsedSteps)
            continue;

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
    if (source.ctx().contains<InitializerContext>())
    {
        auto const& context = source.ctx().get<InitializerContext>();
        if (context.patrolGeometryReady
            && context.patrolTrackOverlay.m_sampleCount >= 2U)
            target.ctx().emplace<ppo::model::sPatrolTrackOverlay>(
                context.patrolTrackOverlay);
    }
}
}
