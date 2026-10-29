#include "airspaceworldrules.hpp"

#include "entity.hpp"
#include "snasystem.hpp"

#include <algorithm>
#include <chrono>

namespace ppo::examples::airspace {

using ppo::model::Entity;
using ppo::model::WorldEntities;
using ppo::model::sEntityState;
using ppo::model::sSNAState;
using ppo::model::sExplosionEvent;
using ppo::model::sExplosionEventBuffer;
using ppo::model::eEntitySide;
using ppo::model::eEntityType;
using ppo::model::SNASystem;

namespace {
/**
* @brief Lifetime of one explosion event in steady-clock seconds.
*/
constexpr double kExplosionDurationSeconds = 2.0;

/**
* @brief Returns the current steady-clock time in seconds.
* @return Current steady-clock time in seconds.
*/
double
GetSteadyTimeSeconds(void)
{
    using Clock = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<Seconds>(Clock::now().time_since_epoch()).count();
}

/**
* @brief Computes the base explosion radius associated with one entity type.
* @param type Destroyed entity type.
* @return Base explosion radius.
*/
float
ExplosionRadiusForType(eEntityType type) noexcept
{
    switch (type)
    {
    case eEntityType::Objective:
        return 2.40F;
    case eEntityType::RestrictedZone:
        return 1.35F;
    case eEntityType::Mobile:
        return 1.10F;
    default:
        return 1.00F;
    }
}

/**
* @brief Computes the explosion plume height scaling for one entity type.
* @param type Destroyed entity type.
* @return Height scaling factor.
*/
float
ExplosionHeightScaleForType(eEntityType type) noexcept
{
    switch (type)
    {
    case eEntityType::Objective:
        return 2.35F;
    case eEntityType::RestrictedZone:
        return 1.45F;
    case eEntityType::Mobile:
        return 1.20F;
    default:
        return 1.00F;
    }
}


} // namespace

/**
* @brief Copies the current destroyed state into the previous-step history.
*
* This is executed once at the start of every simulation step so that any
* destruction detected later during the step keeps a clean transition from
* "alive" to "destroyed".
*/
void
AirspaceWorldRules::BeforeStep(WorldEntities& m_entities)
{
    for (auto& entity : m_entities)
    {
        if (entity == nullptr
            || !entity->has<sEntityState>())
        {
            continue;
        }

        auto [entityState] = entity->get<sEntityState>();
        entityState.m_prevDestroyed = entityState.m_destroyed;
    }
}

/**
* @brief Resolves all destruction events after the world has advanced.
*
* The original entity iteration order and threshold are preserved.
*/
void
AirspaceWorldRules::ResolveDestructions(WorldEntities& m_entities)
{
    // Resolve destruction outcomes only after every entity has advanced.
    for (auto& entity: m_entities)
    {
        if (entity == nullptr
        ||  !entity->has<sEntityState, sSNAState>())
            continue;
    
        auto const [entityState, snaState] = entity->get<sEntityState, sSNAState>();
    
        if (entityState.m_destroyed
            || (!m_resolveHostileRestrictedZoneWeapons
                && entityState.m_type == eEntityType::RestrictedZone
                && entityState.m_side == eEntitySide::Hostile))
            continue;
    
        for (auto& otherEntity : m_entities)
        {
            if (otherEntity == entity
            || !otherEntity->has<sEntityState>())
                continue;
    
            auto [otherEntityState] = otherEntity->get<sEntityState>();
    
            if (otherEntityState.m_destroyed) 
                continue;
    
            if (float pk{ SNASystem::EvaluateDestruction(*entity.get(), *otherEntity.get()) };
                0.95F < pk)
            {
                otherEntityState.m_destroyed = true;
                otherEntityState.m_destroyedById = entity->uniqid();
            } 
        }
    }
}

/**
* @brief Records one explosion event for every fresh destruction.
*
* The resulting event buffer is independent from entity lifetime so that the
* renderer can keep displaying the explosion even when the environment is
* reinitialized immediately afterwards.
*/
void
AirspaceWorldRules::RecordExplosionEvents(WorldEntities& m_entities)
{
    double const now = GetSteadyTimeSeconds();

    for (auto& entity : m_entities)
    {
        if (entity == nullptr
            || !entity->has<sEntityState>())
        {
            continue;
        }

        auto [entityState] = entity->get<sEntityState>();
        if (!entityState.m_destroyed || entityState.m_prevDestroyed)
            continue;

        m_explosionEvents.push_back(sExplosionEvent{
            m_nextExplosionEventId++,
            entityState.m_x,
            entityState.m_y,
            0.F,
            entityState.m_type,
            now,
            ExplosionRadiusForType(entityState.m_type),
            ExplosionHeightScaleForType(entityState.m_type)
        });
    }
}

/**
* @brief Removes expired explosion events from the persistent event buffer.
*/
void
AirspaceWorldRules::AfterStep(void)
{
    double const now = GetSteadyTimeSeconds();
    auto const keepFrom = std::remove_if(
        m_explosionEvents.begin(),
        m_explosionEvents.end(),
        [now](sExplosionEvent const& event)
        {
            return (now - event.m_startTime) > kExplosionDurationSeconds;
        });
    m_explosionEvents.erase(keepFrom, m_explosionEvents.end());
}


void AirspaceWorldRules::Resolve(WorldEntities& entities)
{
    ResolveDestructions(entities);
    RecordExplosionEvents(entities);
}

void AirspaceWorldRules::CopySnapshot(entt::registry& source, entt::registry& destination) const
{
    source.view<ppo::model::sSNAState>().each([&](auto entity, auto const& component)
    {
        destination.emplace<ppo::model::sSNAState>(entity, component);
    });

    source.view<ppo::model::sFlightModelState>().each([&](auto entity, auto const& component)
    {
        destination.emplace<ppo::model::sFlightModelState>(entity, component);
    });

    source.view<ppo::model::sPseudoLidarState>().each([&](auto entity, auto const& component)
    {
        destination.emplace<ppo::model::sPseudoLidarState>(entity, component);
    });

    destination.ctx().emplace<sExplosionEventBuffer>(sExplosionEventBuffer{m_explosionEvents});
}

} // namespace ppo::examples::airspace
