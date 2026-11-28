/**
 * @file Examples/Airspace/Scenario/airspaceworldrules.cpp
 * @brief Construction, données et arbitrage du scénario airspaceworldrules.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "airspaceworldrules.hpp"

#include "entity.hpp"
#include "snasystem.hpp"


namespace ppo::examples::airspace {

using ppo::model::Entity;
using ppo::model::WorldEntities;
using ppo::model::sEntityState;
using ppo::model::sSNAState;
using ppo::model::eEntitySide;
using ppo::model::eEntityType;
using ppo::model::SNASystem;

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
* @brief Nothing to do after a step: the destructions were resolved during it.
*/
void
AirspaceWorldRules::AfterStep(void)
{
}

/**
 * @brief Traite les entrées et applique les effets associés à Resolve.
 * @details Consulte ou modifie des composants ECS.
 */
void AirspaceWorldRules::Resolve(WorldEntities& entities)
{
    ResolveDestructions(entities);
}

} // namespace ppo::examples::airspace
