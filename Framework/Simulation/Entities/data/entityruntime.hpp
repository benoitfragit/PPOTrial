/**
 * @file Framework/Simulation/Entities/data/entityruntime.hpp
 * @brief Interface ou implémentation du composant entityruntime.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "components.hpp"

// This is the simulation-specific adapter for advancing live ECS entities.
// Keep the policy/execution core independent of entity and registry types.
namespace ppo::model {

template<typename Entities, typename GetEntity>
/**
 * @brief Exécute une étape d'évolution ou de calcul du composant.
 */
void AdvanceLivingEntities(Entities& entities, GetEntity&& getEntity)
{
    for (auto& entry : entities)
    {
        auto& entity = getEntity(entry);
        if (entity == nullptr)
            continue;

        if (entity->template has<sEntityState>())
        {
            auto [state] = entity->template get<sEntityState>();
            if (state.m_destroyed)
                continue;
        }
        entity->Run();
    }
}

} // namespace ppo::model
