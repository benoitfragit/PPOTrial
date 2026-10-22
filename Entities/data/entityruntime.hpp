#pragma once

#include "components.hpp"

// This is the simulation-specific adapter for advancing live ECS entities.
// Keep the policy/execution core independent of entity and registry types.
namespace ppo::model {

template<typename Entities, typename GetEntity>
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
