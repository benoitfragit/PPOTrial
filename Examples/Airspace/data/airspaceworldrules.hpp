#pragma once

#include "worldrules.hpp"
#include "components.hpp"

#include <vector>

namespace ppo::examples::airspace {

// All historical world-specific outcome rules and visual events live here.
// The environment invokes these hooks but never interprets the task.
class AirspaceWorldRules final : public ppo::model::IWorldRules
{
public:
    void BeforeStep(ppo::model::WorldEntities& entities) override;
    void Resolve(ppo::model::WorldEntities& entities) override;
    void AfterStep() override;
    void CopySnapshot(entt::registry& world, entt::registry& snapshot) const override;

private:
    void ResolveDestructions(ppo::model::WorldEntities& entities);
    void RecordExplosionEvents(ppo::model::WorldEntities& entities);

    std::vector<ppo::model::sExplosionEvent> m_explosionEvents{};
    unsigned long long m_nextExplosionEventId{1};
};

} // namespace ppo::examples::airspace
