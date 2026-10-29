#pragma once

#include "worldrules.hpp"
#include "components.hpp"
#include "airspacecomponents.hpp"

#include <vector>

namespace ppo::examples::airspace {

// All historical world-specific outcome rules and visual events live here.
// The environment invokes these hooks but never interprets the task.
class AirspaceWorldRules final : public ppo::model::IWorldRules
{
public:
    /**
     * @brief Creates the legacy rule adapter.
     * @param resolveHostileRestrictedZoneWeapons True to retain probabilistic
     * legacy hostile RestrictedZone weapons; false when a scenario owns their
     * deterministic resolution. Friendly zones remain unchanged.
     */
    explicit AirspaceWorldRules(
        bool resolveHostileRestrictedZoneWeapons = true) noexcept
        : m_resolveHostileRestrictedZoneWeapons(
            resolveHostileRestrictedZoneWeapons)
    {
    }

    void BeforeStep(ppo::model::WorldEntities& entities) override;
    void Resolve(ppo::model::WorldEntities& entities) override;
    void AfterStep() override;
    void CopySnapshot(entt::registry& world, entt::registry& snapshot) const override;

private:
    void ResolveDestructions(ppo::model::WorldEntities& entities);
    void RecordExplosionEvents(ppo::model::WorldEntities& entities);

    std::vector<ppo::model::sExplosionEvent> m_explosionEvents{};
    unsigned long long m_nextExplosionEventId{1};
    bool m_resolveHostileRestrictedZoneWeapons{true};
};

} // namespace ppo::examples::airspace
