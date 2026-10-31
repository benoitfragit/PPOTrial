/**
 * @file Examples/BlueDefense/tests/restricted_zone_snapshot_tests.cpp
 * @brief Tests et assertions déterministes pour restricted zone snapshot tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "bluedefenserules.hpp"

#include "components.hpp"
#include "entity.hpp"
#include "environment.hpp"
#include "sharedworldcomponents.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <string_view>
#include <thread>

namespace {

constexpr std::string_view kActiveZoneName{"snapshot_zone_active"};
constexpr std::string_view kDisabledZoneName{"snapshot_zone_disabled"};

/**
 * @brief Result extracted exclusively from the environment's public snapshot.
 */
struct SnapshotResult final
{
    bool activeZonePresent{false};
    bool disabledZonePresent{false};
    bool componentsValid{true};
    unsigned hostileRestrictedZones{0U};
    bool hardBoundaryMatchesWorld{false};
};

/**
 * @brief Adds the components required by simulation snapshots and rendering.
 * @param zone Restricted-zone fixture to configure.
 * @param name Stable scenario identifier exposed through snapshot metadata.
 * @param destroyed True when the zone must be published as disabled.
 * @param north North position in metres.
 */
void ConfigureZone(ppo::model::Entity& zone,
                   std::string_view name,
                   bool destroyed,
                   float north)
{
    zone.add<ppo::model::sEntityMetadata>(ppo::model::sEntityMetadata{
        std::string{name}, "restrictedzone"});
    auto& weapon = zone.add<ppo::model::sSNAState>();
    weapon.m_noescapezone = 5000.F;
    weapon.m_maxNoescapeZone = 5000.F;

    auto [state] = zone.get<ppo::model::sEntityState>();
    state.m_x = north;
    state.m_y = 2500.F;
    state.m_destroyed = destroyed;
    state.m_prevDestroyed = destroyed;
}

/**
 * @brief Reads hostile restricted zones from a renderer-facing ECS snapshot.
 * @param environment Running environment that owns the published snapshot.
 * @return Classification and component-integrity result.
 */
SnapshotResult Capture(ppo::model::Environment const& environment)
{
    SnapshotResult result{};
    environment.ReadSnapshot([&](entt::registry const& snapshot) {
        if (snapshot.ctx().contains<ppo::model::sHardBoundaryOverlay>()
            && snapshot.ctx().contains<ppo::model::sEnvironmentState>())
        {
            auto const& outer = snapshot.ctx().get<
                ppo::model::sHardBoundaryOverlay>();
            auto const& world = snapshot.ctx().get<
                ppo::model::sEnvironmentState>();
            result.hardBoundaryMatchesWorld =
                std::isfinite(outer.m_halfNorth)
                && std::isfinite(outer.m_halfEast)
                && std::fabs(outer.m_halfNorth - world.m_maxNorth) < 1.e-3F
                && std::fabs(outer.m_halfEast - world.m_maxEast) < 1.e-3F;
        }
        snapshot.view<ppo::model::sEntityState,
                      ppo::model::sEntityMetadata>().each(
            [&](entt::entity entity,
                ppo::model::sEntityState const& state,
                ppo::model::sEntityMetadata const& metadata) {
                if (state.m_type != ppo::model::eEntityType::RestrictedZone
                    || state.m_side != ppo::model::eEntitySide::Hostile)
                    return;

                ++result.hostileRestrictedZones;
                auto const* weapon =
                    snapshot.try_get<ppo::model::sSNAState>(entity);
                result.componentsValid &= state.m_entityId == entity
                    && metadata.m_archetype == "restrictedzone"
                    && std::isfinite(state.m_x)
                    && std::isfinite(state.m_y)
                    && weapon != nullptr
                    && std::isfinite(weapon->m_noescapezone)
                    && weapon->m_noescapezone > 0.F;

                if (metadata.m_name == kActiveZoneName)
                    result.activeZonePresent |= !state.m_destroyed;
                else if (metadata.m_name == kDisabledZoneName)
                    result.disabledZonePresent |= state.m_destroyed;
                else
                    result.componentsValid = false;
            });
    });
    return result;
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    using namespace ppo::model;

    Environment environment;
    auto activeZone = environment.CreateEntity<Entity>(
        eEntitySide::Hostile, eEntityType::RestrictedZone);
    auto disabledZone = environment.CreateEntity<Entity>(
        eEntitySide::Hostile, eEntityType::RestrictedZone);
    if (!activeZone || !disabledZone)
        return 1;

    ConfigureZone(*activeZone, kActiveZoneName, false, 12000.F);
    ConfigureZone(*disabledZone, kDisabledZoneName, true, -12000.F);
    if (!environment.SetWorldRules(
            std::make_unique<ppo::examples::bluedefense::BlueDefenseWorldRules>()))
        return 2;

    environment.Play();
    SnapshotResult result{};
    for (unsigned attempt = 0U; attempt < 100U; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        result = Capture(environment);
        if (result.hostileRestrictedZones == 2U)
            break;
    }
    environment.Stop();

    return result.hostileRestrictedZones == 2U
        && result.activeZonePresent
        && result.disabledZonePresent
        && result.componentsValid && result.hardBoundaryMatchesWorld
            ? 0 : 3;
}
