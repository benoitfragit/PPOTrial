#include "initializer.hpp"

#include "components.hpp"
#include "entity.hpp"
#include "sharedworldcomponents.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace {

/**
 * @brief Verifies the real ECS initializer keeps a maximum-size blue team in
 * the patrol rectangle, including its configured boundary margin.
 * @return True when all nine initialized entities satisfy the spawn contract.
 */
bool MaximumBlueTeamFitsWorld()
{
    using namespace ppo::model;
    namespace bluedefense = ppo::examples::bluedefense;
    namespace defenders = bluedefense::defenders;

    constexpr float halfNorth = 22329.375F;
    constexpr float halfEast = 35727.F;
    entt::registry registry;
    auto& environment = registry.ctx().emplace<sEnvironmentState>();
    environment.m_maxNorth = 2.F * halfNorth;
    environment.m_maxEast = 2.F * halfEast;

    auto& context = registry.ctx().emplace<bluedefense::InitializerContext>();
    context.rng.seed(73U);
    context.count = defenders::TeamCapacity::MaximumBlueMobiles;
    context.spawnSettings.blueTeam = bluedefense::BlueTeamSpawnSettings{};
    context.spawnSettings.blueTeam->activeCount =
        defenders::TeamCapacity::MaximumBlueMobiles;
    context.spawnSettings.redTeam = bluedefense::RedTeamSpawnSettings{};
    context.spawnSettings.redTeam->activeCount = 0U;

    std::vector<std::shared_ptr<Entity>> blues;
    blues.reserve(context.count);
    for (std::size_t index = 0U; index < context.count; ++index)
    {
        auto blue = std::make_shared<Entity>(
            &registry, eEntitySide::Friend, eEntityType::Mobile);
        blue->add<sEntityMetadata>(sEntityMetadata{
            "blue_" + std::to_string(index + 1U), "defender_car"});
        blue->add<sPseudoLidarState>();
        blues.push_back(std::move(blue));
    }

    // Insert one nearby friendly after blue_1 has been initialized but
    // before the final initializer refreshes every blue's first LiDAR.
    for (std::size_t index = 0U; index < blues.size(); ++index)
    {
        bluedefense::InitializeEntity(*blues[index]);
        if (index == 0U)
        {
            auto const [state] = blues[0]->get<sEntityState>();
            auto const nearbyId = registry.create();
            auto& nearby = registry.emplace<sEntityState>(nearbyId);
            nearby.m_type = eEntityType::Mobile;
            nearby.m_side = eEntitySide::Friend;
            nearby.m_x = state.m_x + 2500.F * std::cos(state.m_rotation);
            nearby.m_y = state.m_y + 2500.F * std::sin(state.m_rotation);
        }
    }

    auto const [sensor] = blues[0]->get<sPseudoLidarState>();
    if (std::abs(sensor.m_nearestCollisionDistances[30U] - 500.F) > 1.F
        || std::abs(sensor.m_nearestCollisionRanges[30U] - 0.2F) > 0.001F)
    {
        return false;
    }

    if (!context.patrolGeometryReady
        || context.patrolGeometrySamplingFailed
        || context.episodeLayoutGenerationFailed
        || context.next != 0U || !context.blueTeamRoles
        || !context.blueTeamRoles->IsValid())
    {
        return false;
    }

    float const margin = context.patrolGeometrySettings.boundaryMarginMeters;
    defenders::PatrolCurvePose const leader =
        defenders::EvaluatePatrolCurve(
            context.patrolGeometry, context.arclength);
    for (std::size_t index = 0U; index < blues.size(); ++index)
    {
        auto const [state] = blues[index]->get<sEntityState>();
        if (!std::isfinite(state.m_x) || !std::isfinite(state.m_y)
            || std::abs(state.m_x) + margin > halfNorth
            || std::abs(state.m_y) + margin > halfEast)
        {
            return false;
        }

        auto const assignment = context.blueTeamRoles->Assignment(index);
        if (!assignment)
            return false;
        defenders::PatrolCurvePose expected = leader;
        if (assignment->role == defenders::BlueRole::Teammate)
        {
            expected = defenders::EvaluateFormationPose(
                context.patrolGeometry, leader,
                assignment->formationSlot);
        }
        if (std::hypot(state.m_x - expected.north,
                       state.m_y - expected.east) > 0.05F)
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    return MaximumBlueTeamFitsWorld() ? 0 : 1;
}
