#include "initializer.hpp"
#include "spawn.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"
#include "patrol_core.hpp"
#include <iostream>
#include <random>

namespace ppo::examples::twovstwo {

namespace {

/**
 * @brief Samples the leader patrol solely for snapshot visualization.
 * @param config Episode context containing a valid immutable geometry.
 */
void BuildPatrolTrackOverlay(InitializerContext& config) noexcept
{
    auto& overlay = config.patrolTrackOverlay;
    overlay = ppo::model::sPatrolTrackOverlay{};
    overlay.m_sampleCount = overlay.SampleCapacity;
    float const length = defenders::CircuitLength(config.patrolGeometry);
    for (std::size_t sample = 0U; sample < overlay.m_sampleCount; ++sample)
    {
        float const arclength = length * static_cast<float>(sample)
            / static_cast<float>(overlay.m_sampleCount - 1U);
        auto const pose = defenders::EvaluatePatrolCurve(
            config.patrolGeometry, arclength);
        overlay.m_north[sample] = pose.north;
        overlay.m_east[sample] = pose.east;
    }
}

} // namespace

void InitializeEntity(ppo::model::Entity& entity)
{
    auto* registry = entity.registry();
    if (!registry || !registry->ctx().contains<InitializerContext>()
        || !entity.has<ppo::model::sEntityState, ppo::model::sEntityMetadata>())
        return;

    auto& config = registry->ctx().get<InitializerContext>();
    if (config.count == 0 || config.patrolGeometrySamplingFailed)
        return;
    if (config.next == 0)
    {
        config.elapsedSteps = 0U;
        auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
        auto const bounds = WorldBounds{
            0.5F * world.m_maxNorth, 0.5F * world.m_maxEast};
        defenders::Parameters p{};
        p.halfNorth = bounds.halfNorth;
        p.halfEast = bounds.halfEast;
        p.cruiseSpeed = config.patrolCruiseSpeed;
        p.maxRotationRate = config.patrolMaximumRotationRate;
        p.collisionDistance = config.patrolCollisionDistance;
        defenders::PatrolSamplingConstraints const constraints{
            p.halfNorth, p.halfEast, p.cruiseSpeed, p.maxRotationRate,
            p.collisionDistance};
        auto const geometry = defenders::SamplePatrolGeometry(
            config.patrolGeometrySettings, constraints, config.rng);
        if (!geometry)
        {
            config.patrolGeometryReady = false;
            config.patrolGeometrySamplingFailed = true;
            config.patrolTrackOverlay = ppo::model::sPatrolTrackOverlay{};
            config.episodeLayout.reset();
            std::cerr << "[BLUE PATROL] Cannot sample a feasible racetrack after "
                      << config.patrolGeometrySettings.maximumSamplingAttempts
                      << " attempts; check Training/BluePatrol.json.\n";
            return;
        }
        config.patrolGeometry = *geometry;
        config.patrolGeometryReady = true;
        BuildPatrolTrackOverlay(config);
        p.geometry = *geometry;
        config.leaderGuardIndex =
            std::bernoulli_distribution{0.5}(config.rng) ? 0U : 1U;
        ++config.episodeIndex;
        std::clog << "[BLUE PATROL] Episode " << config.episodeIndex
                  << " racetrack length=" << geometry->length
                  << "m width=" << geometry->width
                  << "m heading=" << geometry->heading
                  << "rad anchor=(" << geometry->anchorNorth << ','
                  << geometry->anchorEast << ")m wedge="
                  << geometry->separationDistance << "m/"
                  << geometry->separationAngle << "rad direction="
                  << (geometry->direction > 0 ? "counter-clockwise" : "clockwise")
                  << " leader=blue_guard_" << config.leaderGuardIndex + 1U
                  << '\n';
        if (config.evaluationOverride)
        {
            config.side = config.evaluationSide < 0 ? -1 : 1;
            config.arclength = defenders::PositiveModulo(
                config.evaluationArclength, defenders::CircuitLength(p));
        }
        else
        {
            config.side = std::bernoulli_distribution{0.5}(config.rng) ? 1 : -1;
            config.arclength = std::uniform_real_distribution<float>{
                0.F, defenders::CircuitLength(p)}(config.rng);
        }
        auto spawnSettings = config.spawnSettings;
        if (config.evaluationOverride)
        {
            // Evaluations retain their deterministic immediate 2v1/2v2 cases;
            // stochastic activation schedules belong to training only.
            spawnSettings.threatCurriculum.enabled = false;
            spawnSettings.red2DestroyedProbability =
                config.evaluationSecondAttackerDestroyed ? 1.F : 0.F;
        }
        config.episodeLayout = GenerateEpisodeSpawnLayout(
            config.side, spawnSettings, bounds, config.rng);
    }

    if (!config.episodeLayout)
        return;

    auto const& [metadata] = entity.get<ppo::model::sEntityMetadata>();
    auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
    auto const bounds = WorldBounds{0.5F * world.m_maxNorth, 0.5F * world.m_maxEast};
    auto pose = MakeSpawn(metadata.m_name, *config.episodeLayout,
                          bounds, config.rng);
    if (pose
        && (metadata.m_name == "blue_guard_1" || metadata.m_name == "blue_guard_2"))
    {
        defenders::Parameters p{};
        p.halfNorth = bounds.halfNorth;
        p.halfEast = bounds.halfEast;
        p.cruiseSpeed = config.patrolCruiseSpeed;
        p.maxRotationRate = config.patrolMaximumRotationRate;
        p.collisionDistance = config.patrolCollisionDistance;
        p.geometry = config.patrolGeometry;
        std::size_t const guardIndex =
            metadata.m_name == "blue_guard_1" ? 0U : 1U;
        std::size_t const role = config.RoleForGuard(guardIndex);
        auto const leader = defenders::PatrolReference(
            p, config.side, config.arclength);
        auto const reference = role == 0U
            ? leader : defenders::WingmanReference(p, leader);
        auto const mobile = defenders::OnReference(p, reference);
        pose->north = mobile.north;
        pose->east = mobile.east;
        pose->heading = mobile.heading;
        pose->speed = mobile.speed;
    }
    if (pose)
    {
        auto [state] = entity.get<ppo::model::sEntityState>();
        state.m_x = pose->north;
        state.m_y = pose->east;
        state.m_rotation = pose->heading;
        state.m_speed = pose->speed;
        state.m_destroyed = pose->destroyed;
        state.m_prevDestroyed = pose->destroyed;
        if (entity.has<ppo::model::sFlightModelState>())
        {
            auto [flight] = entity.get<ppo::model::sFlightModelState>();
            flight.m_acc = 0.F;
            flight.m_accpsi = 0.F;
        }
        if (entity.has<ppo::model::sSNAControlState>())
        {
            auto [control] = entity.get<ppo::model::sSNAControlState>();
            control.m_accCmd = 0.F;
            control.m_rotSpeedCmd = 0.F;
        }
    }
    config.next = (config.next + 1U) % config.count;
}
} // namespace ppo::examples::twovstwo
