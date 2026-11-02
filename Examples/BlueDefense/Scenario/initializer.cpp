/**
 * @file Examples/BlueDefense/Scenario/initializer.cpp
 * @brief Initialisation des épisodes, objectifs, rôles et générateur de dispositions.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "initializer.hpp"
#include "spawn.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"
#include "restrictedzonelidar.hpp"
#include "patrol_core.hpp"
#include "blue_team_roles.hpp"
#include <charconv>
#include <iostream>
#include <random>
#include <string_view>

namespace ppo::examples::bluedefense {

namespace {

/**
 * @brief Parses one canonical generalized blue mobile name.
 * @param name Candidate metadata name.
 * @return Zero-based physical index for blue_1..blue_9, otherwise null.
 */
std::optional<std::size_t> GeneralizedBlueIndex(
    std::string_view const name) noexcept
{
    constexpr std::string_view prefix{"blue_"};
    if (name.size() <= prefix.size()
        || name.substr(0U, prefix.size()) != prefix)
    {
        return std::nullopt;
    }
    std::size_t oneBased = 0U;
    char const* first = name.data() + prefix.size();
    char const* last = name.data() + name.size();
    auto const parsed = std::from_chars(first, last, oneBased);
    if (parsed.ec != std::errc{} || parsed.ptr != last || oneBased == 0U
        || oneBased > defenders::TeamCapacity::MaximumBlueMobiles
        || (last - first > 1 && *first == '0'))
    {
        return std::nullopt;
    }
    return oneBased - 1U;
}

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

/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 * @details Consulte ou modifie des composants ECS.
 */
void InitializeEntity(ppo::model::Entity& entity)
{
    auto* registry = entity.registry();
    if (!registry || !registry->ctx().contains<InitializerContext>()
        || !entity.has<ppo::model::sEntityState, ppo::model::sEntityMetadata>())
        return;

    auto& config = registry->ctx().get<InitializerContext>();
    if (config.count == 0 || config.patrolGeometrySamplingFailed
        || config.episodeLayoutGenerationFailed)
        return;
    if (config.next == 0)
    {
        config.elapsedSteps = 0U;
        config.patrolReferenceOverlay =
            ppo::model::sPatrolReferenceOverlay{};
        config.patrolWaypointOverlay.reset();
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
            p.collisionDistance, config.patrolMinimumSeparationMeters,
            config.patrolMaximumSpeed};
        if (!config.spawnSettings.blueTeam)
        {
            config.patrolGeometrySamplingFailed = true;
            std::cerr << "[BLUE PATROL] Missing active Blue team configuration.\n";
            return;
        }
        std::size_t const activeBlueCount =
            config.spawnSettings.blueTeam->activeCount;
        std::optional<float> const evaluationPhase = config.evaluationOverride
            ? std::optional<float>{config.evaluationPhaseFraction}
            : std::nullopt;
        auto const start = defenders::SampleGeneralizedPatrolStart(
            config.patrolGeometrySettings, constraints, activeBlueCount,
            config.rng, evaluationPhase);
        if (!start)
        {
            config.patrolGeometryReady = false;
            config.patrolGeometrySamplingFailed = true;
            config.patrolTrackOverlay = ppo::model::sPatrolTrackOverlay{};
            config.episodeLayout.reset();
            config.blueTeamRoles.reset();
            std::cerr << "[BLUE PATROL] Cannot sample collision-safe P="
                      << activeBlueCount << " racetrack/formation after "
                      << config.patrolGeometrySettings.maximumSamplingAttempts
                      << " geometry attempts ("
                      << (config.evaluationOverride ? "evaluation" : "training")
                      << ", episode=" << config.episodeIndex + 1U
                      << "); check Training/BluePatrol.json.\n";
            return;
        }
        config.patrolGeometry = start->geometry;
        config.patrolGeometryReady = true;
        config.arclength = start->arclength;
        BuildPatrolTrackOverlay(config);
        config.blueTeamRoles = defenders::BlueTeamRoles::Sample(
            activeBlueCount, config.rng);
        if (!config.blueTeamRoles)
        {
            config.patrolGeometryReady = false;
            config.patrolGeometrySamplingFailed = true;
            config.episodeLayout.reset();
            std::cerr << "[BLUE PATROL] Invalid Blue team leader assignment.\n";
            return;
        }
        config.leaderGuardIndex =
            config.blueTeamRoles->LeaderPhysicalIndex();
        config.side = config.evaluationOverride
            ? (config.evaluationSide < 0 ? -1 : 1)
            : (std::bernoulli_distribution{0.5}(config.rng) ? 1 : -1);
        ++config.episodeIndex;
        auto const& geometry = start->geometry;
        std::clog << "[BLUE PATROL] Episode " << config.episodeIndex
                  << " racetrack length=" << geometry.length
                  << "m width=" << geometry.width
                  << "m heading=" << geometry.heading
                  << "rad anchor=(" << geometry.anchorNorth << ','
                  << geometry.anchorEast << ")m wedge="
                  << geometry.separationDistance << "m/"
                  << geometry.separationAngle << "rad direction="
                  << (geometry.direction > 0 ? "counter-clockwise" : "clockwise")
                  << " leader=blue_" << config.leaderGuardIndex + 1U
                  << '\n';
        std::string spawnDiagnostic;
        config.episodeLayout = GenerateEpisodeSpawnLayout(
            config.side, config.spawnSettings, bounds, config.rng,
            &spawnDiagnostic);
        if (!config.episodeLayout)
        {
            config.episodeLayoutGenerationFailed = true;
            std::cerr << "[BLUE_DEFENSE] Cannot generate episode spawns: "
                      << spawnDiagnostic << '\n';
        }
    }

    if (!config.episodeLayout)
        return;

    auto const& [metadata] = entity.get<ppo::model::sEntityMetadata>();
    auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
    auto const bounds = WorldBounds{0.5F * world.m_maxNorth, 0.5F * world.m_maxEast};
    auto pose = MakeSpawn(metadata.m_name, *config.episodeLayout,
                          bounds, config.rng);
    auto const generalizedBlue = GeneralizedBlueIndex(metadata.m_name);
    if (generalizedBlue
        && *generalizedBlue < config.spawnSettings.blueTeam->activeCount)
    {
        pose = Spawn{};
    }
    if (pose && generalizedBlue)
    {
        defenders::Parameters parameters{};
        parameters.halfNorth = bounds.halfNorth;
        parameters.halfEast = bounds.halfEast;
        parameters.cruiseSpeed = config.patrolCruiseSpeed;
        parameters.maxRotationRate = config.patrolMaximumRotationRate;
        parameters.collisionDistance = config.patrolCollisionDistance;
        parameters.geometry = config.patrolGeometry;
        auto const leader = defenders::PatrolReference(
            parameters, config.side, config.arclength);
        defenders::Reference reference = leader;
        auto const assignment =
            config.blueTeamRoles->Assignment(*generalizedBlue);
        if (!assignment)
            return;
        if (assignment->role == defenders::BlueRole::Teammate)
        {
            auto const leaderPose = defenders::EvaluatePatrolCurve(
                parameters.geometry, config.arclength);
            auto const target = defenders::EvaluateFormationReference(
                parameters.geometry, leaderPose, assignment->formationSlot,
                parameters.cruiseSpeed);
            reference.north = target.pose.north;
            reference.east = target.pose.east;
            reference.heading = target.pose.heading;
            pose->speed = target.speed;
        }
        auto const mobile = defenders::OnReference(parameters, reference);
        pose->north = mobile.north;
        pose->east = mobile.east;
        pose->heading = mobile.heading;
        if (assignment->role == defenders::BlueRole::Leader)
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
    if (config.next == 0U)
        ppo::model::UpdateRestrictedZonePseudoLidars(
            *registry, ppo::model::eEntitySide::Friend,
            defenders::Parameters{}.cautionDistance);
}
} // namespace ppo::examples::bluedefense
