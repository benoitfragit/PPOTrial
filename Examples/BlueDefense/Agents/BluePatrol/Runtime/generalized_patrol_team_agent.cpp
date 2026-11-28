/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.cpp
 * @brief Contrôleur ECS MAPPO partagé des Bleus : états, observations, actions, récompenses, évaluations et checkpoints.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "generalized_patrol_team_agent.hpp"
#include "generalized_episode_outcome.hpp"
#include "generalized_leader_succession.hpp"

#include "generalized_cooperative_reward.hpp"
#include "generalized_reward_shaping.hpp"
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ppo::examples::bluedefense::defenders {
namespace {

/**
 * @brief Evaluation campaigns started in this process (their case 0). ppokit-run
 *        creates one world per seed of a campaign, hence one controller per seed: the
 *        campaign numbers of the diagnostic rows are those of the process.
 */
std::atomic<std::size_t> externalCampaignsStarted{0U};

/**
 * @brief Réalise l'opération GeneralizedMobileFromEcs; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
Mobile GeneralizedMobileFromEcs(ppo::model::BaseEntity& entity)
{
    auto const [state] = entity.get<ppo::model::sEntityState>();
    float turnRate = 0.F;
    if (entity.has<ppo::model::sFlightModelState>())
    {
        auto const [flight] = entity.get<ppo::model::sFlightModelState>();
        turnRate = flight.m_accpsi;
    }
    return {state.m_x, state.m_y, state.m_rotation, state.m_speed,
            turnRate, !state.m_destroyed};
}

/**
 * @brief Reward-only zone clearance; matches 1.10 NEZ and 1.75 NEZ
 * thresholds in the historical red zone danger assessment.
 *
 * The ECS scan does not modify the shared 61-ray LiDAR or RLMission.
 */
float GeneralizedRestrictedZoneRisk(
    entt::registry& registry, ppo::model::sEntityState const& own) noexcept
{
    float risk = 0.F;
    auto zones = registry.view<ppo::model::sEntityState,
                               ppo::model::sSNAState>();
    for (auto entity : zones)
    {
        auto const [zone, sna] = zones.get(entity);
        if (zone.m_destroyed
            || zone.m_type != ppo::model::eEntityType::RestrictedZone
            || zone.m_side == own.m_side
            || !std::isfinite(sna.m_noescapezone)
            || sna.m_noescapezone <= 0.F)
            continue;
        float const north = zone.m_x - own.m_x;
        float const east = zone.m_y - own.m_y;
        float const distance = std::hypot(north, east);
        float const alignment = distance > 1.F
            ? (north * std::cos(own.m_rotation)
                + east * std::sin(own.m_rotation)) / distance : 1.F;
        float const candidate = GeneralizedRestrictedZoneProximityRisk(
            distance, sna.m_noescapezone, alignment);
        risk = std::max(risk, candidate);
    }
    return risk;
}

template<std::size_t Capacity>
/**
 * @brief Réalise l'opération AliveCount; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
std::size_t AliveCount(std::array<Mobile, Capacity> const& mobiles,
                       std::size_t const activeCount) noexcept
{
    return static_cast<std::size_t>(std::count_if(
        mobiles.begin(), mobiles.begin() + activeCount,
        [](Mobile const& mobile) { return mobile.available; }));
}

/**
 * @brief Publishes the exact patrol references supplied to policy inference.
 * @param context Scenario context receiving render-only fixed-capacity data.
 * @param targets Physical-index targets computed for the current observation.
 * @param roles Current physical-to-role assignment.
 * @param state Matching observation snapshot; only active Reds hide references.
 */
void PublishPatrolReferenceOverlay(
    InitializerContext& context, GeneralizedPatrolTargets const& targets,
    BlueTeamRoles const& roles, GeneralizedRuntimeState const& state) noexcept
{
    auto& overlay = context.patrolReferenceOverlay;
    overlay = ppo::model::sPatrolReferenceOverlay{};
    if (!roles.IsValid())
        return;

    overlay.m_showInPatrol = !GeneralizedHasActiveRed(state)
        && state.blues[roles.LeaderPhysicalIndex()].available;
    overlay.m_activeReferenceCount = roles.ActiveBlueCount();
    for (std::size_t physicalIndex = 0U;
         physicalIndex < roles.ActiveBlueCount(); ++physicalIndex)
    {
        auto const assignment = roles.Assignment(physicalIndex);
        if (!assignment)
            continue;
        std::size_t const roleIndex = assignment->role == BlueRole::Leader
            ? 0U : assignment->formationSlot + 1U;
        if (roleIndex >= overlay.m_activeReferenceCount)
            continue;

        GeneralizedPatrolTarget const& target = targets[physicalIndex];
        if (!target.valid || !FinitePatrolPose(target.pose)
            || !FinitePatrolPose(target.guidancePose))
        {
            continue;
        }

        overlay.m_references[roleIndex] = {
            target.pose.north,
            target.pose.east,
            target.guidancePose.north,
            target.guidancePose.east,
            target.guidancePose.heading,
            true};
    }
}

/**
 * @brief Publishes the frozen leader waypoint sequence for rendering only.
 * @param context Scenario context receiving the fixed-size snapshot view.
 * @param sequencer Leader-owned RALLY/FOLLOW state.
 */
void PublishPatrolWaypointOverlay(
    InitializerContext& context,
    GeneralizedPatrolWaypointSequencer const& sequencer) noexcept
{
    if (!sequencer.Following())
    {
        context.patrolWaypointOverlay.reset();
        return;
    }
    ppo::model::sPatrolWaypointOverlay overlay{};
    overlay.m_activeWaypointIndex = sequencer.ActiveWaypointIndex();
    overlay.m_validatedWaypointCount = sequencer.ValidatedWaypointCount();
    for (std::size_t index = 0U;
         index < GeneralizedPatrolWaypointSequencer::WaypointCount; ++index)
    {
        PatrolCurvePose const& pose = sequencer.Waypoints()[index].pose;
        overlay.m_waypoints[index] = {pose.north, pose.east, pose.heading};
    }
    context.patrolWaypointOverlay = overlay;
}

/**
 * @brief Réalise l'opération CombatMask; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
GeneralizedCombatStateMask CombatMask(
    GeneralizedRuntimeState const& state) noexcept
{
    GeneralizedCombatStateMask result{};
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        result.blueActive[index] = true;
        result.blueAlive[index] = state.blues[index].available;
    }
    for (std::size_t index = 0U; index < state.counts.redMobiles; ++index)
    {
        result.redActive[index] = true;
        result.redAlive[index] = state.reds[index].available;
    }
    result.objectiveAvailable = state.objectiveAvailable;
    return result;
}

/**
 * @brief Evaluation-only loss classification using the recorded ECS source.
 *
 * Unknown is intentionally explicit when the world did not retain a valid
 * destroying entity. This helper does not alter collision or SNA resolution.
 */
char const* GeneralizedBlueLossCause(
    entt::registry const& registry, entt::entity const blueId) noexcept
{
    using ppo::model::sEntityState;
    using ppo::model::eEntitySide;
    using ppo::model::eEntityType;
    if (!registry.valid(blueId) || !registry.all_of<sEntityState>(blueId))
        return "missing_blue";
    sEntityState const& blue = registry.get<sEntityState>(blueId);
    if (!blue.m_destroyed)
        return "alive";
    entt::entity const sourceId = blue.m_destroyedById;
    if (sourceId == entt::null || !registry.valid(sourceId)
        || !registry.all_of<sEntityState>(sourceId))
        return "unknown";
    sEntityState const& source = registry.get<sEntityState>(sourceId);
    if (source.m_type == eEntityType::Mobile)
        return source.m_side == eEntitySide::Friend
            ? "blue_collision" : "hostile_mobile";
    if (source.m_type == eEntityType::RestrictedZone)
        return source.m_side == eEntitySide::Hostile
            ? "hostile_zone" : "other_zone";
    return "other_source";
}

/**
 * @brief Maps the unchanged legacy JSON weights into distinct generalized
 * responsibilities. No historical 2v2 reward or checkpoint layout is changed.
 */
GeneralizedRewardValues MapGeneralizedRewardWeights(
    PatrolTrainingSettings const& training) noexcept
{
    RewardValues const& legacy = training.rewardWeights;
    GeneralizedRewardValues result{};
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::PatrolTracking)] = legacy[
            static_cast<std::size_t>(RewardChannel::PatrolTracking)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::FormationKeeping)] = legacy[
            static_cast<std::size_t>(RewardChannel::FormationSafety)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::PursuitProgress)] = legacy[
            static_cast<std::size_t>(RewardChannel::Interception)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::ControlEffort)] = legacy[
            static_cast<std::size_t>(RewardChannel::ControlEffort)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::BlueSurvival)] = legacy[
            static_cast<std::size_t>(RewardChannel::MissionOutcome)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::OwnLoss)] = training.ownLossWeight;
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::RedNeutralization)] = legacy[
            static_cast<std::size_t>(RewardChannel::Interception)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::ThreatContainment)] = legacy[
            static_cast<std::size_t>(RewardChannel::ZoneProtection)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::ObjectiveSurvival)] = legacy[
            static_cast<std::size_t>(RewardChannel::ZoneProtection)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::MissionOutcome)] = legacy[
            static_cast<std::size_t>(RewardChannel::MissionOutcome)];
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::Timeout)] = legacy[
            static_cast<std::size_t>(RewardChannel::Timeout)];
    // Independent blue-only safety weights; old ten channel indices are stable.
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::BlueSeparation)] = 1.F;
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::RestrictedZoneClearance)] = 1.F;
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::BoundaryClearance)] =
            kGeneralizedBoundaryClearanceWeight;
    result[static_cast<std::size_t>(
        GeneralizedRewardChannel::LowSpeedRisk)] = training.lowSpeedRiskWeight;
    return result;
}

} // namespace

GeneralizedPatrolMappoAgent::GeneralizedPatrolMappoAgent(
    std::array<ppo::model::BaseEntity*,
               TeamCapacity::MaximumBlueMobiles> const& guards,
    std::size_t const blueCount,
    std::array<ppo::model::BaseEntity*,
               TeamCapacity::MaximumRedMobiles> const& attackers,
    std::size_t const redCount,
    ppo::model::BaseEntity& objective,
    Parameters parameters,
    PatrolTrainingSettings training)
    : registry_(objective.registry()), objectiveId_(objective.uniqid()),
      counts_{blueCount, redCount}, parameters_(parameters),
      training_(training)
{
    guardIds_.fill(entt::null);
    attackerIds_.fill(entt::null);
    if (registry_ == nullptr || !Valid(counts_) || !Valid(parameters_)
        || !Valid(training_))
    {
        throw std::invalid_argument(
            "Invalid generalized blue MAPPO configuration");
    }
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        if (guards[index] == nullptr || guards[index]->registry() != registry_)
            throw std::invalid_argument("Invalid generalized blue mobile");
        guardIds_[index] = guards[index]->uniqid();
    }
    for (std::size_t index = 0U; index < counts_.redMobiles; ++index)
    {
        if (attackers[index] == nullptr
            || attackers[index]->registry() != registry_)
        {
            throw std::invalid_argument("Invalid generalized red track");
        }
        attackerIds_[index] = attackers[index]->uniqid();
    }
    // The learner, the production actor and the campaigns live outside the world.
    rewardWeights_ = MapGeneralizedRewardWeights(training_);
}

void GeneralizedPatrolMappoAgent::Init()
{
    episodeStep_ = 0U;
    ++diagnosticEpisode_;
    followTransitionStep_ = 0U;
    waypointSequencer_.Reset();
    noThreatLap_.Reset();
    roleRewardSums_ = {};
    roleRewardSteps_ = {};
    hasBeforeStep_ = false;
    externalPrepared_ = false;
    episodeDone_ = false;
}

/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées de GeneralizedPatrolMappoAgent.
 * @details Consulte ou modifie des composants ECS.
 */
Parameters GeneralizedPatrolMappoAgent::ReadParameters() const
{
    Parameters result = parameters_;
    if (registry_->ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& world = registry_->ctx().get<
            ppo::model::sEnvironmentState>();
        result.halfNorth = 0.5F * world.m_maxNorth;
        result.halfEast = 0.5F * world.m_maxEast;
        result.dt = world.m_dt;
    }
    if (registry_->ctx().contains<InitializerContext>())
    {
        auto const& context = registry_->ctx().get<InitializerContext>();
        if (context.patrolGeometryReady)
            result.geometry = context.patrolGeometry;
    }
    ppo::model::BaseEntity first{registry_, guardIds_[0]};
    if (first.has<ppo::model::sFlightModelState>())
    {
        auto const [flight] = first.get<ppo::model::sFlightModelState>();
        result.maxSpeed = flight.m_maxSpeed;
        result.maxAcceleration = flight.m_maxAcc;
        result.maxRotationRate = flight.m_maxAccRot;
    }
    return result;
}

/**
 * @brief Projette les composants ECS en tableaux compacts de Bleus, Rouges, rayons LiDAR, objectif et disponibilité; vérifie la validité cinématique.
 * @details Consulte ou modifie des composants ECS. Peut signaler un état invalide par exception.
 */
GeneralizedRuntimeState GeneralizedPatrolMappoAgent::ReadState(
    Parameters const& parameters) const
{
    if (registry_->ctx().contains<InitializerContext>())
    {
        auto const& context = registry_->ctx().get<InitializerContext>();
        if (context.patrolGeometrySamplingFailed
            || context.episodeLayoutGenerationFailed
            || !context.patrolGeometryReady)
        {
            throw std::runtime_error(
                "Generalized episode initialization failed (geometry="
                + std::to_string(context.patrolGeometrySamplingFailed)
                + ", red_layout="
                + std::to_string(context.episodeLayoutGenerationFailed)
                + ", ready=" + std::to_string(context.patrolGeometryReady)
                + ", evaluation=" + std::to_string(context.evaluationOverride)
                + ", episode=" + std::to_string(context.episodeIndex) + ")");
        }
    }
    GeneralizedRuntimeState result{};
    result.counts = counts_;
    result.step = episodeStep_;
    if (registry_->ctx().contains<InitializerContext>())
        result.side = registry_->ctx().get<InitializerContext>().side;

    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        ppo::model::BaseEntity blue{registry_, guardIds_[index]};
        result.blues[index] = GeneralizedMobileFromEcs(blue);
        auto const [ownState] = blue.get<ppo::model::sEntityState>();
        result.restrictedZoneRisk[index] = GeneralizedRestrictedZoneRisk(
            *registry_, ownState);
        if (blue.has<ppo::model::sPseudoLidarState>())
        {
            auto const [sensor] = blue.get<ppo::model::sPseudoLidarState>();
            bool const finite = std::all_of(
                sensor.m_nearestCollisionRanges.begin(),
                sensor.m_nearestCollisionRanges.end(),
                [](float const value) { return std::isfinite(value); });
            if (finite)
            {
                result.lidar[index] = sensor.m_nearestCollisionRanges;
                result.lidarAvailable[index] = true;
            }
        }
    }
    result.failed = GeneralizedBlueTeamFailed(
        result, parameters, Roles().LeaderPhysicalIndex());
    for (std::size_t index = 0U; index < counts_.redMobiles; ++index)
    {
        if (attackerIds_[index] == entt::null
            || !registry_->valid(attackerIds_[index]))
        {
            result.reds[index] = {0.F, 0.F, 0.F, 0.F, 0.F, false};
            continue;
        }
        ppo::model::BaseEntity red{registry_, attackerIds_[index]};
        result.reds[index] = GeneralizedMobileFromEcs(red);
    }
    ppo::model::BaseEntity objective{registry_, objectiveId_};
    auto const [objectiveState] = objective.get<ppo::model::sEntityState>();
    result.objectiveNorth = objectiveState.m_x;
    result.objectiveEast = objectiveState.m_y;
    result.objectiveAvailable = !objectiveState.m_destroyed;
    return result;
}

/**
 * @brief Réalise l'opération Roles de GeneralizedPatrolMappoAgent; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS. Peut signaler un état invalide par exception.
 */
BlueTeamRoles const& GeneralizedPatrolMappoAgent::Roles() const
{
    if (!registry_->ctx().contains<InitializerContext>())
        throw std::runtime_error("Missing generalized initializer context");
    auto const& context = registry_->ctx().get<InitializerContext>();
    if (!context.blueTeamRoles
        || context.blueTeamRoles->ActiveBlueCount() != counts_.blueMobiles)
    {
        throw std::runtime_error("Missing generalized blue role assignment");
    }
    return *context.blueTeamRoles;
}

/**
 * @brief Calcule les observations de chacun des P Bleus et refuse les masques, rôles et valeurs invalides.
 * @details Peut signaler un état invalide par exception.
 */
void GeneralizedPatrolMappoAgent::BuildObservations(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    std::array<GeneralizedObservation,
               TeamCapacity::MaximumBlueMobiles>& observations,
    GeneralizedPatrolTargets const& targets) const
{
    if (!Valid(parameters))
        throw std::runtime_error("Invalid generalized blue parameters");
    if (!Valid(state))
        throw std::runtime_error(
            "Invalid generalized blue runtime state: "
            + DescribeInvalidGeneralizedRuntimeState(state));

    BlueTeamRoles const& roles = Roles();
    if (!roles.IsValid())
        throw std::runtime_error("Invalid generalized blue role assignment");

    // The generalized runtime is deliberately stricter than the pure-domain
    // helper: every controlled ECS mobile must consume the sensor populated by
    // world rules so hostile restricted zones cannot disappear behind a
    // boundary-only fallback.
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        if (!state.lidarAvailable[index])
        {
            throw std::runtime_error(
                "Missing or non-finite pseudo-LiDAR for blue_"
                + std::to_string(index + 1U));
        }
    }

    if (BuildGeneralizedObservations(
            parameters, state, roles, observations, &targets))
    {
        return;
    }

    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        if (!Valid(observations[index]))
        {
            throw std::runtime_error(
                "Invalid encoded observation for blue_"
                + std::to_string(index + 1U));
        }
    }
    throw std::runtime_error("Invalid generalized blue observation");
}

/**
 * @brief Construit les métriques d'un épisode de campagne à partir de l'objectif, de la survie et de la qualité du suivi de patrouille et de formation.
 * @details Applique des bornes numériques.
 */
GeneralizedEvaluationSample
GeneralizedPatrolMappoAgent::BuildEvaluationSample(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    bool const missionFailed, bool const timeout,
    bool const patrolCompleted, GeneralizedPatrolTargets const& targets) const
{
    GeneralizedEvaluationSample result{};
    result.finalState = CombatMask(state);
    result.missionFailed = missionFailed;
    result.timeout = timeout;
    result.patrolCompleted = patrolCompleted;

    BlueTeamRoles const& roles = Roles();
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        auto const error = GeneralizedNoThreatError(
            state, targets[index], index, training_.noThreatPatrol);
        float const quality = targets[index].valid
            ? Clamp(1.F - error.BoundedCost(), 0.F, 1.F) : 0.F;
        BlueRoleAssignment const assignment = *roles.Assignment(index);
        if (assignment.role == BlueRole::Leader)
            result.leaderPatrolQuality = quality;
        else
            result.formationSlotQuality[assignment.formationSlot] = quality;
    }
    return result;
}

void GeneralizedPatrolMappoAgent::WritePatrolProgress(
    Parameters const& parameters, GeneralizedWaypointUpdate const update) const
{
    char const* event = nullptr;
    switch (update)
    {
    case GeneralizedWaypointUpdate::EnteredFollow: event = "follow_entered"; break;
    case GeneralizedWaypointUpdate::Advanced: event = "waypoint_advanced"; break;
    case GeneralizedWaypointUpdate::Completed: event = "sequence_completed"; break;
    case GeneralizedWaypointUpdate::Invalidated: event = "sequence_invalidated"; break;
    default: return;
    }
    float const qualifiedPercent = Clamp(100.F * noThreatLap_.QualifiedDistanceMeters()
        / CircuitLength(parameters.geometry), 0.F, 100.F);
    std::size_t const waypoint = waypointSequencer_.Following()
        ? std::min(waypointSequencer_.ActiveWaypointIndex() + 1U,
            GeneralizedPatrolWaypointSequencer::WaypointCount) : 0U;
    std::cout << "[MAPPO_PATROL_PROGRESS] phase="
              << (EvaluationEpisode() ? "evaluation" : "training")
              << " episode=" << diagnosticEpisode_
              << " P=" << counts_.blueMobiles << " N=" << counts_.redMobiles
              << " steps=" << episodeStep_ << " event=" << event
              << " mode=" << (waypointSequencer_.Following() ? "FOLLOW" : "RALLY")
              << " active_waypoint=" << waypoint
              << " validated_waypoints=" << waypointSequencer_.ValidatedWaypointCount()
              << " rally_seconds=" << static_cast<float>(followTransitionStep_ == 0U
                    ? episodeStep_ : followTransitionStep_) * parameters.dt
              << " arclength_m=" << waypointSequencer_.DistanceMeters()
              << " qualified_lap_percent=" << qualifiedPercent << '\n';
}

void GeneralizedPatrolMappoAgent::WritePatrolDiagnostics(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    GeneralizedPatrolTargets const& targets, char const* const phase,
    char const* const terminationCause) const
{
    float spacing = 0.F;
    bool pair = false;
    for (std::size_t first = 0U; first < counts_.blueMobiles; ++first)
        for (std::size_t second = first + 1U; second < counts_.blueMobiles; ++second)
        {
            if (!state.blues[first].available || !state.blues[second].available)
                continue;
            float const distance = std::hypot(
                state.blues[first].north - state.blues[second].north,
                state.blues[first].east - state.blues[second].east);
            spacing = pair ? std::min(spacing, distance) : distance;
            pair = true;
        }
    auto const& roles = Roles();
    std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
    Mobile const& leader = state.blues[leaderIndex];
    PatrolProjection const projection = ProjectOnPatrolCurve(
        parameters.geometry, leader.north, leader.east);
    GeneralizedPatrolWaypoint const* const active =
        waypointSequencer_.ActiveWaypoint();
    PatrolCurvePose const& headingReference = active != nullptr
        ? active->pose : projection.pose;
    std::size_t const displayedWaypoint = waypointSequencer_.Following()
        ? std::min(waypointSequencer_.ActiveWaypointIndex() + 1U,
            GeneralizedPatrolWaypointSequencer::WaypointCount)
        : 0U;
    std::cout << "[MAPPO_PATROL_EPISODE] phase=" << phase
              << " episode=" << diagnosticEpisode_
              << " P=" << counts_.blueMobiles << " N=" << counts_.redMobiles
              << " steps=" << episodeStep_
              << " mode=" << (waypointSequencer_.Following()
                    ? "FOLLOW" : "RALLY")
              << " active_waypoint=" << displayedWaypoint
              << " validated_waypoints="
              << waypointSequencer_.ValidatedWaypointCount()
              << " waypoint_distance_m="
              << waypointSequencer_.DistanceToActiveWaypoint(state, roles)
              << " cross_track_m=" << projection.signedCrossTrackError
              << " heading_error_rad=" << std::abs(std::remainder(
                    leader.heading - headingReference.heading, kPatrolTwoPi))
              << " speed_mps=" << leader.speed
              << " arclength_m=" << waypointSequencer_.DistanceMeters()
              << " rally_seconds=" << static_cast<float>(followTransitionStep_ == 0U
                    ? episodeStep_ : followTransitionStep_) * parameters.dt
              << " qualified_lap_percent=" << Clamp(
                    100.F * noThreatLap_.QualifiedDistanceMeters()
                        / CircuitLength(parameters.geometry), 0.F, 100.F)
              << " distance_m=" << noThreatLap_.DistanceMeters()
              << " qualified_m=" << noThreatLap_.QualifiedDistanceMeters()
              << " circuit_m=" << CircuitLength(parameters.geometry)
              << " resets=" << noThreatLap_.ResetCount()
              << " reset_cause=" << noThreatLap_.LastResetCause()
              << " tracking_violations=" << noThreatLap_.TrackingViolationSteps()
              << " termination=" << terminationCause
              << " spacing_applicable=" << pair
              << " spacing_margin_m=" << (pair ? spacing - std::max(
                    training_.noThreatPatrol.minimumSeparationMeters,
                    parameters.collisionDistance) : 0.F) << '\n';
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        auto const error = GeneralizedNoThreatError(
            state, targets[index], index, training_.noThreatPatrol);
        std::cout << "[MAPPO_PATROL_AGENT] blue=" << index + 1U
                  << " role=" << (roles.Assignment(index)->role == BlueRole::Leader
                        ? "leader" : "teammate")
                  << " valid=" << targets[index].valid
                  << " position_tolerances=" << error.position
                  << " speed_tolerances=" << error.speed
                  << " heading_tolerances=" << error.heading << '\n';
    }
    for (std::size_t role = 0U; role < roleRewardSteps_.size(); ++role)
    {
        if (roleRewardSteps_[role] == 0U)
            continue;
        std::cout << "[MAPPO_ROLE_REWARD] role=" << (role == 0U ? "leader" : "teammate")
                  << " agent_steps=" << roleRewardSteps_[role];
        for (auto channel : {GeneralizedRewardChannel::PatrolTracking,
                GeneralizedRewardChannel::FormationKeeping,
                GeneralizedRewardChannel::ControlEffort,
                GeneralizedRewardChannel::LowSpeedRisk})
        {
            auto const index = static_cast<std::size_t>(channel);
            std::cout << " [" << kGeneralizedRewardChannelNames[index] << "]="
                      << roleRewardSums_[role][index]
                            / static_cast<float>(roleRewardSteps_[role]);
        }
        std::cout << '\n';
    }
}

/**
 * @brief Applies the actions ppokit-run's policy wrote for the observations of
 *        PrepareExternalStep; errors reach the caller of the world step.
 */
void GeneralizedPatrolMappoAgent::ChooseAction()
{
    if (episodeDone_ || registry_ == nullptr)
        return;
    if (!externalPrepared_)
        throw std::logic_error("Blue step without PrepareExternalStep");
    lastActions_ = exchange_.actions;
    ApplyActions();
    externalPrepared_ = false;
}

/**
 * @brief Reads the world before the step: state, targets, overlays, observations, blues alive.
 */
void GeneralizedPatrolMappoAgent::PrepareStep(
    std::array<GeneralizedObservation, TeamCapacity::MaximumBlueMobiles>& observations)
{
    Parameters const parameters = ReadParameters();
    beforeStep_ = ReadState(parameters);
    BlueTeamRoles const& roles = Roles();
    beforeTargets_ = waypointSequencer_.PhysicalTargets(
        parameters, beforeStep_, roles, training_.noThreatPatrol);
    if (registry_->ctx().contains<InitializerContext>())
    {
        auto& context = registry_->ctx().get<InitializerContext>();
        PublishPatrolReferenceOverlay(
            context, beforeTargets_, roles, beforeStep_);
        PublishPatrolWaypointOverlay(context, waypointSequencer_);
    }
    BuildObservations(parameters, beforeStep_, observations, beforeTargets_);

    beforeAvailability_.fill(false);
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        beforeAvailability_[index] = beforeStep_.blues[index].available;
}

/**
 * @brief Applies lastActions_, clamped, to the blue controls; the step awaits SetNewState.
 */
void GeneralizedPatrolMappoAgent::ApplyActions()
{
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        ppo::model::BaseEntity blue{registry_, guardIds_[index]};
        auto [control] = blue.get<ppo::model::sSNAControlState>();
        control.m_rotSpeedCmd = Clamp(lastActions_[index][0], -1.F, 1.F);
        control.m_accCmd = Clamp(lastActions_[index][1], -1.F, 1.F);
    }
    hasBeforeStep_ = true;
}

bool GeneralizedPatrolMappoAgent::EvaluationEpisode() const noexcept
{
    return externalEvaluation_;
}

void GeneralizedPatrolMappoAgent::PrepareExternalStep()
{
    if (registry_ == nullptr)
        throw std::logic_error("Blue controller without a world");
    PrepareStep(exchange_.observations);
    exchange_.available = beforeAvailability_;
    CaptureScene(beforeStep_, beforeTargets_);
    externalPrepared_ = true;
}

void GeneralizedPatrolMappoAgent::CaptureScene(
    GeneralizedRuntimeState const& state, GeneralizedPatrolTargets const& targets)
{
    GeneralizedTeamScene& scene = exchange_.scene;
    scene.state = state;
    scene.targets = targets;
    BlueTeamRoles const& roles = Roles();
    scene.leader = roles.IsValid() ? roles.LeaderPhysicalIndex() : 0U;
    // The defended area and the outer failure rectangle: half the world's bounds, the outer
    // one scaled by hardBoundaryScale.
    scene.hasDefendedArea = registry_->ctx().contains<ppo::model::sEnvironmentState>();
    if (scene.hasDefendedArea)
    {
        auto const& world = registry_->ctx().get<ppo::model::sEnvironmentState>();
        scene.defendedArea = {0.F, 0.F, 0.5F * world.m_maxNorth, 0.5F * world.m_maxEast};
        scene.hardHalfNorth = scene.defendedArea.m_halfNorth * parameters_.hardBoundaryScale;
        scene.hardHalfEast = scene.defendedArea.m_halfEast * parameters_.hardBoundaryScale;
    }
    scene.followingWaypoints = waypointSequencer_.Following();
    scene.activeWaypoint = waypointSequencer_.ActiveWaypointIndex();
    scene.validatedWaypoints = waypointSequencer_.ValidatedWaypointCount();
    for (std::size_t index = 0U; index < scene.waypoints.size(); ++index)
        scene.waypoints[index] = waypointSequencer_.Waypoints()[index].pose;
    CaptureSceneOverlays(scene);
}

void GeneralizedPatrolMappoAgent::CaptureSceneOverlays(GeneralizedTeamScene& scene) const
{
    BlueTeamRoles const& roles = Roles();
    for (std::size_t slot = 0U; slot < TeamCapacity::MaximumBlueMobiles; ++slot)
    {
        scene.roles[slot] = GeneralizedTeamScene::NoRole;
        scene.blueCones[slot] = {};
        scene.lidars[slot] = {};
        if (slot >= counts_.blueMobiles)
            continue;
        if (roles.IsValid())
        {
            auto const assignment = roles.Assignment(slot);
            if (assignment && assignment->role == BlueRole::Leader)
                scene.roles[slot] = GeneralizedTeamScene::LeaderRole;
            else if (assignment)
                scene.roles[slot] = static_cast<int>(assignment->formationSlot) + 1;
        }
        entt::entity const blue = guardIds_[slot];
        if (!registry_->valid(blue))
            continue;
        if (auto const* sna = registry_->try_get<ppo::model::sSNAState>(blue))
            scene.blueCones[slot] = {sna->m_noescapezone, sna->m_maxShootAngle};
        if (auto const* lidar = registry_->try_get<ppo::model::sPseudoLidarState>(blue))
        {
            SceneLidar& target = scene.lidars[slot];
            target.ranges = lidar->m_nearestCollisionRanges;
            target.distances = lidar->m_nearestCollisionDistances;
            target.span = lidar->m_angleSpanRadians;
            target.valid = true;
        }
    }
    for (std::size_t red = 0U; red < TeamCapacity::MaximumRedMobiles; ++red)
    {
        scene.redCones[red] = {};
        entt::entity const id = attackerIds_[red];
        if (red < counts_.redMobiles && registry_->valid(id))
        {
            if (auto const* sna = registry_->try_get<ppo::model::sSNAState>(id))
                scene.redCones[red] = {sna->m_noescapezone, sna->m_maxShootAngle};
        }
    }
    scene.zoneCount = 0U;
    auto zones = registry_->view<ppo::model::sEntityState, ppo::model::sSNAState>();
    for (entt::entity const zone : zones)
    {
        auto const& state = zones.get<ppo::model::sEntityState>(zone);
        if (state.m_type != ppo::model::eEntityType::RestrictedZone || state.m_destroyed)
            continue;
        if (scene.zoneCount < GeneralizedTeamScene::ZoneCapacity)
        {
            scene.zones[scene.zoneCount] = {state.m_x, state.m_y,
                zones.get<ppo::model::sSNAState>(zone).m_noescapezone,
                state.m_side == ppo::model::eEntitySide::Hostile};
        }
        ++scene.zoneCount;
    }
    scene.track = {};
    scene.references = {};
    if (registry_->ctx().contains<InitializerContext>())
    {
        auto const& context = registry_->ctx().get<InitializerContext>();
        if (context.patrolGeometryReady && context.patrolTrackOverlay.m_sampleCount >= 2U)
            scene.track = context.patrolTrackOverlay;
        if (context.patrolReferenceOverlay.Valid())
            scene.references = context.patrolReferenceOverlay;
    }
}

void GeneralizedPatrolMappoAgent::PublishExternalStep(
    std::array<GeneralizedObservation, TeamCapacity::MaximumBlueMobiles> const& observations,
    std::array<bool, TeamCapacity::MaximumBlueMobiles> const& availability,
    std::array<float, TeamCapacity::MaximumBlueMobiles> const& rewards,
    GeneralizedCooperativeRewards const& breakdown, bool const terminated, bool const truncated,
    GeneralizedEvaluationSample const& sample)
{
    exchange_.next = observations;
    exchange_.nextAvailable = availability;
    exchange_.rewards = rewards;
    // Weighted channels per slot (before the clamp), for the journal of ppokit-run.
    for (std::size_t index = 0U; index < TeamCapacity::MaximumBlueMobiles; ++index)
    {
        for (std::size_t channel = 0U; channel < rewardWeights_.size(); ++channel)
        {
            exchange_.weightedChannels[index][channel] = index < counts_.blueMobiles
                ? rewardWeights_[channel] * breakdown.perBlue[index].values[channel]
                : 0.F;
        }
    }
    exchange_.terminated = terminated;
    exchange_.truncated = truncated;
    if (!terminated && !truncated)
        return;
    // The episode results with the roles after the step.
    if (EvaluateGeneralizedEpisode(Roles(), sample, exchange_.episode)
        != GeneralizedEvaluationStatus::Recorded)
    {
        throw std::runtime_error("Invalid generalized patrol evaluation sample");
    }
}

void GeneralizedPatrolMappoAgent::StartExternalEvaluationCase(
    GeneralizedEvaluationCase const& test, std::size_t const caseIndex)
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<InitializerContext>())
    {
        throw std::runtime_error(
            "Missing initializer context for generalized evaluation case");
    }
    // The case's reset: its patrol side and racetrack phase, from its seed.
    auto& context = registry_->ctx().get<InitializerContext>();
    context.evaluationOverride = true;
    context.evaluationSide = test.side;
    context.evaluationPhaseFraction = test.phaseFraction;
    context.rng.seed(test.seed);
    context.next = 0U;
    externalEvaluation_ = true;
    // The case numbers of the diagnostic rows: a campaign starts at its case 0.
    externalCampaign_ = caseIndex == 0U
        ? externalCampaignsStarted.fetch_add(1U) + 1U
        : externalCampaignsStarted.load();
    externalCase_ = test;
    externalCaseIndex_ = caseIndex;
}

void GeneralizedPatrolMappoAgent::EndExternalEvaluation() noexcept
{
    if (!externalEvaluation_)
        return;
    RestoreTrainingInitializer();
    externalEvaluation_ = false;
}

GeneralizedPatrolMappoAgent::EvaluationCaseLabel
GeneralizedPatrolMappoAgent::ExternalCaseLabel() const noexcept
{
    return {externalCampaign_, externalCaseIndex_ + 1U, externalCase_.seed};
}

char const* GeneralizedPatrolMappoAgent::EvaluationTerminal(
    GeneralizedRuntimeState const& after, bool const noThreatRemains,
    bool const patrolCompleted, bool const truncated) const noexcept
{
    std::size_t const livingBlue = AliveCount(after.blues, counts_.blueMobiles);
    if (!after.objectiveAvailable)
        return "objective_lost";
    if (livingBlue == 0U)
        return "all_blues_lost";
    if (noThreatRemains)
        return "reds_cleared";
    if (after.failed)
        return "boundary_failure";
    if (patrolCompleted)
        return "patrol_complete";
    return truncated ? "timeout" : "unknown";
}

void GeneralizedPatrolMappoAgent::WriteEvaluationLosses(
    EvaluationCaseLabel const& label, GeneralizedRuntimeState const& after) const
{
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        if (!beforeStep_.blues[index].available
            || after.blues[index].available)
            continue;

        entt::entity const blueId = guardIds_[index];
        auto const* flight = registry_->valid(blueId)
            ? registry_->try_get<ppo::model::sFlightModelState>(blueId)
            : nullptr;
        bool const lowSpeed = flight != nullptr
            && 2.F * after.blues[index].speed < flight->m_minSpeed;
        bool const fuelEmpty = flight != nullptr
            && flight->m_fuel
                < std::numeric_limits<float>::epsilon();

        // Each row in one write: other output never cuts it.
        std::ostringstream row;
        row << "[MAPPO_EVAL_LOSS]";
        row << " campaign=" << label.campaign
                  << " case=" << label.caseNumber
                  << " seed=" << label.seed
                  << " step=" << episodeStep_
                  << " blue=blue_" << index + 1U
                  << " source=" << GeneralizedBlueLossCause(
                         *registry_, blueId)
                  << " speed_before=" << beforeStep_.blues[index].speed
                  << " speed=" << after.blues[index].speed
                  << " accel_cmd=" << lastActions_[index][1]
                  << " flight_model=" << static_cast<int>(flight != nullptr)
                  << " low_speed=" << static_cast<int>(lowSpeed)
                  << " fuel_empty=" << static_cast<int>(fuelEmpty);
        if (flight != nullptr)
            row << " min_speed=" << flight->m_minSpeed
                      << " low_speed_threshold=" << 0.5F * flight->m_minSpeed
                      << " fuel_remaining=" << flight->m_fuel
                      << " fuel_capacity=" << flight->m_maxfuel;
        row << '\n';
        std::cout << row.str() << std::flush;
    }
}

void GeneralizedPatrolMappoAgent::WriteEvaluationEpisode(
    EvaluationCaseLabel const& label, GeneralizedRuntimeState const& after,
    char const* const terminal, GeneralizedEvaluationSample const& sample) const
{
    std::size_t const livingBlue = AliveCount(after.blues, counts_.blueMobiles);
    std::size_t const livingRed = AliveCount(after.reds, counts_.redMobiles);
    // The row in one write: other output never cuts it.
    std::ostringstream row;
    row << "[MAPPO_EVAL_EPISODE]";
    row << " campaign=" << label.campaign
              << " case=" << label.caseNumber
              << " seed=" << label.seed
              << " steps=" << episodeStep_
              << " P=" << counts_.blueMobiles
              << " N=" << counts_.redMobiles
              << " blue_alive=" << livingBlue
              << " red_alive=" << livingRed
              << " objective_alive=" << after.objectiveAvailable
              << " leader_alive="
              << after.blues[Roles().LeaderPhysicalIndex()].available
              << " terminal=" << terminal;
    for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
    {
        row << " blue_" << index + 1U << '='
                  << GeneralizedBlueLossCause(
                         *registry_, guardIds_[index]);
    }
    // The episode's normalized qualities, as the campaign averages them.
    GeneralizedEpisodeEvaluation episodeResult{};
    if (EvaluateGeneralizedEpisode(Roles(), sample, episodeResult)
        == GeneralizedEvaluationStatus::Recorded)
    {
        row << std::setprecision(9)
                  << " patrol_quality=" << episodeResult.patrolQuality
                  << " formation_quality=" << episodeResult.formationQuality
                  << std::setprecision(6);
    }
    row << '\n';
    std::cout << row.str() << std::flush;
}

void GeneralizedPatrolMappoAgent::SetRewardWeights(GeneralizedRewardValues const& weights)
{
    for (std::size_t index = 0; index < weights.size(); ++index)
    {
        float const weight = weights[index];
        // The historical slider limits only changed channels. The configured
        // own-loss weight is 24 and must survive changes to other channels.
        if (!std::isfinite(weight) || weight < 0.F
            || (weight > 10.F && weight != rewardWeights_[index]))
            throw std::invalid_argument("Changed Blue reward weights must be finite, from 0 to 10");
    }
    rewardWeights_ = weights;
}

/**
 * @brief Traite le résultat ECS; l'entraînement produit les rewards et transitions, tandis que l'inférence ne conserve que la terminaison.
 * @details Consulte ou modifie des composants ECS.
 */
bool GeneralizedPatrolMappoAgent::SetNewState()
{
    if (!hasBeforeStep_)
        return episodeDone_;
    ++episodeStep_;
    try
    {
        Parameters const parameters = ReadParameters();
        GeneralizedRuntimeState const after = ReadState(parameters);
        std::array<GeneralizedObservation,
                   TeamCapacity::MaximumBlueMobiles> observations{};
        // The action and its reward belong to the roles before physics. The
        // next observation may already describe a promoted physical teammate.
        BlueTeamRoles const roles = Roles();
        std::size_t const leaderIndex = roles.LeaderPhysicalIndex();
        GeneralizedPatrolTargets const baselineFrozenTargets =
            beforeTargets_[leaderIndex].valid
            ? ComputePatrolWaypointTargets(parameters, after, roles,
                beforeTargets_[leaderIndex].guidancePose)
            : ComputePatrolTargets(parameters, after, roles,
                training_.noThreatPatrol.lookaheadSeconds);
        GeneralizedPatrolTargets const frozenAfterTargets =
            ComputePhysicalWedgeTargets(
                parameters, after, roles, baselineFrozenTargets);
        bool const sequenceWasValid = waypointSequencer_.CircuitValid();
        GeneralizedWaypointUpdate const waypointUpdate =
            counts_.redMobiles == 0U
            ? waypointSequencer_.Observe(parameters, beforeStep_, after,
                roles, training_.noThreatPatrol)
            : GeneralizedWaypointUpdate::None;
        if (waypointUpdate == GeneralizedWaypointUpdate::EnteredFollow)
        {
            followTransitionStep_ = episodeStep_;
            noThreatLap_.Reset();
        }
        GeneralizedPatrolTargets afterTargets =
            waypointSequencer_.PhysicalTargets(
                parameters, after, roles, training_.noThreatPatrol);
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        {
            if (roles.Assignment(index)->role != BlueRole::Teammate
                || !afterTargets[index].valid)
                continue;
            afterTargets[index].physicallyFeasible =
                GeneralizedPhysicalSlotRotationFeasible(
                    parameters, beforeTargets_[index], afterTargets[index]);
        }
        GeneralizedPatrolTargets const rewardAfterTargets = afterTargets;
        auto& context = registry_->ctx().get<InitializerContext>();
        if (RelieveGeneralizedLeader(after, *context.blueTeamRoles,
                waypointSequencer_, noThreatLap_))
        {
            followTransitionStep_ = 0U;
            afterTargets = waypointSequencer_.PhysicalTargets(
                parameters, after, Roles(), training_.noThreatPatrol);
        }
        BuildObservations(parameters, after, observations, afterTargets);
        // Refresh from the resolved state, including spawn/loss transitions,
        // so the next renderer snapshot cannot retain pre-step visibility.
        PublishPatrolReferenceOverlay(context, afterTargets, Roles(), after);
        PublishPatrolWaypointOverlay(context, waypointSequencer_);

        std::array<bool, TeamCapacity::MaximumBlueMobiles> availability{};
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
            availability[index] = after.blues[index].available;

        bool pendingThreats = false;
        if (registry_->ctx().contains<InitializerContext>())
        {
            auto const& context = registry_->ctx().get<InitializerContext>();
            pendingThreats = context.episodeLayout
                && HasPendingThreats(*context.episodeLayout,
                                     context.elapsedSteps);
        }
        // N=0 can end successfully after one correctly flown lap.
        // Pending red activations never trigger the no-red mission outcome.
        bool lapCompleted = false;
        if (counts_.redMobiles == 0U && !pendingThreats
            && waypointSequencer_.Following()
            && waypointUpdate != GeneralizedWaypointUpdate::EnteredFollow)
        {
            lapCompleted = noThreatLap_.Observe(
                parameters, beforeStep_, after, roles,
                training_.noThreatPatrol, beforeTargets_, frozenAfterTargets);
        }
        auto const outcome = generalized_episode_detail::EvaluateOutcome(
            after, pendingThreats, waypointSequencer_,
            Roles().LeaderPhysicalIndex(), lapCompleted,
            training_.noThreatPatrol, parameters.dt, episodeStep_,
            EvaluationEpisode()
                ? training_.evaluation.maxSteps : parameters.episodeSteps);
        bool const noThreatRemains = outcome.noThreatRemains;
        bool const patrolCompleted = outcome.patrolCompleted;
        bool const missionFailed = outcome.missionFailed;
        bool const terminated = outcome.terminated;
        bool const truncated = outcome.truncated;

        if (waypointUpdate != GeneralizedWaypointUpdate::Invalidated
            || sequenceWasValid)
        {
            WritePatrolProgress(parameters, waypointUpdate);
        }

        GeneralizedRewardTransition transition{};
        transition.before = CombatMask(beforeStep_);
        transition.after = CombatMask(after);
        transition.missionFailed = missionFailed;
        transition.pendingThreats = pendingThreats;
        transition.timeout = truncated;
        transition.patrolCompleted = patrolCompleted;
        // Dense shared signal: interception of an imminent attacker reduces
        // objective danger more than neutralizing a distant attacker.
        transition.responsibilities.threatContainment = Clamp(
            ObjectiveThreatLevel(beforeStep_, parameters)
                - ObjectiveThreatLevel(after, parameters), -1.F, 1.F);
        DefenseTrack const objectiveBefore{beforeStep_.objectiveNorth,
            beforeStep_.objectiveEast, 0.F, 0.F, beforeStep_.objectiveAvailable};
        for (std::size_t index = 0U; index < counts_.redMobiles; ++index)
        {
            Mobile const& red = beforeStep_.reds[index];
            if (red.available && !after.reds[index].available)
                transition.responsibilities.redNeutralizationRiskBefore[index] =
                    ObjectiveThreatRisk(parameters.defense,
                        {red.north, red.east, red.heading, red.speed, red.available},
                        objectiveBefore);
        }
        float const patrolWeight = rewardWeights_[static_cast<std::size_t>(
            GeneralizedRewardChannel::PatrolTracking)];
        float const formationWeight = rewardWeights_[static_cast<std::size_t>(
            GeneralizedRewardChannel::FormationKeeping)];
        // Reward the objective applicable to the observed physical state.
        // This is NOT a controller switch: the same MAPPO actor always picks
        // actions using all Red masks, objective and guidance observations.
        // Never reward tracking an unrelated racetrack during an interception.
        bool const patrolRelevant =
            GeneralizedPatrolRewardRelevant(beforeStep_, after);
        transition.responsibilities.leaderPatrol = patrolRelevant
            ? GeneralizedUnifiedPatrolConvergenceShaping(
                parameters, beforeStep_, after, beforeTargets_, rewardAfterTargets,
                roles, roles.LeaderPhysicalIndex(), training_.noThreatPatrol)
                / std::max(patrolWeight, 1.F)
            : 0.F;
        transition.responsibilities.leaderOnlyPursuit = false;
        // No distance gate: a far-away live Red still requires interception.
        transition.responsibilities.pursuitProgress =
            GeneralizedPursuitProgress(parameters, beforeStep_, after);
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        {
            BlueRoleAssignment const assignment = *roles.Assignment(index);
            if (assignment.role == BlueRole::Teammate)
            {
                transition.responsibilities.formationSlots[assignment.formationSlot] =
                    patrolRelevant
                    ? GeneralizedUnifiedPatrolConvergenceShaping(
                        parameters, beforeStep_, after, beforeTargets_, rewardAfterTargets,
                        roles, index, training_.noThreatPatrol)
                        / std::max(formationWeight, 1.F)
                    : GeneralizedCooperationWeight(parameters, beforeStep_)
                        * GeneralizedFormationProgressReward(
                            parameters, beforeStep_, after, roles, index);
            }
            transition.responsibilities.controlEffort[index] =
                GeneralizedControlEffortPenalty(
                    parameters, lastActions_[index]);
            if (beforeAvailability_[index])
            {
                transition.responsibilities.blueSeparation[index] =
                    GeneralizedSafetyPenalty(
                        GeneralizedBlueSeparationRisk(
                            parameters, beforeStep_, index),
                        GeneralizedBlueSeparationRisk(parameters, after, index));
                transition.responsibilities.restrictedZoneClearance[index] =
                    GeneralizedSafetyPenalty(
                        beforeStep_.restrictedZoneRisk[index],
                        after.restrictedZoneRisk[index]);
                transition.responsibilities.boundaryClearance[index] =
                    GeneralizedSafetyPenalty(
                        GeneralizedBoundaryRisk(
                            parameters, beforeStep_.blues[index]),
                        GeneralizedBoundaryRisk(parameters, after.blues[index]));
                transition.responsibilities.lowSpeedRisk[index] =
                    GeneralizedLowSpeedTransitionPenalty(
                        parameters, beforeStep_.blues[index], after.blues[index]);
            }
        }

        GeneralizedRewardSettings rewardSettings{};
        rewardSettings.weights = rewardWeights_;
        rewardSettings.clampMagnitude = training_.rewardClamp;
        GeneralizedCooperativeRewards rewardBreakdown{};
        if (AggregateGeneralizedCooperativeRewards(
                roles, transition, rewardSettings, rewardBreakdown)
            != GeneralizedRewardStatus::Applied)
        {
            throw std::runtime_error("Invalid generalized reward transition");
        }
        std::array<float, TeamCapacity::MaximumBlueMobiles> rewards{};
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        {
            rewards[index] = rewardBreakdown.perBlue[index].Total(
                rewardSettings);
            if (beforeAvailability_[index])
            {
                std::size_t const role = roles.Assignment(index)->role == BlueRole::Leader
                    ? 0U : 1U;
                ++roleRewardSteps_[role];
                for (std::size_t channel = 0U; channel < rewardWeights_.size(); ++channel)
                    roleRewardSums_[role][channel] += rewardWeights_[channel]
                        * rewardBreakdown.perBlue[index].values[channel];
            }
        }
        if (externalEvaluation_)
            WriteEvaluationLosses(ExternalCaseLabel(), after);
        if (terminated || truncated)
            WritePatrolDiagnostics(parameters, after, afterTargets,
                externalEvaluation_ ? "evaluation" : "training", outcome.DiagnosticCause());

        GeneralizedEvaluationSample sample{};
        if (terminated || truncated)
            sample = BuildEvaluationSample(parameters, after,
                missionFailed, truncated, patrolCompleted, afterTargets);
        if (externalEvaluation_ && (terminated || truncated))
        {
            WriteEvaluationEpisode(ExternalCaseLabel(), after,
                EvaluationTerminal(after, noThreatRemains, patrolCompleted, truncated), sample);
        }
        PublishExternalStep(observations, availability, rewards,
            rewardBreakdown, terminated, truncated, sample);
        CaptureScene(after, afterTargets);
        hasBeforeStep_ = false;
        episodeDone_ = terminated || truncated;
        return episodeDone_;
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO V10][MAPPO_ERROR] caught_at="
                  << __FILE__ << ':' << __LINE__
                  << " phase=SetNewState episode_step=" << episodeStep_
                  << " cause=" << error.what() << '\n';
        hasBeforeStep_ = false;
        episodeDone_ = true;
        return true;
    }
}

/** @brief Ends an evaluation case's reset overrides: the next reset draws a training episode. */
void GeneralizedPatrolMappoAgent::RestoreTrainingInitializer() noexcept
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<InitializerContext>())
    {
        return;
    }
    auto& context = registry_->ctx().get<InitializerContext>();
    context.evaluationOverride = false;
    context.next = 0U;
    context.ClearEpisodeInitializationFailures();
}

} // namespace ppo::examples::bluedefense::defenders
