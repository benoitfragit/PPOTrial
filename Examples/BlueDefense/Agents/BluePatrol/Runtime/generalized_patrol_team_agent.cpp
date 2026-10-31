#include "generalized_patrol_team_agent.hpp"

#include "generalized_cooperative_reward.hpp"
#include "generalized_patrol_learner.hpp"
#include "generalized_reward_shaping.hpp"
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ppo::examples::bluedefense::defenders {
namespace {

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
std::size_t AliveCount(std::array<Mobile, Capacity> const& mobiles,
                       std::size_t const activeCount) noexcept
{
    return static_cast<std::size_t>(std::count_if(
        mobiles.begin(), mobiles.begin() + activeCount,
        [](Mobile const& mobile) { return mobile.available; }));
}

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
    PatrolTrainingSettings training,
    std::filesystem::path actorCheckpoint,
    std::filesystem::path trainingCheckpoint,
    std::filesystem::path productionCheckpoint,
    std::filesystem::path legacyTrainingV4,
    std::filesystem::path legacyLatestV7,
    std::filesystem::path legacyProductionV7)
    : registry_(objective.registry()), objectiveId_(objective.uniqid()),
      counts_{blueCount, redCount}, parameters_(parameters),
      training_(training), actorCheckpoint_(std::move(actorCheckpoint)),
      trainingCheckpoint_(std::move(trainingCheckpoint)),
      productionCheckpoint_(std::move(productionCheckpoint)),
      evaluation_(training.evaluation)
{
    guardIds_.fill(entt::null);
    attackerIds_.fill(entt::null);
    if (registry_ == nullptr || !Valid(counts_) || !Valid(parameters_)
        || !Valid(training_) || actorCheckpoint_.empty()
        || trainingCheckpoint_.empty() || productionCheckpoint_.empty())
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

    rewardWeights_ = MapGeneralizedRewardWeights(training_);
    learner_ = std::make_unique<GeneralizedPatrolLearner>(
        counts_, training_.seed, training_.rolloutSteps);
    if (std::filesystem::exists(trainingCheckpoint_))
    {
        // A V5 value function cannot be resumed with the new per-blue input.
        // Keep an untouched backup and migrate only the compatible actor.
        std::ifstream probe{trainingCheckpoint_};
        std::string magic;
        probe >> magic;
        if (magic == "BLUE_TEAM_TRAINING_V5")
        {
            auto backup = trainingCheckpoint_;
            backup += ".v5.bak";
            if (!std::filesystem::exists(backup))
                std::filesystem::copy_file(trainingCheckpoint_, backup);
            learner_->MigrateV5TrainingActor(trainingCheckpoint_);
            std::cout << "[BLUE MAPPO V8] Migrated V5 actor; old state backed up "
                      << backup.string()
                      << ". New 256-wide critic and optimizers start fresh.\n";
        }
        else
        {
            learner_->LoadTrainingState(trainingCheckpoint_);
            std::cout << "[BLUE MAPPO V8] Resumed V6 generalized training update "
                      << learner_->StepCount() << ".\n";
        }
    }
    else if (std::filesystem::exists(actorCheckpoint_))
    {
        learner_->LoadActor(actorCheckpoint_);
        std::cout << "[BLUE MAPPO V8] Loaded generalized actor.\n";
    }
    else
    {
        if (std::filesystem::exists(legacyTrainingV4))
        {
            GeneralizedTrainingMigrationReport const report =
                learner_->MigrateV4TrainingState(
                    legacyTrainingV4, training_.seed);
            std::cout << "[BLUE MAPPO V8] Migrated v4 actor parameters="
                      << report.actorParametersMigrated
                      << " compatible-Adam-moments="
                      << report.compatibleActorMomentsMigrated
                      << "; critic, optimizer scalars, RNG, update counter "
                         "and rollout reset for generalized training.\n";
        }
        else
        {
            std::filesystem::path const legacy =
                std::filesystem::exists(legacyLatestV7)
                ? legacyLatestV7 : legacyProductionV7;
            if (std::filesystem::exists(legacy))
            {
                learner_->LoadV7Actor(legacy);
                std::cout << "[BLUE MAPPO V8] Migrated compatible v7 actor "
                             "weights; critic and optimizer state start fresh.\n";
            }
        }
    }
    lastSavedUpdate_ = learner_->StepCount();
    finished_ = learner_->StepCount() >= training_.maxUpdates;
    LoadPromotedScore();
}

GeneralizedPatrolMappoAgent::~GeneralizedPatrolMappoAgent()
{
    SaveAtUpdateBoundary();
}

void GeneralizedPatrolMappoAgent::Init()
{
    episodeStep_ = 0U;
    currentReward_ = 0.F;
    lastRewardChannels_.fill(0.F);
    hasBeforeStep_ = false;
    episodeDone_ = false;
}

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

void GeneralizedPatrolMappoAgent::BuildObservations(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    std::array<GeneralizedObservation,
               TeamCapacity::MaximumBlueMobiles>& observations) const
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
            parameters, state, roles, observations))
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

GeneralizedEvaluationSample
GeneralizedPatrolMappoAgent::BuildEvaluationSample(
    Parameters const& parameters, GeneralizedRuntimeState const& state,
    bool const missionFailed, bool const timeout) const
{
    GeneralizedEvaluationSample result{};
    result.finalState = CombatMask(state);
    result.missionFailed = missionFailed;
    result.timeout = timeout;

    BlueTeamRoles const& roles = Roles();
    Mobile const& leader = state.blues[roles.LeaderPhysicalIndex()];
    PatrolProjection const projection = ProjectOnPatrolCurve(
        parameters.geometry, leader.north, leader.east);
    float const positionScale = std::max(
        training_.evaluation.trackingTolerance
            * std::hypot(parameters.halfNorth, parameters.halfEast),
        1.F);
    float const positionQuality = Clamp(
        1.F - projection.distance / positionScale, 0.F, 1.F);
    float const headingQuality = Clamp(
        0.5F * (1.F + std::cos(
            leader.heading - projection.pose.heading)), 0.F, 1.F);
    float const speedQuality = Clamp(
        leader.speed / std::max(parameters.cruiseSpeed, 1.F), 0.F, 1.F);
    result.leaderPatrolQuality = leader.available
        ? (positionQuality + headingQuality + speedQuality) / 3.F : 0.F;

    for (std::size_t physicalIndex = 0U;
         physicalIndex < counts_.blueMobiles; ++physicalIndex)
    {
        BlueRoleAssignment const assignment = *roles.Assignment(physicalIndex);
        if (assignment.role != BlueRole::Teammate)
            continue;
        result.formationSlotQuality[assignment.formationSlot] = Clamp(
            GeneralizedFormationQuality(parameters, state, roles, physicalIndex),
            0.F, 1.F);
    }
    return result;
}

void GeneralizedPatrolMappoAgent::ChooseAction()
{
    if (episodeDone_ || (finished_ && !evaluation_.InProgress())
        || registry_ == nullptr)
        return;
    try
    {
        Parameters const parameters = ReadParameters();
        beforeStep_ = ReadState(parameters);
        std::array<GeneralizedObservation,
                   TeamCapacity::MaximumBlueMobiles> observations{};
        BuildObservations(parameters, beforeStep_, observations);

        beforeAvailability_.fill(false);
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
            beforeAvailability_[index] = beforeStep_.blues[index].available;
        lastActions_ = evaluation_.InProgress()
            ? learner_->EvaluateActions(observations, beforeAvailability_)
            : learner_->ChooseActions(observations, beforeAvailability_);
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        {
            ppo::model::BaseEntity blue{registry_, guardIds_[index]};
            auto [control] = blue.get<ppo::model::sSNAControlState>();
            control.m_rotSpeedCmd = Clamp(lastActions_[index][0], -1.F, 1.F);
            control.m_accCmd = Clamp(lastActions_[index][1], -1.F, 1.F);
        }
        hasBeforeStep_ = true;
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO V8][MAPPO_ERROR] caught_at="
                  << __FILE__ << ':' << __LINE__
                  << " phase=ChooseAction update="
                  << (learner_ ? learner_->StepCount() : 0U)
                  << " episode_step=" << episodeStep_
                  << " cause=" << error.what() << '\n';
        if (evaluation_.InProgress())
        {
            evaluation_.Cancel();
            RestoreTrainingInitializer();
        }
        if (learner_)
        {
            learner_->AbortPendingStep();
            if (learner_->OptimizationFailed())
                finished_ = true; // Do not replay a possibly partial Adam update.
        }
        hasBeforeStep_ = false;
        episodeDone_ = true;
    }
}

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
        BuildObservations(parameters, after, observations);

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
        bool const noThreatRemains = counts_.redMobiles > 0U
            && AliveCount(after.reds, counts_.redMobiles) == 0U
            && !pendingThreats;
        bool const terminated = after.failed || !after.objectiveAvailable
            || noThreatRemains;
        std::size_t const episodeLimit = evaluation_.InProgress()
            ? training_.evaluation.maxSteps : parameters.episodeSteps;
        bool const truncated = !terminated
            && episodeStep_ >= episodeLimit;

        if (evaluation_.InProgress())
        {
            evaluation_.ObserveStep();
            // Diagnostics only: record the first alive-to-destroyed transition.
            // The ECS remains authoritative; no physics or training input changes.
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
                auto const test = evaluation_.CurrentCase();

                std::cout << "[MAPPO_EVAL_LOSS] update="
                          << learner_->StepCount()
                          << " campaign=" << evaluation_.CompletedCampaigns() + 1U
                          << " case=" << evaluation_.TestsFinished() + 1U
                          << " seed=" << test.seed
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
                    std::cout << " min_speed=" << flight->m_minSpeed
                              << " low_speed_threshold=" << 0.5F * flight->m_minSpeed
                              << " fuel_remaining=" << flight->m_fuel
                              << " fuel_capacity=" << flight->m_maxfuel;
                std::cout << '\n';
            }
            hasBeforeStep_ = false;
            episodeDone_ = terminated || truncated;
            currentReward_ = 0.F;
            if (episodeDone_)
            {
                // One bounded diagnostic row per evaluation episode. The
                // production physics and the training rollout stay untouched.
                auto const test = evaluation_.CurrentCase();
                std::size_t const livingBlue =
                    AliveCount(after.blues, counts_.blueMobiles);
                std::size_t const livingRed =
                    AliveCount(after.reds, counts_.redMobiles);
                char const* const outcome = !after.objectiveAvailable
                    ? "objective_lost"
                    : livingBlue == 0U ? "all_blues_lost"
                    : after.failed ? "boundary_failure"
                    : noThreatRemains ? "reds_cleared"
                    : truncated ? "timeout" : "unknown";
                std::cout << "[MAPPO_EVAL_EPISODE] update="
                          << learner_->StepCount()
                          << " campaign=" << evaluation_.CompletedCampaigns() + 1U
                          << " case=" << evaluation_.TestsFinished() + 1U
                          << " seed=" << test.seed
                          << " steps=" << episodeStep_
                          << " P=" << counts_.blueMobiles
                          << " N=" << counts_.redMobiles
                          << " blue_alive=" << livingBlue
                          << " red_alive=" << livingRed
                          << " objective_alive=" << after.objectiveAvailable
                          << " leader_alive="
                          << after.blues[Roles().LeaderPhysicalIndex()].available
                          << " terminal=" << outcome;
                for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
                {
                    std::cout << " blue_" << index + 1U << '='
                              << GeneralizedBlueLossCause(
                                     *registry_, guardIds_[index]);
                }
                std::cout << '\n';
                bool const hasMore = evaluation_.FinishEpisode(
                    Roles(), BuildEvaluationSample(
                        parameters, after,
                        after.failed || !after.objectiveAvailable, truncated));
                if (hasMore)
                {
                    PrepareEvaluationCase();
                }
                else
                {
                    RestoreTrainingInitializer();
                    if (evaluation_.Improved())
                        PromoteEvaluatedActor();
                    GeneralizedEvaluationMetrics const metrics =
                        evaluation_.Metrics();
                    std::cout << "[BLUE MAPPO V8] Evaluation #"
                              << evaluation_.CompletedCampaigns()
                              << " P=" << counts_.blueMobiles
                              << " N=" << counts_.redMobiles
                              << " score=" << metrics.score
                              << " success=" << metrics.missionSuccessRate
                              << " blue-survival=" << metrics.blueSurvivalRate
                              << " red-neutralization="
                              << metrics.redNeutralizationRate
                              << " objective-survival="
                              << metrics.objectiveSurvivalRate << '\n';
                }
            }
            return episodeDone_;
        }

        GeneralizedRewardTransition transition{};
        transition.before = CombatMask(beforeStep_);
        transition.after = CombatMask(after);
        transition.missionFailed = after.failed || !after.objectiveAvailable;
        transition.pendingThreats = pendingThreats;
        transition.timeout = truncated;
        // Dense shared signal: interception of an imminent attacker reduces
        // objective danger more than neutralizing a distant attacker.
        transition.responsibilities.threatContainment = Clamp(
            ObjectiveThreatLevel(beforeStep_, parameters)
                - ObjectiveThreatLevel(after, parameters), -1.F, 1.F);
        BlueTeamRoles const& roles = Roles();
        float const cooperation =
            beforeStep_.blues[roles.LeaderPhysicalIndex()].available
            && after.blues[roles.LeaderPhysicalIndex()].available
            ? GeneralizedCooperationWeight(parameters, beforeStep_) : 0.F;
        transition.responsibilities.leaderPatrol = cooperation
            * GeneralizedLeaderTrackingReward(
                parameters, beforeStep_, after, roles);
        transition.responsibilities.pursuitProgress =
            GeneralizedEngagementWeight(parameters, beforeStep_)
            * GeneralizedPursuitProgress(parameters, beforeStep_, after);
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        {
            BlueRoleAssignment const assignment = *roles.Assignment(index);
            if (assignment.role == BlueRole::Teammate)
            {
                transition.responsibilities.formationSlots[
                    assignment.formationSlot] = cooperation
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
        GeneralizedPatrolLearner::Rewards rewards{};
        float rewardSum = 0.F;
        std::size_t rewardCount = 0U;
        for (std::size_t index = 0U; index < counts_.blueMobiles; ++index)
        {
            rewards[index] = rewardBreakdown.perBlue[index].Total(
                rewardSettings);
            if (beforeAvailability_[index])
            {
                rewardSum += rewards[index];
                ++rewardCount;
            }
        }
        currentReward_ = rewardCount == 0U ? 0.F
            : rewardSum / static_cast<float>(rewardCount);
        lastRewardChannels_ = rewardBreakdown.team.values;

        bool const updated = learner_->Observe(
            observations, rewards, availability, terminated, truncated);
        hasBeforeStep_ = false;
        episodeDone_ = terminated || truncated;
        if (updated)
        {
            finished_ = learner_->StepCount() >= training_.maxUpdates;
            bool const evaluate = ShouldEvaluateGeneralizedUpdate(
                learner_->StepCount(), training_.checkpointEvery, finished_);
            SaveAtUpdateBoundary(evaluate);
            if (evaluate)
            {
                StartEvaluation();
                episodeDone_ = true;
            }
        }
        // A PPO update is an optimizer boundary, not an episode boundary.
        // Continuing here is essential when an episode spans several
        // rollouts (for example N=0 with a 4096-step horizon and a 2048-step
        // rollout). Only the physical mission outcome requests an ECS reset.
        return episodeDone_;
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO V8][MAPPO_ERROR] caught_at="
                  << __FILE__ << ':' << __LINE__
                  << " phase=SetNewState update="
                  << (learner_ ? learner_->StepCount() : 0U)
                  << " episode_step=" << episodeStep_
                  << " collected="
                  << (learner_ ? learner_->CollectedTeamSteps() : 0U)
                  << " capacity="
                  << (learner_ ? learner_->RolloutSize() : 0U)
                  << " cause=" << error.what() << '\n';
        if (evaluation_.InProgress())
        {
            evaluation_.Cancel();
            RestoreTrainingInitializer();
        }
        if (learner_)
        {
            learner_->AbortPendingStep();
            if (learner_->OptimizationFailed())
                finished_ = true; // Fail-stop until a valid checkpoint is loaded.
        }
        hasBeforeStep_ = false;
        episodeDone_ = true;
        return true;
    }
}

void GeneralizedPatrolMappoAgent::SaveAtUpdateBoundary(
    bool const force) noexcept
{
    if (!learner_ || learner_->HasPendingAction()
        || learner_->CollectedTeamSteps() != 0U
        || learner_->StepCount() == lastSavedUpdate_)
    {
        return;
    }
    if (!force
        && learner_->StepCount() % training_.saveEveryUpdates != 0U
        && learner_->StepCount() < training_.maxUpdates)
    {
        return;
    }
    try
    {
        learner_->SaveActor(actorCheckpoint_);
        learner_->SaveTrainingState(trainingCheckpoint_);
        lastSavedUpdate_ = learner_->StepCount();
        std::cout << "[BLUE MAPPO V8] Saved actor and training checkpoint at update "
                  << lastSavedUpdate_ << '\n';
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO V8] Checkpoint failed: "
                  << error.what() << '\n';
    }
}

void GeneralizedPatrolMappoAgent::StartEvaluation()
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<InitializerContext>())
    {
        throw std::runtime_error(
            "Missing initializer context for generalized evaluation");
    }
    auto& context = registry_->ctx().get<InitializerContext>();
    trainingRng_ = context.rng;
    trainingRngSaved_ = true;
    evaluation_.Start();
    try
    {
        PrepareEvaluationCase();
    }
    catch (...)
    {
        evaluation_.Cancel();
        RestoreTrainingInitializer();
        throw;
    }
    std::cout << "[BLUE MAPPO V8] Starting deterministic P="
              << counts_.blueMobiles << " N=" << counts_.redMobiles
              << " evaluation after update " << learner_->StepCount()
              << " (" << evaluation_.TestCount() << " cases)\n";
}

void GeneralizedPatrolMappoAgent::PrepareEvaluationCase()
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<InitializerContext>())
    {
        throw std::runtime_error(
            "Missing initializer context for generalized evaluation case");
    }
    GeneralizedEvaluationCase const test = evaluation_.CurrentCase();
    auto& context = registry_->ctx().get<InitializerContext>();
    context.evaluationOverride = true;
    context.evaluationSide = test.side;
    context.evaluationPhaseFraction = test.phaseFraction;
    context.rng.seed(test.seed);
    context.next = 0U;
}

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
    if (trainingRngSaved_)
    {
        context.rng = trainingRng_;
        trainingRngSaved_ = false;
    }
}

void GeneralizedPatrolMappoAgent::PromoteEvaluatedActor() noexcept
{
    try
    {
        learner_->SaveActor(productionCheckpoint_);
        SavePromotedScore();
        std::cout << "[BLUE MAPPO V8] Promoted evaluated generalized actor to "
                  << productionCheckpoint_.string() << '\n';
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO V8] Actor promotion failed: "
                  << error.what() << '\n';
    }
}

void GeneralizedPatrolMappoAgent::LoadPromotedScore() noexcept
{
    try
    {
        if (!std::filesystem::exists(productionCheckpoint_))
            return;
        auto scorePath = productionCheckpoint_;
        scorePath += ".score";
        std::ifstream stream{scorePath};
        std::string magic;
        std::size_t blueCount = 0U;
        std::size_t redCount = 0U;
        float score = 0.F;
        if (stream >> magic >> blueCount >> redCount >> score
            && magic == "BLUE_PATROL_GENERALIZED_EVALUATION_V1"
            && blueCount == counts_.blueMobiles
            && redCount == counts_.redMobiles)
        {
            evaluation_.RestoreBestScore(score);
        }
    }
    catch (...)
    {
        // Optional malformed metadata only forces a fresh evaluation baseline.
    }
}

void GeneralizedPatrolMappoAgent::SavePromotedScore()
{
    auto scorePath = productionCheckpoint_;
    scorePath += ".score";
    if (scorePath.has_parent_path())
        std::filesystem::create_directories(scorePath.parent_path());
    auto temporary = scorePath;
    temporary += ".tmp";
    auto backup = scorePath;
    backup += ".bak";
    {
        std::ofstream stream{temporary, std::ios::trunc};
        if (!stream)
            throw std::runtime_error(
                "Cannot create generalized blue evaluation score");
        stream << "BLUE_PATROL_GENERALIZED_EVALUATION_V1 "
               << counts_.blueMobiles << ' ' << counts_.redMobiles << ' '
               << std::setprecision(9) << evaluation_.BestScore() << '\n';
        if (!stream)
            throw std::runtime_error(
                "Cannot write generalized blue evaluation score");
    }
    std::error_code ignored;
    std::filesystem::remove(backup, ignored);
    bool const existed = std::filesystem::exists(scorePath);
    if (existed)
        std::filesystem::rename(scorePath, backup);
    try
    {
        std::filesystem::rename(temporary, scorePath);
    }
    catch (...)
    {
        if (existed)
            std::filesystem::rename(backup, scorePath);
        throw;
    }
    if (existed)
        std::filesystem::remove(backup, ignored);
}

void GeneralizedPatrolMappoAgent::ProcessCommand(
    ppo::model::sAgentRealTimeCommand const& command) noexcept
{
    if (command.m_id != id_)
        return;
    if (command.m_type
        == ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardClamp
        && std::isfinite(command.m_rewardClamp)
        && command.m_rewardClamp > 0.F)
    {
        training_.rewardClamp = command.m_rewardClamp;
    }
    else if (command.m_type
        == ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardScaling
        && command.m_rewardScalingType < rewardWeights_.size()
        && std::isfinite(command.m_rewardScaling)
        && command.m_rewardScaling >= 0.F
        && command.m_rewardScaling <= 10.F)
    {
        rewardWeights_[command.m_rewardScalingType] =
            command.m_rewardScaling;
    }
}

ppo::model::sAgentStateSnaphot
GeneralizedPatrolMappoAgent::GetSnapshot() const noexcept
{
    ppo::model::sAgentStateSnaphot snapshot{};
    snapshot.m_id = id_;
    snapshot.m_name = "Blue generalized shared MAPPO";
    snapshot.m_common.m_step = static_cast<long long>(episodeStep_);
    snapshot.m_common.m_maxStep = static_cast<long long>(training_.rolloutSteps);
    snapshot.m_common.m_reward = currentReward_;
    snapshot.m_common.m_rewardClamp = training_.rewardClamp;
    snapshot.m_common.m_checkpoint = static_cast<unsigned int>(
        training_.checkpointEvery);
    // The renderer uses m_version as the completed PPO update counter.
    // The observation schema version remains in the V8 policy checkpoint.
    snapshot.m_common.m_version = learner_
        ? static_cast<unsigned int>(std::min<std::uint64_t>(
            learner_->StepCount(), std::numeric_limits<unsigned int>::max()))
        : 0U;
    snapshot.m_common.m_done = episodeDone_;
    snapshot.m_common.m_training = !finished_ && !evaluation_.InProgress();
    if (learner_)
    {
        snapshot.m_common.m_step = static_cast<long long>(
            learner_->CollectedTeamSteps());
    }
    GeneralizedEvaluationMetrics const metrics = evaluation_.Metrics();
    snapshot.m_common.m_notation = evaluation_.BestScore();
    snapshot.m_common.m_latestEvaluationNotation = metrics.score;
    snapshot.m_common.m_completedEvaluations =
        evaluation_.CompletedCampaigns();
    snapshot.m_common.m_hasLatestEvaluation = evaluation_.HasLatest();
    snapshot.m_common.m_latestEvaluationSuccessRate =
        metrics.missionSuccessRate;
    snapshot.m_common.m_latestEvaluationSurvivalRate =
        metrics.blueSurvivalRate;
    snapshot.m_common.m_latestEvaluationTestCount = static_cast<unsigned int>(
        evaluation_.TestCount());
    snapshot.m_common.m_evaluationTestsFinished = static_cast<unsigned int>(
        evaluation_.TestsFinished());
    snapshot.m_common.m_evaluationTimeoutTests = static_cast<unsigned int>(
        std::lround(metrics.timeoutRate
            * static_cast<float>(metrics.episodes)));
    snapshot.m_common.m_latestEvaluationTimeoutRate = metrics.timeoutRate;
    snapshot.m_common.m_evaluationStepsExecuted =
        evaluation_.StepsExecuted();
    snapshot.m_common.m_evaluationStepsPerSecond =
        evaluation_.StepsPerSecond();
    snapshot.m_common.m_evaluationElapsedSeconds =
        evaluation_.ElapsedSeconds();
    snapshot.m_common.m_evaluationprogress = evaluation_.Progress();
    snapshot.m_common.m_evaluationinprogress = evaluation_.InProgress();
    snapshot.m_scalings.assign(
        rewardWeights_.begin(), rewardWeights_.end());
    snapshot.m_rewards.assign(
        lastRewardChannels_.begin(), lastRewardChannels_.end());
    snapshot.m_rewardnames.assign(kGeneralizedRewardChannelNames.begin(),
                                  kGeneralizedRewardChannelNames.end());
    return snapshot;
}

} // namespace ppo::examples::bluedefense::defenders
