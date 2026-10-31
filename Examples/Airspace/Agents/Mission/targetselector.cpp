/**
* @file Agents/mission/src/targetselector.cpp
* @brief Implementation for targetselector.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "targetselector.hpp"
#include "missiontrackfilter.hpp"
#include "airspacecomponents.hpp"
#include "utils.hpp"
#include "snasystem.hpp"
#include "restrictedzonelidar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace rl {

namespace {

/**
* @brief Number of bins used by the mission pseudo-lidar.
*/
constexpr std::size_t kPseudoLidarBinCount = MissionObservationLayout::kPseudoLidarBinCount;
static_assert(kPseudoLidarBinCount == ppo::model::kRestrictedZoneLidarBins,
              "RLMission pseudo-LiDAR observation count must remain unchanged");
constexpr float kPseudoLidarAngleSpan = static_cast<float>(ppo::model::TWOPI);

constexpr float kTrackRangeReference = 12000.F;
constexpr float kClosingSpeedReference = 900.F;
constexpr float kRestrictedZoneDangerMultiplier = 1.10F;
constexpr float kSamCautionMultiplier = 1.75F;
constexpr float kNormalizedCautionMargin =
    (kSamCautionMultiplier - kRestrictedZoneDangerMultiplier) / kRestrictedZoneDangerMultiplier;

constexpr float kMainTargetDestructionReward = 40.F;
constexpr float kSecondaryTargetDestructionReward = 6.F;
constexpr float kOwnshipDestructionReward = -45.F;
constexpr float kWingmanDestructionReward = -10.F;
constexpr float kCivilianDestructionReward = -12.F;
constexpr float kMissionFailureReward = -20.F;
constexpr float kInterceptReward = 12.F;
constexpr float kRangeRewardGain = 0.01F;
constexpr float kDangerBasePenalty = -2.5F;
constexpr float kNearDangerBasePenalty = -0.35F;
constexpr float kDangerFacingPenalty = -1.75F;
constexpr float kNearDangerFacingPenalty = -0.85F;
constexpr float kOutOfAreaGraceTime = 1.0F;
constexpr float kOutOfAreaPenaltyDistance = 2500.F;

/**
* @brief Ranked observation extracted for one visible track candidate.
*/
struct sTrackObservation
{
    entt::entity            m_entity{ entt::null };
    ppo::model::eEntityType m_type{ ppo::model::eEntityType::None };
    float                   m_priority{ 0.F };
    float                   m_distance{ 0.F };
    float                   m_dx{ 0.F };
    float                   m_dy{ 0.F };
    float                   m_relativeBearing{ 0.F };
    float                   m_closingSpeed{ 0.F };
    float                   m_threatToOwnship{ 0.F };
    float                   m_isPrimaryTarget{ 0.F };
};

/**
* @brief Aggregated threat estimate built from hostile SAM entities.
*/
struct sRestrictedZoneThreatAssessment
{
    bool  m_hasThreat{ false };
    float m_maxThreatPk{ 0.F };
    float m_nearestDangerMargin{ std::numeric_limits<float>::infinity() };
    float m_forwardAlignmentToNearestThreat{ 0.F };
};

/**
* @brief Maps one distance to a bounded range feature.
*/
float
NormalizeRange(float distance, float rangeReference) noexcept
{
    return distance / (distance + std::max(rangeReference, 1.F));
}

/**
* @brief Computes the target bearing relative to the ownship heading.
*/
float
ComputeRelativeBearing(ppo::model::sEntityState const& ownState,
                       float dx,
                       float dy) noexcept
{
    float const distance = std::sqrt(dx * dx + dy * dy);
    float const ux = dx / (distance + 1.0e-6F);
    float const uy = dy / (distance + 1.0e-6F);

    float azimuth = std::atan2(uy, ux);
    azimuth = static_cast<float>(ppo::model::wrapMinusPi2Pi(azimuth));
    return static_cast<float>(ppo::model::wrapMinusPi2Pi(azimuth - ownState.m_rotation));
}

/**
* @brief Evaluates how dangerous another entity is for the ownship.
*/
float
ComputeThreatToOwnship(ppo::model::BaseEntity& otherEntity,
                       ppo::model::BaseEntity& ownEntity,
                       ppo::model::sEntityState const& ownState,
                       ppo::model::sEntityState const& otherState) noexcept
{
    if (!otherEntity.has<ppo::model::sSNAState>()
    ||  otherState.m_side == ppo::model::eEntitySide::Unknown
    ||  otherState.m_side == ownState.m_side)
    {
        return 0.F;
    }

    return ppo::model::SNASystem::EvaluateDestruction(otherEntity, ownEntity);
}

/**
* @brief Builds one ranked track observation from another entity.
*/
sTrackObservation
BuildTrackObservation(ppo::model::BaseEntity& ownEntity,
                      ppo::model::sEntityState const& ownState,
                      ppo::model::sSNAState const& ownSnaState,
                      ppo::model::BaseEntity& otherEntity)
{
    auto const [otherState] = otherEntity.get<ppo::model::sEntityState>();

    sTrackObservation track{};
    track.m_entity = otherEntity.uniqid();
    track.m_type = otherState.m_type;
    track.m_dx = otherState.m_x - ownState.m_x;
    track.m_dy = otherState.m_y - ownState.m_y;
    track.m_distance = std::sqrt(track.m_dx * track.m_dx + track.m_dy * track.m_dy);
    track.m_relativeBearing = ComputeRelativeBearing(ownState, track.m_dx, track.m_dy);

    float const ux = track.m_dx / (track.m_distance + 1.0e-6F);
    float const uy = track.m_dy / (track.m_distance + 1.0e-6F);
    float const vrx = otherState.m_dx - ownState.m_dx;
    float const vry = otherState.m_dy - ownState.m_dy;
    track.m_closingSpeed = -(vrx * ux + vry * uy);

    track.m_threatToOwnship = ComputeThreatToOwnship(otherEntity, ownEntity, ownState, otherState);
    track.m_isPrimaryTarget = ownSnaState.m_target == otherEntity.uniqid() ? 1.F : 0.F;

    float priority = -track.m_distance;
    priority += 6000.F * track.m_isPrimaryTarget;
    priority += 4000.F * track.m_threatToOwnship;
    if (track.m_type == ppo::model::eEntityType::RestrictedZone)
        priority += 1500.F;
    else if (track.m_type == ppo::model::eEntityType::Mobile)
        priority += 500.F;

    track.m_priority = priority;
    return track;
}

/**
* @brief Aggregates nearest and strongest hostile SAM threats.
*/
sRestrictedZoneThreatAssessment
AssessRestrictedZoneThreats(entt::registry& registry,
                 ppo::model::BaseEntity& ownEntity,
                 ppo::model::sEntityState const& ownState) noexcept
{
    sRestrictedZoneThreatAssessment assessment{};

    auto view = registry.view<ppo::model::sEntityState, ppo::model::sSNAState>();
    for (auto entity : view)
    {
        auto const [otherState, otherSnaState] = view.get(entity);

        if (entity == ownEntity.uniqid()
        ||  otherState.m_destroyed
        ||  otherState.m_type != ppo::model::eEntityType::RestrictedZone
        ||  otherState.m_side == ownState.m_side)
        {
            continue;
        }

        float const dx = otherState.m_x - ownState.m_x;
        float const dy = otherState.m_y - ownState.m_y;
        float const distance = std::sqrt(dx * dx + dy * dy);
        float const dangerRadius = std::max(kRestrictedZoneDangerMultiplier * otherSnaState.m_noescapezone, 1.F);
        float const normalizedMargin = (distance - dangerRadius) / dangerRadius;
        float const relativeBearing = ComputeRelativeBearing(ownState, dx, dy);
        float const forwardAlignment = std::max(std::cos(relativeBearing), 0.F);

        ppo::model::BaseEntity otherEntity{ &registry, entity };
        assessment.m_hasThreat = true;
        if (normalizedMargin < assessment.m_nearestDangerMargin)
        {
            assessment.m_nearestDangerMargin = normalizedMargin;
            assessment.m_forwardAlignmentToNearestThreat = forwardAlignment;
        }
        assessment.m_maxThreatPk =
            std::max(assessment.m_maxThreatPk,
                     ppo::model::SNASystem::EvaluateDestruction(otherEntity, ownEntity));
    }

    return assessment;
}

} // namespace

/**
* @brief Resets mission-specific adapter state before a new episode.
*/
void
StatesAndActionsForMission::Init(void)
{
    if (auto* registry = m_entity.registry();
        registry != nullptr && m_entity.uniqid() != entt::null)
    {
        auto& pseudoLidarState =
            registry->get_or_emplace<ppo::model::sPseudoLidarState>(m_entity.uniqid());
        pseudoLidarState.m_nearestCollisionRanges.fill(1.F);
        pseudoLidarState.m_nearestCollisionDistances.fill(0.F);
        pseudoLidarState.m_angleSpanRadians = kPseudoLidarAngleSpan;
    }
}

/**
* @brief Applies the action vector to the controlled entity.
*/
void
StatesAndActionsForMission::SetNextAction(ACTIONS const& action) noexcept
{
    if (!m_entity.has<ppo::model::sSNAControlState>())
        return;

    auto [snaControlState] = m_entity.get<ppo::model::sSNAControlState>();

    snaControlState.m_rotSpeedCmd = std::clamp(action[0], -1.F, 1.F);
    snaControlState.m_accCmd = std::clamp(action[1], -1.F, 1.F);

    // action 2 for further use is missile departure
}

/**
* @brief Rebuilds the mission observation vector from the current world state.
*/
void
StatesAndActionsForMission::SetNewState(STATES& state) noexcept
{
    std::fill(state.begin(), state.end(), 0.F);

    auto* registry = m_entity.registry();
    if (registry == nullptr
    || !registry->ctx().contains<ppo::model::sEnvironmentState>())
        return;

    auto [aircraftState, snaState, _] = m_entity.get<ppo::model::sEntityState,
                                                     ppo::model::sSNAState,
                                                     ppo::model::sSNAControlState>();
    ppo::model::BaseEntity ownEntity{ registry, m_entity.uniqid() };
    auto const& envState = registry->ctx().get<ppo::model::sEnvironmentState>();

    std::array<float, kPseudoLidarBinCount> nearestCollisionRanges{};
    std::array<float, kPseudoLidarBinCount> nearestCollisionDistances{};
    ppo::model::BuildRestrictedZonePseudoLidar(*registry,
                        ownEntity,
                        aircraftState,
                        nearestCollisionRanges,
                        nearestCollisionDistances);
    ppo::model::BuildZoneBoundaryPseudoLidar(aircraftState,
                                 envState,
                                 nearestCollisionRanges,
                                 nearestCollisionDistances);

    std::vector<sTrackObservation> tracks{};
    tracks.reserve(32);

    auto view = registry->view<ppo::model::sEntityState>();
    for (auto entity : view)
    {
        ppo::model::BaseEntity otherEntity{ registry, entity };
        auto [entityState] = otherEntity.get<ppo::model::sEntityState>();

        if (!detail::IsMissionTrackCandidate(
                aircraftState, entityState, m_entity.uniqid()))
            continue;

        tracks.push_back(BuildTrackObservation(ownEntity, aircraftState, snaState, otherEntity));
    }

    std::sort(
        tracks.begin(),
        tracks.end(),
        [](sTrackObservation const& lhs, sTrackObservation const& rhs)
        {
            if (lhs.m_priority != rhs.m_priority)
                return lhs.m_priority > rhs.m_priority;

            return lhs.m_distance < rhs.m_distance;
        });

    float const rangeReference =
        std::max(std::max(snaState.m_escapezone, snaState.m_noescapezone), kTrackRangeReference);

    unsigned int trackIndex{ 0U };
    for (auto const& track : tracks)
    {
        if (trackIndex >= MAX_TRACKS)
            break;

        unsigned int const offset =
            trackIndex * static_cast<unsigned int>(eFeatures::_Size);
        float const distance = std::max(track.m_distance, 1.0e-6F);

        state[offset + static_cast<unsigned int>(eFeatures::NorthDirection)] = track.m_dx / distance;
        state[offset + static_cast<unsigned int>(eFeatures::EastDirection)] = track.m_dy / distance;
        state[offset + static_cast<unsigned int>(eFeatures::Range)] =
            NormalizeRange(track.m_distance, rangeReference);
        state[offset + static_cast<unsigned int>(eFeatures::CosBearing)] =
            std::cos(track.m_relativeBearing);
        state[offset + static_cast<unsigned int>(eFeatures::SinBearing)] =
            std::sin(track.m_relativeBearing);
        state[offset + static_cast<unsigned int>(eFeatures::ClosingSpeed)] =
            std::tanh(track.m_closingSpeed / kClosingSpeedReference);
        state[offset + static_cast<unsigned int>(eFeatures::Threat)] = track.m_threatToOwnship;
        state[offset + static_cast<unsigned int>(eFeatures::PrimaryTarget)] = track.m_isPrimaryTarget;
        state[offset + static_cast<unsigned int>(eFeatures::Available)] = 1.F;

        ++trackIndex;
    }

    state[MAX_TRACKS * static_cast<unsigned int>(eFeatures::_Size)] =
        static_cast<float>(snaState.m_nbmissiles)
        / std::max(static_cast<float>(snaState.m_maxnbmissiles), 1.0e-6F);

    for (std::size_t iang = 0; iang < nearestCollisionRanges.size(); ++iang)
    {
        state[MAX_TRACKS * static_cast<unsigned int>(eFeatures::_Size)
              + static_cast<unsigned int>(iang) + 1U] = nearestCollisionRanges[iang];
    }

    auto& pseudoLidarState =
        registry->get_or_emplace<ppo::model::sPseudoLidarState>(m_entity.uniqid());
    pseudoLidarState.m_nearestCollisionRanges = nearestCollisionRanges;
    pseudoLidarState.m_nearestCollisionDistances = nearestCollisionDistances;
    pseudoLidarState.m_angleSpanRadians = kPseudoLidarAngleSpan;
}

/**
 * @brief Initialise ou rétablit l'état de TargetSelectorTrainingAgent selon le contexte courant.
 * @details Consulte ou modifie des composants ECS.
 */
void
TargetSelectorTrainingAgent::Init(void)
{
    Base::Init();

    m_stepMemoryInitialized = false;
    m_interceptReward = false;
    m_threatMarginInitialized = false;
    m_prevTargetRange = 0.F;
    m_timeOutOfArea = 0.F;
    m_prevThreatMargin = 0.F;

    if (!m_entity.has<StatesAndActionsForMission>())
        return;

    auto [stateActions] =
        m_entity.get<StatesAndActionsForMission>();

    stateActions.Init();
}

bool
TargetSelectorTrainingAgent::RewardFunction(void) noexcept
{
    entt::registry* registry{ m_entity.registry() };

    if (registry == nullptr
    || !registry->ctx().contains<ppo::model::sEnvironmentState>()
    || m_entity.uniqid() == entt::null
    || !m_entity.has<ppo::model::sEntityState,
                     ppo::model::sFlightModelState,
                     ppo::model::sSNAState>())
    {
        return false;
    }

    auto const [entityState, flightModelState, snaState] =
        m_entity.get<ppo::model::sEntityState,
                     ppo::model::sFlightModelState,
                     ppo::model::sSNAState>();

    auto const& envState{ registry->ctx().get<ppo::model::sEnvironmentState>() };

    float const dt{ envState.m_dt };
    float const x{ entityState.m_x };
    float const y{ entityState.m_y };
    float d{ 0.F };

    ppo::model::EntityId const targetId{ snaState.m_target };
    float const vref{ 800.F };
    float const fuelconsumption{ flightModelState.m_fuelconsumption };
    float const fuelref{ ppo::model::C0 + ppo::model::CV3 * vref * vref * vref };
    float const R_fuel{ std::clamp(-fuelconsumption / (fuelref * dt), -1.F, 0.F) };

    bool done{ false };
    bool targetDestroyed{ false };
    bool targetDestroyedByOwnship{ false };

    float R_Intercept{ 0.F };
    float R_Range{ 0.F };
    float R_OutOfArea{ 0.F };

    std::array<float, static_cast<unsigned char>(StatesAndActionsForMission::RewardType::_Size)> rewards{};
    rewards.fill(0.F);

    float const northLimit = 0.5F * envState.m_maxNorth;
    float const eastLimit = 0.5F * envState.m_maxEast;
    float const northOverflow = std::max(std::fabs(entityState.m_x) - northLimit, 0.F);
    float const eastOverflow = std::max(std::fabs(entityState.m_y) - eastLimit, 0.F);
    float const overflow = std::max(northOverflow, eastOverflow);

    if (overflow > 0.F)
    {
        m_timeOutOfArea += dt;

        if (m_timeOutOfArea >= kOutOfAreaGraceTime)
        {
            float const overflowRatio =
                std::clamp(overflow / kOutOfAreaPenaltyDistance, 0.F, 1.5F);
            float const persistence =
                std::clamp((m_timeOutOfArea - kOutOfAreaGraceTime) / 4.F, 0.F, 1.F);
            R_OutOfArea = -(0.4F + 0.8F * persistence) * overflowRatio;
            rewards[static_cast<int>(StatesAndActionsForMission::RewardType::OutOfArea)] = R_OutOfArea;
        }
    }
    else
    {
        m_timeOutOfArea = 0.F;
    }

    ppo::model::BaseEntity targetEntity{ registry, targetId };
    if (targetEntity.has<ppo::model::sEntityState>())
    {
        auto const [targetState] = targetEntity.get<ppo::model::sEntityState>();

        float const tx = targetState.m_x - x;
        float const ty = targetState.m_y - y;
        d = std::sqrt(tx * tx + ty * ty);

        if (!m_stepMemoryInitialized)
        {
            m_prevTargetRange = d;
        }
        else if (!m_interceptReward && d < snaState.m_escapezone)
        {
            R_Intercept = kInterceptReward;
            m_interceptReward = true;
        }

        R_Range = std::clamp(kRangeRewardGain * (m_prevTargetRange - d), -2.5F, 2.5F);
        m_prevTargetRange = d;

        targetDestroyed = targetState.m_destroyed;
        targetDestroyedByOwnship =
            targetDestroyed && (targetState.m_destroyedById == entityState.m_entityId);
    }

    unsigned char nbEntitiesToBeDestroyed{ 0 };
    auto view = registry->view<ppo::model::sEntityState>();
    for (auto entity : view)
    {
        ppo::model::BaseEntity otherEntity{ registry, entity };

        if (!otherEntity.has<ppo::model::sEntityState>()
        ||  entity == m_entity.uniqid())
        {
            continue;
        }

        auto [otherEntityState] = otherEntity.get<ppo::model::sEntityState>();

        if (!otherEntityState.m_destroyed
        &&  otherEntityState.m_type != ppo::model::eEntityType::RestrictedZone
        &&  otherEntityState.m_side != ppo::model::eEntitySide::Unknown
        &&  otherEntityState.m_side != entityState.m_side)
        {
            ++nbEntitiesToBeDestroyed;
        }

        if (entity == snaState.m_target)
            continue;

        if (otherEntityState.m_destroyed
        && !otherEntityState.m_prevDestroyed
        &&  otherEntityState.m_destroyedById == m_entity.uniqid())
        {
            if (otherEntityState.m_side == entityState.m_side)
            {
                rewards[static_cast<int>(StatesAndActionsForMission::RewardType::WingmanDestruction)] +=
                    kWingmanDestructionReward;
            }
            else if (otherEntityState.m_side != ppo::model::eEntitySide::Unknown
                 &&  otherEntityState.m_type != ppo::model::eEntityType::RestrictedZone)
            {
                rewards[static_cast<int>(StatesAndActionsForMission::RewardType::SecondaryTargetDestruction)] +=
                    kSecondaryTargetDestructionReward;
            }
            else
            {
                rewards[static_cast<int>(StatesAndActionsForMission::RewardType::CivilianDestruction)] +=
                    kCivilianDestructionReward;
            }
        }
    }

    if (targetDestroyedByOwnship)
    {
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::MainTargetDestruction)] +=
            kMainTargetDestructionReward;
    }

    if (entityState.m_destroyed)
    {
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::OwnshipDestruction)] +=
            kOwnshipDestructionReward;
    }

    if (!entityState.m_destroyed && !targetDestroyed)
    {
        ppo::model::BaseEntity ownEntity{ registry, m_entity.uniqid() };
        sRestrictedZoneThreatAssessment const samThreat = AssessRestrictedZoneThreats(*registry, ownEntity, entityState);
        if (samThreat.m_hasThreat)
        {
            float const currentThreatMargin = samThreat.m_nearestDangerMargin;
            float approachPenalty{ 0.F };
            if (m_threatMarginInitialized)
                approachPenalty = std::max(m_prevThreatMargin - currentThreatMargin, 0.F);

            if (currentThreatMargin <= 0.F)
            {
                float const penetration = std::clamp(-currentThreatMargin, 0.F, 2.F);
                float const facingPenalty =
                    std::max(samThreat.m_forwardAlignmentToNearestThreat, 0.F);
                rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Danger)] =
                    kDangerBasePenalty
                    - 2.5F * penetration
                    - 4.F * samThreat.m_maxThreatPk
                    + kDangerFacingPenalty * facingPenalty;
            }
            else if (currentThreatMargin < kNormalizedCautionMargin)
            {
                float const proximity =
                    1.F - std::clamp(currentThreatMargin / kNormalizedCautionMargin, 0.F, 1.F);
                float const facingPenalty =
                    proximity * std::max(samThreat.m_forwardAlignmentToNearestThreat, 0.F);
                rewards[static_cast<int>(StatesAndActionsForMission::RewardType::NearDanger)] =
                    kNearDangerBasePenalty
                    - 0.75F * proximity
                    - 1.5F * approachPenalty
                    - 1.5F * samThreat.m_maxThreatPk
                    + kNearDangerFacingPenalty * facingPenalty;
            }

            m_prevThreatMargin = currentThreatMargin;
            m_threatMarginInitialized = true;
        }
        else
        {
            m_threatMarginInitialized = false;
            m_prevThreatMargin = 0.F;
        }

        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Intercept)] = R_Intercept;
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Range)] = R_Range;
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Fuel)] = R_fuel;
    }

    if (IsLastStep() && !targetDestroyed)
    {
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::MissionFailure)] =
            kMissionFailureReward;
    }

    for (auto i = 0; i < rewards.size(); ++i)
        AddReward(static_cast<StatesAndActionsForMission::RewardType>(i), rewards[i]);

    done = entityState.m_destroyed || targetDestroyed || (nbEntitiesToBeDestroyed == 0);
    m_stepMemoryInitialized = true;

    if (done)
    {
        m_interceptReward = false;
        m_stepMemoryInitialized = false;
        m_threatMarginInitialized = false;
        m_prevTargetRange = 0.F;
        m_timeOutOfArea = 0.F;
        m_prevThreatMargin = 0.F;
    }

    SetDone(done);
    return done;
}

/**
 * @brief Initialise ou rétablit l'état de TargetSelectorProdAgent selon le contexte courant.
 */
void
TargetSelectorProdAgent::Init()
{
    Base::Init();

    if (!m_entity.has<StatesAndActionsForMission>())
        return;

    auto [stateActions] =
        m_entity.get<StatesAndActionsForMission>();

    stateActions.Init();
}

}
