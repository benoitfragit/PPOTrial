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
        if (config.spawnSettings.blueTeam)
        {
            config.blueTeamRoles = defenders::BlueTeamRoles::Sample(
                config.spawnSettings.blueTeam->activeCount, config.rng);
            if (!config.blueTeamRoles)
            {
                config.patrolGeometrySamplingFailed = true;
                return;
            }
            config.leaderGuardIndex =
                config.blueTeamRoles->LeaderPhysicalIndex();
        }
        else
        {
            // Keep the archived Bernoulli draw byte-for-byte isolated from the
            // generalized uniform leader sampler.
            config.blueTeamRoles.reset();
            config.leaderGuardIndex =
                std::bernoulli_distribution{0.5}(config.rng) ? 0U : 1U;
        }
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
                  << " leader="
                  << (config.spawnSettings.blueTeam ? "blue_" : "blue_guard_")
                  << config.leaderGuardIndex + 1U
                  << '\n';
        if (config.evaluationOverride)
        {
            config.side = config.evaluationSide < 0 ? -1 : 1;
            config.arclength = defenders::PositiveModulo(
                config.evaluationArclength, defenders::CircuitLength(p));
            if (config.spawnSettings.blueTeam
                && !defenders::GeneralizedFormationFitsBounds(
                    config.patrolGeometry, config.patrolGeometrySettings,
                    constraints,
                    config.spawnSettings.blueTeam->activeCount,
                    config.arclength))
            {
                config.patrolGeometryReady = false;
                config.patrolGeometrySamplingFailed = true;
                config.patrolTrackOverlay =
                    ppo::model::sPatrolTrackOverlay{};
                config.episodeLayout.reset();
                std::cerr << "[BLUE PATROL] Evaluation phase places an active "
                          << "generalized blue outside patrol bounds.\n";
                return;
            }
        }
        else
        {
            config.side = std::bernoulli_distribution{0.5}(config.rng) ? 1 : -1;
            if (config.spawnSettings.blueTeam)
            {
                auto const arclength =
                    defenders::SampleGeneralizedFormationArclength(
                        config.patrolGeometry,
                        config.patrolGeometrySettings, constraints,
                        config.spawnSettings.blueTeam->activeCount,
                        config.rng);
                if (!arclength)
                {
                    config.patrolGeometryReady = false;
                    config.patrolGeometrySamplingFailed = true;
                    config.patrolTrackOverlay =
                        ppo::model::sPatrolTrackOverlay{};
                    config.episodeLayout.reset();
                    std::cerr << "[BLUE PATROL] Cannot sample a bounded "
                              << config.spawnSettings.blueTeam->activeCount
                              << "-mobile formation after "
                              << config.patrolGeometrySettings
                                     .maximumSamplingAttempts
                              << " attempts.\n";
                    return;
                }
                config.arclength = *arclength;
            }
            else
            {
                // Preserve the archived 2v2 RNG sequence exactly when the
                // optional generalized blue-team configuration is absent.
                config.arclength = std::uniform_real_distribution<float>{
                    0.F, defenders::CircuitLength(p)}(config.rng);
            }
        }
        auto spawnSettings = config.spawnSettings;
        if (config.evaluationOverride && !spawnSettings.redTeam)
        {
            // Evaluations retain their deterministic immediate 2v1/2v2 cases;
            // stochastic activation schedules belong to training only. This
            // branch is intentionally limited to the archived contract.
            spawnSettings.threatCurriculum.enabled = false;
            spawnSettings.red2DestroyedProbability =
                config.evaluationSecondAttackerDestroyed ? 1.F : 0.F;
        }
        std::string spawnDiagnostic;
        config.episodeLayout = GenerateEpisodeSpawnLayout(
            config.side, spawnSettings, bounds, config.rng,
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
    auto const generalizedBlue = config.spawnSettings.blueTeam
        ? GeneralizedBlueIndex(metadata.m_name) : std::nullopt;
    if (generalizedBlue
        && *generalizedBlue < config.spawnSettings.blueTeam->activeCount)
    {
        pose = Spawn{};
    }
    bool const legacyBlue = metadata.m_name == "blue_guard_1"
        || metadata.m_name == "blue_guard_2";
    if (pose && (legacyBlue || generalizedBlue.has_value()))
    {
        defenders::Parameters p{};
        p.halfNorth = bounds.halfNorth;
        p.halfEast = bounds.halfEast;
        p.cruiseSpeed = config.patrolCruiseSpeed;
        p.maxRotationRate = config.patrolMaximumRotationRate;
        p.collisionDistance = config.patrolCollisionDistance;
        p.geometry = config.patrolGeometry;
        std::size_t const guardIndex = generalizedBlue.value_or(
            metadata.m_name == "blue_guard_1" ? 0U : 1U);
        auto const leader = defenders::PatrolReference(
            p, config.side, config.arclength);
        defenders::Reference reference = leader;
        if (generalizedBlue)
        {
            if (!config.blueTeamRoles)
                return;
            auto const assignment = config.blueTeamRoles->Assignment(guardIndex);
            if (!assignment)
                return;
            if (assignment->role == defenders::BlueRole::Teammate)
            {
                defenders::PatrolCurvePose const leaderPose{
                    leader.north, leader.east, leader.heading,
                    config.arclength, leader.phase, 0.F};
                defenders::PatrolCurvePose const formationPose =
                    defenders::EvaluateFormationPose(
                        p.geometry, leaderPose,
                        assignment->formationSlot);
                reference.north = formationPose.north;
                reference.east = formationPose.east;
            }
        }
        else if (config.RoleForGuard(guardIndex) != 0U)
        {
            reference = defenders::WingmanReference(p, leader);
        }
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
    if (config.next == 0U)
        ppo::model::UpdateRestrictedZonePseudoLidars(
            *registry, ppo::model::eEntitySide::Friend);
}
} // namespace ppo::examples::bluedefense
