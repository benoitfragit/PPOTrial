#pragma once
#include "entity.hpp"
#include "blue_team_roles.hpp"
#include "patrol_geometry.hpp"
#include "sharedworldcomponents.hpp"
#include "spawn.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

namespace ppo::examples::bluedefense {
/**
 * @brief Mutable episode-initialization state shared by the scenario loader.
 */
struct InitializerContext {
    std::mt19937 rng{std::random_device{}()};
    std::size_t count{};
    std::size_t next{};
    /** @brief Number of ECS team steps already resolved in this episode. */
    std::size_t elapsedSteps{};
    int side{1};
    SpawnSettings spawnSettings{};
    std::optional<EpisodeSpawnLayout> episodeLayout{};
    bool patrolOnly{false};
    /** @brief Configured initial/reference speed of P blue mobiles. */
    float patrolCruiseSpeed{700.F};
    /** @brief Maximum blue yaw rate used to reject infeasible turns. */
    float patrolMaximumRotationRate{0.70000059604645F};
    /** @brief Blue collision radius used to validate the sampled wedge. */
    float patrolCollisionDistance{1000.F};
    /** @brief Configured family sampled exactly once per episode. */
    defenders::PatrolGeometrySettings patrolGeometrySettings{};
    /** @brief Immutable continuous racetrack used by the current episode. */
    defenders::PatrolGeometry patrolGeometry{};
    /** @brief Snapshot-only rendering samples generated once per episode. */
    ppo::model::sPatrolTrackOverlay patrolTrackOverlay{};
    /** @brief True after a feasible geometry has been sampled for the reset. */
    bool patrolGeometryReady{false};
    /** @brief Signals an exhausted bounded sampling attempt sequence. */
    bool patrolGeometrySamplingFailed{false};
    /** @brief Signals that bounded red-team placement failed for the episode. */
    bool episodeLayoutGenerationFailed{false};
    /** @brief Monotonic episode number used by diagnostics. */
    std::uint64_t episodeIndex{};
    /** @brief Physical blue-guard index assigned leader role for this episode. */
    std::size_t leaderGuardIndex{};
    /** @brief Exactly one sampled leader and P-1 immutable formation roles. */
    std::optional<defenders::BlueTeamRoles> blueTeamRoles{};
    float arclength{0.F};
    bool evaluationOverride{false};
    int evaluationSide{1};
    /** @brief Preferred fractional phase of the newly generated racetrack. */
    float evaluationPhaseFraction{0.F};

    /** @brief Lets a fresh ECS reset recover after a failed evaluation spawn. */
    void ClearEpisodeInitializationFailures() noexcept
    {
        patrolGeometrySamplingFailed = false;
        episodeLayoutGenerationFailed = false;
    }
};

/**
 * @brief Initializes one scenario entity for a random training episode or a
 * deterministic evaluation case.
 * @param entity Entity whose ECS state is reset.
 */
void InitializeEntity(ppo::model::Entity& entity);
}
