#pragma once
#include "entity.hpp"
#include "patrol_geometry.hpp"
#include "sharedworldcomponents.hpp"
#include "spawn.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

namespace ppo::examples::twovstwo {
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
    /** @brief Configured initial/reference speed of both blue guards. */
    float patrolCruiseSpeed{300.F};
    /** @brief Maximum blue yaw rate used to reject infeasible turns. */
    float patrolMaximumRotationRate{0.7000006F};
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
    /** @brief Monotonic episode number used by diagnostics. */
    std::uint64_t episodeIndex{};
    float arclength{0.F};
    bool evaluationOverride{false};
    int evaluationSide{1};
    float evaluationArclength{0.F};
    bool evaluationSecondAttackerDestroyed{false};
};

/**
 * @brief Initializes one scenario entity for a random training episode or a
 * deterministic evaluation case.
 * @param entity Entity whose ECS state is reset.
 */
void InitializeEntity(ppo::model::Entity& entity);
}
