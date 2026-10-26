#pragma once
#include "entity.hpp"
#include "spawn.hpp"
#include <cstddef>
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
