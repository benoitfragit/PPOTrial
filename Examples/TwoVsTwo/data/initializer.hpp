#pragma once
#include "entity.hpp"
#include <cstddef>
#include <random>

namespace ppo::examples::twovstwo {
struct InitializerContext {
    std::mt19937 rng{std::random_device{}()};
    std::size_t count{};
    std::size_t next{};
    int side{1};
    float red2DestroyedProbability{0.15F};
    bool patrolOnly{false};
    bool policyReady{false};
    float arclength{0.F};
};
void InitializeEntity(ppo::model::Entity& entity);
}
