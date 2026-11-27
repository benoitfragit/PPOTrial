/**
 * @file Framework/RL/Core/include/ppo/core/environment.hpp
 * @brief Observation, action and policy values shared by the agents of the ECS world.
 */
#pragma once

#include <vector>

namespace ppo::core {

// Domain-independent, owned observation/action values. No tensor or ECS lifetime escapes.
using Observation = std::vector<float>;
using Action = std::vector<float>;

/**
 * @brief A policy: the action for an observation (production agents of the ECS world).
 */
class IPolicy {
public:
    virtual ~IPolicy() = default;
    /**
     * @brief The action for `observation`.
     * @param observation The agent's observation.
     * @return The action.
     */
    virtual Action ChooseAction(Observation const& observation) = 0;
};

} // namespace ppo::core
