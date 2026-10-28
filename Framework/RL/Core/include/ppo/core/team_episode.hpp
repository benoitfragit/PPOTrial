#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ppo/core/environment.hpp"

namespace ppo::core {

// A domain-independent contract for parameter-sharing PPO and future MAPPO.
// Actors consume only their local observation. A centralized critic may consume
// the separate global state without changing an actor or the environment.
template<std::size_t N>
struct TeamStep
{
    static_assert(N > 0, "An episode must contain at least one agent");
    std::array<Observation, N> observations{};
    std::array<float, N> individualRewards{};
    float teamReward = 0.F;
    bool terminated = false;
    bool truncated = false;
    bool Done() const noexcept { return terminated || truncated; }
};

template<std::size_t N>
class ITeamEnvironment
{
public:
    virtual ~ITeamEnvironment() = default;
    virtual TeamStep<N> Reset() = 0;
    virtual TeamStep<N> Step(std::array<Action, N> const& actions) = 0;
    virtual std::size_t LocalObservationSize() const noexcept = 0;
    virtual std::size_t LocalActionSize() const noexcept = 0;
    // Reserved for a centralized critic; local PPO can ignore it.
    virtual Observation GlobalState() const = 0;
};

template<std::size_t N>
void ValidateTeamStep(ITeamEnvironment<N> const& environment,
                      TeamStep<N> const& step)
{
    for (auto const& obs : step.observations)
        if (obs.size() != environment.LocalObservationSize())
            throw std::invalid_argument("Team environment returned invalid observation size");
}
} // namespace ppo::core
