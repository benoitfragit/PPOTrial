#pragma once

#include <cstddef>
#include <vector>

namespace ppo::core {

// Domain-independent, owned observation/action values. No tensor or ECS lifetime escapes.
using Observation = std::vector<float>;
using Action = std::vector<float>;

struct Transition {
    Observation observation;
    float reward = 0.F;
    bool done = false;
};

// Reset and Step are executed by the caller on a single thread; no implicit reset.
class IEnvironment {
public:
    virtual ~IEnvironment() = default;
    virtual std::size_t ObservationSize() const noexcept = 0;
    virtual std::size_t ActionSize() const noexcept = 0;
    virtual Observation Reset() = 0;
    virtual Transition Step(Action const& action) = 0;
};

class IPolicy {
public:
    virtual ~IPolicy() = default;
    virtual Action ChooseAction(Observation const& observation) = 0;
};

// Optional observer; no assumptions about the optimization algorithm or rewards.
class ILearner : public IPolicy {
public:
    virtual void Observe(Transition const& transition) = 0;
};

} // namespace ppo::core
