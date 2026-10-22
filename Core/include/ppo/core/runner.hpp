#pragma once

#include "ppo/core/environment.hpp"

#include <stdexcept>
#include <utility>

namespace ppo::core {

// Single-agent adapter for future environments. It does not alter PPO's existing
// terminal/bootstrap semantics. An adapter defines what its domain 'done' means.
class EpisodeRunner {
public:
    // Inference-only. Passing an ILearner instead selects the overload below,
    // ensuring feedback cannot be accidentally omitted from training.
    EpisodeRunner(IEnvironment& environment, IPolicy& policy)
        : environment_(environment), policy_(policy) {}

    // Acting policy and learner are the same object: no mismatched transitions.
    EpisodeRunner(IEnvironment& environment, ILearner& learner)
        : environment_(environment), policy_(learner), learner_(&learner) {}

    Observation const& Reset() {
        Observation next = environment_.Reset();
        ValidateObservation(next);
        observation_ = std::move(next);
        initialized_ = true;
        finished_ = false;
        return observation_;
    }

    Transition const& Advance() {
        if (!initialized_ || finished_)
            throw std::logic_error("EpisodeRunner: reset required before advance");
        // Any exception after entering policy selection may leave learner state
        // inconsistent; requiring Reset prevents reusing a stale observation.
        initialized_ = false;
        Action action = policy_.ChooseAction(observation_);
        if (action.size() != environment_.ActionSize())
            throw std::invalid_argument("EpisodeRunner: invalid action size");
        // A failing step may also have mutated the world.
        Transition next = environment_.Step(action);
        ValidateObservation(next.observation);
        transition_ = std::move(next);
        observation_ = transition_.observation;
        finished_ = transition_.done;
        if (learner_ != nullptr)
            learner_->Observe(transition_);
        initialized_ = true;
        return transition_;
    }

    bool Finished() const noexcept { return finished_; }
    Observation const& CurrentObservation() const noexcept { return observation_; }

private:
    void ValidateObservation(Observation const& value) const {
        if (value.size() != environment_.ObservationSize())
            throw std::invalid_argument("EpisodeRunner: invalid observation size");
    }

    IEnvironment& environment_;
    IPolicy& policy_;
    ILearner* learner_ = nullptr;
    Observation observation_;
    Transition transition_;
    bool initialized_ = false;
    bool finished_ = true;
};

// One shared sequencing primitive for ECS training and evaluation loops.
// All decisions must precede world advancement; feedback follows outcomes.
// Callers retain ownership of entities, rules, and episode-reset semantics.
template<class Choose, class Advance, class Resolve, class Feedback>
void StepWorld(Choose&& choose, Advance&& advance, Resolve&& resolve, Feedback&& feedback)
{
    std::forward<Choose>(choose)();
    std::forward<Advance>(advance)();
    std::forward<Resolve>(resolve)();
    std::forward<Feedback>(feedback)();
}

} // namespace ppo::core
