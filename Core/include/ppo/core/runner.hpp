#pragma once

#include "ppo/core/environment.hpp"

#include <stdexcept>
#include <utility>

namespace ppo::core {

// Single-agent adapter for future environments. It does not alter PPO's existing
// terminal/bootstrap semantics. An adapter defines what its domain 'done' means.
class EpisodeRunner {
public:
    EpisodeRunner(IEnvironment& environment, IPolicy& policy, ILearner* learner = nullptr)
        : environment_(environment), policy_(policy), learner_(learner) {}

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
        Action action = policy_.ChooseAction(observation_);
        if (action.size() != environment_.ActionSize())
            throw std::invalid_argument("EpisodeRunner: invalid action size");
        Transition next = environment_.Step(action);
        ValidateObservation(next.observation);
        transition_ = std::move(next);
        observation_ = transition_.observation;
        finished_ = transition_.done;
        if (learner_ != nullptr)
            learner_->Observe(transition_);
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
    ILearner* learner_;
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
