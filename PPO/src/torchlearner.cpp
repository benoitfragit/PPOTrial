#include "torchlearner.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ppo::model {

namespace {
long long CheckedDimension(std::size_t value)
{
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<long long>::max()))
        throw std::invalid_argument("TorchLearner: dimensions must fit a positive int64");
    return static_cast<long long>(value);
}

std::shared_ptr<ITrainingNet> CheckedPolicy(std::shared_ptr<ITrainingNet> policy)
{
    if (!policy)
        throw std::invalid_argument("TorchLearner: null training policy");
    return policy;
}

long long CheckedRollout(std::int64_t value)
{
    if (value <= 0)
        throw std::invalid_argument("TorchLearner: rollout size must be positive");
    return static_cast<long long>(value);
}
} // namespace

TorchLearner::TorchLearner(
    std::shared_ptr<ITrainingNet> policy,
    std::string const& modelpath,
    std::size_t observation_size,
    std::size_t action_size,
    std::int64_t rollout_size,
    PPOParamters const& parameters,
    bool load_existing_model)
    : observation_(observation_size, 0.F),
      action_(action_size, 0.F),
      agent_(CheckedPolicy(std::move(policy)),
             modelpath,
             CheckedDimension(observation_size),
             CheckedDimension(action_size),
             CheckedRollout(rollout_size),
             parameters,
             observation_.data(),
             action_.data(),
             load_existing_model)
{
    agent_.Init();
}

ppo::core::Action
TorchLearner::ChooseAction(ppo::core::Observation const& observation)
{
    if (observation.size() != observation_.size())
        throw std::invalid_argument("TorchLearner: observation dimension mismatch");
    if (RolloutComplete())
        throw std::logic_error("TorchLearner: reset rollout before collecting next action");
    if (pending_action_)
        throw std::logic_error("TorchLearner: previous action has no transition");

    std::copy(observation.begin(), observation.end(), observation_.begin());
    agent_.ChooseAction();
    pending_action_ = true;
    return action_;
}

void
TorchLearner::Observe(ppo::core::Transition const& transition)
{
    if (transition.observation.size() != observation_.size())
        throw std::invalid_argument("TorchLearner: next observation dimension mismatch");
    if (RolloutComplete())
        throw std::logic_error("TorchLearner: rollout has already completed");
    if (!pending_action_)
        throw std::logic_error("TorchLearner: no action to observe");

    // Preserve RLAgent's next-state bootstrap input and original done masking.
    std::copy(transition.observation.begin(), transition.observation.end(), observation_.begin());
    agent_.SetNewState(transition.reward, transition.done);
    pending_action_ = false;
}

void TorchLearner::ResetRollout() { agent_.Init(); pending_action_ = false; }
bool TorchLearner::RolloutComplete() const noexcept { return !agent_.IsTraining(); }
long long TorchLearner::RolloutStep() const noexcept { return agent_.GetCurrentStep(); }
PPOOutputs const& TorchLearner::Outputs() const noexcept { return agent_.GetOutputs(); }
void TorchLearner::Save(std::filesystem::path const& path) { agent_.Save(path); }

} // namespace ppo::model
