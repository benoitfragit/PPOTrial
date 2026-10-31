/**
 * @file Framework/RL/PPO/src/torchlearner.cpp
 * @brief Interface ou implémentation du composant torchlearner.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "torchlearner.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ppo::model {

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
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

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
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
    : observation_(static_cast<std::size_t>(CheckedDimension(observation_size)), 0.F),
      action_(static_cast<std::size_t>(CheckedDimension(action_size)), 0.F),
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
/**
 * @brief Réalise l'opération ChooseAction de TorchLearner; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
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

/**
 * @brief Initialise ou rétablit l'état de TorchLearner selon le contexte courant.
 * @details Utilise des fichiers ou chemins.
 */
void TorchLearner::ResetRollout() { agent_.Init(); pending_action_ = false; }
/**
 * @brief Réalise l'opération RolloutComplete de TorchLearner; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
bool TorchLearner::RolloutComplete() const noexcept { return !agent_.IsTraining(); }
/**
 * @brief Réalise l'opération RolloutStep de TorchLearner; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
long long TorchLearner::RolloutStep() const noexcept { return agent_.GetCurrentStep(); }
/**
 * @brief Réalise l'opération Outputs de TorchLearner; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
PPOOutputs const& TorchLearner::Outputs() const noexcept { return agent_.GetOutputs(); }
/**
 * @brief Sérialise ou écrit les données associées à Save selon les garanties implémentées.
 */
void TorchLearner::Save(std::filesystem::path const& path) { agent_.Save(path); }

} // namespace ppo::model
