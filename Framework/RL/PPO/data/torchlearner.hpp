/**
 * @file Framework/RL/PPO/data/torchlearner.hpp
 * @brief Interface ou implémentation du composant torchlearner.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "ppo/core/environment.hpp"
#include "rlagent.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ppo::model {

// Generic policy adapter: any ITrainingNet works with any ppo::core::IEnvironment
// exposing the matching flattened float observation and action dimensions.
// The adapter owns fixed-size buffers for RLAgent's torch::from_blob view.
// It is deliberately non-movable to preserve tensor aliases.
/** @brief Déclare le type TorchLearner, utilisé par Interface ou implémentation du composant torchlearner. */
class TorchLearner final : public ppo::core::ILearner
{
public:
    TorchLearner(std::shared_ptr<ITrainingNet> policy,
                 std::string const& modelpath,
                 std::size_t observation_size,
                 std::size_t action_size,
                 std::int64_t rollout_size,
                 PPOParamters const& parameters,
                 bool load_existing_model = true);

    TorchLearner(TorchLearner const&) = delete;
    TorchLearner& operator=(TorchLearner const&) = delete;
    TorchLearner(TorchLearner&&) = delete;
    TorchLearner& operator=(TorchLearner&&) = delete;

/**
 * @brief Réalise l'opération ChooseAction; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override;
/**
 * @brief Réalise l'opération Observe; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    void Observe(ppo::core::Transition const& transition) override;

    // Reset only the rollout collector, not the environment or policy weights.
    // Call at the same boundaries as the legacy RLAgent::Init().
/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 * @details Utilise des fichiers ou chemins.
 */
    void ResetRollout();
/**
 * @brief Réalise l'opération RolloutComplete; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    bool RolloutComplete() const noexcept;
/**
 * @brief Réalise l'opération RolloutStep; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    long long RolloutStep() const noexcept;
/**
 * @brief Réalise l'opération RolloutSize; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    long long RolloutSize() const noexcept { return agent_.GetMaxStep(); }
/**
 * @brief Réalise l'opération Outputs; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    PPOOutputs const& Outputs() const noexcept;
/**
 * @brief Sérialise ou écrit les données associées à Save selon les garanties implémentées.
 */
    void Save(std::filesystem::path const& path);

private:
    std::vector<float> observation_;
    std::vector<float> action_;
    RLAgent agent_;
    bool pending_action_ = false;
};

} // namespace ppo::model
