/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_network.hpp
 * @brief Implémentation des calculs et contrats d'entraînement generalized critic network.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "generalized_critic_observation.hpp"
#include "patrol_rltools.hpp"

#include <rl_tools/nn_models/mlp/network.h>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Dedicated 256-wide value network for generalized P/N training.
 *        Historical 128-wide v7 actor/critic remain unmodified. The generalized
 *        actor also remains 128-wide so saved v8 actor weights can be migrated.
 */
struct GeneralizedCriticConfig final
{
    static constexpr Index HiddenDim = 256;
    static constexpr Index Layers = 3;
    using Structure = rlt::nn_models::mlp::StructureSpecification<
        float, Index, static_cast<Index>(GeneralizedCriticLayout::Size),
        1, Layers, HiddenDim,
        rlt::nn::activation_functions::ActivationFunction::RELU,
        rlt::nn::activation_functions::IDENTITY,
        TrainingParameters::BATCH_SIZE>;
    using Spec = rlt::nn_models::mlp::AdamSpecification<Structure>;
    using Network = rlt::nn_models::mlp::NeuralNetworkAdam<Spec>;
};

} // namespace ppo::examples::bluedefense::defenders
