/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_network.hpp
 * @brief Réseau de valeur centralisé 470-vers-1 utilisé par le MAPPO Blue.
 * @details Le réseau est décrit dans BluePatrol/graph.json (MLP de deux couches cachées
 * de 256 unités ReLU, sortie identité) et généré par PPOKit (ppokit_generate_networks).
 * L'acteur à attention reste écrit à la main ("model": "external").
 */
#pragma once

#include "generalized_critic_observation.hpp"
#include "patrol_rltools.hpp"

#include <rl_tools/nn_models/mlp/network.h>

// Generated from task.json + graph.json; requires the rl_tools headers included above.
#include "blue_patrol_networks.hpp"

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Dedicated 256-wide value network for generalized P/N training.
 *        The generalized actor remains 128-wide; checkpoint readers still
 *        enforce their own observation and guidance semantic versions.
 */
struct GeneralizedCriticConfig final
{
    using Description = task::networks::Critic;
    static_assert(Description::Inputs == GeneralizedCriticLayout::Size,
        "graph.json critic input and GeneralizedCriticLayout disagree");
    /** @brief Hidden width, written in the training checkpoint header. */
    static constexpr Index HiddenDim = static_cast<Index>(Description::DefaultWidth);
    /** @brief rl_tools layer count (input, hidden and output layers), written in the checkpoint header. */
    static constexpr Index Layers = static_cast<Index>(Description::HiddenLayers + 1);
    using Network = Description::Network<Index, TrainingParameters::BATCH_SIZE>;
};

} // namespace ppo::examples::bluedefense::defenders
