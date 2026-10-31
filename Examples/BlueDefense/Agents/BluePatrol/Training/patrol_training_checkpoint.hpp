/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/patrol_training_checkpoint.hpp
 * @brief Implémentation des calculs et contrats d'entraînement patrol training checkpoint.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "patrol_checkpoint.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ppo::examples::bluedefense::defenders {

/** @brief Version of the complete blue MAPPO training-state contract. */
inline constexpr unsigned kPatrolTrainingCheckpointVersion = 4U;

namespace training_checkpoint_detail {

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
inline void RequirePatrolCheckpointToken(std::istream& stream,
                                         char const* expected)
{
    std::string token;
    if (!(stream >> token) || token != expected)
        throw std::runtime_error(
            std::string{"Invalid blue MAPPO training section: "} + expected);
}

template<typename PARAMETER>
/**
 * @brief Sérialise ou écrit les données associées à WritePatrolMoments selon les garanties implémentées.
 */
void WritePatrolMoments(std::ostream& stream, PARAMETER const& parameter)
{
    WritePatrolMatrix(stream, parameter.gradient_first_order_moment);
    WritePatrolMatrix(stream, parameter.gradient_second_order_moment);
}

template<typename PARAMETER>
void ReadPatrolMoments(std::istream& stream, PARAMETER& parameter)
{
    ReadPatrolMatrix(stream, parameter.gradient_first_order_moment);
    ReadPatrolMatrix(stream, parameter.gradient_second_order_moment);
}

template<typename LAYER>
/**
 * @brief Sérialise ou écrit les données associées à WritePatrolLayerMoments selon les garanties implémentées.
 */
void WritePatrolLayerMoments(std::ostream& stream, LAYER const& layer)
{
    WritePatrolMoments(stream, layer.weights);
    WritePatrolMoments(stream, layer.biases);
}

template<typename LAYER>
void ReadPatrolLayerMoments(std::istream& stream, LAYER& layer)
{
    ReadPatrolMoments(stream, layer.weights);
    ReadPatrolMoments(stream, layer.biases);
}

template<typename ATTENTION>
/**
 * @brief Sérialise ou écrit les données associées à WritePatrolAttentionMoments selon les garanties implémentées.
 */
void WritePatrolAttentionMoments(std::ostream& stream,
                                 ATTENTION const& attention)
{
    WritePatrolMoments(stream, attention.encoderW);
    WritePatrolMoments(stream, attention.encoderB);
    WritePatrolMoments(stream, attention.queryW);
    WritePatrolMoments(stream, attention.queryB);
}

template<typename ATTENTION>
void ReadPatrolAttentionMoments(std::istream& stream, ATTENTION& attention)
{
    ReadPatrolMoments(stream, attention.encoderW);
    ReadPatrolMoments(stream, attention.encoderB);
    ReadPatrolMoments(stream, attention.queryW);
    ReadPatrolMoments(stream, attention.queryB);
}

template<typename NETWORK>
/**
 * @brief Sérialise ou écrit les données associées à WritePatrolMlpState selon les garanties implémentées.
 */
void WritePatrolMlpState(std::ostream& stream, NETWORK const& network)
{
    WritePatrolLayer(stream, network.input_layer);
    WritePatrolLayerMoments(stream, network.input_layer);
    for (auto const& layer : network.hidden_layers)
    {
        WritePatrolLayer(stream, layer);
        WritePatrolLayerMoments(stream, layer);
    }
    WritePatrolLayer(stream, network.output_layer);
    WritePatrolLayerMoments(stream, network.output_layer);
}

template<typename NETWORK>
/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées.
 * @details Peut signaler un état invalide par exception.
 */
void ReadPatrolMlpState(std::istream& stream, NETWORK& network)
{
    ReadPatrolLayer(stream, network.input_layer);
    ReadPatrolLayerMoments(stream, network.input_layer);
    for (auto& layer : network.hidden_layers)
    {
        ReadPatrolLayer(stream, layer);
        ReadPatrolLayerMoments(stream, layer);
    }
    ReadPatrolLayer(stream, network.output_layer);
    ReadPatrolLayerMoments(stream, network.output_layer);
}

template<typename OPTIMIZER>
/**
 * @brief Sérialise ou écrit les données associées à WritePatrolOptimizer selon les garanties implémentées.
 * @details Peut signaler un état invalide par exception.
 */
void WritePatrolOptimizer(std::ostream& stream, OPTIMIZER const& optimizer)
{
    auto const& p = optimizer.parameters;
    stream << p.alpha << ' ' << p.beta_1 << ' ' << p.beta_2 << ' '
           << p.epsilon << ' ' << p.weight_decay << ' '
           << p.weight_decay_input << ' ' << p.weight_decay_output << ' '
           << p.bias_lr_factor << ' ' << optimizer.age << '\n';
}

template<typename OPTIMIZER>
void ReadPatrolOptimizer(std::istream& stream, OPTIMIZER& optimizer)
{
    auto parameters = optimizer.parameters;
    typename OPTIMIZER::TI age{};
    if (!(stream >> parameters.alpha >> parameters.beta_1 >> parameters.beta_2
          >> parameters.epsilon >> parameters.weight_decay
          >> parameters.weight_decay_input >> parameters.weight_decay_output
          >> parameters.bias_lr_factor >> age))
        throw std::runtime_error("Invalid blue MAPPO optimizer state");
    auto const finite = [](auto value) {
        return std::isfinite(static_cast<double>(value));
    };
    if (!finite(parameters.alpha) || parameters.alpha <= 0
        || !finite(parameters.beta_1) || parameters.beta_1 < 0
        || parameters.beta_1 >= 1 || !finite(parameters.beta_2)
        || parameters.beta_2 < 0 || parameters.beta_2 >= 1
        || !finite(parameters.epsilon) || parameters.epsilon <= 0
        || !finite(parameters.weight_decay) || parameters.weight_decay < 0
        || !finite(parameters.weight_decay_input)
        || parameters.weight_decay_input < 0
        || !finite(parameters.weight_decay_output)
        || parameters.weight_decay_output < 0
        || !finite(parameters.bias_lr_factor)
        || parameters.bias_lr_factor <= 0 || age == 0)
        throw std::runtime_error("Nonfinite blue MAPPO optimizer state");
    optimizer.parameters = parameters;
    optimizer.age = age;
}

} // namespace training_checkpoint_detail

/**
 * @brief Serializes a complete update-boundary training state.
 *
 * Rollout samples and a pending ECS transition are deliberately excluded. A
 * loaded learner starts a fresh rollout with the restored optimization state.
 *
 * @tparam ACTOR RLTools shared Actor type.
 * @tparam CRITIC RLTools centralized Critic type.
 * @tparam ACTOR_OPTIMIZER RLTools Actor Adam type.
 * @tparam CRITIC_OPTIMIZER RLTools Critic Adam type.
 * @tparam RNG RLTools random-engine type.
 * @param stream Destination stream.
 * @param actor Shared Actor, including Adam moments.
 * @param critic Centralized Critic, including Adam moments.
 * @param actorOptimizer Actor optimizer scalar state.
 * @param criticOptimizer Critic optimizer scalar state.
 * @param rng RLTools sampling engine.
 * @param updates Number of completed PPO updates.
 */
template<typename ACTOR, typename CRITIC, typename ACTOR_OPTIMIZER,
         typename CRITIC_OPTIMIZER, typename RNG>
/**
 * @brief Sérialise ou écrit les données associées à WritePatrolTrainingState selon les garanties implémentées.
 */
void WritePatrolTrainingState(std::ostream& stream,
                              ACTOR const& actor,
                              CRITIC const& critic,
                              ACTOR_OPTIMIZER const& actorOptimizer,
                              CRITIC_OPTIMIZER const& criticOptimizer,
                              RNG const& rng,
                              std::uint64_t updates)
{
    stream << "BLUE_PATROL_TRAINING_V4 "
           << kPatrolTrainingCheckpointVersion << ' '
           << ObservationLayout::Version << ' '
           << ObservationLayout::Size << ' '
           << ObservationLayout::ActionSize << ' '
           << TrainingParameters::ACTOR_HIDDEN_DIM << ' '
           << TrainingParameters::ACTOR_NUM_LAYERS << ' '
           << TrainingParameters::CRITIC_HIDDEN_DIM << ' '
           << TrainingParameters::CRITIC_NUM_LAYERS << '\n'
           << std::setprecision(std::numeric_limits<float>::max_digits10);

    stream << "ACTOR\n";
    WritePatrolPolicy(stream, actor);
    stream << "ACTOR_MOMENTS\n";
    training_checkpoint_detail::WritePatrolAttentionMoments(stream, actor.red);
    training_checkpoint_detail::WritePatrolAttentionMoments(
        stream, actor.friendlies);
    training_checkpoint_detail::WritePatrolLayerMoments(
        stream, actor.input_layer);
    for (auto const& layer : actor.hidden_layers)
        training_checkpoint_detail::WritePatrolLayerMoments(stream, layer);
    training_checkpoint_detail::WritePatrolLayerMoments(
        stream, actor.output_layer);
    training_checkpoint_detail::WritePatrolMoments(stream, actor.log_std);

    stream << "CRITIC\n";
    training_checkpoint_detail::WritePatrolMlpState(stream, critic);
    stream << "OPTIMIZERS\n";
    training_checkpoint_detail::WritePatrolOptimizer(stream, actorOptimizer);
    training_checkpoint_detail::WritePatrolOptimizer(stream, criticOptimizer);
    stream << "UPDATES " << updates << "\nRNG\n" << rng
           << "\nEND_BLUE_PATROL_TRAINING\n";
    if (!stream)
        throw std::runtime_error("Failed to serialize blue MAPPO training state");
}

/**
 * @brief Restores a complete update-boundary training state.
 * @tparam ACTOR RLTools shared Actor type.
 * @tparam CRITIC RLTools centralized Critic type.
 * @tparam ACTOR_OPTIMIZER RLTools Actor Adam type.
 * @tparam CRITIC_OPTIMIZER RLTools Critic Adam type.
 * @tparam RNG RLTools random-engine type.
 * @param stream Source stream.
 * @param actor Shared Actor to restore.
 * @param critic Centralized Critic to restore.
 * @param actorOptimizer Actor optimizer to restore.
 * @param criticOptimizer Critic optimizer to restore.
 * @param rng RLTools sampling engine to restore.
 * @return Number of completed PPO updates stored in the checkpoint.
 */
template<typename ACTOR, typename CRITIC, typename ACTOR_OPTIMIZER,
         typename CRITIC_OPTIMIZER, typename RNG>
std::uint64_t ReadPatrolTrainingState(std::istream& stream,
                                     ACTOR& actor,
                                     CRITIC& critic,
                                     ACTOR_OPTIMIZER& actorOptimizer,
                                     CRITIC_OPTIMIZER& criticOptimizer,
                                     RNG& rng)
{
    std::string magic;
    unsigned formatVersion = 0U, observationVersion = 0U;
    unsigned input = 0U, output = 0U, actorHidden = 0U, actorLayers = 0U;
    unsigned criticHidden = 0U, criticLayers = 0U;
    if (!(stream >> magic >> formatVersion >> observationVersion
          >> input >> output >> actorHidden >> actorLayers
          >> criticHidden >> criticLayers)
        || magic != "BLUE_PATROL_TRAINING_V4"
        || formatVersion != kPatrolTrainingCheckpointVersion
        || observationVersion != ObservationLayout::Version
        || input != ObservationLayout::Size
        || output != ObservationLayout::ActionSize
        || actorHidden != TrainingParameters::ACTOR_HIDDEN_DIM
        || actorLayers != TrainingParameters::ACTOR_NUM_LAYERS
        || criticHidden != TrainingParameters::CRITIC_HIDDEN_DIM
        || criticLayers != TrainingParameters::CRITIC_NUM_LAYERS)
        throw std::runtime_error(
            "Incompatible blue MAPPO training checkpoint contract");

    training_checkpoint_detail::RequirePatrolCheckpointToken(stream, "ACTOR");
    ReadPatrolPolicyParameters(stream, actor);
    training_checkpoint_detail::RequirePatrolCheckpointToken(
        stream, "ACTOR_MOMENTS");
    training_checkpoint_detail::ReadPatrolAttentionMoments(stream, actor.red);
    training_checkpoint_detail::ReadPatrolAttentionMoments(
        stream, actor.friendlies);
    training_checkpoint_detail::ReadPatrolLayerMoments(
        stream, actor.input_layer);
    for (auto& layer : actor.hidden_layers)
        training_checkpoint_detail::ReadPatrolLayerMoments(stream, layer);
    training_checkpoint_detail::ReadPatrolLayerMoments(
        stream, actor.output_layer);
    training_checkpoint_detail::ReadPatrolMoments(stream, actor.log_std);

    training_checkpoint_detail::RequirePatrolCheckpointToken(stream, "CRITIC");
    training_checkpoint_detail::ReadPatrolMlpState(stream, critic);
    training_checkpoint_detail::RequirePatrolCheckpointToken(
        stream, "OPTIMIZERS");
    training_checkpoint_detail::ReadPatrolOptimizer(stream, actorOptimizer);
    training_checkpoint_detail::ReadPatrolOptimizer(stream, criticOptimizer);
    training_checkpoint_detail::RequirePatrolCheckpointToken(stream, "UPDATES");
    std::uint64_t updates = 0U;
    if (!(stream >> updates))
        throw std::runtime_error("Invalid blue MAPPO update count");
    training_checkpoint_detail::RequirePatrolCheckpointToken(stream, "RNG");
    // Some standard engines clear skipws while reading their state.
    if (!(stream >> std::ws >> rng))
        throw std::runtime_error("Invalid blue MAPPO random-engine state");
    training_checkpoint_detail::RequirePatrolCheckpointToken(
        stream, "END_BLUE_PATROL_TRAINING");
    std::string unexpected;
    if (stream >> unexpected)
        throw std::runtime_error("Trailing data in blue MAPPO training checkpoint");
    return updates;
}

} // namespace ppo::examples::bluedefense::defenders
