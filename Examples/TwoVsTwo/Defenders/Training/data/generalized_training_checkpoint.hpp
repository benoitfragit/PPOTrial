#pragma once

#include "generalized_checkpoint.hpp"
#include "patrol_training_checkpoint.hpp"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ppo::examples::twovstwo::defenders {

/** @brief Version of the complete generalized P/N training-state contract. */
inline constexpr unsigned kGeneralizedTrainingCheckpointVersion = 5U;

/**
 * @brief Explicit scope of a legacy training-state migration.
 *
 * The v4 critic encoded a fixed 2v2 state, so its parameters, Adam moments and
 * optimizer state are incompatible with the generalized critic. Rollout data,
 * update count and RNG are likewise intentionally restarted.
 */
struct GeneralizedTrainingMigrationReport final
{
    bool actorParametersMigrated{true};
    bool compatibleActorMomentsMigrated{true};
    bool criticResetRequired{true};
    bool optimizerScalarsResetRequired{true};
    bool rolloutAndRngResetRequired{true};
};

/**
 * @brief Migrates only compatible v7/v4 actor state for generalized training.
 * @tparam LegacyActor Historical allocated actor type.
 * @tparam NewActor Generalized allocated actor type.
 * @param legacy Loaded v7 actor or the actor loaded from a v4 training file.
 * @param migrated Destination actor.
 * @return Report documenting every deliberately reset training component.
 */
template<typename LegacyActor, typename NewActor>
GeneralizedTrainingMigrationReport MigrateV7TrainingActor(
    LegacyActor const& legacy, NewActor& migrated)
{
    MigrateV7Actor(legacy, migrated);
    return {};
}

/**
 * @brief Serializes a full generalized update-boundary training state.
 * @tparam Actor Generalized shared actor type.
 * @tparam Critic Generalized centralized critic type.
 * @tparam ActorOptimizer Actor Adam optimizer type.
 * @tparam CriticOptimizer Critic Adam optimizer type.
 * @tparam RandomEngine Sampling engine type.
 * @param stream Destination stream.
 * @param actor Actor parameters and Adam moments.
 * @param critic Critic parameters and Adam moments.
 * @param actorOptimizer Actor optimizer scalar state.
 * @param criticOptimizer Critic optimizer scalar state.
 * @param randomEngine Sampling engine state.
 * @param completedUpdates Number of completed generalized PPO updates.
 * @param metadata Active P/N and compiled capacities.
 */
template<typename Actor, typename Critic, typename ActorOptimizer,
         typename CriticOptimizer, typename RandomEngine>
void WriteGeneralizedTrainingState(
    std::ostream& stream, Actor const& actor, Critic const& critic,
    ActorOptimizer const& actorOptimizer,
    CriticOptimizer const& criticOptimizer,
    RandomEngine const& randomEngine, std::uint64_t const completedUpdates,
    GeneralizedCheckpointMetadata const metadata)
{
    stream << "BLUE_TEAM_TRAINING_V5 "
           << kGeneralizedTrainingCheckpointVersion << ' '
           << GeneralizedObservationLayout::Version << ' '
           << GeneralizedObservationLayout::Size << ' '
           << ObservationLayout::Size << ' '
           << GeneralizedObservationLayout::ActionSize << ' '
           << TrainingParameters::ACTOR_HIDDEN_DIM << ' '
           << TrainingParameters::ACTOR_NUM_LAYERS << ' '
           << Critic::INPUT_DIM << ' '
           << TrainingParameters::CRITIC_HIDDEN_DIM << ' '
           << TrainingParameters::CRITIC_NUM_LAYERS << ' ';
    generalized_checkpoint_detail::WriteMetadata(stream, metadata);
    stream << '\n'
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "ACTOR\n";
    WriteGeneralizedPolicy(stream, actor, metadata);
    stream << "ACTOR_MOMENTS\n";
    training_checkpoint_detail::WritePatrolAttentionMoments(stream,
                                                             actor.red);
    training_checkpoint_detail::WritePatrolAttentionMoments(
        stream, actor.friendlies);
    training_checkpoint_detail::WritePatrolLayerMoments(stream,
                                                         actor.input_layer);
    for (auto const& layer : actor.hidden_layers)
        training_checkpoint_detail::WritePatrolLayerMoments(stream, layer);
    training_checkpoint_detail::WritePatrolLayerMoments(stream,
                                                         actor.output_layer);
    training_checkpoint_detail::WritePatrolMoments(stream, actor.log_std);

    stream << "CRITIC\n";
    training_checkpoint_detail::WritePatrolMlpState(stream, critic);
    stream << "OPTIMIZERS\n";
    training_checkpoint_detail::WritePatrolOptimizer(stream, actorOptimizer);
    training_checkpoint_detail::WritePatrolOptimizer(stream, criticOptimizer);
    stream << "UPDATES " << completedUpdates << "\nRNG\n" << randomEngine
           << "\nEND_BLUE_TEAM_TRAINING\n";
    if (!stream)
        throw std::runtime_error(
            "Failed to serialize generalized blue training state");
}

/**
 * @brief Restores a strict generalized update-boundary training state.
 * @tparam Actor Generalized shared actor type.
 * @tparam Critic Generalized centralized critic type.
 * @tparam ActorOptimizer Actor Adam optimizer type.
 * @tparam CriticOptimizer Critic Adam optimizer type.
 * @tparam RandomEngine Sampling engine type.
 * @param stream Source stream.
 * @param actor Actor receiving parameters and moments.
 * @param critic Critic receiving parameters and moments.
 * @param actorOptimizer Actor optimizer receiving scalar state.
 * @param criticOptimizer Critic optimizer receiving scalar state.
 * @param randomEngine Sampling engine receiving saved state.
 * @param metadata Receives validated P/N metadata.
 * @return Number of completed PPO updates.
 */
template<typename Actor, typename Critic, typename ActorOptimizer,
         typename CriticOptimizer, typename RandomEngine>
std::uint64_t ReadGeneralizedTrainingState(
    std::istream& stream, Actor& actor, Critic& critic,
    ActorOptimizer& actorOptimizer, CriticOptimizer& criticOptimizer,
    RandomEngine& randomEngine, GeneralizedCheckpointMetadata& metadata)
{
    std::string magic;
    unsigned formatVersion{}, observationVersion{};
    std::size_t rawInput{}, compactInput{}, output{}, actorHidden{};
    std::size_t actorLayers{}, criticInput{}, criticHidden{}, criticLayers{};
    if (!(stream >> magic >> formatVersion >> observationVersion
          >> rawInput >> compactInput >> output >> actorHidden >> actorLayers
          >> criticInput >> criticHidden >> criticLayers)
        || magic != "BLUE_TEAM_TRAINING_V5"
        || formatVersion != kGeneralizedTrainingCheckpointVersion
        || observationVersion != GeneralizedObservationLayout::Version
        || rawInput != GeneralizedObservationLayout::Size
        || compactInput != ObservationLayout::Size
        || output != GeneralizedObservationLayout::ActionSize
        || actorHidden != TrainingParameters::ACTOR_HIDDEN_DIM
        || actorLayers != TrainingParameters::ACTOR_NUM_LAYERS
        || criticInput != Critic::INPUT_DIM
        || criticHidden != TrainingParameters::CRITIC_HIDDEN_DIM
        || criticLayers != TrainingParameters::CRITIC_NUM_LAYERS)
    {
        throw std::runtime_error(
            "Incompatible generalized blue training checkpoint contract");
    }
    metadata = generalized_checkpoint_detail::ReadMetadata(stream);

    training_checkpoint_detail::RequirePatrolCheckpointToken(stream,
                                                              "ACTOR");
    auto const actorMetadata = ReadGeneralizedPolicyParameters(stream, actor);
    if (actorMetadata.activeBlueMobiles != metadata.activeBlueMobiles
        || actorMetadata.activeRedMobiles != metadata.activeRedMobiles)
    {
        throw std::runtime_error(
            "Mismatched generalized actor and training P/N metadata");
    }
    training_checkpoint_detail::RequirePatrolCheckpointToken(
        stream, "ACTOR_MOMENTS");
    training_checkpoint_detail::ReadPatrolAttentionMoments(stream,
                                                            actor.red);
    training_checkpoint_detail::ReadPatrolAttentionMoments(
        stream, actor.friendlies);
    training_checkpoint_detail::ReadPatrolLayerMoments(stream,
                                                        actor.input_layer);
    for (auto& layer : actor.hidden_layers)
        training_checkpoint_detail::ReadPatrolLayerMoments(stream, layer);
    training_checkpoint_detail::ReadPatrolLayerMoments(stream,
                                                        actor.output_layer);
    training_checkpoint_detail::ReadPatrolMoments(stream, actor.log_std);

    training_checkpoint_detail::RequirePatrolCheckpointToken(stream,
                                                              "CRITIC");
    training_checkpoint_detail::ReadPatrolMlpState(stream, critic);
    training_checkpoint_detail::RequirePatrolCheckpointToken(stream,
                                                              "OPTIMIZERS");
    training_checkpoint_detail::ReadPatrolOptimizer(stream, actorOptimizer);
    training_checkpoint_detail::ReadPatrolOptimizer(stream, criticOptimizer);
    training_checkpoint_detail::RequirePatrolCheckpointToken(stream,
                                                              "UPDATES");
    std::uint64_t completedUpdates{};
    if (!(stream >> completedUpdates))
        throw std::runtime_error("Invalid generalized PPO update count");
    training_checkpoint_detail::RequirePatrolCheckpointToken(stream, "RNG");
    if (!(stream >> randomEngine))
        throw std::runtime_error("Invalid generalized random-engine state");
    training_checkpoint_detail::RequirePatrolCheckpointToken(
        stream, "END_BLUE_TEAM_TRAINING");
    std::string trailing;
    if (stream >> trailing)
        throw std::runtime_error(
            "Trailing data in generalized blue training checkpoint");
    return completedUpdates;
}

/**
 * @brief Atomically saves a generalized training-state file.
 */
template<typename Actor, typename Critic, typename ActorOptimizer,
         typename CriticOptimizer, typename RandomEngine>
void SaveGeneralizedTrainingState(
    std::filesystem::path const& path, Actor const& actor,
    Critic const& critic, ActorOptimizer const& actorOptimizer,
    CriticOptimizer const& criticOptimizer,
    RandomEngine const& randomEngine, std::uint64_t const completedUpdates,
    GeneralizedCheckpointMetadata const metadata)
{
    SavePatrolAtomically(path, [&](std::ostream& stream) {
        WriteGeneralizedTrainingState(
            stream, actor, critic, actorOptimizer, criticOptimizer,
            randomEngine, completedUpdates, metadata);
    });
}

} // namespace ppo::examples::twovstwo::defenders
