#pragma once

#include "generalized_attention_actor.hpp"
#include "patrol_checkpoint.hpp"

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ppo::examples::twovstwo::defenders {

/** @brief Version of the generalized P/N actor checkpoint contract. */
inline constexpr unsigned kGeneralizedPolicyCheckpointVersion = 8U;

/**
 * @brief Maximum absolute action-mean drift accepted after a 2v2 migration.
 *
 * Zeroing only formerly masked input columns makes the operation exact in
 * real arithmetic. The tolerance covers floating-point pooling order only.
 */
inline constexpr float kV7MigrationOutputTolerance = 1.0e-6F;

/**
 * @brief Episode/training-domain metadata embedded in every generalized file.
 */
struct GeneralizedCheckpointMetadata final
{
    /** @brief Blue mobiles P used by the saved update or evaluation run. */
    std::size_t activeBlueMobiles{};
    /** @brief Red mobiles N used by the saved update or evaluation run. */
    std::size_t activeRedMobiles{};
};

/**
 * @brief Validates checkpoint counts against compiled fixed capacities.
 * @param metadata P/N metadata parsed from a checkpoint.
 * @return True when the counts can be represented without truncation.
 */
constexpr bool Valid(GeneralizedCheckpointMetadata const metadata) noexcept
{
    return Valid(ActiveTeamCounts{metadata.activeBlueMobiles,
                                  metadata.activeRedMobiles});
}

namespace generalized_checkpoint_detail {

template<typename Source, typename Destination>
void CopyMatrix(Source const& source, Destination& destination)
{
    static_assert(Source::SPEC::ROWS == Destination::SPEC::ROWS);
    static_assert(Source::SPEC::COLS == Destination::SPEC::COLS);
    for (typename Destination::SPEC::TI row = 0;
         row < Destination::SPEC::ROWS; ++row)
    {
        for (typename Destination::SPEC::TI column = 0;
             column < Destination::SPEC::COLS; ++column)
        {
            rlt::set(destination, row, column,
                     static_cast<typename Destination::SPEC::T>(
                         rlt::get(source, row, column)));
        }
    }
}

template<typename Source, typename Destination>
void CopyParameter(Source const& source, Destination& destination)
{
    CopyMatrix(source.parameters, destination.parameters);
    CopyMatrix(source.gradient_first_order_moment,
               destination.gradient_first_order_moment);
    CopyMatrix(source.gradient_second_order_moment,
               destination.gradient_second_order_moment);
    for (typename Destination::CONTAINER::SPEC::TI row = 0;
         row < Destination::CONTAINER::SPEC::ROWS; ++row)
    {
        for (typename Destination::CONTAINER::SPEC::TI column = 0;
             column < Destination::CONTAINER::SPEC::COLS; ++column)
        {
            rlt::set(destination.gradient, row, column, 0.F);
        }
    }
}

template<typename Source, typename Destination>
void CopyLayer(Source const& source, Destination& destination)
{
    CopyParameter(source.weights, destination.weights);
    CopyParameter(source.biases, destination.biases);
}

template<typename Source, typename Destination>
void CopyAttention(Source const& source, Destination& destination)
{
    CopyParameter(source.encoderW, destination.encoderW);
    CopyParameter(source.encoderB, destination.encoderB);
    CopyParameter(source.queryW, destination.queryW);
    CopyParameter(source.queryB, destination.queryB);
}

template<typename Parameter>
void ClearParameterColumn(Parameter& parameter, std::size_t const column)
{
    using Matrix = typename Parameter::CONTAINER;
    static_assert(Matrix::SPEC::COLS == ObservationLayout::Size);
    for (typename Matrix::SPEC::TI row = 0; row < Matrix::SPEC::ROWS; ++row)
    {
        rlt::set(parameter.parameters, row,
                 static_cast<typename Matrix::SPEC::TI>(column), 0.F);
        rlt::set(parameter.gradient, row,
                 static_cast<typename Matrix::SPEC::TI>(column), 0.F);
        rlt::set(parameter.gradient_first_order_moment, row,
                 static_cast<typename Matrix::SPEC::TI>(column), 0.F);
        rlt::set(parameter.gradient_second_order_moment, row,
                 static_cast<typename Matrix::SPEC::TI>(column), 0.F);
    }
}

inline void WriteMetadata(std::ostream& stream,
                          GeneralizedCheckpointMetadata const metadata)
{
    if (!Valid(metadata))
        throw std::invalid_argument("Invalid P/N checkpoint metadata");
    stream << metadata.activeBlueMobiles << ' '
           << metadata.activeRedMobiles << ' '
           << TeamCapacity::MaximumBlueMobiles << ' '
           << TeamCapacity::MaximumRedMobiles << ' '
           << GeneralizedObservationLayout::BlueTracks << ' '
           << GeneralizedObservationLayout::ObjectiveTracks << ' '
           << GeneralizedObservationLayout::Lidar;
}

inline GeneralizedCheckpointMetadata ReadMetadata(std::istream& stream)
{
    GeneralizedCheckpointMetadata metadata{};
    std::size_t blueCapacity{}, redCapacity{}, otherBlueCapacity{};
    std::size_t objectiveCapacity{}, lidarRays{};
    if (!(stream >> metadata.activeBlueMobiles >> metadata.activeRedMobiles
          >> blueCapacity >> redCapacity >> otherBlueCapacity
          >> objectiveCapacity >> lidarRays)
        || !Valid(metadata)
        || blueCapacity != TeamCapacity::MaximumBlueMobiles
        || redCapacity != TeamCapacity::MaximumRedMobiles
        || otherBlueCapacity != GeneralizedObservationLayout::BlueTracks
        || objectiveCapacity != GeneralizedObservationLayout::ObjectiveTracks
        || lidarRays != GeneralizedObservationLayout::Lidar)
    {
        throw std::runtime_error(
            "Incompatible generalized blue MAPPO P/N capacities");
    }
    return metadata;
}

} // namespace generalized_checkpoint_detail

/**
 * @brief Migrates compatible v7 actor parameters and Adam moments to v8.
 *
 * Track encoders, queries, the dense trunk, output and log standard deviation
 * retain their parameters and matching first/second Adam moments. Three input
 * columns that were guaranteed zero in v7 (formation slot and P/N counts) are
 * explicitly zero-initialized with zero moments so a migrated 2v2 policy is
 * output-equivalent before further training. Transient gradients are reset.
 * No critic, optimizer scalar, RNG, rollout or update counter is migrated.
 *
 * @tparam LegacyActor Historical AttentionActor type.
 * @tparam NewActor GeneralizedAttentionActor type.
 * @param legacy Loaded v7 actor or v4 training actor.
 * @param migrated Allocated v8 actor receiving compatible state.
 */
template<typename LegacyActor, typename NewActor>
void MigrateV7Actor(LegacyActor const& legacy, NewActor& migrated)
{
    generalized_checkpoint_detail::CopyAttention(legacy.red, migrated.red);
    generalized_checkpoint_detail::CopyAttention(legacy.friendlies,
                                                  migrated.friendlies);
    generalized_checkpoint_detail::CopyLayer(legacy.input_layer,
                                              migrated.input_layer);
    static_assert(LegacyActor::NUM_HIDDEN_LAYERS
                  == NewActor::NUM_HIDDEN_LAYERS);
    for (std::size_t layer = 0U;
         layer < LegacyActor::NUM_HIDDEN_LAYERS; ++layer)
    {
        generalized_checkpoint_detail::CopyLayer(
            legacy.hidden_layers[layer], migrated.hidden_layers[layer]);
    }
    generalized_checkpoint_detail::CopyLayer(legacy.output_layer,
                                              migrated.output_layer);
    generalized_checkpoint_detail::CopyParameter(legacy.log_std,
                                                  migrated.log_std);

    generalized_checkpoint_detail::ClearParameterColumn(
        migrated.input_layer.weights,
        GeneralizedCompactFeatures::FormationSlot);
    generalized_checkpoint_detail::ClearParameterColumn(
        migrated.input_layer.weights,
        GeneralizedCompactFeatures::ActiveBlueCount);
    generalized_checkpoint_detail::ClearParameterColumn(
        migrated.input_layer.weights,
        GeneralizedCompactFeatures::ActiveRedCount);
}

/**
 * @brief Writes a generalized actor section without accepting legacy magic.
 * @tparam Actor Compatible generalized actor type.
 * @param stream Destination stream.
 * @param actor Actor parameters.
 * @param metadata Active P/N and compiled capacities.
 */
template<typename Actor>
void WriteGeneralizedPolicy(std::ostream& stream, Actor const& actor,
                            GeneralizedCheckpointMetadata const metadata)
{
    stream << "BLUE_TEAM_MAPPO_V8 " << kGeneralizedPolicyCheckpointVersion
           << ' ' << GeneralizedObservationLayout::Version << ' '
           << GeneralizedObservationLayout::Size << ' '
           << ObservationLayout::Size << ' '
           << GeneralizedObservationLayout::ActionSize << ' '
           << TrainingParameters::ACTOR_HIDDEN_DIM << ' '
           << TrainingParameters::ACTOR_NUM_LAYERS << ' ';
    generalized_checkpoint_detail::WriteMetadata(stream, metadata);
    stream << '\n'
           << std::setprecision(std::numeric_limits<float>::max_digits10);
    WritePatrolAttention(stream, actor.red);
    WritePatrolAttention(stream, actor.friendlies);
    WritePatrolLayer(stream, actor.input_layer);
    for (auto const& layer : actor.hidden_layers)
        WritePatrolLayer(stream, layer);
    WritePatrolLayer(stream, actor.output_layer);
    WritePatrolMatrix(stream, actor.log_std.parameters);
    if (!stream)
        throw std::runtime_error("Failed to serialize generalized blue actor");
}

/**
 * @brief Reads generalized actor parameters and returns their P/N metadata.
 * @tparam Actor Compatible allocated generalized actor type.
 * @param stream Source stream.
 * @param actor Actor receiving parameters.
 * @return Validated P/N metadata.
 */
template<typename Actor>
GeneralizedCheckpointMetadata ReadGeneralizedPolicyParameters(
    std::istream& stream, Actor& actor)
{
    std::string magic;
    unsigned checkpointVersion{}, observationVersion{};
    std::size_t rawInput{}, compactInput{}, output{}, hidden{}, layers{};
    if (!(stream >> magic >> checkpointVersion >> observationVersion
          >> rawInput >> compactInput >> output >> hidden >> layers)
        || magic != "BLUE_TEAM_MAPPO_V8"
        || checkpointVersion != kGeneralizedPolicyCheckpointVersion
        || observationVersion != GeneralizedObservationLayout::Version
        || rawInput != GeneralizedObservationLayout::Size
        || compactInput != ObservationLayout::Size
        || output != GeneralizedObservationLayout::ActionSize
        || hidden != TrainingParameters::ACTOR_HIDDEN_DIM
        || layers != TrainingParameters::ACTOR_NUM_LAYERS)
    {
        throw std::runtime_error(
            "Incompatible generalized blue MAPPO actor contract");
    }
    auto const metadata = generalized_checkpoint_detail::ReadMetadata(stream);
    ReadPatrolAttention(stream, actor.red);
    ReadPatrolAttention(stream, actor.friendlies);
    ReadPatrolLayer(stream, actor.input_layer);
    for (auto& layer : actor.hidden_layers)
        ReadPatrolLayer(stream, layer);
    ReadPatrolLayer(stream, actor.output_layer);
    ReadPatrolMatrix(stream, actor.log_std.parameters);
    return metadata;
}

/**
 * @brief Strictly reads one inference-only generalized actor checkpoint.
 * @tparam Actor Compatible allocated generalized actor type.
 * @param stream Source stream.
 * @param actor Actor receiving parameters.
 * @return Validated P/N metadata.
 */
template<typename Actor>
GeneralizedCheckpointMetadata ReadGeneralizedPolicy(std::istream& stream,
                                                     Actor& actor)
{
    auto const metadata = ReadGeneralizedPolicyParameters(stream, actor);
    std::string trailing;
    if (stream >> trailing)
        throw std::runtime_error(
            "Trailing data in generalized blue MAPPO actor checkpoint");
    return metadata;
}

/**
 * @brief Atomically saves one generalized inference actor.
 * @tparam Actor Compatible generalized actor type.
 * @param path Destination file path.
 * @param actor Actor parameters.
 * @param metadata Active P/N and compiled capacities.
 */
template<typename Actor>
void SaveGeneralizedPolicy(std::filesystem::path const& path,
                           Actor const& actor,
                           GeneralizedCheckpointMetadata const metadata)
{
    SavePatrolAtomically(path, [&](std::ostream& stream) {
        WriteGeneralizedPolicy(stream, actor, metadata);
    });
}

} // namespace ppo::examples::twovstwo::defenders
