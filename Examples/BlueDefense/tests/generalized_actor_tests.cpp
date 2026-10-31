/**
 * @file Examples/BlueDefense/tests/generalized_actor_tests.cpp
 * @brief Tests et assertions déterministes pour generalized actor tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "generalized_training_checkpoint.hpp"

#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace blue = ppo::examples::bluedefense::defenders;
namespace rlt = rl_tools;

namespace {

template<typename Callable>
/**
 * @brief Réalise l'opération ThrowsRuntimeError; voir les préconditions et effets dans son corps C++.
 */
bool ThrowsRuntimeError(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (std::runtime_error const&)
    {
        return true;
    }
    return false;
}

template<typename Matrix>
/**
 * @brief Réalise l'opération CopyObservationToMatrix; voir les préconditions et effets dans son corps C++.
 */
void CopyObservationToMatrix(blue::GeneralizedRawObservation const& source,
                             Matrix& destination)
{
    for (blue::Index column = 0; column < Matrix::SPEC::COLS; ++column)
        rlt::set(destination, 0, column, source[column]);
}

template<typename Matrix>
void SwapRawTracks(Matrix& matrix, std::size_t const first,
                   std::size_t const second)
{
    using Layout = blue::GeneralizedObservationLayout;
    for (std::size_t feature = 0; feature < Layout::RedFeatures; ++feature)
    {
        auto const firstColumn = static_cast<blue::Index>(
            Layout::RedOffset + first * Layout::RedFeatures + feature);
        auto const secondColumn = static_cast<blue::Index>(
            Layout::RedOffset + second * Layout::RedFeatures + feature);
        float const temporary = rlt::get(matrix, 0, firstColumn);
        rlt::set(matrix, 0, firstColumn,
                 rlt::get(matrix, 0, secondColumn));
        rlt::set(matrix, 0, secondColumn, temporary);
    }
}

/**
 * @brief Réalise l'opération Excluded; voir les préconditions et effets dans son corps C++.
 */
bool Excluded(std::size_t const column,
              std::initializer_list<std::size_t> const excluded)
{
    return std::find(excluded.begin(), excluded.end(), column)
        != excluded.end();
}

template<typename Parameter>
void SeedMoments(Parameter& parameter, float const seed)
{
    using Matrix = typename Parameter::CONTAINER;
    for (typename Matrix::SPEC::TI row = 0; row < Matrix::SPEC::ROWS; ++row)
    {
        for (typename Matrix::SPEC::TI column = 0;
             column < Matrix::SPEC::COLS; ++column)
        {
            float const offset = 0.001F * static_cast<float>(
                row * Matrix::SPEC::COLS + column);
            rlt::set(parameter.gradient_first_order_moment, row, column,
                     seed + offset);
            rlt::set(parameter.gradient_second_order_moment, row, column,
                     seed + 1.F + offset);
        }
    }
}

template<typename Source, typename Destination>
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool SameMigratedParameter(
    Source const& source, Destination const& destination,
    std::initializer_list<std::size_t> const excluded = {})
{
    using Matrix = typename Source::CONTAINER;
    static_assert(Matrix::SPEC::ROWS == Destination::CONTAINER::SPEC::ROWS);
    static_assert(Matrix::SPEC::COLS == Destination::CONTAINER::SPEC::COLS);
    for (typename Matrix::SPEC::TI row = 0; row < Matrix::SPEC::ROWS; ++row)
    {
        for (typename Matrix::SPEC::TI column = 0;
             column < Matrix::SPEC::COLS; ++column)
        {
            if (Excluded(column, excluded))
                continue;
            if (rlt::get(source.parameters, row, column)
                    != rlt::get(destination.parameters, row, column)
                || rlt::get(source.gradient_first_order_moment, row, column)
                    != rlt::get(
                        destination.gradient_first_order_moment, row, column)
                || rlt::get(source.gradient_second_order_moment, row, column)
                    != rlt::get(
                        destination.gradient_second_order_moment, row, column))
            {
                return false;
            }
        }
    }
    return true;
}

template<typename Parameter>
/**
 * @brief Réalise l'opération ZeroParameterColumn; voir les préconditions et effets dans son corps C++.
 */
bool ZeroParameterColumn(Parameter const& parameter,
                         std::size_t const column)
{
    using Matrix = typename Parameter::CONTAINER;
    for (typename Matrix::SPEC::TI row = 0; row < Matrix::SPEC::ROWS; ++row)
    {
        auto const typedColumn =
            static_cast<typename Matrix::SPEC::TI>(column);
        if (rlt::get(parameter.parameters, row, typedColumn) != 0.F
            || rlt::get(parameter.gradient, row, typedColumn) != 0.F
            || rlt::get(parameter.gradient_first_order_moment, row,
                        typedColumn) != 0.F
            || rlt::get(parameter.gradient_second_order_moment, row,
                        typedColumn) != 0.F)
        {
            return false;
        }
    }
    return true;
}

template<typename Source, typename Destination>
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
bool SameParameterColumn(Source const& source, Destination const& destination,
                         std::size_t const column)
{
    using Matrix = typename Source::CONTAINER;
    auto const typedColumn = static_cast<typename Matrix::SPEC::TI>(column);
    for (typename Matrix::SPEC::TI row = 0; row < Matrix::SPEC::ROWS; ++row)
    {
        if (rlt::get(source.parameters, row, typedColumn)
                != rlt::get(destination.parameters, row, typedColumn)
            || rlt::get(source.gradient_first_order_moment, row, typedColumn)
                != rlt::get(destination.gradient_first_order_moment, row,
                            typedColumn)
            || rlt::get(source.gradient_second_order_moment, row, typedColumn)
                != rlt::get(destination.gradient_second_order_moment, row,
                            typedColumn))
        {
            return false;
        }
    }
    return true;
}

template<typename Attention>
/**
 * @brief Réalise l'opération SeedAttentionMoments; voir les préconditions et effets dans son corps C++.
 */
void SeedAttentionMoments(Attention& attention, float const seed)
{
    SeedMoments(attention.encoderW, seed);
    SeedMoments(attention.encoderB, seed + 0.1F);
    SeedMoments(attention.queryW, seed + 0.2F);
    SeedMoments(attention.queryB, seed + 0.3F);
}

template<typename Source, typename Destination>
bool SameAttention(Source const& source, Destination const& destination)
{
    return SameMigratedParameter(source.encoderW, destination.encoderW)
        && SameMigratedParameter(source.encoderB, destination.encoderB)
        && SameMigratedParameter(source.queryW, destination.queryW)
        && SameMigratedParameter(source.queryB, destination.queryB);
}

template<typename Layer>
/**
 * @brief Réalise l'opération SeedLayerMoments; voir les préconditions et effets dans son corps C++.
 */
void SeedLayerMoments(Layer& layer, float const seed)
{
    SeedMoments(layer.weights, seed);
    SeedMoments(layer.biases, seed + 0.1F);
}

template<typename Source, typename Destination>
bool SameLayer(Source const& source, Destination const& destination)
{
    return SameMigratedParameter(source.weights, destination.weights)
        && SameMigratedParameter(source.biases, destination.biases);
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    using RawLayout = blue::GeneralizedObservationLayout;
    using LegacyLayout = blue::ObservationLayout;
    static_assert(RawLayout::Lidar == 61U);
    static_assert(RawLayout::RedTracks == 10U);
    static_assert(RawLayout::BlueTracks == 8U);
    static_assert(RawLayout::ObjectiveTracks == 1U);
    static_assert(RawLayout::Version != LegacyLayout::Version);
    static_assert(blue::kGeneralizedPolicyCheckpointVersion == 8U);
    static_assert(blue::kGeneralizedTrainingCheckpointVersion == 6U);

    blue::Observation legacyObservation{};
    for (std::size_t index = 0; index < LegacyLayout::Own; ++index)
        legacyObservation[index] = 0.03F * static_cast<float>(index + 1U);
    for (std::size_t ray = 0; ray < LegacyLayout::Lidar; ++ray)
    {
        legacyObservation[LegacyLayout::Own + ray] =
            static_cast<float>(ray) / static_cast<float>(LegacyLayout::Lidar);
    }
    for (std::size_t track = 0; track < LegacyLayout::RedTracks; ++track)
    {
        std::size_t const base = LegacyLayout::RedOffset
            + track * LegacyLayout::RedFeatures;
        for (std::size_t feature = 0;
             feature + 1U < LegacyLayout::RedFeatures; ++feature)
        {
            legacyObservation[base + feature] =
                0.1F * static_cast<float>(track + 1U)
                + 0.005F * static_cast<float>(feature);
        }
        legacyObservation[base + LegacyLayout::RedFeatures - 1U] = 1.F;
    }
    for (std::size_t track = 0; track < LegacyLayout::FriendlyTracks; ++track)
    {
        std::size_t const base = LegacyLayout::FriendOffset
            + track * LegacyLayout::FriendlyFeatures;
        for (std::size_t feature = 0;
             feature + 1U < LegacyLayout::FriendlyFeatures; ++feature)
        {
            legacyObservation[base + feature] =
                -0.15F * static_cast<float>(track + 1U)
                + 0.004F * static_cast<float>(feature);
        }
        legacyObservation[base + LegacyLayout::FriendlyFeatures - 1U] = 1.F;
    }
    for (std::size_t feature = 0; feature < LegacyLayout::Patrol; ++feature)
    {
        legacyObservation[LegacyLayout::PatrolOffset + feature] =
            0.02F * static_cast<float>(feature);
    }
    legacyObservation[LegacyLayout::PatrolOffset + 8U] = -1.F;
    legacyObservation[LegacyLayout::GlobalOffset] = 0.75F;
    legacyObservation[LegacyLayout::GlobalOffset + 1U] = 0.F;
    for (std::size_t column = LegacyLayout::CentralOffset;
         column < LegacyLayout::Size; ++column)
    {
        legacyObservation[column] =
            0.25F + 0.01F * static_cast<float>(column
                                               - LegacyLayout::CentralOffset);
    }
    auto const upgraded =
        blue::UpgradeLegacy2v2Observation(legacyObservation);
    if (!blue::Valid(upgraded))
        return 1;

    auto badMask = upgraded;
    badMask.values[blue::GeneralizedTrackMaskIndex(
        RawLayout::RedOffset, 0U, RawLayout::RedFeatures)] = 0.5F;
    if (blue::Valid(badMask))
        return 2;

    blue::Device device{};
    using ActorSpecification = blue::TrainingConfig::NN::ACTOR_SPEC;
    using LegacyActor = blue::TrainingConfig::NN::ACTOR_TYPE;
    using GeneralizedActor = blue::GeneralizedAttentionActor<ActorSpecification>;
    LegacyActor legacyActor{};
    GeneralizedActor generalizedActor{};
    rlt::malloc(device, legacyActor);
    rlt::malloc(device, generalizedActor);
    auto randomEngine = rlt::random::default_engine(
        typename blue::Device::SPEC::RANDOM{}, 918273U);
    rlt::init_weights(device, legacyActor, randomEngine);
    rlt::init_weights(device, generalizedActor, randomEngine);
    typename blue::TrainingConfig::NN::ACTOR_OPTIMIZER optimizer{};
    rlt::_reset_optimizer_state(device, legacyActor, optimizer);
    blue::_reset_optimizer_state(device, generalizedActor, optimizer);

    SeedAttentionMoments(legacyActor.red, 0.1F);
    SeedAttentionMoments(legacyActor.friendlies, 0.5F);
    SeedLayerMoments(legacyActor.input_layer, 0.9F);
    for (std::size_t layer = 0U;
         layer < LegacyActor::NUM_HIDDEN_LAYERS; ++layer)
    {
        SeedLayerMoments(legacyActor.hidden_layers[layer],
                         1.3F + static_cast<float>(layer));
    }
    SeedLayerMoments(legacyActor.output_layer, 3.7F);
    SeedMoments(legacyActor.log_std, 4.1F);
    blue::MigrateV7Actor(legacyActor, generalizedActor);
    std::initializer_list<std::size_t> const newColumns{
        blue::GeneralizedCompactFeatures::FormationSlot,
        blue::GeneralizedCompactFeatures::ActiveBlueCount,
        blue::GeneralizedCompactFeatures::ActiveRedCount};
    if (!SameAttention(legacyActor.red, generalizedActor.red)
        || !SameAttention(legacyActor.friendlies,
                          generalizedActor.friendlies)
        || !SameMigratedParameter(legacyActor.input_layer.weights,
                                  generalizedActor.input_layer.weights,
                                  newColumns)
        || !SameMigratedParameter(legacyActor.input_layer.biases,
                                  generalizedActor.input_layer.biases))
    {
        return 3;
    }
    for (std::size_t layer = 0U;
         layer < LegacyActor::NUM_HIDDEN_LAYERS; ++layer)
    {
        if (!SameLayer(legacyActor.hidden_layers[layer],
                       generalizedActor.hidden_layers[layer]))
        {
            return 4;
        }
    }
    if (!SameLayer(legacyActor.output_layer, generalizedActor.output_layer)
        || !SameMigratedParameter(legacyActor.log_std,
                                  generalizedActor.log_std))
    {
        return 5;
    }
    for (std::size_t const column : newColumns)
    {
        if (!ZeroParameterColumn(generalizedActor.input_layer.weights,
                                 column))
        {
            return 6;
        }
    }
    // Every centralized v7 column was masked and remains zero in the new
    // actor adapter. Its parameters and both moments remain byte-identical.
    for (std::size_t column = LegacyLayout::CentralOffset;
         column < LegacyLayout::Size; ++column)
    {
        if (!SameParameterColumn(legacyActor.input_layer.weights,
                                 generalizedActor.input_layer.weights,
                                 column))
        {
            return 7;
        }
    }

    using LegacyInputSpecification = rlt::matrix::Specification<
        float, blue::Index, 1, LegacyLayout::Size>;
    using RawInputSpecification = rlt::matrix::Specification<
        float, blue::Index, 1, RawLayout::Size>;
    using OutputSpecification = rlt::matrix::Specification<
        float, blue::Index, 1, LegacyLayout::ActionSize>;
    rlt::MatrixDynamic<LegacyInputSpecification> legacyInput{};
    rlt::MatrixDynamic<RawInputSpecification> rawInput{};
    rlt::MatrixDynamic<RawInputSpecification> permutedInput{};
    rlt::MatrixDynamic<OutputSpecification> legacyOutput{};
    rlt::MatrixDynamic<OutputSpecification> generalizedOutput{};
    rlt::MatrixDynamic<OutputSpecification> permutedOutput{};
    rlt::malloc(device, legacyInput);
    rlt::malloc(device, rawInput);
    rlt::malloc(device, permutedInput);
    rlt::malloc(device, legacyOutput);
    rlt::malloc(device, generalizedOutput);
    rlt::malloc(device, permutedOutput);
    for (blue::Index column = 0; column < LegacyLayout::Size; ++column)
        rlt::set(legacyInput, 0, column, legacyObservation[column]);
    CopyObservationToMatrix(upgraded.values, rawInput);
    CopyObservationToMatrix(upgraded.values, permutedInput);
    SwapRawTracks(permutedInput, 0U, 1U);

    typename LegacyActor::template Buffer<1> legacyBuffer{};
    typename GeneralizedActor::template Buffer<1> generalizedBuffer{};
    rlt::malloc(device, legacyBuffer);
    rlt::malloc(device, generalizedBuffer);
    rlt::evaluate(device, legacyActor, legacyInput, legacyOutput,
                  legacyBuffer);
    rlt::evaluate(device, generalizedActor, rawInput, generalizedOutput,
                  generalizedBuffer);
    rlt::evaluate(device, generalizedActor, permutedInput, permutedOutput,
                  generalizedBuffer);
    for (blue::Index action = 0; action < LegacyLayout::ActionSize; ++action)
    {
        float const legacyValue = rlt::get(legacyOutput, 0, action);
        float const migratedValue = rlt::get(generalizedOutput, 0, action);
        if (std::abs(legacyValue - migratedValue)
            > blue::kV7MigrationOutputTolerance)
        {
            std::cerr << "2v2 migration error="
                      << std::abs(legacyValue - migratedValue) << '\n';
            return 8;
        }
        if (std::abs(migratedValue
                     - rlt::get(permutedOutput, 0, action))
            > blue::kV7MigrationOutputTolerance)
        {
            return 9;
        }
    }

    using TrainingRawSpecification = rlt::matrix::Specification<
        float, blue::Index, ActorSpecification::BATCH_SIZE, RawLayout::Size>;
    using TrainingOutputSpecification = rlt::matrix::Specification<
        float, blue::Index, ActorSpecification::BATCH_SIZE,
        LegacyLayout::ActionSize>;
    rlt::MatrixDynamic<TrainingRawSpecification> trainingRaw{};
    rlt::MatrixDynamic<TrainingOutputSpecification> trainingOutput{};
    rlt::malloc(device, trainingRaw);
    rlt::malloc(device, trainingOutput);
    for (blue::Index row = 0; row < ActorSpecification::BATCH_SIZE; ++row)
    {
        for (blue::Index column = 0; column < RawLayout::Size; ++column)
            rlt::set(trainingRaw, row, column, upgraded.values[column]);
    }
    typename GeneralizedActor::template Buffer<
        ActorSpecification::BATCH_SIZE> trainingBuffer{};
    rlt::malloc(device, trainingBuffer);
    rlt::forward(device, generalizedActor, trainingRaw, trainingOutput);
    for (blue::Index action = 0; action < LegacyLayout::ActionSize; ++action)
    {
        if (std::abs(rlt::get(trainingOutput, 0, action)
                     - rlt::get(generalizedOutput, 0, action))
            > blue::kV7MigrationOutputTolerance)
        {
            return 10;
        }
    }
    rlt::set_all(device, trainingOutput, 0.25F);
    rlt::zero_gradient(device, generalizedActor);
    rlt::backward(device, generalizedActor, trainingRaw, trainingOutput,
                  trainingBuffer);
    if (!std::isfinite(rlt::get(
            generalizedActor.red.encoderW.gradient, 0, 0)))
    {
        return 11;
    }
    rlt::free(device, trainingBuffer);
    rlt::free(device, trainingOutput);
    rlt::free(device, trainingRaw);

    std::stringstream policyStream;
    blue::WriteGeneralizedPolicy(policyStream, generalizedActor, {2U, 2U});
    std::string const serializedPolicy = policyStream.str();
    GeneralizedActor loadedActor{};
    rlt::malloc(device, loadedActor);
    std::stringstream policyInput{serializedPolicy};
    auto const loadedMetadata =
        blue::ReadGeneralizedPolicy(policyInput, loadedActor);
    if (loadedMetadata.activeBlueMobiles != 2U
        || loadedMetadata.activeRedMobiles != 2U
        || rlt::get(loadedActor.red.encoderW.parameters, 0, 0)
               != rlt::get(generalizedActor.red.encoderW.parameters, 0, 0))
    {
        return 12;
    }

    std::string wrongVersion = serializedPolicy;
    auto const headerVersion = wrongVersion.find("BLUE_TEAM_MAPPO_V8 8");
    if (headerVersion == std::string::npos)
        return 13;
    wrongVersion[headerVersion + std::string{"BLUE_TEAM_MAPPO_V8 "}.size()]
        = '9';
    if (!ThrowsRuntimeError([&] {
            std::stringstream input{wrongVersion};
            blue::ReadGeneralizedPolicy(input, loadedActor);
        }))
    {
        return 14;
    }
    if (!ThrowsRuntimeError([&] {
            std::stringstream input{
                serializedPolicy.substr(0U, serializedPolicy.size() / 2U)};
            blue::ReadGeneralizedPolicy(input, loadedActor);
        }))
    {
        return 15;
    }

    auto const atomicPath = std::filesystem::temp_directory_path()
        / "ppo_generalized_actor_checkpoint_test.txt";
    std::error_code ignored;
    std::filesystem::remove(atomicPath, ignored);
    blue::SaveGeneralizedPolicy(atomicPath, generalizedActor, {2U, 2U});
    {
        std::ifstream atomicInput{atomicPath};
        if (!atomicInput
            || blue::ReadGeneralizedPolicy(atomicInput, loadedActor)
                   .activeBlueMobiles != 2U)
        {
            return 16;
        }
    }
    std::filesystem::remove(atomicPath, ignored);

    using Critic = blue::GeneralizedCriticConfig::Network;
    Critic critic{};
    Critic loadedCritic{};
    rlt::malloc(device, critic);
    rlt::malloc(device, loadedCritic);
    rlt::init_weights(device, critic, randomEngine);
    typename blue::TrainingConfig::NN::CRITIC_OPTIMIZER criticOptimizer{};
    rlt::_reset_optimizer_state(device, critic, criticOptimizer);
    std::stringstream trainingStream;
    blue::WriteGeneralizedTrainingState(
        trainingStream, generalizedActor, critic, optimizer,
        criticOptimizer, randomEngine, 17U, {2U, 2U});
    std::string const serializedTraining = trainingStream.str();
    GeneralizedActor loadedTrainingActor{};
    rlt::malloc(device, loadedTrainingActor);
    typename blue::TrainingConfig::NN::ACTOR_OPTIMIZER loadedActorOptimizer{};
    typename blue::TrainingConfig::NN::CRITIC_OPTIMIZER
        loadedCriticOptimizer{};
    auto loadedRandomEngine = rlt::random::default_engine(
        typename blue::Device::SPEC::RANDOM{}, 1U);
    blue::GeneralizedCheckpointMetadata trainingMetadata{};
    std::stringstream trainingInput{serializedTraining};
    auto const completedUpdates = blue::ReadGeneralizedTrainingState(
        trainingInput, loadedTrainingActor, loadedCritic,
        loadedActorOptimizer, loadedCriticOptimizer, loadedRandomEngine,
        trainingMetadata);
    if (completedUpdates != 17U
        || trainingMetadata.activeBlueMobiles != 2U
        || trainingMetadata.activeRedMobiles != 2U)
    {
        return 17;
    }

    std::string wrongTrainingVersion = serializedTraining;
    auto const trainingVersion =
        wrongTrainingVersion.find("BLUE_TEAM_TRAINING_V6 6");
    if (trainingVersion == std::string::npos)
        return 18;
    wrongTrainingVersion[
        trainingVersion + std::string{"BLUE_TEAM_TRAINING_V6 "}.size()] = '5';
    if (!ThrowsRuntimeError([&] {
            std::stringstream input{wrongTrainingVersion};
            blue::ReadGeneralizedTrainingState(
                input, loadedTrainingActor, loadedCritic,
                loadedActorOptimizer, loadedCriticOptimizer,
                loadedRandomEngine, trainingMetadata);
        }))
    {
        return 19;
    }
    if (!ThrowsRuntimeError([&] {
            std::stringstream input{serializedTraining.substr(
                0U, serializedTraining.size() / 2U)};
            blue::ReadGeneralizedTrainingState(
                input, loadedTrainingActor, loadedCritic,
                loadedActorOptimizer, loadedCriticOptimizer,
                loadedRandomEngine, trainingMetadata);
        }))
    {
        return 20;
    }

    auto const migrationReport =
        blue::MigrateV7TrainingActor(legacyActor, loadedTrainingActor);
    if (!migrationReport.actorParametersMigrated
        || !migrationReport.compatibleActorMomentsMigrated
        || !migrationReport.criticResetRequired
        || !migrationReport.optimizerScalarsResetRequired
        || !migrationReport.rolloutAndRngResetRequired)
    {
        return 21;
    }

    rlt::free(device, loadedTrainingActor);
    rlt::free(device, loadedCritic);
    rlt::free(device, critic);
    rlt::free(device, loadedActor);
    rlt::free(device, generalizedBuffer);
    rlt::free(device, legacyBuffer);
    rlt::free(device, permutedOutput);
    rlt::free(device, generalizedOutput);
    rlt::free(device, legacyOutput);
    rlt::free(device, permutedInput);
    rlt::free(device, rawInput);
    rlt::free(device, legacyInput);
    rlt::free(device, generalizedActor);
    rlt::free(device, legacyActor);
    return 0;
}
