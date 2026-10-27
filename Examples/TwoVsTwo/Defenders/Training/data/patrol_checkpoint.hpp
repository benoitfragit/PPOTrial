#pragma once
// Canonical checkpoint serialization shared by GUI learning and the
// optional headless utility. Version 5 uses continuous analytical racetrack
// observations and deliberately rejects policies trained with older semantics.
#include "patrol_rltools.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace ppo::examples::twovstwo::defenders {
template<typename MATRIX> void ReadPatrolMatrix(std::istream& stream, MATRIX& matrix)
{
    for (typename MATRIX::SPEC::TI i = 0; i < MATRIX::SPEC::ROWS; ++i)
        for (typename MATRIX::SPEC::TI j = 0; j < MATRIX::SPEC::COLS; ++j)
        {
            float value = 0.F;
            if (!(stream >> value) || !std::isfinite(value))
                throw std::runtime_error("Invalid blue MAPPO weight matrix");
            rlt::set(matrix, i, j, value);
        }
}
template<typename LAYER> void ReadPatrolLayer(std::istream& stream, LAYER& layer)
{
    ReadPatrolMatrix(stream, layer.weights.parameters);
    ReadPatrolMatrix(stream, layer.biases.parameters);
}
template<typename BLOCK> void ReadPatrolAttention(std::istream& stream, BLOCK& block)
{
    ReadPatrolMatrix(stream, block.encoderW.parameters);
    ReadPatrolMatrix(stream, block.encoderB.parameters);
    ReadPatrolMatrix(stream, block.queryW.parameters);
    ReadPatrolMatrix(stream, block.queryB.parameters);
}

/**
 * @brief Loads versioned blue MAPPO Actor parameters from a text stream.
 *
 * The stream may contain a following training-state section. Use
 * ReadPatrolPolicy for a strict inference-only checkpoint.
 * @tparam ACTOR Compatible RLTools attention Actor type.
 * @param stream Source checkpoint stream.
 * @param actor Destination Actor whose parameters are replaced.
 */
template<typename ACTOR>
void ReadPatrolPolicyParameters(std::istream& stream, ACTOR& actor)
{
    std::string magic;
    unsigned version = 0U;
    unsigned input = 0U, output = 0U, hidden = 0U, layers = 0U;
    if (!(stream >> magic >> version >> input >> output >> hidden >> layers)
        || magic != "BLUE_PATROL_MAPPO_V5"
        || version != ObservationLayout::Version
        || input != ObservationLayout::Size
        || output != ObservationLayout::ActionSize
        || hidden != TrainingParameters::ACTOR_HIDDEN_DIM
        || layers != TrainingParameters::ACTOR_NUM_LAYERS)
        throw std::runtime_error("Incompatible blue MAPPO checkpoint contract");

    ReadPatrolAttention(stream, actor.red);
    ReadPatrolAttention(stream, actor.friendlies);
    ReadPatrolLayer(stream, actor.input_layer);
    for (auto& layer : actor.hidden_layers)
        ReadPatrolLayer(stream, layer);
    ReadPatrolLayer(stream, actor.output_layer);
    ReadPatrolMatrix(stream, actor.log_std.parameters);
}

/**
 * @brief Loads one strict inference-only blue MAPPO Actor checkpoint.
 * @tparam ACTOR Compatible RLTools attention Actor type.
 * @param stream Source checkpoint stream.
 * @param actor Destination Actor whose parameters are replaced.
 */
template<typename ACTOR> void ReadPatrolPolicy(std::istream& stream, ACTOR& actor)
{
    ReadPatrolPolicyParameters(stream, actor);
    std::string unexpected;
    if (stream >> unexpected)
        throw std::runtime_error("Trailing data in blue MAPPO checkpoint");
}

template<typename MATRIX> void WritePatrolMatrix(std::ostream& os, MATRIX const& m)
{
    for (typename MATRIX::SPEC::TI i=0;i<MATRIX::SPEC::ROWS;++i)
        for (typename MATRIX::SPEC::TI j=0;j<MATRIX::SPEC::COLS;++j)
            os << rlt::get(m,i,j) << ' ';
    os << '\n';
}
template<typename LAYER> void WritePatrolLayer(std::ostream& os,LAYER const& layer)
{
    WritePatrolMatrix(os,layer.weights.parameters);
    WritePatrolMatrix(os,layer.biases.parameters);
}
template<typename BLOCK> void WritePatrolAttention(std::ostream& os,BLOCK const& block)
{
    WritePatrolMatrix(os,block.encoderW.parameters);
    WritePatrolMatrix(os,block.encoderB.parameters);
    WritePatrolMatrix(os,block.queryW.parameters);
    WritePatrolMatrix(os,block.queryB.parameters);
}
template<typename ACTOR> void WritePatrolPolicy(std::ostream& os,ACTOR const& actor)
{
    os << "BLUE_PATROL_MAPPO_V5 " << ObservationLayout::Version << ' '
       << ObservationLayout::Size << ' ' << ObservationLayout::ActionSize << ' '
       << TrainingParameters::ACTOR_HIDDEN_DIM << ' '
       << TrainingParameters::ACTOR_NUM_LAYERS << '\n'
       << std::setprecision(std::numeric_limits<float>::max_digits10);
    WritePatrolAttention(os,actor.red);
    WritePatrolAttention(os,actor.friendlies);
    WritePatrolLayer(os,actor.input_layer);
    for (auto const& layer:actor.hidden_layers) WritePatrolLayer(os,layer);
    WritePatrolLayer(os,actor.output_layer);
    WritePatrolMatrix(os,actor.log_std.parameters);
    if (!os) throw std::runtime_error("Failed to serialize patrol actor");
}

/**
 * @brief Atomically replaces a checkpoint while retaining rollback on rename.
 * @tparam Writer Callable that serializes to the supplied output stream.
 * @param path Final checkpoint path.
 * @param writer Serialization callback.
 */
template<typename Writer>
void SavePatrolAtomically(std::filesystem::path const& path, Writer&& writer)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    auto const temp = std::filesystem::path{path.string() + ".tmp"};
    auto const backup = std::filesystem::path{path.string() + ".bak"};
    try
    {
        std::ofstream file{temp, std::ios::trunc};
        if (!file)
            throw std::runtime_error("Cannot create patrol checkpoint");
        std::forward<Writer>(writer)(file);
        file.flush();
        if (!file)
            throw std::runtime_error("Cannot flush patrol checkpoint");
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        throw;
    }

    std::error_code ignored;
    std::filesystem::remove(backup, ignored);
    bool const existed = std::filesystem::exists(path);
    if (existed)
        std::filesystem::rename(path, backup);
    try
    {
        std::filesystem::rename(temp, path);
    }
    catch (...)
    {
        if (existed)
            std::filesystem::rename(backup, path);
        std::filesystem::remove(temp, ignored);
        throw;
    }
    if (existed)
        std::filesystem::remove(backup, ignored);
}

template<typename ACTOR>
void SavePatrolPolicy(std::filesystem::path const& path,ACTOR const& actor)
{
    SavePatrolAtomically(path, [&](std::ostream& stream) {
        WritePatrolPolicy(stream, actor);
    });
}
} // namespace ppo::examples::twovstwo::defenders
