/**
 * @file PPOKit/include/ppokit/graph/checkpoint_io.hpp
 * @brief Text sections of a graph checkpoint: parameter tensors, Adam moments, a counter.
 * @details Floats are written with 9 significant digits, enough to read back the same
 * float bit for bit on every IEEE platform; the file stays readable and diffable. A
 * section starts with its label, so a damaged or foreign file is reported with the
 * section that does not fit. Reading fills staging vectors: nothing is applied to a
 * network until the whole file has been read.
 */
#pragma once

#include "ppokit/artifacts/random_state.hpp"
#include "ppokit/graph/adam.hpp"
#include "ppokit/graph/blocks.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::graph::checkpoint {

/** @brief Values of each tensor of a section, in order. */
using Tensors = std::vector<std::vector<float>>;

inline void WriteTensors(std::ostream& out, char const* label, std::vector<Parameter*> const& parameters)
{
    out << label << ' ' << parameters.size() << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (Parameter const* parameter : parameters)
    {
        out << parameter->value.size();
        for (float const value : parameter->value)
        {
            out << ' ' << value;
        }
        out << '\n';
    }
}

inline void WriteMoments(std::ostream& out, char const* label, Tensors const& moments)
{
    out << label << ' ' << moments.size() << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::vector<float> const& tensor : moments)
    {
        out << tensor.size();
        for (float const value : tensor)
        {
            out << ' ' << value;
        }
        out << '\n';
    }
}

/** @brief Reads the label of the next section and checks it. */
inline void RequireLabel(std::istream& in, char const* label, std::string const& source)
{
    std::string found;
    if (!(in >> found) || found != label)
    {
        throw std::runtime_error(source + ": damaged checkpoint: expected the section '" + label + "', found '"
            + found + "'");
    }
}

/**
 * @brief Reads a section of tensors whose sizes must match `expected`.
 * @throws std::runtime_error naming the section and the tensor that does not fit.
 */
inline Tensors ReadTensors(std::istream& in, char const* label, std::vector<std::size_t> const& expected,
    std::string const& source)
{
    RequireLabel(in, label, source);
    std::size_t count = 0;
    in >> count;
    if (!in || count != expected.size())
    {
        throw std::runtime_error(source + ": damaged checkpoint: '" + label + "' holds " + std::to_string(count)
            + " tensors, the network has " + std::to_string(expected.size()));
    }
    Tensors tensors(count);
    for (std::size_t tensor = 0; tensor < count; ++tensor)
    {
        std::size_t size = 0;
        in >> size;
        if (!in || size != expected[tensor])
        {
            throw std::runtime_error(source + ": damaged checkpoint: tensor " + std::to_string(tensor) + " of '"
                + label + "' holds " + std::to_string(size) + " values, the network expects "
                + std::to_string(expected[tensor]));
        }
        tensors[tensor].resize(size);
        for (float& value : tensors[tensor])
        {
            if (!(in >> value) || !std::isfinite(value))
            {
                throw std::runtime_error(source + ": damaged checkpoint: invalid value in '" + label + "'");
            }
        }

    }
    return tensors;
}

/** @brief Sizes of the tensors, for ReadTensors. */
inline std::vector<std::size_t> SizesOf(std::vector<Parameter*> const& parameters)
{
    std::vector<std::size_t> sizes;
    for (Parameter const* parameter : parameters)
    {
        sizes.push_back(parameter->value.size());
    }
    return sizes;
}

/** @brief Copies staged values into the parameters (sizes already checked). */
inline void Apply(Tensors const& tensors, std::vector<Parameter*> const& parameters) noexcept
{
    for (std::size_t tensor = 0; tensor < parameters.size(); ++tensor)
    {
        parameters[tensor]->value = tensors[tensor];
    }
}

/** @brief The section "rng" and the engine's state in the portable (standard) form. */
inline void WriteRandomState(std::ostream& out, std::mt19937 const& engine)
{
    out << "rng\n" << artifacts::StandardRandomText(engine) << '\n';
}

/**
 * @brief Reads the section "rng" (standard or libstdc++ form, from any platform).
 * @throws std::runtime_error naming `source` for a missing or damaged state.
 */
inline std::mt19937 ReadRandomState(std::istream& in, std::string const& source)
{
    RequireLabel(in, "rng", source);
    std::string line;
    in >> std::ws;
    std::getline(in, line);
    return artifacts::ParseRandomText(line, source);
}

/** @brief Writes an optimizer's state as section `label`: its step count, then both moments. */
inline void WriteAdam(std::ostream& out, char const* label, Adam::State const& state)
{
    out << label << ' ' << state.steps << '\n';
    WriteMoments(out, "first", state.first);
    WriteMoments(out, "second", state.second);
}

/**
 * @brief Reads section `label` written by WriteAdam, for an optimizer of `parameters`.
 * @throws std::runtime_error naming `source` for a missing or damaged section.
 */
inline Adam::State ReadAdam(std::istream& in, char const* label, std::vector<Parameter*> const& parameters,
    std::string const& source)
{
    RequireLabel(in, label, source);
    Adam::State state;
    std::string count;
    in >> count;
    auto const parsed = std::from_chars(count.data(), count.data() + count.size(), state.steps);
    if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size())
    {
        throw std::runtime_error(source + ": invalid Adam step count");
    }
    std::vector<std::size_t> const sizes = SizesOf(parameters);
    state.first = ReadTensors(in, "first", sizes, source);
    state.second = ReadTensors(in, "second", sizes, source);
    return state;
}

} // namespace ppokit::graph::checkpoint
