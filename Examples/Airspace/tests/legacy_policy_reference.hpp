/**
 * @file Examples/Airspace/tests/legacy_policy_reference.hpp
 * @brief Reference observations and actions of the red policies (fixtures/legacy_policy_reference.txt).
 * @details Written by tools/legacy_torch/reference_outputs.py, shared by the tests that compare
 *          the C++ policies with it and, while LibTorch was still built, with LibTorch.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl::test {

/** @brief One observation and the action it must give. */
struct ReferenceCase final
{
    std::vector<float> observation;
    std::vector<float> action;
};

/** @brief Cases of each model, by "RLMission/prod" style names. */
using ReferenceCases = std::map<std::string, std::vector<ReferenceCase>>;

inline std::vector<float> ReadReferenceRow(std::istream& stream)
{
    std::string line;
    if (!std::getline(stream, line))
    {
        throw std::runtime_error("legacy policy reference: unexpected end of file");
    }
    std::istringstream values(line);
    std::vector<float> row;
    for (float value = 0.F; values >> value;)
    {
        row.push_back(value);
    }
    return row;
}

inline ReferenceCases ReadReferenceCases(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    std::string line;
    if (!std::getline(stream, line) || line != "ppokit.legacy_policy_reference/1")
    {
        throw std::runtime_error("cannot read legacy policy reference " + path.string());
    }
    ReferenceCases cases;
    while (std::getline(stream, line) && line != "end")
    {
        std::istringstream header(line);
        std::string keyword;
        std::string model;
        std::size_t count = 0;
        if (!(header >> keyword >> model >> count) || keyword != "model")
        {
            throw std::runtime_error("legacy policy reference: expected \"model <name> <count>\"");
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            ReferenceCase item;
            item.observation = ReadReferenceRow(stream);
            item.action = ReadReferenceRow(stream);
            cases[model].push_back(std::move(item));
        }
    }
    return cases;
}

/**
 * @brief Same value to float rounding: relative 1e-4 above 1, absolute 1e-4 below.
 * @details The no-track RLMission case reaches 5e8 (the -1e9 of absent tracks feeds the
 *          heads), where only the relative error is meaningful.
 */
inline bool SameValue(float actual, float expected) noexcept
{
    float const scale = std::max(1.F, std::abs(expected));
    return std::isfinite(actual) && std::abs(actual - expected) <= 1e-4F * scale;
}

} // namespace rl::test
