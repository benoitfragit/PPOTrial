/**
 * @file Examples/Airspace/Agents/Policy/legacy_policy_file.cpp
 * @brief Reader of legacy policy files and the linear layer they feed.
 */
#include "legacy_policy_file.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rl {
namespace {

constexpr char Format[] = "ppokit.legacy_policy/1";
/** @brief Largest parameter accepted, against a corrupt size line (the mission network has 16 K). */
constexpr std::size_t MaximumValues = 1U << 22U;

/** @brief Reads the file line by line, naming the line in every error. */
class LineReader final
{
public:
    explicit LineReader(std::filesystem::path const& path) : path_(path), stream_(path)
    {
        if (!stream_)
        {
            throw std::runtime_error("cannot open legacy policy " + path.string());
        }
    }

    std::string Next()
    {
        std::string line;
        if (!std::getline(stream_, line))
        {
            Fail("unexpected end of file");
        }
        ++number_;
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        return line;
    }

    [[noreturn]] void Fail(std::string const& message) const
    {
        throw std::runtime_error(path_.string() + ":" + std::to_string(number_) + ": " + message);
    }

private:
    std::filesystem::path path_;
    std::ifstream stream_;
    std::size_t number_ = 0;
};

/** @brief Reads `count` finite values spread over the following lines. */
std::vector<float> ReadValues(LineReader& reader, std::size_t rows, std::size_t columns)
{
    std::vector<float> values;
    values.reserve(rows * columns);
    for (std::size_t row = 0; row < rows; ++row)
    {
        std::istringstream line(reader.Next());
        for (std::size_t column = 0; column < columns; ++column)
        {
            float value = 0.F;
            if (!(line >> value) || !std::isfinite(value))
            {
                reader.Fail("expected " + std::to_string(columns) + " finite values");
            }
            values.push_back(value);
        }
        std::string extra;
        if (line >> extra)
        {
            reader.Fail("more than " + std::to_string(columns) + " values");
        }
    }
    return values;
}

} // namespace

LegacyPolicyFile::LegacyPolicyFile(std::filesystem::path const& path) : path_(path)
{
    LineReader reader(path);
    if (reader.Next() != Format)
    {
        reader.Fail(std::string("expected \"") + Format + "\"");
    }
    std::istringstream header(reader.Next());
    std::string keyword;
    if (!(header >> keyword >> architecture_) || keyword != "architecture")
    {
        reader.Fail("expected \"architecture <name>\"");
    }
    for (std::string line = reader.Next(); line != "end"; line = reader.Next())
    {
        std::istringstream fields(line);
        std::string name;
        LegacyMatrix matrix;
        if (!(fields >> keyword >> name >> matrix.rows >> matrix.columns) || keyword != "parameter")
        {
            reader.Fail("expected \"parameter <name> <rows> <columns>\" or \"end\"");
        }
        if (matrix.rows == 0 || matrix.columns == 0 || matrix.rows * matrix.columns > MaximumValues)
        {
            reader.Fail("parameter " + name + " has an invalid shape");
        }
        matrix.values = ReadValues(reader, matrix.rows, matrix.columns);
        if (!parameters_.emplace(name, std::move(matrix)).second)
        {
            reader.Fail("parameter " + name + " appears twice");
        }
    }
}

LegacyMatrix const& LegacyPolicyFile::Require(std::string const& name, std::size_t rows, std::size_t columns) const
{
    auto const found = parameters_.find(name);
    if (found == parameters_.end())
    {
        throw std::runtime_error(path_.string() + ": parameter " + name + " is missing");
    }
    if (found->second.rows != rows || found->second.columns != columns)
    {
        throw std::runtime_error(path_.string() + ": parameter " + name + " is " + std::to_string(found->second.rows)
            + "x" + std::to_string(found->second.columns) + ", expected " + std::to_string(rows) + "x"
            + std::to_string(columns));
    }
    return found->second;
}

LegacyDense::LegacyDense(LegacyPolicyFile const& file, std::string const& module, std::size_t inputs,
    std::size_t outputs)
    : inputs_(inputs), outputs_(outputs), weight_(file.Require(module + ".weight", outputs, inputs).values),
      bias_(file.Require(module + ".bias", 1, outputs).values)
{
}

void LegacyDense::Apply(float const* input, float* output) const noexcept
{
    for (std::size_t row = 0; row < outputs_; ++row)
    {
        float const* weights = weight_.data() + row * inputs_;
        float sum = 0.F;
        for (std::size_t column = 0; column < inputs_; ++column)
        {
            sum += weights[column] * input[column];
        }
        output[row] = sum + bias_[row];
    }
}

void ApplyRelu(float* values, std::size_t count) noexcept
{
    for (std::size_t index = 0; index < count; ++index)
    {
        values[index] = std::max(values[index], 0.F);
    }
}

} // namespace rl
