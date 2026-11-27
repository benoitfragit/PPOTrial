/**
 * @file Examples/Airspace/Agents/Policy/legacy_policy_file.hpp
 * @brief Weights of the red policies first trained with LibTorch, read without LibTorch.
 *
 * tools/legacy_torch/convert_legacy_model.py writes them from the old model.pt archives:
 *
 *     ppokit.legacy_policy/1
 *     architecture <name>
 *     parameter <module>.<weight|bias> <rows> <columns>
 *     <rows x columns values, row-major>
 *     end
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace rl {

/** @brief One weight matrix or bias row, row-major. */
struct LegacyMatrix final
{
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<float> values;
};

/** @brief A legacy policy file: its architecture name and its parameters by name. */
class LegacyPolicyFile final
{
public:
    /**
     * @brief Reads and checks a legacy policy file.
     * @param path File written by convert_legacy_model.py.
     * @throw std::runtime_error when it cannot be read or is malformed (with the line).
     */
    explicit LegacyPolicyFile(std::filesystem::path const& path);

    /** @brief Architecture the weights belong to, e.g. "mission_track_attention". */
    std::string const& Architecture() const noexcept { return architecture_; }

    /**
     * @brief A parameter of the expected shape.
     * @param name "<module>.weight" or "<module>.bias".
     * @param rows Expected rows (1 for a bias).
     * @param columns Expected columns.
     * @return Its values.
     * @throw std::runtime_error when it is missing or has another shape.
     */
    LegacyMatrix const& Require(std::string const& name, std::size_t rows, std::size_t columns) const;

private:
    std::filesystem::path path_;
    std::string architecture_;
    std::map<std::string, LegacyMatrix> parameters_;
};

/**
 * @brief A linear layer of a legacy policy: y = W x + b, W of shape [outputs x inputs].
 * @details Accumulates in float, in input order, as the float32 LibTorch layers did (the
 *          results agree with LibTorch to rounding).
 */
class LegacyDense final
{
public:
    LegacyDense() = default;
    /**
     * @brief Takes `<module>.weight` and `<module>.bias` from the file.
     * @param file Legacy policy file.
     * @param module Module name, e.g. "enc1".
     * @param inputs Expected input width.
     * @param outputs Expected output width.
     */
    LegacyDense(LegacyPolicyFile const& file, std::string const& module, std::size_t inputs, std::size_t outputs);

    /**
     * @brief Writes W x + b into `output`.
     * @param input `Inputs()` values.
     * @param output `Outputs()` values.
     */
    void Apply(float const* input, float* output) const noexcept;

    std::size_t Inputs() const noexcept { return inputs_; }
    std::size_t Outputs() const noexcept { return outputs_; }

private:
    std::size_t inputs_ = 0;
    std::size_t outputs_ = 0;
    std::vector<float> weight_;
    std::vector<float> bias_;
};

/** @brief max(0, x) on `count` values, in place. */
void ApplyRelu(float* values, std::size_t count) noexcept;

} // namespace rl
