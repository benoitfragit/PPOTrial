#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ppo::rl {

/// Owned values; no tensor or simulator lifetime crosses the contract.
using Value = std::variant<std::vector<float>, std::int64_t>;
using InfoValue = std::variant<bool, std::int64_t, double, std::string>;
using Info = std::map<std::string, InfoValue>;

/// Float32 Box with flattened row-major bounds and an explicit shape.
class Box
{
public:
    /// Throws invalid_argument for invalid dimensions/bounds; overflow_error on size overflow.
    /// Bounds may be infinite; observations/actions must be finite.
    Box(std::vector<std::size_t> shape, std::vector<float> low, std::vector<float> high);

    const std::vector<std::size_t>& shape() const noexcept;
    const std::vector<float>& low() const noexcept;
    const std::vector<float>& high() const noexcept;
    bool contains(const Value& value) const noexcept;

private:
    std::vector<std::size_t> shape_;
    std::vector<float> low_;
    std::vector<float> high_;
};

/// Int64 space [start, start + count), without floating-point conversion.
class Discrete
{
public:
    /// Throws invalid_argument for a nonpositive count; overflow_error if the last value overflows.
    explicit Discrete(std::int64_t count, std::int64_t start = 0);
    std::int64_t count() const noexcept;
    std::int64_t start() const noexcept;
    bool contains(const Value& value) const noexcept;

private:
    std::int64_t count_;
    std::int64_t start_;
    std::int64_t last_;
};

using Space = std::variant<Box, Discrete>;
bool contains(const Space& space, const Value& value) noexcept;

struct ResetOptions
{
    std::optional<std::uint64_t> seed;
    Info options;
};

struct ResetResult
{
    Value observation;
    Info info;
};

struct StepResult
{
    Value observation;
    double reward = 0.0;
    bool terminated = false;
    bool truncated = false;
    Info info;
};

/// Synchronous single-environment contract, usable by local or remote adapters.
/// Call reset before step and after either episode-end flag. No implicit autoreset.
/// A seeded reset must reproduce the same initial state and subsequent sequence
/// for the same actions and options. An omitted seed must not reseed the RNG.
/// Returned observations belong to the caller and satisfy observation_space().
/// step returns the final observation on episode end and a finite reward.
/// Invalid actions throw invalid_argument; invalid lifecycle throws logic_error.
/// Implementations are not required to be thread-safe; use one instance per worker.
class IEnvironment
{
public:
    virtual ~IEnvironment() = default;
    virtual ResetResult reset(const ResetOptions& options = {}) = 0;
    virtual StepResult step(const Value& action) = 0;
    virtual const Space& observation_space() const noexcept = 0;
    virtual const Space& action_space() const noexcept = 0;
};

} // namespace ppo::rl
