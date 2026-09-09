#include "ppo/rl/environment.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ppo::rl {

Box::Box(std::vector<std::size_t> shape, std::vector<float> low, std::vector<float> high)
    : shape_(std::move(shape)), low_(std::move(low)), high_(std::move(high))
{
    // An empty shape denotes a scalar, stored as one float.
    std::size_t size = 1;
    for (const auto dimension : shape_)
    {
        if (dimension == 0)
            throw std::invalid_argument("Box dimensions must be positive");
        if (dimension > std::numeric_limits<std::size_t>::max() / size)
            throw std::overflow_error("Box shape overflows size_t");
        size *= dimension;
    }
    if (low_.size() != size || high_.size() != size)
        throw std::invalid_argument("Box bounds must match its flattened shape");
    for (std::size_t index = 0; index < size; ++index)
    {
        if (std::isnan(low_[index]) || std::isnan(high_[index])
            || low_[index] > high_[index]
            || low_[index] == std::numeric_limits<float>::infinity()
            || high_[index] == -std::numeric_limits<float>::infinity())
            throw std::invalid_argument("Box bounds must admit finite values");
    }
}

const std::vector<std::size_t>& Box::shape() const noexcept { return shape_; }
const std::vector<float>& Box::low() const noexcept { return low_; }
const std::vector<float>& Box::high() const noexcept { return high_; }

bool Box::contains(const Value& value) const noexcept
{
    const auto* elements = std::get_if<std::vector<float>>(&value);
    if (elements == nullptr || elements->size() != low_.size())
        return false;
    for (std::size_t index = 0; index < elements->size(); ++index)
    {
        const float element = (*elements)[index];
        if (!std::isfinite(element) || element < low_[index] || element > high_[index])
            return false;
    }
    return true;
}

Discrete::Discrete(std::int64_t count, std::int64_t start)
    : count_(count), start_(start), last_(start)
{
    if (count <= 0)
        throw std::invalid_argument("Discrete count must be positive");
    if (start > std::numeric_limits<std::int64_t>::max() - (count - 1))
        throw std::overflow_error("Discrete last value overflows int64");
    last_ = start + (count - 1);
}

std::int64_t Discrete::count() const noexcept { return count_; }
std::int64_t Discrete::start() const noexcept { return start_; }

bool Discrete::contains(const Value& value) const noexcept
{
    const auto* integer = std::get_if<std::int64_t>(&value);
    return integer != nullptr && *integer >= start_ && *integer <= last_;
}

bool contains(const Space& space, const Value& value) noexcept
{
    if (const auto* box = std::get_if<Box>(&space))
        return box->contains(value);
    if (const auto* discrete = std::get_if<Discrete>(&space))
        return discrete->contains(value);
    return false;
}

} // namespace ppo::rl
