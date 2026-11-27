/**
 * @file PPOKit/tests/components/test_components.cpp
 * @brief A component DLL written with the SDK, for the component tests: a trainable
 *        scale, a set counter without gradient, a set gate, a stateful memory and a
 *        component whose forward fails, and two observers that check what they are given.
 */
#include "ppokit/component/sdk.hpp"

#include <array>
#include <cmath>
#include <string>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using ppokit::sdk::Call;
using ppokit::sdk::Config;
using ppokit::sdk::GradientCall;
using ppokit::sdk::Manifest;
using ppokit::sdk::Shapes;

/** @brief y = w * x, element by element; w is trained. */
class Scale final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("scale", 1);
        manifest.Input("x", {"N"}).Output("y", {"N"});
        manifest.Trainable();
        manifest.Number("initial", 1.0, -10.0, 10.0);
        return manifest;
    }
    Scale(Config const& config, Shapes const& shapes, std::uint64_t /*seed*/)
        : weights_(shapes["N"], static_cast<float>(config.Number("initial")))
    {
    }
    void Forward(Call const& call) override
    {
        // The input is kept in the cache: backward needs it.
        auto* const cache = static_cast<float*>(call.Cache());
        for (std::size_t index = 0; index < weights_.size(); ++index)
        {
            cache[index] = call.Input(0)[index];
            call.Output(0)[index] = weights_[index] * call.Input(0)[index];
        }
    }
    std::size_t CacheSize(std::uint32_t batch) const override { return batch * weights_.size() * sizeof(float); }
    void Backward(GradientCall const& call) override
    {
        auto const* const input = static_cast<float const*>(call.Cache());
        for (std::size_t index = 0; index < weights_.size(); ++index)
        {
            call.InputGradient(0)[index] += weights_[index] * call.OutputGradient(0)[index];
            call.ParameterGradients()[index] += input[index] * call.OutputGradient(0)[index];
        }
    }
    float* Parameters(std::size_t& count) override
    {
        count = weights_.size();
        return weights_.data();
    }

private:
    std::vector<float> weights_;
};

/** @brief Number of present entities of a set; no gradient. */
class CountPresent final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("count_present", 1);
        manifest.Input("tracks", {"N", "F"}, "presence").Mask("presence", {"N"}).Output("count", {1});
        return manifest;
    }
    CountPresent(Config const&, Shapes const& shapes, std::uint64_t) : capacity_(shapes["N"]) {}
    void Forward(Call const& call) override
    {
        float count = 0.F;
        for (std::size_t entity = 0; entity < capacity_; ++entity)
        {
            count += call.Mask(1)[entity] != 0 ? 1.F : 0.F;
        }
        call.Output(0)[0] = count;
    }

private:
    std::size_t capacity_;
};

/** @brief Multiplies every feature of a set by a fixed gate; differentiable, follows the input mask. */
class SetGate final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("set_gate", 1);
        manifest.Input("tracks", {"N", "F"}, "presence").Mask("presence", {"N"});
        manifest.Output("gated", {"N", "F"}, "presence");
        manifest.Differentiable();
        manifest.Number("gate", 0.5, 0.0, 1.0);
        return manifest;
    }
    SetGate(Config const& config, Shapes const& shapes, std::uint64_t)
        : gate_(static_cast<float>(config.Number("gate"))), size_(shapes["N"] * shapes["F"])
    {
    }
    void Forward(Call const& call) override
    {
        for (std::size_t index = 0; index < size_; ++index)
        {
            call.Output(0)[index] = gate_ * call.Input(0)[index];
        }
    }
    void Backward(GradientCall const& call) override
    {
        for (std::size_t index = 0; index < size_; ++index)
        {
            call.InputGradient(0)[index] += gate_ * call.OutputGradient(0)[index];
        }
    }

private:
    float gate_;
    std::size_t size_;
};

/** @brief Stateful: graphs refuse it. */
class Memory final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("memory", 1);
        manifest.Input("x", {"N"}).Output("y", {"N"});
        manifest.Stateful();
        return manifest;
    }
    Memory(Config const&, Shapes const& shapes, std::uint64_t) : size_(shapes["N"]) {}
    void Forward(Call const& call) override { std::memcpy(call.Output(0), call.Input(0), size_ * sizeof(float)); }

private:
    std::size_t size_;
};

/** @brief Its forward always fails, with a message PPOKit must report. */
class Failing final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("failing", 1);
        manifest.Input("x", {"N"}).Output("y", {"N"});
        return manifest;
    }
    Failing(Config const&, Shapes const&, std::uint64_t) {}
    void Forward(Call const&) override { throw std::runtime_error("sensor offline"); }
};

/**
 * @brief An observer that checks what it is given: the sum of every value of every agent
 *        and the number of agents must be the configured ones, otherwise it fails.
 */
class Checksum final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("checksum", 1);
        manifest.Input("pose", {3}).Input("tracks", {"N", 3}).Input("controls", {3}).Input("rewards", {2});
        manifest.Observer();
        manifest.Number("expected_sum", 0.0, -1.0e6, 1.0e6);
        manifest.Integer("expected_agents", 1, 1, 16);
        return manifest;
    }
    Checksum(Config const& config, Shapes const& shapes, std::uint64_t)
        : expectedSum_(config.Number("expected_sum")),
          expectedAgents_(static_cast<std::uint32_t>(config.Integer("expected_agents"))),
          sizes_{3, shapes["N"] * 3, 3, 2}
    {
    }
    void Forward(Call const& call) override
    {
        if (call.Batch() != expectedAgents_)
        {
            throw std::runtime_error("expected " + std::to_string(expectedAgents_) + " agents, got "
                + std::to_string(call.Batch()));
        }
        double sum = 0.0;
        for (std::size_t port = 0; port < sizes_.size(); ++port)
        {
            float const* const values = call.Input(port);
            for (std::size_t index = 0; index < sizes_[port] * call.Batch(); ++index)
            {
                sum += values[index];
            }
        }
        if (std::abs(sum - expectedSum_) > 1e-4)
        {
            throw std::runtime_error("sum " + std::to_string(sum) + " instead of " + std::to_string(expectedSum_));
        }
    }

private:
    double expectedSum_;
    std::uint32_t expectedAgents_;
    std::array<std::size_t, 4> sizes_;
};

/** @brief An observer that fails when a value it is given exceeds its configured limit. */
class Bounded final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("bounded", 1);
        manifest.Input("values", {"N"});
        manifest.Observer();
        manifest.Number("limit", 1.0, 0.0, 1.0e6);
        return manifest;
    }
    Bounded(Config const& config, Shapes const& shapes, std::uint64_t)
        : limit_(config.Number("limit")), size_(shapes["N"])
    {
    }
    void Forward(Call const& call) override
    {
        for (std::size_t index = 0; index < size_ * call.Batch(); ++index)
        {
            if (std::abs(call.Input(0)[index]) > limit_)
            {
                throw std::runtime_error("value " + std::to_string(call.Input(0)[index]) + " above the limit");
            }
        }
    }

private:
    double limit_;
    std::size_t size_;
};

} // namespace

PPOKIT_COMPONENTS(Scale, CountPresent, SetGate, Memory, Failing, Checksum, Bounded)
