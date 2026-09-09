#include "ppo/rl/environment.hpp"

#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void test_box()
{
    const ppo::rl::Box box({2}, {-1.F, 0.F}, {1.F, 2.F});
    require(box.contains(std::vector<float>{-1.F, 2.F}), "inclusive Box bounds");
    require(!box.contains(std::vector<float>{0.F}), "Box shape mismatch");
    require(!box.contains(std::int64_t{0}), "Box rejects integer storage");
    require(!box.contains(std::vector<float>{2.F, 1.F}), "Box out of bounds");
    require(!box.contains(std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.F}), "Box rejects NaN");
    const float infinity = std::numeric_limits<float>::infinity();
    const ppo::rl::Box unbounded({}, {-infinity}, {infinity});
    require(unbounded.contains(std::vector<float>{42.F}), "scalar unbounded Box");
    require(!unbounded.contains(std::vector<float>{infinity}), "observations must be finite");
    const ppo::rl::Space space = box;
    require(ppo::rl::contains(space, std::vector<float>{0.F, 1.F}), "Box dispatch");
}

bool invalid_box(std::vector<std::size_t> shape, std::vector<float> low, std::vector<float> high)
{
    try
    {
        const ppo::rl::Box box(shape, low, high);
        require(!box.low().empty(), "valid Box has bounds");
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

void test_invalid_spaces()
{
    require(invalid_box({0}, {}, {}), "zero dimension rejected");
    require(invalid_box({2}, {0.F}, {1.F}), "wrong bound size rejected");
    require(invalid_box({1}, {2.F}, {1.F}), "reversed bounds rejected");
    require(invalid_box({1}, {std::numeric_limits<float>::quiet_NaN()}, {1.F}), "NaN bound rejected");
    const float infinity = std::numeric_limits<float>::infinity();
    require(invalid_box({1}, {infinity}, {infinity}), "empty finite domain rejected");
    bool overflow = false;
    try
    {
        const ppo::rl::Box box({std::numeric_limits<std::size_t>::max(), 2}, {}, {});
        require(!box.shape().empty(), "shape exists");
    }
    catch (const std::overflow_error&)
    {
        overflow = true;
    }
    require(overflow, "shape product overflow rejected");

    bool invalid_count = false;
    try
    {
        const ppo::rl::Discrete space(0);
        require(space.count() > 0, "positive count");
    }
    catch (const std::invalid_argument&)
    {
        invalid_count = true;
    }
    require(invalid_count, "zero discrete count rejected");

    overflow = false;
    try
    {
        const ppo::rl::Discrete space(2, std::numeric_limits<std::int64_t>::max());
        require(space.count() > 0, "positive count");
    }
    catch (const std::overflow_error&)
    {
        overflow = true;
    }
    require(overflow, "discrete endpoint overflow rejected");
}

void test_discrete()
{
    const ppo::rl::Discrete space(3, -1);
    require(space.contains(std::int64_t{-1}) && space.contains(std::int64_t{1}), "discrete endpoints");
    require(!space.contains(std::int64_t{2}), "discrete upper bound excluded");
    require(!space.contains(std::vector<float>{0.F}), "discrete rejects float storage");
    const auto maximum = std::numeric_limits<std::int64_t>::max();
    require(ppo::rl::Discrete(1, maximum).contains(maximum), "int64 maximum allowed");
    const auto minimum = std::numeric_limits<std::int64_t>::min();
    require(ppo::rl::Discrete(2, minimum).contains(minimum), "int64 minimum allowed");
    require(ppo::rl::contains(ppo::rl::Space{space}, std::int64_t{0}), "Discrete dispatch");
}

// Neutral test fixture: action 1 completes a task, action 0 waits, with a two-step limit.
class CounterEnvironment final : public ppo::rl::IEnvironment
{
public:
    ppo::rl::ResetResult reset(const ppo::rl::ResetOptions& options = {}) override
    {
        if (!options.options.empty())
            throw std::invalid_argument("CounterEnvironment has no custom reset options");
        if (options.seed)
            random_.seed(*options.seed);
        initial_ = static_cast<float>(random_() % 100);
        steps_ = 0;
        active_ = true;
        return {std::vector<float>{initial_}, {}};
    }

    ppo::rl::StepResult step(const ppo::rl::Value& action) override
    {
        if (!active_)
            throw std::logic_error("reset required");
        if (!ppo::rl::contains(actions_, action))
            throw std::invalid_argument("invalid counter action");
        ++steps_;
        const bool terminated = std::get<std::int64_t>(action) == 1;
        const bool truncated = steps_ == 2;
        active_ = !(terminated || truncated);
        return {std::vector<float>{initial_ + static_cast<float>(steps_)},
                terminated ? 1.0 : 0.0, terminated, truncated, {{"success", terminated}}};
    }

    const ppo::rl::Space& observation_space() const noexcept override { return observations_; }
    const ppo::rl::Space& action_space() const noexcept override { return actions_; }

private:
    ppo::rl::Space observations_ = ppo::rl::Box({1}, {0.F}, {101.F});
    ppo::rl::Space actions_ = ppo::rl::Discrete(2);
    std::mt19937_64 random_{0};
    float initial_ = 0.F;
    int steps_ = 0;
    bool active_ = false;
};

bool rejects_step_before_reset(ppo::rl::IEnvironment& environment)
{
    try
    {
        environment.step(std::int64_t{0});
    }
    catch (const std::logic_error&)
    {
        return true;
    }
    return false;
}

void test_lifecycle()
{
    CounterEnvironment implementation;
    ppo::rl::IEnvironment& environment = implementation;
    require(rejects_step_before_reset(environment), "initial reset required");
    const auto initial = environment.reset({42, {}});
    const auto first = environment.step(std::int64_t{0});
    const auto timeout = environment.step(std::int64_t{0});
    require(!first.terminated && !first.truncated, "ordinary transition");
    require(!timeout.terminated && timeout.truncated, "timeout is not termination");
    require(ppo::rl::contains(environment.observation_space(), timeout.observation), "final observation in space");
    require(std::get<std::vector<float>>(timeout.observation)[0]
        == std::get<std::vector<float>>(initial.observation)[0] + 2.F, "final observation preserved");
    require(rejects_step_before_reset(environment), "reset required after truncation");
    const auto repeated = environment.reset({42, {}});
    require(repeated.observation == initial.observation, "seeded reset reproducible");
    const auto terminal = environment.step(std::int64_t{1});
    require(terminal.terminated && !terminal.truncated, "MDP termination distinct");
    require(terminal.reward == 1.0 && std::get<bool>(terminal.info.at("success")), "reward and info delivered");
    require(rejects_step_before_reset(environment), "reset required after termination");
    environment.reset();
    environment.step(std::int64_t{0});
    const auto both = environment.step(std::int64_t{1});
    require(both.terminated && both.truncated, "flags may coexist");
    environment.reset();
    bool rejected = false;
    try
    {
        environment.step(std::int64_t{2});
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected, "invalid action rejected");
    require(!environment.step(std::int64_t{0}).truncated, "invalid action did not advance state");
}

} // namespace

int main()
{
    try
    {
        test_box();
        test_invalid_spaces();
        test_discrete();
        test_lifecycle();
        std::cout << "4 environment contract test groups passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
