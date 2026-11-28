/**
 * @file PPOKit/tests/adam_rltools_parity_tests.cpp
 * @brief The graph Adam in its RlTools arithmetic takes the steps of rl_tools' Adam bit for bit.
 * @details One rl_tools dense layer and two PPOKit tensors start from the same weights and
 * receive the same gradients for many steps (moments grow, shrink and change sign); after
 * every step the weights and both moments must be equal. The Reference arithmetic, kept as
 * the default, must round differently somewhere, so that the test would notice a mode that
 * does nothing. The train.json "optimizer" section is read into the learner settings.
 */
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "ppokit/graph/adam.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace rlt = rl_tools;
using Device = rlt::devices::DEVICE_FACTORY<>;
using Index = typename Device::index_t;

constexpr Index Inputs = 7;
constexpr Index Outputs = 5;
constexpr int Steps = 400;

using LayerSpecification = rlt::nn::layers::dense::Specification<float, Index, Inputs, Outputs,
    rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::parameters::Adam, 1>;
using Layer = rlt::nn::layers::dense::LayerBackwardGradient<LayerSpecification>;
using OptimizerSpecification = rlt::nn::optimizers::adam::Specification<float, Index>;
using Optimizer = rlt::nn::optimizers::Adam<OptimizerSpecification>;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAILED: " << description << '\n';
    }
}

bool SameBits(float a, float b) noexcept
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

/** @brief rl_tools layer, its Adam, and the same tensors as PPOKit parameters. */
class Pair final
{
public:
    explicit Pair(ppokit::core::AdamArithmetic arithmetic) : weights_(Inputs * Outputs), biases_(Outputs)
    {
        rlt::malloc(device_, layer_);
        std::mt19937 rng(11);
        rlt::init_weights(device_, layer_, rng);
        rlt::reset_optimizer_state(device_, optimizer_, layer_);
        optimizer_.parameters.alpha = 3e-4F;
        for (Index out = 0; out < Outputs; ++out)
        {
            for (Index in = 0; in < Inputs; ++in)
            {
                weights_.value[Flat(out, in)] = rlt::get(layer_.weights.parameters, out, in);
            }
            biases_.value[out] = rlt::get(layer_.biases.parameters, 0, out);
        }
        ppokit::core::PpoSettings settings;
        settings.adamArithmetic = arithmetic;
        settings.adamEpsilon = ppokit::core::RlToolsAdamEpsilon;
        adam_ = std::make_unique<ppokit::graph::Adam>(
            std::vector<ppokit::graph::Parameter*>{&weights_, &biases_}, ppokit::graph::AdamFor(settings, 3e-4));
    }

    ~Pair() { rlt::free(device_, layer_); }
    Pair(Pair const&) = delete;
    Pair& operator=(Pair const&) = delete;

    /** @brief The same gradients to both, then one step of each. */
    void Step(std::mt19937& rng, float scale)
    {
        std::normal_distribution<float> gradient(0.F, scale);
        rlt::zero_gradient(device_, layer_);
        for (Index out = 0; out < Outputs; ++out)
        {
            for (Index in = 0; in < Inputs; ++in)
            {
                float const value = gradient(rng);
                rlt::set(layer_.weights.gradient, out, in, value);
                weights_.gradient[Flat(out, in)] = value;
            }
            float const value = gradient(rng);
            rlt::set(layer_.biases.gradient, 0, out, value);
            biases_.gradient[out] = value;
        }
        rlt::step(device_, optimizer_, layer_);
        adam_->Step();
    }

    /** @brief Whether weights and both moments carry the same bits. */
    bool Identical() const
    {
        ppokit::graph::Adam::State const state = adam_->Save();
        bool same = true;
        for (Index out = 0; out < Outputs; ++out)
        {
            for (Index in = 0; in < Inputs; ++in)
            {
                std::size_t const flat = Flat(out, in);
                same = same && SameBits(weights_.value[flat], rlt::get(layer_.weights.parameters, out, in))
                    && SameBits(state.first[0][flat], rlt::get(layer_.weights.gradient_first_order_moment, out, in))
                    && SameBits(state.second[0][flat], rlt::get(layer_.weights.gradient_second_order_moment, out, in));
            }
            same = same && SameBits(biases_.value[out], rlt::get(layer_.biases.parameters, 0, out))
                && SameBits(state.first[1][out], rlt::get(layer_.biases.gradient_first_order_moment, 0, out))
                && SameBits(state.second[1][out], rlt::get(layer_.biases.gradient_second_order_moment, 0, out));
        }
        return same;
    }

private:
    static std::size_t Flat(Index out, Index in) noexcept
    {
        return static_cast<std::size_t>(out) * Inputs + static_cast<std::size_t>(in);
    }

    Device device_{};
    Layer layer_{};
    Optimizer optimizer_{};
    ppokit::graph::Parameter weights_;
    ppokit::graph::Parameter biases_;
    std::unique_ptr<ppokit::graph::Adam> adam_;
};

/** @brief Gradients of changing size: small, large, then small again. */
float ScaleOf(int step) noexcept
{
    if (step < 100)
    {
        return 1e-3F;
    }
    if (step < 200)
    {
        return 10.F;
    }
    return 1e-2F;
}

void RlToolsArithmeticIsBitIdentical()
{
    Pair pair(ppokit::core::AdamArithmetic::RlTools);
    Check(pair.Identical(), "adam parity: same starting weights");
    std::mt19937 rng(3);
    int firstDifference = -1;
    for (int step = 0; step < Steps && firstDifference < 0; ++step)
    {
        pair.Step(rng, ScaleOf(step));
        firstDifference = pair.Identical() ? -1 : step;
    }
    Check(firstDifference < 0,
        "adam parity: rl_tools arithmetic equals rl_tools' Adam (first difference at step "
            + std::to_string(firstDifference) + ")");
}

void ReferenceArithmeticRoundsDifferently()
{
    Pair pair(ppokit::core::AdamArithmetic::Reference);
    std::mt19937 rng(3);
    bool differed = false;
    for (int step = 0; step < Steps && !differed; ++step)
    {
        pair.Step(rng, ScaleOf(step));
        differed = !pair.Identical();
    }
    Check(differed, "adam parity: the reference arithmetic is another rounding (the mode matters)");
}

void TrainJsonSelectsTheArithmetic()
{
    ppokit::schema::TaskSchema const task = ppokit::schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
    auto const settingsOf = [&](std::string const& optimizer) {
        std::string const text = std::string(R"({"schema": "ppokit.train/1")") + optimizer + "}";
        return ppokit::schema::ParseTrainingPlan(text, "train.json", task).ppo.Settings();
    };
    ppokit::core::PpoSettings const reference = settingsOf("");
    Check(reference.adamArithmetic == ppokit::core::AdamArithmetic::Reference && reference.adamEpsilon == 1e-8,
        "train.json: without \"optimizer\", the reference Adam with epsilon 1e-8");
    ppokit::core::PpoSettings const rltools = settingsOf(R"(, "optimizer": {"adam": "rltools"})");
    Check(rltools.adamArithmetic == ppokit::core::AdamArithmetic::RlTools && rltools.adamEpsilon == 1e-7,
        "train.json: \"rltools\" takes rl_tools' epsilon 1e-7 by default");
    ppokit::core::PpoSettings const chosen = settingsOf(R"(, "optimizer": {"adam": "rltools", "epsilon": 1e-6})");
    Check(chosen.adamEpsilon == 1e-6, "train.json: an explicit epsilon wins");
    for (char const* bad : {R"(, "optimizer": {"adam": "sgd"})", R"(, "optimizer": {"epsilon": 0})",
             R"(, "optimizer": {"adam": "rltools", "beta": 0.9})"})
    {
        bool rejected = false;
        try
        {
            (void)settingsOf(bad);
        }
        catch (ppokit::schema::SchemaError const&)
        {
            rejected = true;
        }
        Check(rejected, std::string("train.json: rejects ") + bad);
    }
}

} // namespace

int main()
{
    RlToolsArithmeticIsBitIdentical();
    ReferenceArithmeticRoundsDifferently();
    TrainJsonSelectsTheArithmetic();
    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Adam rl_tools parity tests passed\n";
    return EXIT_SUCCESS;
}
