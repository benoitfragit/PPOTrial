/**
 * @file PPOKit/include/ppokit/backend/rltools/gaussian_mlp_learner.hpp
 * @brief PPO learner for a Gaussian MLP actor and an MLP critic generated from graph.json.
 * @details Backend layer: this header uses rl_tools directly and is meant for learner
 * translation units only, never for task code. Include the rl_tools operation headers
 * (and the generated networks header) before it:
 *
 * @code
 * #include <rl_tools/operations/cpu_mux.h>
 * #include <rl_tools/nn/operations_cpu_mux.h>
 * #include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
 * #include <rl_tools/nn/optimizers/adam/operations_generic.h>
 * #include "my_networks.hpp"                       // ppokit_generate_networks
 * #include "ppokit/backend/rltools/gaussian_mlp_learner.hpp"
 *
 * using Learner = ppokit::backend::rltools::GaussianMlpLearner<networks::Actor, networks::Critic>;
 * Learner learner(settings);
 * RolloutCollector<MyTask, Learner, Learner::Buffer> collector(task, learner);
 * collector.Collect(buffer);
 * auto const statistics = learner.Update(buffer);
 * @endcode
 *
 * One update: GAE, explained variance, advantage normalization, then `epochs` passes
 * over shuffled minibatches. For each sample, the clipped surrogate (ppokit/core) is
 * chained with the Gaussian head gradients (ppokit/heads) into the actor, and the
 * squared error to the value target into the critic. Both networks take one Adam step
 * per minibatch. Everything is reproducible from `settings.seed`.
 */
#pragma once

#ifndef RL_TOOLS_RL_TOOLS_H
#error "Include the rl_tools operation headers before gaussian_mlp_learner.hpp"
#endif

#include "ppokit/backend/rltools/mlp_probes.hpp"
#include "ppokit/core/kl.hpp"
#include "ppokit/core/ppo_settings.hpp"
#include "ppokit/core/ppo_surrogate.hpp"
#include "ppokit/heads/gaussian.hpp"
#include "ppokit/runtime/collector.hpp"
#include "ppokit/runtime/rollout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ppokit::backend::rltools {

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

/** @brief Shared PPO hyper-parameters (ppokit/core/ppo_settings.hpp). */
using PpoSettings = core::PpoSettings;
/** @brief Shared update diagnostics (ppokit/core/ppo_settings.hpp). */
using UpdateStatistics = core::UpdateStatistics;

/**
 * @brief PPO learner on generated networks.
 * @tparam ActorModel Generated `Actor` description (graph.json, "model": "mlp").
 * @tparam CriticModel Generated `Critic` description reading the actor observation.
 */
template<typename ActorModel, typename CriticModel>
class GaussianMlpLearner final
{
public:
    static constexpr std::size_t ObservationSize = ActorModel::Inputs;
    static constexpr std::size_t ActionSize = ActorModel::Outputs;
    static_assert(CriticModel::Inputs == ObservationSize,
        "GaussianMlpLearner collects one observation per step: the critic must read the actor observation");
    static_assert(CriticModel::Outputs == 1, "the critic outputs one value");
    static_assert(ActorModel::GaussianOnly,
        "GaussianMlpLearner trains Gaussian heads only; tasks with Bernoulli actions are not supported yet");

    using Observation = std::array<float, ObservationSize>;
    using Action = std::array<float, ActionSize>;
    using Buffer = runtime::RolloutBuffer<double, ObservationSize, ActionSize>;

    /** @throws std::invalid_argument for inconsistent settings. */
    explicit GaussianMlpLearner(PpoSettings const& settings)
        : settings_(core::ValidatePpoSettings(settings)), networks_(std::make_unique<Networks>()), rng_(settings.seed)
    {
        if (settings_.maxGradientNorm > 0.0)
        {
            throw std::invalid_argument("PPOKit learner: max_gradient_norm is for graph networks (set it to 0)");
        }
        Initialize();
    }

    /** @brief Samples an action; also returns its log-probability and the critic value. */
    runtime::PolicyDecision<double, ActionSize> Decide(Observation const& observation)
    {
        Action const means = Mean(observation);
        runtime::PolicyDecision<double, ActionSize> decision;
        for (std::size_t axis = 0; axis < ActionSize; ++axis)
        {
            double const logStd = LogStd(axis);
            std::normal_distribution<double> distribution(means[axis], std::exp(logStd));
            decision.action[axis] = static_cast<float>(distribution(rng_));
            decision.logProbability += heads::GaussianAxisLogDensity(static_cast<double>(decision.action[axis]),
                static_cast<double>(means[axis]), logStd);
        }
        decision.value = Value(observation);
        return decision;
    }

    /** @brief Deterministic action: the Gaussian means (inference and evaluation). */
    Action Mean(Observation const& observation)
    {
        Input(observation);
        rlt::evaluate(device_, networks_->actor, networks_->input, networks_->actorOutput, networks_->actorBuffer);
        Action means{};
        for (std::size_t axis = 0; axis < ActionSize; ++axis)
        {
            means[axis] = rlt::get(networks_->actorOutput, 0, static_cast<Index>(axis));
            if (!std::isfinite(means[axis]))
            {
                throw std::runtime_error("PPOKit learner: non-finite actor output");
            }
        }
        return means;
    }

    /** @brief Critic value of an observation. */
    double Value(Observation const& observation)
    {
        Input(observation);
        rlt::evaluate(device_, networks_->critic, networks_->input, networks_->valueOutput, networks_->criticBuffer);
        return rlt::get(networks_->valueOutput, 0, 0);
    }

    /**
     * @brief Runs one PPO update on a full rollout, then empties it.
     * @throws std::runtime_error when a loss becomes non-finite.
     */
    UpdateStatistics Update(Buffer& buffer)
    {
        buffer.ComputeAdvantages(settings_.gamma, settings_.lambda);
        UpdateStatistics statistics;
        statistics.explainedVariance = runtime::ExplainedVariance(buffer);
        buffer.NormalizeAdvantages();
        std::vector<std::size_t> order(buffer.Size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::size_t optimizedSamples = 0;
        for (std::size_t epoch = 0; epoch < settings_.epochs; ++epoch)
        {
            std::shuffle(order.begin(), order.end(), rng_);
            for (std::size_t start = 0; start < order.size(); start += settings_.minibatch)
            {
                std::size_t const end = std::min(start + settings_.minibatch, order.size());
                optimizedSamples += TrainMinibatch(buffer, order, start, end, statistics);
            }
        }
        if (optimizedSamples > 0)
        {
            double const inverse = 1.0 / static_cast<double>(optimizedSamples);
            statistics.policyLoss *= inverse;
            statistics.entropy *= inverse;
            statistics.approximateKl *= inverse;
            statistics.clipFraction *= inverse;
        }
        // Every sample, learnable or not, trains the critic.
        statistics.valueLoss /= static_cast<double>(settings_.epochs * buffer.Size());
        buffer.Clear();
        statistics.updates = ++updates_;
        return statistics;
    }

    /** @brief Current log standard deviation of one action component. */
    double LogStd(std::size_t axis) const
    {
        return rlt::get(networks_->actor.log_std.parameters, 0, static_cast<Index>(axis));
    }

private:
    using Device = rlt::devices::DEVICE_FACTORY<>;
    using Index = typename Device::index_t;
    static constexpr Index SingleSample = 1;
    using ActorNetwork = typename ActorModel::template Network<Index, SingleSample, ActorModel::DefaultWidth,
        rlt::MatrixStaticTag>;
    using CriticNetwork = typename CriticModel::template Network<Index, SingleSample, CriticModel::DefaultWidth,
        rlt::MatrixStaticTag>;

public:
    /** @brief Probes of the actor and critic layers (mlp_probes.hpp). */
    using ActorProbes = MlpProbes<ActorNetwork>;
    using CriticProbes = MlpProbes<CriticNetwork>;

    /**
     * @brief Probes fed by the next updates (null: off, the default). Read-only: a probed
     *        update gives the same numbers as an unprobed one. Reset them before an update.
     */
    void SetProbes(ActorProbes* actor, CriticProbes* critic) noexcept
    {
        actorProbes_ = actor;
        criticProbes_ = critic;
    }

private:
    using Optimizer = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<float, Index>>;
    template<std::size_t Columns>
    using Row = rlt::MatrixStatic<rlt::matrix::Specification<float, Index, 1, static_cast<Index>(Columns)>>;

    /** @brief rl_tools objects; static matrices hold self-pointers, so they live on the heap and never move. */
    struct Networks final
    {
        ActorNetwork actor;
        CriticNetwork critic;
        Optimizer actorOptimizer;
        Optimizer criticOptimizer;
        Row<ObservationSize> input;
        Row<ActionSize> actorOutput;
        Row<ActionSize> actorDerivative;
        Row<1> valueOutput;
        Row<1> valueDerivative;
        typename ActorNetwork::template Buffer<SingleSample, rlt::MatrixStaticTag> actorBuffer;
        typename CriticNetwork::template Buffer<SingleSample, rlt::MatrixStaticTag> criticBuffer;
    };

    /** @brief Weights from the seed, initial log-std, optimizers; the order is part of reproducibility. */
    void Initialize()
    {
        rlt::init_weights(device_, networks_->actor, rng_);
        rlt::init_weights(device_, networks_->critic, rng_);
        for (std::size_t axis = 0; axis < ActionSize; ++axis)
        {
            rlt::set(networks_->actor.log_std.parameters, 0, static_cast<Index>(axis),
                static_cast<float>(settings_.initialLogStd));
        }
        rlt::reset_optimizer_state(device_, networks_->actorOptimizer, networks_->actor);
        rlt::reset_optimizer_state(device_, networks_->criticOptimizer, networks_->critic);
        networks_->actorOptimizer.parameters.alpha = static_cast<float>(settings_.actorLearningRate);
        networks_->criticOptimizer.parameters.alpha = static_cast<float>(settings_.criticLearningRate);
    }

    void Input(Observation const& observation)
    {
        for (std::size_t index = 0; index < ObservationSize; ++index)
        {
            if (!std::isfinite(observation[index]))
            {
                throw std::invalid_argument("PPOKit learner: non-finite observation");
            }
            rlt::set(networks_->input, 0, static_cast<Index>(index), observation[index]);
        }
    }

    /** @brief One optimizer step of the actor and of the critic; returns the number of actor samples. */
    std::size_t TrainMinibatch(Buffer const& buffer, std::vector<std::size_t> const& order, std::size_t start,
        std::size_t end, UpdateStatistics& statistics)
    {
        rlt::zero_gradient(device_, networks_->actor);
        rlt::zero_gradient(device_, networks_->critic);
        double const size = static_cast<double>(end - start);
        std::size_t actorSamples = 0;
        minibatchSampledKl_ = 0.0;
        for (std::size_t position = start; position < end; ++position)
        {
            auto const& row = buffer[order[position]];
            Input(row.observation);
            if (row.learnable)
            {
                AccumulateActorSample(row, size, statistics);
                ++actorSamples;
            }
            AccumulateCriticSample(row, size, statistics);
        }
        // The KL guard: a minibatch that moved the policy too far does not step the actor.
        if (actorProbes_ != nullptr)
        {
            actorProbes_->ObserveGradients(networks_->actor);
        }
        if (criticProbes_ != nullptr)
        {
            criticProbes_->ObserveGradients(networks_->critic);
        }
        if (core::ActorStepAllowed(minibatchSampledKl_, settings_.targetKl))
        {
            rlt::step(device_, networks_->actorOptimizer, networks_->actor);
            ClampLogStd();
        }
        else
        {
            ++statistics.actorKlSkips;
        }
        rlt::step(device_, networks_->criticOptimizer, networks_->critic);
        return actorSamples;
    }

    /** @brief Chains the clipped surrogate with the Gaussian head into the actor gradients. */
    void AccumulateActorSample(typename Buffer::Row const& row, double minibatchSize, UpdateStatistics& statistics)
    {
        rlt::forward(device_, networks_->actor, networks_->input, networks_->actorOutput);
        if (actorProbes_ != nullptr)
        {
            actorProbes_->ObserveForward(networks_->actor, SingleSample);
        }
        double logProbability = 0;
        for (std::size_t axis = 0; axis < ActionSize; ++axis)
        {
            logProbability += heads::GaussianAxisLogDensity(static_cast<double>(row.action[axis]), MeanOf(axis),
                LogStd(axis));
        }
        auto const surrogate = core::EvaluateClippedSurrogate(logProbability, row.logProbability, row.advantage,
            settings_.clip);
        if (!std::isfinite(surrogate.objective))
        {
            throw std::runtime_error("PPOKit learner: non-finite PPO objective");
        }
        double entropy = 0;
        for (std::size_t axis = 0; axis < ActionSize; ++axis)
        {
            double const logStd = LogStd(axis);
            auto const gradient =
                heads::GaussianAxisLogDensityGradient(static_cast<double>(row.action[axis]), MeanOf(axis), logStd);
            // Mean over the minibatch: each sample contributes derivative / minibatch size.
            double const meanDerivative = surrogate.logProbabilityGradient * gradient.mean / minibatchSize;
            rlt::set(networks_->actorDerivative, 0, static_cast<Index>(axis), static_cast<float>(meanDerivative));
            // The entropy bonus adds d entropy / d log-std = 1 per component.
            double const logStdDerivative =
                (surrogate.logProbabilityGradient * gradient.logStd - settings_.entropyCoefficient) / minibatchSize;
            rlt::increment(networks_->actor.log_std.gradient, 0, static_cast<Index>(axis),
                static_cast<float>(logStdDerivative));
            entropy += heads::GaussianAxisEntropy(logStd);
        }
        rlt::backward(device_, networks_->actor, networks_->input, networks_->actorDerivative,
            networks_->actorBuffer);
        statistics.policyLoss += -surrogate.objective - settings_.entropyCoefficient * entropy;
        statistics.entropy += entropy;
        statistics.approximateKl += core::ApproximateKl(surrogate.logRatio);
        minibatchSampledKl_ += core::SampledKl(surrogate.logRatio) / minibatchSize;
        statistics.clipFraction += surrogate.outsideClipRange ? 1.0 : 0.0;
    }

    /** @brief Gradient of (V(s) - target)^2, averaged over the minibatch. */
    void AccumulateCriticSample(typename Buffer::Row const& row, double minibatchSize, UpdateStatistics& statistics)
    {
        rlt::forward(device_, networks_->critic, networks_->input, networks_->valueOutput);
        if (criticProbes_ != nullptr)
        {
            criticProbes_->ObserveForward(networks_->critic, SingleSample);
        }
        double const error = rlt::get(networks_->valueOutput, 0, 0) - row.valueTarget;
        if (!std::isfinite(error))
        {
            throw std::runtime_error("PPOKit learner: non-finite value error");
        }
        rlt::set(networks_->valueDerivative, 0, 0, static_cast<float>(2.0 * error / minibatchSize));
        rlt::backward(device_, networks_->critic, networks_->input, networks_->valueDerivative,
            networks_->criticBuffer);
        statistics.valueLoss += error * error;
    }

    double MeanOf(std::size_t axis) const
    {
        return rlt::get(networks_->actorOutput, 0, static_cast<Index>(axis));
    }

    void ClampLogStd() noexcept
    {
        for (std::size_t axis = 0; axis < ActionSize; ++axis)
        {
            float const clamped = std::clamp(static_cast<float>(LogStd(axis)),
                static_cast<float>(settings_.minimumLogStd), static_cast<float>(settings_.maximumLogStd));
            rlt::set(networks_->actor.log_std.parameters, 0, static_cast<Index>(axis), clamped);
        }
    }

    PpoSettings settings_;
    Device device_{};
    std::unique_ptr<Networks> networks_;
    std::mt19937 rng_;
    std::uint64_t updates_ = 0;
    /** @brief Sampled KL of the minibatch being accumulated, for the KL guard. */
    double minibatchSampledKl_ = 0.0;
    ActorProbes* actorProbes_ = nullptr;
    CriticProbes* criticProbes_ = nullptr;
};

} // namespace ppokit::backend::rltools
