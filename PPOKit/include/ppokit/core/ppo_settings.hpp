/**
 * @file PPOKit/include/ppokit/core/ppo_settings.hpp
 * @brief PPO hyper-parameters and update diagnostics shared by every PPOKit learner.
 * @details Independent of any backend: the rl_tools learner and the graph learner read
 * the same settings and report the same statistics.
 */
#pragma once

#include "ppokit/core/ppo_parameter_specs.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ppokit::core {

/**
 * @brief Order of the floating-point operations of an Adam step. Both compute the same
 *        update; they round differently.
 */
enum class AdamArithmetic : std::uint8_t
{
    /** @brief m / (1 - beta1^t) and v / (1 - beta2^t), as Kingma and Ba write it (default). */
    Reference,
    /**
     * @brief As rl_tools::nn::optimizers::Adam: the rate times 1 / (1 - beta1^t) times m, over
     *        sqrt(v times 1 / (1 - beta2^t)) plus epsilon. A graph learner then takes the
     *        steps of an rl_tools learner bit for bit.
     */
    RlTools
};

/** @brief Adam epsilon of rl_tools' default parameters (TensorFlow's). */
inline constexpr double RlToolsAdamEpsilon = 1e-7;

/** @brief PPO hyper-parameters; see docs/architecture/ppokit-framework-design.md §13.4 for their effect. */
struct PpoSettings final
{
    /** @brief Discount factor: weight of future rewards. */
    double gamma = 0.99;
    /** @brief GAE bias/variance trade-off. */
    double lambda = 0.95;
    /** @brief PPO clipping range epsilon. */
    double clip = 0.2;
    /** @brief Adam learning rate of the actor. */
    double actorLearningRate = 3e-4;
    /** @brief Adam learning rate of the critic. */
    double criticLearningRate = 1e-3;
    /** @brief Weight of the entropy bonus (exploration). */
    double entropyCoefficient = 0.0;
    /**
     * @brief Weight of the value loss in the one loss of an actor and a critic that share
     *        blocks (graph learner); unused when they are optimized apart.
     */
    double valueCoefficient = 0.5;
    /**
     * @brief Largest global L2 norm of the gradients of one optimizer step; larger ones are
     *        scaled down (as torch's clip_grad_norm_). 0 turns it off. Graph learner only.
     */
    double maxGradientNorm = 0.0;
    /**
     * @brief Target KL per minibatch: the actor step is skipped when the sampled KL reaches
     *        1.5 x this value (ppokit/core/kl.hpp); 0 turns the guard off.
     */
    double targetKl = 0.0;
    /** @brief Passes over each rollout. */
    std::size_t epochs = 4;
    /** @brief Samples per optimizer step. */
    std::size_t minibatch = 64;
    /** @brief Initial log standard deviation of every Gaussian action component. */
    double initialLogStd = -0.5;
    /** @brief Bounds of the log standard deviation, kept after every actor step. */
    double minimumLogStd = -5.0;
    double maximumLogStd = 2.0;
    /** @brief Seed of weights, sampling and minibatch shuffling. */
    std::uint32_t seed = 1;
    /** @brief Arithmetic of the Adam steps (train.json "optimizer"). Graph learners only. */
    AdamArithmetic adamArithmetic = AdamArithmetic::Reference;
    /** @brief Epsilon added to sqrt(v) by Adam (train.json "optimizer"). Graph learners only. */
    double adamEpsilon = 1e-8;
};

/** @brief Diagnostics of one update, averaged over the optimized samples. */
struct UpdateStatistics final
{
    std::uint64_t updates = 0;
    double policyLoss = 0;
    double valueLoss = 0;
    double entropy = 0;
    /** @brief Mean of r - 1 - log r, an estimate of KL(old || new). */
    double approximateKl = 0;
    /** @brief Fraction of samples outside the clipping range. */
    double clipFraction = 0;
    /** @brief Critic quality before the update (1 is perfect). */
    double explainedVariance = 0;
    /** @brief Minibatches whose actor step the KL guard skipped. */
    std::size_t actorKlSkips = 0;
};

/**
 * @brief Returns the settings when they are consistent.
 * @throws std::invalid_argument naming the learner for out-of-range values.
 */
inline PpoSettings ValidatePpoSettings(PpoSettings const& settings)
{
    auto const require = [](PpoParameter parameter, double value) {
        if (!AcceptsValue(parameter, value))
        {
            throw std::invalid_argument(std::string("invalid PPO parameter: ") + SpecOf(parameter).name);
        }
    };
    require(PpoParameter::Gamma, settings.gamma);
    require(PpoParameter::Lambda, settings.lambda);
    require(PpoParameter::Clip, settings.clip);
    require(PpoParameter::ActorLearningRate, settings.actorLearningRate);
    require(PpoParameter::CriticLearningRate, settings.criticLearningRate);
    require(PpoParameter::EntropyCoefficient, settings.entropyCoefficient);
    require(PpoParameter::ValueCoefficient, settings.valueCoefficient);
    require(PpoParameter::TargetKl, settings.targetKl);
    require(PpoParameter::MaxGradientNorm, settings.maxGradientNorm);
    require(PpoParameter::InitialLogStd, settings.initialLogStd);
    require(PpoParameter::MinimumLogStd, settings.minimumLogStd);
    require(PpoParameter::MaximumLogStd, settings.maximumLogStd);
    require(PpoParameter::Epochs, static_cast<double>(settings.epochs));
    require(PpoParameter::Minibatch, static_cast<double>(settings.minibatch));
    if (!(settings.adamEpsilon > 0.0 && settings.adamEpsilon <= 1e-3))
    {
        throw std::invalid_argument("invalid Adam epsilon: it must lie in (0, 0.001]");
    }
    if (settings.minimumLogStd >= settings.maximumLogStd || settings.initialLogStd < settings.minimumLogStd
        || settings.initialLogStd > settings.maximumLogStd)
    {
        throw std::invalid_argument("initial_log_std must lie inside the ordered log-std bounds");
    }
    return settings;
}

/**
 * @brief Settings to run a policy without training it: the learning rates play no part, so
 *        the critic's is set to the actor's, which a critic sharing actor blocks requires.
 */
inline PpoSettings InferenceSettings(PpoSettings settings) noexcept
{
    settings.criticLearningRate = settings.actorLearningRate;
    return settings;
}

} // namespace ppokit::core
