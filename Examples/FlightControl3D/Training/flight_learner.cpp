/**
 * @file flight_learner.cpp
 * @brief PPO with GAE on RLtools MLPs: collection buffer, update loop, KL guard and checkpoints.
 *
 * One update (Optimize), in order:
 * 1. GAE advantages and value targets on the full rollout (ComputeAdvantages);
 * 2. advantage normalization to zero mean and unit variance (epsilon 1e-8);
 * 3. for each epoch, a shuffle of the sample order with the learner RNG, then for each
 *    minibatch:
 *    a. per sample: actor forward, clipped PPO loss and its derivatives (flight_policy.hpp),
 *       sampled and approximate KL accumulation, actor backward, critic forward and
 *       backward on the squared error to the target;
 *    b. KL guard: the actor step is skipped when the sampled minibatch KL reaches 1.5 x
 *       targetKl; the critic, an independent network, always steps;
 *    c. log-std clamped back into its bounds after each actor step; every parameter and
 *       Adam moment is checked for finiteness;
 * 4. rollout statistics (mean reward per channel) and an empty buffer.
 *
 * Any exception during an update marks the learner as failed: it must be reloaded from a
 * valid checkpoint, because the networks may hold a partial update.
 *
 * Read next: flight_policy.hpp (loss formulas), flight_model_store.cpp (BEST),
 * Runtime/flight_session.cpp (collection loop).
 */
#include "flight_learner.hpp"
#include "flight_atomic_file.hpp"
#include "flight_rng_checkpoint.hpp"
// Upstream CPU headers define non-inline symbols. Use its supported namespace
// wrapper so this backend can coexist with BlueDefense in the same executable.
#define RL_TOOLS_NAMESPACE_WRAPPER ppo_flight_backend
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <vector>

namespace ppo::examples::flight {
namespace rlt = ppo_flight_backend::rl_tools;

void ComputeAdvantages(Transition* transitions, std::size_t count, TrainingSettings const& settings) noexcept
{
    double advantage = 0;
    for (std::size_t index = count; index > 0; --index)
    {
        auto& row = transitions[index - 1];
        // The recursion restarts at every episode boundary: the next row
        // belongs to another episode, whatever the reason the episode ended.
        bool const episodeEnded = row.terminated || row.truncated;
        advantage = AdvantageDelta(row.reward, row.value, row.nextValue, row.terminated, settings.gamma)
            + (episodeEnded ? 0 : settings.gamma * settings.lambda * advantage);
        row.advantage = advantage;
        row.target = advantage + row.value;
    }
}

/** @brief Backend-independent learner state; Learner<Width> adds the networks. */
struct FlightLearner::Impl
{
    Config config;
    /** @brief Learner RNG: weight initialization, action sampling and minibatch shuffles. */
    std::mt19937 rng;
    /** @brief Fixed-capacity rollout buffer. */
    std::vector<Transition> rows;
    /** @brief Sample order, shuffled at each epoch. */
    std::vector<std::size_t> order;
    std::size_t collected{};
    LearningStats stats{};
    double rewardSum{};
    std::array<double, RewardCount> rewardSums{};
    /** @brief Set when an update threw; the learner must be reloaded. */
    bool failed{};

    explicit Impl(Config const& configuration)
        : config(configuration), rng(configuration.training.seed), rows(configuration.training.rolloutSteps),
          order(rows.size())
    {
        std::iota(order.begin(), order.end(), 0U);
    }
    virtual ~Impl() = default;
    virtual PolicyDistribution Evaluate(Observation const&) = 0;
    virtual double Value(Observation const&) = 0;
    virtual void Optimize() = 0;
    virtual void Write(std::ostream&) const = 0;
    virtual void Read(std::istream&) = 0;

    void Check() const
    {
        if (failed)
        {
            throw std::runtime_error("FlightControl3D: learner failed; restart from a valid checkpoint");
        }
    }
};

namespace {
using Device = rlt::devices::DEVICE_FACTORY<>;
using Index = typename Device::index_t;
using Optimizer = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<float, Index>>;

/** @brief Input layer, two hidden layers and output layer: three weight layers after the input. */
constexpr Index NetworkLayers = 3;
/** @brief Batch size of the per-sample forward and backward passes. */
constexpr Index SingleSampleBatch = 1;

template<Index Width, Index Outputs>
using Structure = rlt::nn_models::mlp::StructureSpecification<float, Index, ObservationSize, Outputs,
    NetworkLayers,
    Width, rlt::nn::activation_functions::TANH, rlt::nn::activation_functions::IDENTITY, SingleSampleBatch,
    rlt::MatrixStaticTag>;
template<Index Width, Index Outputs>
using Network = rlt::nn_models::mlp::NeuralNetworkAdam<rlt::nn_models::mlp::AdamSpecification<Structure<Width,
    Outputs>>>;
template<Index Width>
using Actor = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<
    rlt::nn_models::mlp::AdamSpecification<Structure<Width, PolicyOutputs>>>;
template<Index Columns>
using Matrix = rlt::MatrixStatic<rlt::matrix::Specification<float, Index, 1, Columns>>;

/** @brief Header tag of the serialized actor, critic, optimizer and RNG state. */
constexpr char CheckpointMagic[] = "PPO_FLIGHT_POLICY_V3";
/** @brief Added to the advantage variance before the square root. */
constexpr double AdvantageEpsilon = 1e-8;
/** @brief The actor step is skipped when the sampled minibatch KL reaches this multiple of targetKl. */
constexpr double KlSkipFactor = 1.5;
/** @brief Derivative of the squared critic error (v - target)^2 with respect to v is 2 (v - target). */
constexpr double SquaredErrorDerivativeFactor = 2;
/** @brief Value above which a binary observation flag is set. */
constexpr float FlagThreshold = 0.5F;

void CheckObservation(Observation const& observation)
{
    for (std::size_t index = 0; index < observation.size(); ++index)
    {
        if (!std::isfinite(observation[index]) || std::abs(observation[index]) > ObservationBound)
        {
            throw std::invalid_argument("FlightControl3D: observation is not normalized");
        }
    }
}

/** @brief Calls visit on every layer, in input, hidden, output order (the checkpoint order). */
template<typename Net, typename Visitor>
void VisitLayers(Net& net, Visitor&& visit)
{
    visit(net.input_layer);
    for (auto& layer : net.hidden_layers)
    {
        visit(layer);
    }
    visit(net.output_layer);
}

/** @brief Calls visit on a parameter and its two Adam moments (the checkpoint order). */
template<typename Parameter, typename Visitor>
void VisitParameter(Parameter& parameter, Visitor&& visit)
{
    visit(parameter.parameters);
    visit(parameter.gradient_first_order_moment);
    visit(parameter.gradient_second_order_moment);
}

/**
 * @brief Throws when a matrix holds a non-finite value, or a negative one when it must not.
 * @param values Matrix to check.
 * @param nonnegative True for Adam second moments.
 * @param message Error message.
 */
template<typename M>
void ValidateMatrix(M const& values, bool nonnegative, char const* message)
{
    for (Index row = 0; row < M::ROWS; ++row)
    {
        for (Index column = 0; column < M::COLS; ++column)
        {
            float const value = rlt::get(values, row, column);
            if (!std::isfinite(value) || (nonnegative && value < 0))
            {
                throw std::runtime_error(message);
            }
        }
    }
}

/** @brief Checks a parameter and its Adam moments; the second moment must be non-negative. */
template<typename Parameter>
void ValidateParameter(Parameter const& parameter, char const* message)
{
    ValidateMatrix(parameter.parameters, false, message);
    ValidateMatrix(parameter.gradient_first_order_moment, false, message);
    ValidateMatrix(parameter.gradient_second_order_moment, true, message);
}

/** @brief Writes a matrix as one line of space-separated values, row-major. */
template<typename M>
void WriteMatrix(std::ostream& stream, M const& matrix)
{
    bool first = true;
    for (Index row = 0; row < M::ROWS; ++row)
    {
        for (Index column = 0; column < M::COLS; ++column)
        {
            if (!first)
            {
                stream << ' ';
            }
            stream << rlt::get(matrix, row, column);
            first = false;
        }
    }
    stream << '\n';
}

/** @brief Reads a matrix written by WriteMatrix; every value must be finite. */
template<typename M>
void ReadMatrix(std::istream& stream, M& matrix)
{
    for (Index row = 0; row < M::ROWS; ++row)
    {
        for (Index column = 0; column < M::COLS; ++column)
        {
            float value{};
            if (!(stream >> value) || !std::isfinite(value))
            {
                throw std::runtime_error("FlightControl3D: corrupt checkpoint parameter");
            }
            rlt::set(matrix, row, column, value);
        }
    }
}

template<typename Parameter>
void WriteParameter(std::ostream& stream, Parameter const& parameter)
{
    VisitParameter(parameter, [&stream](auto const& matrix) { WriteMatrix(stream, matrix); });
}

template<typename Parameter>
void ReadParameter(std::istream& stream, Parameter& parameter)
{
    VisitParameter(parameter, [&stream](auto& matrix) { ReadMatrix(stream, matrix); });
}

/** @brief Writes every layer: weights then biases, each with its Adam moments. */
template<typename Net>
void WriteNetwork(std::ostream& stream, Net const& net)
{
    VisitLayers(net, [&stream](auto const& layer) {
        WriteParameter(stream, layer.weights);
        WriteParameter(stream, layer.biases);
    });
}

template<typename Net>
void ReadNetwork(std::istream& stream, Net& net)
{
    VisitLayers(net, [&stream](auto& layer) {
        ReadParameter(stream, layer.weights);
        ReadParameter(stream, layer.biases);
    });
}

/** @brief Checks every weight, bias and Adam moment of a network. */
template<typename Net>
void ValidateNetwork(Net const& net)
{
    VisitLayers(net, [](auto const& layer) {
        ValidateParameter(layer.weights, "FlightControl3D: invalid model parameter or Adam moment");
        ValidateParameter(layer.biases, "FlightControl3D: invalid model parameter or Adam moment");
    });
}

/** @brief Checks the log-std parameter, its Adam moments and the flight log-std bounds. */
template<typename Parameter>
void ValidateLogStandardDeviations(Parameter const& parameter)
{
    ValidateParameter(parameter, "FlightControl3D: invalid global log standard deviation or Adam moment");
    for (Index action = 0; action < ContinuousActions; ++action)
    {
        float const value = rlt::get(parameter.parameters, 0, action);
        if (value < MinimumLogStandardDeviation || value > MaximumLogStandardDeviation)
        {
            throw std::runtime_error("FlightControl3D: global log standard deviation outside flight bounds");
        }
    }
}

/** @brief Learner for one hidden width (the RLtools networks are statically sized). */
template<Index Width>
class Learner final : public FlightLearner::Impl
{
    Device device_{};
    Actor<Width> actor_{};
    Network<Width, 1> critic_{};
    Optimizer actorOptimizer_{};
    Optimizer criticOptimizer_{};
    Matrix<ObservationSize> input_{};
    Matrix<PolicyOutputs> actorOutput_{};
    Matrix<PolicyOutputs> actorDerivative_{};
    Matrix<1> valueOutput_{};
    Matrix<1> valueDerivative_{};
    typename Actor<Width>::template Buffer<1, rlt::MatrixStaticTag> actorBuffer_{};
    typename Network<Width, 1>::template Buffer<1, rlt::MatrixStaticTag> criticBuffer_{};

    /** @brief Copies a checked observation into the network input. */
    void Input(Observation const& observation)
    {
        CheckObservation(observation);
        for (Index index = 0; index < ObservationSize; ++index)
        {
            rlt::set(input_, 0, index, observation[index]);
        }
    }

    void Setup(Optimizer& optimizer, double rate)
    {
        optimizer.parameters.alpha = static_cast<float>(rate);
        optimizer.parameters.beta_1 = static_cast<float>(config.training.adamBeta1);
        optimizer.parameters.beta_2 = static_cast<float>(config.training.adamBeta2);
        optimizer.parameters.epsilon = static_cast<float>(config.training.adamEpsilon);
    }

    /** @brief Projects the log-std back into [MinimumLogStandardDeviation, MaximumLogStandardDeviation]. */
    void ClampActiveLogStandardDeviations() noexcept
    {
        for (Index action = 0; action < ContinuousActions; ++action)
        {
            rlt::set(actor_.log_std.parameters, 0, action, std::clamp(
                rlt::get(actor_.log_std.parameters, 0, action),
                static_cast<float>(MinimumLogStandardDeviation),
                static_cast<float>(MaximumLogStandardDeviation)));
        }
    }

    /** @brief Training-mode actor forward pass on the current input (keeps activations for backward). */
    PolicyDistribution ForwardActor()
    {
        rlt::forward(device_, actor_, input_, actorOutput_);
        PolicyDistribution policy{};
        for (Index output = 0; output < PolicyOutputs; ++output)
        {
            policy.output[output] = rlt::get(actorOutput_, 0, output);
        }
        for (Index action = 0; action < ContinuousActions; ++action)
        {
            policy.logStandardDeviations[action] = rlt::get(actor_.log_std.parameters, 0, action);
        }
        return policy;
    }

    /** @brief Shift and scale the advantages to zero mean and unit variance (population variance). */
    void NormalizeAdvantages()
    {
        double mean = 0;
        double variance = 0;
        for (auto const& row : rows)
        {
            mean += row.advantage / static_cast<double>(collected);
        }
        for (auto const& row : rows)
        {
            variance += (row.advantage - mean) * (row.advantage - mean) / static_cast<double>(collected);
        }
        double const advantageScale = 1 / std::sqrt(variance + AdvantageEpsilon);
        for (auto& row : rows)
        {
            row.advantage = (row.advantage - mean) * advantageScale;
        }
    }

    void ResetUpdateStatistics() noexcept
    {
        stats.policyLoss = 0;
        stats.valueLoss = 0;
        stats.entropy = 0;
        stats.maximumApproximateKl = 0;
        stats.maximumSampledKl = std::numeric_limits<double>::lowest();
        stats.actorSteps = 0;
        stats.criticSteps = 0;
        stats.actorKlSkips = 0;
    }

    /**
     * @brief Accumulates the actor gradient of one sample and its KL estimates.
     *
     * sampledKl accumulates -log r / minibatch (the sampled KL(old || new));
     * approximateKl accumulates max(0, r - 1 - log r) / minibatch with log r bounded to +/-20.
     * @return Clipped PPO loss of the sample.
     */
    PolicyLoss AccumulateActorSample(Transition const& row, double minibatchSize, double& approximateKl,
        double& sampledKl)
    {
        auto const policy = ForwardActor();
        auto const loss = ClippedLoss(policy, row.action, row.advantage, config.training);
        double const logRatio = LogProbability(policy, row.action) - row.action.logProbability;
        if (!std::isfinite(logRatio))
        {
            throw std::runtime_error("FlightControl3D: invalid PPO log ratio");
        }
        sampledKl -= logRatio / minibatchSize;
        double const boundedLogRatio = std::clamp(logRatio, -MaximumAbsoluteLogRatio,
            MaximumAbsoluteLogRatio);
        approximateKl += std::max(0.0, std::expm1(boundedLogRatio) - boundedLogRatio) / minibatchSize;
        if (!std::isfinite(loss.loss))
        {
            throw std::runtime_error("FlightControl3D: invalid PPO loss");
        }
        // Mean over the minibatch: each sample contributes derivative / minibatch.
        for (Index output = 0; output < PolicyOutputs; ++output)
        {
            rlt::set(actorDerivative_, 0, output, static_cast<float>(loss.outputDerivative[output]
                / minibatchSize));
        }
        rlt::backward(device_, actor_, input_, actorDerivative_, actorBuffer_);
        for (Index action = 0; action < ContinuousActions; ++action)
        {
            rlt::increment(actor_.log_std.gradient, 0, action,
                static_cast<float>(loss.logStandardDeviationDerivative[action] / minibatchSize));
        }
        return loss;
    }

    /**
     * @brief Accumulates the critic gradient of one sample on (V(s) - target)^2.
     * @return V(s) - target.
     */
    double AccumulateCriticSample(Transition const& row, double minibatchSize)
    {
        rlt::forward(device_, critic_, input_, valueOutput_);
        double const difference = rlt::get(valueOutput_, 0, 0) - row.target;
        rlt::set(valueDerivative_, 0, 0,
            static_cast<float>(SquaredErrorDerivativeFactor * difference / minibatchSize));
        rlt::backward(device_, critic_, input_, valueDerivative_, criticBuffer_);
        return difference;
    }

    /**
     * @brief Applies the optimizer steps of one minibatch, with the KL guard on the actor.
     *
     * Matches RLMission's sampled old-logp minus new-logp guard. The separate critic can
     * still learn from this minibatch.
     */
    void StepMinibatch(double approximateKl, double sampledKl)
    {
        if (!std::isfinite(sampledKl))
        {
            throw std::runtime_error("FlightControl3D: invalid minibatch KL");
        }
        stats.maximumApproximateKl = std::max(stats.maximumApproximateKl, approximateKl);
        stats.maximumSampledKl = std::max(stats.maximumSampledKl, sampledKl);
        if (sampledKl >= KlSkipFactor * config.training.targetKl)
        {
            ++stats.actorKlSkips;
        }
        else
        {
            rlt::step(device_, actorOptimizer_, actor_);
            ClampActiveLogStandardDeviations();
            ++stats.actorSteps;
        }
        rlt::step(device_, criticOptimizer_, critic_);
        ++stats.criticSteps;
        ValidateNetwork(actor_);
        ValidateLogStandardDeviations(actor_.log_std);
        ValidateNetwork(critic_);
    }

    /**
     * @brief Accumulates the gradients of one minibatch, then steps the optimizers.
     * @param start Index of the first sample in the shuffled order.
     * @param samples Total samples of the update (rollout size x epochs), for the mean statistics.
     */
    void TrainMinibatch(std::size_t start, double samples)
    {
        rlt::zero_gradient(device_, actor_);
        rlt::zero_gradient(device_, critic_);
        double const minibatchSize = static_cast<double>(config.training.minibatch);
        double approximateKl = 0;
        double sampledKl = 0;
        for (std::size_t sample = 0; sample < config.training.minibatch; ++sample)
        {
            auto const& row = rows[order[start + sample]];
            Input(row.observation);
            auto const loss = AccumulateActorSample(row, minibatchSize, approximateKl, sampledKl);
            double const difference = AccumulateCriticSample(row, minibatchSize);
            // Reported policy loss excludes the entropy bonus.
            stats.policyLoss += (loss.loss + config.training.entropy * loss.entropy) / samples;
            stats.entropy += loss.entropy / samples;
            stats.valueLoss += difference * difference / samples;
        }
        StepMinibatch(approximateKl, sampledKl);
    }

    /** @brief Publishes the rollout reward means and empties the buffer. */
    void FinishRollout()
    {
        if (!std::isfinite(stats.policyLoss) || !std::isfinite(stats.valueLoss))
        {
            throw std::runtime_error("FlightControl3D: invalid training diagnostics");
        }
        stats.meanReward = rewardSum / static_cast<double>(collected);
        for (std::size_t channel = 0; channel < RewardCount; ++channel)
        {
            stats.meanRewards[channel] = rewardSums[channel] / static_cast<double>(collected);
        }
        if (!std::isfinite(stats.meanReward))
        {
            throw std::runtime_error("FlightControl3D: invalid mean reward diagnostic");
        }
        for (double const reward : stats.meanRewards)
        {
            if (!std::isfinite(reward))
            {
                throw std::runtime_error("FlightControl3D: invalid mean reward-channel diagnostic");
            }
        }
        stats.hasRewardMeans = true;
        rewardSum = 0;
        rewardSums.fill(0);
        ++stats.updates;
        collected = 0;
    }

public:
    explicit Learner(Config const& configuration) : Impl(configuration)
    {
        // Initialization order is part of the reproducibility contract:
        // actor weights, then critic weights, from the learner RNG.
        rlt::init_weights(device_, actor_, rng);
        rlt::init_weights(device_, critic_, rng);
        for (Index action = 0; action < ContinuousActions; ++action)
        {
            rlt::set(actor_.log_std.parameters, 0, action, static_cast<float>(InitialLogStandardDeviation));
        }
        rlt::reset_optimizer_state(device_, actorOptimizer_, actor_);
        rlt::reset_optimizer_state(device_, criticOptimizer_, critic_);
        Setup(actorOptimizer_, configuration.training.actorRate);
        Setup(criticOptimizer_, configuration.training.criticRate);
        rlt::set_all(device_, input_, 0.F);
        rlt::zero_gradient(device_, actor_);
        rlt::zero_gradient(device_, critic_);
        rlt::forward(device_, actor_, input_, actorOutput_);
        rlt::forward(device_, critic_, input_, valueOutput_);
    }

    PolicyDistribution Evaluate(Observation const& observation) override
    {
        Check();
        Input(observation);
        rlt::evaluate(device_, actor_, input_, actorOutput_, actorBuffer_);
        PolicyDistribution policy{};
        for (Index output = 0; output < PolicyOutputs; ++output)
        {
            policy.output[output] = rlt::get(actorOutput_, 0, output);
            if (!std::isfinite(policy.output[output]))
            {
                throw std::runtime_error("FlightControl3D: invalid actor");
            }
        }
        for (Index action = 0; action < ContinuousActions; ++action)
        {
            policy.logStandardDeviations[action] = rlt::get(actor_.log_std.parameters, 0, action);
            if (!std::isfinite(policy.logStandardDeviations[action]))
            {
                throw std::runtime_error("FlightControl3D: invalid global log standard deviation");
            }
        }
        return policy;
    }

    double Value(Observation const& observation) override
    {
        Check();
        Input(observation);
        rlt::evaluate(device_, critic_, input_, valueOutput_, criticBuffer_);
        double const value = rlt::get(valueOutput_, 0, 0);
        if (!std::isfinite(value))
        {
            throw std::runtime_error("FlightControl3D: invalid critic");
        }
        return value;
    }

    void Optimize() override
    {
        Check();
        if (collected != rows.size())
        {
            throw std::logic_error("FlightControl3D: incomplete rollout");
        }
        try
        {
            ComputeAdvantages(rows.data(), collected, config.training);
            NormalizeAdvantages();
            ResetUpdateStatistics();
            double const samples = static_cast<double>(collected * config.training.epochs);
            for (std::size_t epoch = 0; epoch < config.training.epochs; ++epoch)
            {
                std::shuffle(order.begin(), order.end(), rng);
                for (std::size_t start = 0; start < collected; start += config.training.minibatch)
                {
                    TrainMinibatch(start, samples);
                }
            }
            FinishRollout();
        }
        catch (...)
        {
            failed = true;
            throw;
        }
    }

    void Write(std::ostream& stream) const override
    {
        Check();
        ValidateNetwork(actor_);
        ValidateLogStandardDeviations(actor_.log_std);
        ValidateNetwork(critic_);
        // Line 1: format header. Line 2: quoted contract signature. Then floats
        // at full precision, Adam ages, RNG state and the parameters.
        stream << CheckpointMagic << ' ' << ObservationSize << ' '
            << ContinuousActions << ' ' << PolicyOutputs << ' ' << Width << ' '
            << stats.updates << '\n';
        stream << std::quoted(config.checkpointSignature) << '\n'
            << std::setprecision(std::numeric_limits<float>::max_digits10);
        stream << actorOptimizer_.age << ' ' << criticOptimizer_.age << '\n' << rng << '\n';
        WriteNetwork(stream, actor_);
        WriteParameter(stream, actor_.log_std);
        WriteNetwork(stream, critic_);
    }

    void Read(std::istream& stream) override
    {
        Check();
        std::string magic;
        std::string signature;
        std::size_t observations{};
        std::size_t actions{};
        std::size_t outputs{};
        std::size_t width{};
        std::uint64_t updates{};
        Index actorAge{};
        Index criticAge{};
        std::mt19937 candidateRng;
        bool const headerRead = static_cast<bool>(stream >> magic >> observations >> actions >> outputs
            >> width            >> updates >> std::quoted(signature) >> actorAge >> criticAge);
        if (!headerRead || magic != CheckpointMagic || observations != ObservationSize
            || actions != ContinuousActions || outputs != PolicyOutputs || width != Width
            || !CompatibleTrainingSignature(signature, config.checkpointSignature) || actorAge == 0
                || criticAge == 0)
        {
            throw std::runtime_error("FlightControl3D: incompatible/corrupt checkpoint header");
        }
        candidateRng = detail::ReadCheckpointRng(stream);
        // RLtools static matrices contain self-pointers: never use C++ copies.
        // Stage independent networks on the heap (including on Windows' small stack).
        auto candidateActor = std::make_unique<Actor<Width>>();
        auto candidateCritic = std::make_unique<Network<Width, 1>>();
        rlt::forward(device_, actor_, input_, actorOutput_);
        rlt::forward(device_, critic_, input_, valueOutput_);
        rlt::copy(device_, device_, actor_, *candidateActor);
        rlt::copy(device_, device_, critic_, *candidateCritic);
        ReadNetwork(stream, *candidateActor);
        ReadParameter(stream, candidateActor->log_std);
        ReadNetwork(stream, *candidateCritic);
        ValidateNetwork(*candidateActor);
        ValidateLogStandardDeviations(candidateActor->log_std);
        ValidateNetwork(*candidateCritic);
        stream >> std::ws;
        if (!stream.eof())
        {
            throw std::runtime_error("FlightControl3D: trailing checkpoint data");
        }
        // Commit only after the whole file was read and validated.
        rlt::copy(device_, device_, *candidateActor, actor_);
        rlt::copy(device_, device_, *candidateCritic, critic_);
        rng = candidateRng;
        actorOptimizer_.age = actorAge;
        criticOptimizer_.age = criticAge;
        stats = {};
        stats.updates = updates;
        collected = 0;
        rewardSum = 0;
        rewardSums.fill(0);
    }
};
} // namespace

FlightLearner::FlightLearner(Config const& config)
{
    Validate(config);
    if (config.training.hiddenWidth == 32)
    {
        impl_ = std::make_unique<Learner<32>>(config);
    }
    else if (config.training.hiddenWidth == 64)
    {
        impl_ = std::make_unique<Learner<64>>(config);
    }
    else if (config.training.hiddenWidth == 128)
    {
        impl_ = std::make_unique<Learner<128>>(config);
    }
    else
    {
        impl_ = std::make_unique<Learner<256>>(config);
    }
}

FlightLearner::~FlightLearner() = default;

PolicyDistribution FlightLearner::Evaluate(Observation const& observation)
{
    return impl_->Evaluate(observation);
}

PolicyAction FlightLearner::Choose(Observation const& observation, bool deterministic)
{
    bool const rollout = observation[RolloutObservation] > FlagThreshold;
    // The gear decision exists only while its extension is not yet latched.
    bool const gearActive = observation[GearCommandObservation] <= FlagThreshold;
    return Sample(impl_->Evaluate(observation), impl_->rng, deterministic, rollout, gearActive);
}

bool FlightLearner::Ready() const noexcept
{
    return impl_->collected == impl_->rows.size();
}

std::size_t FlightLearner::Collected() const noexcept
{
    return impl_->collected;
}

void FlightLearner::SetRewardContract(RewardSettings const& settings, std::string const& signature)
{
    if (impl_->collected != 0)
    {
        throw std::logic_error("FlightControl3D: reward changes require a rollout boundary");
    }
    impl_->config.reward = settings;
    impl_->config.checkpointSignature = signature;
}

LearningStats const& FlightLearner::Stats() const noexcept
{
    return impl_->stats;
}

void FlightLearner::Optimize()
{
    impl_->Optimize();
}

void FlightLearner::Record(Observation const& observation, PolicyAction const& action, StepResult const& step,
    Observation const& nextObservation)
{
    impl_->Check();
    if (Ready())
    {
        throw std::logic_error("FlightControl3D: rollout full");
    }
    if (!std::isfinite(step.reward) || !std::isfinite(action.logProbability) || (step.terminated
        && step.truncated))
    {
        throw std::invalid_argument("FlightControl3D: invalid transition");
    }
    for (double const parameter : action.policy.output)
    {
        if (!std::isfinite(parameter))
        {
            throw std::invalid_argument("FlightControl3D: invalid collection policy");
        }
    }
    for (double const parameter : action.policy.logStandardDeviations)
    {
        if (!std::isfinite(parameter))
        {
            throw std::invalid_argument("FlightControl3D: invalid collection deviation");
        }
    }
    for (double const reward : step.rewards)
    {
        if (!std::isfinite(reward))
        {
            throw std::invalid_argument("FlightControl3D: invalid reward channel");
        }
    }
    auto& row = impl_->rows[impl_->collected];
    // V(s') is not evaluated after an absorbing end: its value is zero by definition.
    double const value = impl_->Value(observation);
    double nextValue = 0;
    if (!step.terminated)
    {
        nextValue = impl_->Value(nextObservation);
    }
    row = {observation, action, step.reward, value, nextValue, 0, 0, step.terminated, step.truncated};
    impl_->rewardSum += step.reward;
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        impl_->rewardSums[channel] += step.rewards[channel];
    }
    ++impl_->collected;
}

void FlightLearner::Save(std::filesystem::path const& path) const
{
    if (impl_->collected != 0)
    {
        throw std::logic_error("FlightControl3D: save at an update boundary");
    }
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        impl_->Write(stream);
        stream.flush();
        if (!stream)
        {
            throw std::runtime_error("FlightControl3D: checkpoint write failed");
        }
    }
    detail::CommitTemporaryFile(temporary, path);
}

void FlightLearner::Load(std::filesystem::path const& path)
{
    if (impl_->collected != 0)
    {
        throw std::logic_error("FlightControl3D: load at an update boundary");
    }
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("FlightControl3D: checkpoint not found");
    }
    impl_->Read(stream);
}
} // namespace ppo::examples::flight
