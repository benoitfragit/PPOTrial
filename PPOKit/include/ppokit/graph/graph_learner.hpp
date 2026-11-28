/**
 * @file PPOKit/include/ppokit/graph/graph_learner.hpp
 * @brief PPO learner on PPOKit block graphs: Gaussian and Bernoulli heads, any block wiring.
 * @details Same interface and same update as the rl_tools GaussianMlpLearner (Decide, Mean,
 * Value, Update, LogStd), so the collector, the evaluator and the examples use either one.
 * The networks are built at load time from graph.json (ppokit.graph/2):
 *
 * @code
 * auto const task = ppokit::schema::LoadTaskSchema("task.json");
 * auto const graphs = ppokit::schema::LoadBlockGraph("graph.json", task);
 * using Learner = ppokit::graph::GraphLearner<ObservationSize, ActionSize>;
 * Learner learner(ppokit::schema::BuildGraph(graphs.actor), ppokit::schema::BuildGraph(graphs.critic),
 *     ppokit::schema::MakePolicyLayout(graphs, task), settings);
 * @endcode
 *
 * One update: GAE, explained variance, advantage normalization, then `epochs` passes over
 * shuffled minibatches. For each learnable sample the clipped surrogate (ppokit/core) is
 * chained with the head gradients (ppokit/heads) into the actor graph; every sample trains
 * the critic on the squared error to its value target. One Adam step per minibatch for
 * each network. Everything is reproducible from `settings.seed`. Bernoulli actions are
 * stored as 1 or 0 in the action vector.
 *
 * Components active only in some states (task.json "active", PolicyLayout::activities)
 * are decided from the observation: an inactive one draws no random number, is 0 in the
 * action and takes no part in the likelihood, the entropy or the gradients, at collection
 * and again at update time (the observation is stored, so the test gives the same answer).
 *
 * A Gaussian head may compute its log-std from the state (LogStdSettings): the learner then
 * reads it per sample and back-propagates its gradient into the actor.
 *
 * Shared blocks (PolicyLayout::sharedBlocks, network.json "share"): a critic block may use
 * the parameters of an actor block, as an actor-critic with a common trunk. The learner
 * then optimizes one loss, policy + value_coefficient * value, with one Adam over the
 * actor and the critic's own parameters at the actor's learning rate; the shared blocks
 * receive the gradients of both networks. A minibatch over the KL bound steps nothing.
 *
 * Checkpoints are versioned artifacts (ppokit/artifacts/artifact.hpp): SaveTraining keeps
 * everything a resume needs (the next update gives the same weights, bit for bit, as if
 * training had not stopped); SavePolicy keeps the actor. Loading refuses a checkpoint of
 * another task or network with a message saying what changed.
 */
#pragma once

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/core/kl.hpp"
#include "ppokit/core/ppo_settings.hpp"
#include "ppokit/core/ppo_surrogate.hpp"
#include "ppokit/graph/adam.hpp"
#include "ppokit/graph/checkpoint_io.hpp"
#include "ppokit/graph/graph.hpp"
#include "ppokit/graph/policy_layout.hpp"
#include "ppokit/graph/probes.hpp"
#include "ppokit/heads/bernoulli.hpp"
#include "ppokit/heads/gaussian.hpp"
#include "ppokit/runtime/collector.hpp"
#include "ppokit/runtime/rollout.hpp"
#include "ppokit/runtime/ppo_update_engine.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppokit::graph {

/** @brief Dimension-independent graph PPO engine. All per-step storage is owned by the caller or preallocated. */
class GraphLearnerCore
{
public:

    /**
     * @param actor Graph with the Gaussian head as output 0 (if any), then the Bernoulli head.
     * @param critic Graph with one value output.
     * @throws std::invalid_argument when the layout, the graphs and the sizes disagree.
     */
    GraphLearnerCore(Graph actor, Graph critic, PolicyLayout layout, core::PpoSettings const& settings,
        std::size_t observationSize, std::size_t actionSize)
        : observationSize_(observationSize), actionSize_(actionSize),
          settings_(core::ValidatePpoSettings(settings)), actor_(std::move(actor)), critic_(std::move(critic)),
          gaussianAxes_(std::move(layout.gaussianAxes)), bernoulliAxes_(std::move(layout.bernoulliAxes)),
          actorInputs_(std::move(layout.actorInputs), observationSize_),
          criticInputs_(std::move(layout.criticInputs), observationSize_),
          activities_(std::move(layout.activities)), active_(actionSize_), oneOnTie_(actionSize_), rng_(settings.seed)
    {
        if (observationSize_ == 0 || actionSize_ == 0)
            throw std::invalid_argument("PPOKit graph learner: dimensions must be positive");
        for (std::size_t axis : layout.oneOnTie)
        {
            oneOnTie_.at(axis) = true;
        }
        CheckLayout();
        TieSharedBlocks(layout.sharedBlocks);
        Initialize();
        if (Joint())
        {
            jointOptimizer_ =
                std::make_unique<Adam>(JointParameters(), AdamFor(settings_, settings_.actorLearningRate));
            return;
        }
        actorOptimizer_ = std::make_unique<Adam>(actor_.Parameters(), AdamFor(settings_, settings_.actorLearningRate));
        criticOptimizer_ =
            std::make_unique<Adam>(critic_.Parameters(), AdamFor(settings_, settings_.criticLearningRate));
    }

    /** @brief Samples into actionSize values; observation contains observationSize values. */
    void Decide(float const* observation, float* action, double& logProbability, double& value)
    {
        RunActor(observation);
        FindActive(observation);
        std::fill_n(action, actionSize_, 0.F);
        logProbability = 0.0;
        for (std::size_t component = 0; component < gaussianAxes_.size(); ++component)
        {
            if (!active_[gaussianAxes_[component]])
            {
                continue;
            }
            double const mean = GaussianMean(component);
            double const logStd = LogStd(component);
            std::normal_distribution<double> distribution(mean, std::exp(logStd));
            float const sample = static_cast<float>(distribution(rng_));
            action[gaussianAxes_[component]] = sample;
            logProbability += heads::GaussianAxisLogDensity(static_cast<double>(sample), mean, logStd);
        }
        for (std::size_t component = 0; component < bernoulliAxes_.size(); ++component)
        {
            if (!active_[bernoulliAxes_[component]])
            {
                continue;
            }
            double const logit = BernoulliLogit(component);
            std::bernoulli_distribution distribution(heads::Sigmoid(logit));
            bool const drawn = distribution(rng_);
            action[bernoulliAxes_[component]] = drawn ? 1.F : 0.F;
            logProbability += heads::BernoulliLogProbability(logit, drawn);
        }
        value = Value(observation);
    }

    /**
     * @brief Deterministic action: Gaussian means, and 1 where a Bernoulli logit is positive;
     *        0 for the inactive components.
     */
    void Mean(float const* observation, float* action)
    {
        RunActor(observation);
        FindActive(observation);
        std::fill_n(action, actionSize_, 0.F);
        for (std::size_t component = 0; component < gaussianAxes_.size(); ++component)
        {
            std::size_t const axis = gaussianAxes_[component];
            if (active_[axis])
            {
                action[axis] = static_cast<float>(GaussianMean(component));
            }
        }
        for (std::size_t component = 0; component < bernoulliAxes_.size(); ++component)
        {
            std::size_t const axis = bernoulliAxes_[component];
            double const logit = BernoulliLogit(component);
            if (active_[axis] && (logit > 0.0 || (logit == 0.0 && oneOnTie_[axis])))
            {
                action[axis] = 1.F;
            }
        }
    }

    /** @brief Critic value of observationSize values. */
    double Value(float const* observation)
    {
        RequireFinite(observation);
        double const value = critic_.Forward(criticInputs_.Bind(observation)).values[0];
        if (!std::isfinite(value))
        {
            throw std::runtime_error("PPOKit graph learner: non-finite critic output");
        }
        return value;
    }

    /**
     * @brief Runs one PPO update on a full rollout, then empties it.
     * @details On failure the rollout stays full. All trained weights, Adam states, the
     *          update count and the shuffle RNG retain their values from before the update.
     * @throws std::runtime_error when a loss or optimizer step becomes non-finite.
     */
    template<typename Buffer>
    core::UpdateStatistics Update(Buffer& buffer)
    {
        if (!buffer.Full())
        {
            throw std::invalid_argument("PPOKit graph learner: an update requires a full rollout");
        }
        if (buffer[0].observation.size() != observationSize_ || buffer[0].action.size() != actionSize_)
        {
            throw std::invalid_argument("PPOKit graph learner: rollout dimensions differ from the learner");
        }
        core::UpdateStatistics statistics;
        statistics.explainedVariance = runtime::PpoUpdateEngine::Prepare(buffer, settings_.gamma, settings_.lambda,
            [](std::size_t) {});
        updateEngine_.ResetOrder(buffer.Size());
        policyBackup_.Capture(Joint() ? *jointOptimizer_ : *actorOptimizer_);
        if (criticOptimizer_)
        {
            criticBackup_.Capture(*criticOptimizer_);
        }
        auto const randomBefore = rng_;
        runtime::PpoSampleCounts counts;
        ResetProbes();
        try
        {
            counts = TrainEpochs(buffer, statistics);
        }
        catch (...)
        {
            policyBackup_.Restore();
            criticBackup_.Restore();
            CopySharedValues();
            rng_ = randomBefore;
            probing_ = false;
            ResetProbes();
            throw;
        }
        probing_ = false;
        runtime::PpoUpdateEngine::Finalize(statistics, counts);
        buffer.Clear();
        statistics.updates = ++updates_;
        return statistics;
    }

    /**
     * @brief Log standard deviation of one Gaussian component (head order): the learned one,
     *        or, when it depends on the state, the one of the last actor forward pass.
     */
    double LogStd(std::size_t component) const { return gaussianHead_->CurrentLogStd(component); }

    /** @brief True when critic blocks share actor parameters: one loss and one optimizer. */
    bool Joint() const noexcept { return !ties_.empty(); }

    /**
     * @brief Probes measured during the next updates (null: off, the default).
     * @details The probes must probe this learner's Actor() and Critic(); they are reset
     * at the start of each update and fed during its last epoch.
     */
    void SetProbes(GraphProbes* actor, GraphProbes* critic) noexcept
    {
        actorProbes_ = actor;
        criticProbes_ = critic;
    }

    /**
     * @brief Writes everything a resume needs, atomically: networks, Adam, random state, update count.
     * @param identity What these networks are (schema::ArtifactIdentityOf of the task and graph).
     */
    void SaveTraining(std::filesystem::path const& path, artifacts::ArtifactIdentity const& identity)
    {
        Write(path, identity, artifacts::ArtifactKind::Training);
    }

    /** @brief Writes the actor alone, for evaluation and inference. */
    void SavePolicy(std::filesystem::path const& path, artifacts::ArtifactIdentity const& identity)
    {
        Write(path, identity, artifacts::ArtifactKind::Policy);
    }

    /**
     * @brief Resumes from a training checkpoint.
     * @param expected Identity of the networks this learner was built from.
     * @throws artifacts::ArtifactMismatch for another task or network; std::runtime_error
     *         for a policy checkpoint or a damaged file. The learner is unchanged on failure.
     */
    void LoadTraining(std::filesystem::path const& path, artifacts::ArtifactIdentity const& expected)
    {
        Read(path, expected, true);
    }

    /**
     * @brief Loads the actor of a policy or training checkpoint; the critic and optimizers are
     *        unchanged, except the shared blocks, which are the actor's.
     */
    void LoadPolicy(std::filesystem::path const& path, artifacts::ArtifactIdentity const& expected)
    {
        Read(path, expected, false);
    }

    /** @brief Actor graph, for probes and checkpoints. */
    /** @brief Updates made, including those of a loaded training state. */
    std::uint64_t Updates() const noexcept { return updates_; }
    Graph& Actor() noexcept { return actor_; }
    /** @brief Critic graph, for probes and checkpoints. */
    Graph& Critic() noexcept { return critic_; }

private:
    /** @brief Checks that heads, axes, inputs and sizes describe the same policy. */
    void CheckLayout()
    {
        std::size_t const expectedOutputs = (gaussianAxes_.empty() ? 0 : 1) + (bernoulliAxes_.empty() ? 0 : 1);
        if (actor_.OutputCount() != expectedOutputs || expectedOutputs == 0)
        {
            throw std::invalid_argument("PPOKit graph learner: the actor needs one output per head kind, Gaussian "
                                        "first, then Bernoulli");
        }
        if (!gaussianAxes_.empty())
        {
            gaussianHead_ = dynamic_cast<HeadNode*>(&actor_.At(actor_.OutputId(0)));
            RequireHead(gaussianHead_, HeadKind::Gaussian, gaussianAxes_.size(), "Gaussian");
        }
        if (!bernoulliAxes_.empty())
        {
            std::size_t const output = gaussianAxes_.empty() ? 0 : 1;
            bernoulliOutput_ = output;
            RequireHead(dynamic_cast<HeadNode*>(&actor_.At(actor_.OutputId(output))), HeadKind::Bernoulli,
                bernoulliAxes_.size(), "Bernoulli");
        }
        std::vector<bool> covered(actionSize_, false);
        for (std::vector<std::size_t> const* axes : {&gaussianAxes_, &bernoulliAxes_})
        {
            for (std::size_t const axis : *axes)
            {
                if (axis >= actionSize_ || covered[axis])
                {
                    throw std::invalid_argument("PPOKit graph learner: action index " + std::to_string(axis)
                        + " is out of range or produced twice");
                }
                covered[axis] = true;
            }
        }
        if (std::find(covered.begin(), covered.end(), false) != covered.end())
        {
            throw std::invalid_argument("PPOKit graph learner: some action components have no head");
        }
        for (ComponentActivity const& activity : activities_)
        {
            if (activity.axis >= actionSize_ || activity.feature >= observationSize_)
            {
                throw std::invalid_argument("PPOKit graph learner: an action activity refers to action "
                    + std::to_string(activity.axis) + " or observation " + std::to_string(activity.feature)
                    + ", out of range");
            }
        }
        RequireInputs(actor_, actorInputs_, "actor");
        RequireInputs(critic_, criticInputs_, "critic");
        if (critic_.OutputCount() != 1 || critic_.OutputShape() != Shape::Vector(1))
        {
            throw std::invalid_argument("PPOKit graph learner: the critic outputs one value");
        }
    }

    static void RequireHead(HeadNode const* head, HeadKind kind, std::size_t components, char const* name)
    {
        if (head == nullptr || head->Kind() != kind || head->OutputShape() != Shape::Vector(components))
        {
            throw std::invalid_argument(std::string("PPOKit graph learner: the ") + name + " head must give "
                + std::to_string(components) + " output(s)");
        }
    }

    static void RequireInputs(Graph const& network, InputBinder const& inputs, char const* name)
    {
        bool matches = network.InputCount() == inputs.Slices().size();
        for (std::size_t input = 0; matches && input < inputs.Slices().size(); ++input)
        {
            matches = network.InputShape(input) == inputs.Slices()[input].shape;
        }
        if (!matches)
        {
            throw std::invalid_argument(std::string("PPOKit graph learner: the ") + name
                + " inputs differ from its layout slices");
        }
    }

    /** @brief Weights from the seed, then the initial log-std; the order is part of reproducibility. */
    void Initialize()
    {
        actor_.Initialize(rng_);
        critic_.Initialize(rng_);
        CopySharedValues();
        if (gaussianHead_ != nullptr && !gaussianHead_->StateDependentLogStd())
        {
            std::vector<float>& logStd = gaussianHead_->LogStd().value;
            std::fill(logStd.begin(), logStd.end(), static_cast<float>(settings_.initialLogStd));
        }
        gaussianGradient_.assign(gaussianAxes_.size(), 0.F);
        bernoulliGradient_.assign(bernoulliAxes_.size(), 0.F);
        outputGradients_.assign(actor_.OutputCount(), nullptr);
        if (!gaussianAxes_.empty())
        {
            outputGradients_[0] = gaussianGradient_.data();
        }
        if (!bernoulliAxes_.empty())
        {
            outputGradients_[bernoulliOutput_] = bernoulliGradient_.data();
        }
    }

    void RequireFinite(float const* observation) const
    {
        for (std::size_t index = 0; index < observationSize_; ++index)
        {
            if (!std::isfinite(observation[index]))
            {
                throw std::invalid_argument("PPOKit graph learner: non-finite observation");
            }
        }
    }

    /** @brief Forward pass of the actor; heads are then read with GaussianMean and BernoulliLogit. */
    void RunActor(float const* observation)
    {
        RequireFinite(observation);
        actor_.Forward(actorInputs_.Bind(observation));
        for (std::size_t output = 0; output < actor_.OutputCount(); ++output)
        {
            Signal const head = actor_.Output(output);
            std::size_t const width = actor_.OutputShape(output).width;
            for (std::size_t index = 0; index < width; ++index)
            {
                if (!std::isfinite(head.values[index]))
                {
                    throw std::runtime_error("PPOKit graph learner: non-finite actor output");
                }
            }
        }
    }

    /** @brief Marks every action component active, then applies the activity rules to `observation`. */
    void FindActive(float const* observation) noexcept
    {
        std::fill(active_.begin(), active_.end(), true);
        for (ComponentActivity const& activity : activities_)
        {
            active_[activity.axis] = activity.Active(observation);
        }
    }

    double GaussianMean(std::size_t component) const { return actor_.Output(0).values[component]; }
    double BernoulliLogit(std::size_t component) const { return actor_.Output(bernoulliOutput_).values[component]; }

    /** @brief Shuffle and train each epoch, measuring block probes during the final pass. */
    template<typename Buffer>
    runtime::PpoSampleCounts TrainEpochs(Buffer const& buffer, core::UpdateStatistics& statistics)
    {
        return updateEngine_.RunEpochs(settings_.epochs, settings_.minibatch,
            [this](auto& order) { std::shuffle(order.begin(), order.end(), rng_); },
            [this](std::size_t epoch) {
                probing_ = epoch + 1 == settings_.epochs && (actorProbes_ != nullptr || criticProbes_ != nullptr);
            },
            [this, &buffer, &statistics](std::size_t start, std::size_t end) {
                return TrainMinibatch(buffer, start, end, statistics);
            },
            [](std::size_t) {});
    }

    /** @brief One optimizer step of each network; returns the number of actor samples. */
    template<typename Buffer>
    std::size_t TrainMinibatch(Buffer const& buffer, std::size_t start, std::size_t end,
        core::UpdateStatistics& statistics)
    {
        actor_.ZeroGradients();
        critic_.ZeroGradients();
        double const size = static_cast<double>(end - start);
        std::size_t actorSamples = 0;
        minibatchSampledKl_ = 0.0;
        for (std::size_t position = start; position < end; ++position)
        {
            auto const& row = buffer[updateEngine_.IndexAt(position)];
            if (row.learnable)
            {
                AccumulateActorSample(row, size, statistics);
                ++actorSamples;
            }
            AccumulateCriticSample(row, size, statistics);
        }
        if (probing_)
        {
            ObserveProbeGradients();
        }
        // The KL guard: a minibatch that moved the policy too far does not step the actor
        // (nor, with shared blocks, anything: the loss is one).
        bool const actorAllowed = core::ActorStepAllowed(minibatchSampledKl_, settings_.targetKl);
        if (!actorAllowed)
        {
            ++statistics.actorKlSkips;
        }
        if (Joint())
        {
            if (actorAllowed)
            {
                StepJoint();
            }
            return actorSamples;
        }
        if (actorAllowed)
        {
            ClipGradients(*actorOptimizer_);
            actorOptimizer_->Step();
            ClampLogStd();
        }
        ClipGradients(*criticOptimizer_);
        criticOptimizer_->Step();
        return actorSamples;
    }

    /** @brief The shared blocks sum both gradients, one Adam step, then the critic copies follow. */
    void StepJoint()
    {
        for (auto const& [actorParameter, criticParameter] : ties_)
        {
            for (std::size_t index = 0; index < actorParameter->gradient.size(); ++index)
            {
                actorParameter->gradient[index] += criticParameter->gradient[index];
            }
        }
        ClipGradients(*jointOptimizer_);
        jointOptimizer_->Step();
        ClampLogStd();
        CopySharedValues();
    }

    /** @brief Applies max_gradient_norm to the tensors of one optimizer (one loss), if set. */
    void ClipGradients(Adam& optimizer) const
    {
        if (settings_.maxGradientNorm > 0.0)
        {
            optimizer.ClipGradientNorm(settings_.maxGradientNorm);
        }
    }

    /** @brief Chains the clipped surrogate with every head into the actor gradients. */
    template<typename Row>
    void AccumulateActorSample(Row const& row, double minibatchSize,
        core::UpdateStatistics& statistics)
    {
        RunActor(row.observation.data());
        FindActive(row.observation.data());
        if (probing_ && actorProbes_ != nullptr)
        {
            actorProbes_->ObserveForward();
        }
        double logProbability = 0;
        for (std::size_t component = 0; component < gaussianAxes_.size(); ++component)
        {
            if (!active_[gaussianAxes_[component]])
            {
                continue;
            }
            double const sample = static_cast<double>(row.action[gaussianAxes_[component]]);
            logProbability += heads::GaussianAxisLogDensity(sample, GaussianMean(component), LogStd(component));
        }
        for (std::size_t component = 0; component < bernoulliAxes_.size(); ++component)
        {
            if (!active_[bernoulliAxes_[component]])
            {
                continue;
            }
            logProbability += heads::BernoulliLogProbability(BernoulliLogit(component), Drawn(row, component));
        }
        auto const surrogate =
            core::EvaluateClippedSurrogate(logProbability, row.logProbability, row.advantage, settings_.clip);
        if (!std::isfinite(surrogate.objective))
        {
            throw std::runtime_error("PPOKit graph learner: non-finite PPO objective");
        }
        double const entropy = AccumulateGaussianGradients(row, surrogate.logProbabilityGradient, minibatchSize)
            + AccumulateBernoulliGradients(row, surrogate.logProbabilityGradient, minibatchSize);
        actor_.Backward(outputGradients_.data(), outputGradients_.size());
        statistics.policyLoss += -surrogate.objective - settings_.entropyCoefficient * entropy;
        statistics.entropy += entropy;
        statistics.approximateKl += core::ApproximateKl(surrogate.logRatio);
        minibatchSampledKl_ += core::SampledKl(surrogate.logRatio) / minibatchSize;
        statistics.clipFraction += surrogate.outsideClipRange ? 1.0 : 0.0;
    }

    /**
     * @brief Writes d loss / d mean into the head gradient and adds d loss / d log-std
     *        (zero for an inactive component).
     * @return Entropy of the active Gaussian components.
     */
    template<typename Row>
    double AccumulateGaussianGradients(Row const& row, double logProbabilityGradient,
        double minibatchSize)
    {
        if (gaussianAxes_.empty())
        {
            return 0.0;
        }
        double entropy = 0;
        // A learned log-std accumulates its gradient; a computed one takes this sample's, then Backward.
        bool const stateDependent = gaussianHead_->StateDependentLogStd();
        std::vector<float>& logStdGradient =
            stateDependent ? gaussianHead_->LogStdGradient() : gaussianHead_->LogStd().gradient;
        for (std::size_t component = 0; component < gaussianAxes_.size(); ++component)
        {
            if (!active_[gaussianAxes_[component]])
            {
                gaussianGradient_[component] = 0.F;
                if (stateDependent)
                {
                    logStdGradient[component] = 0.F;
                }
                continue;
            }
            double const logStd = LogStd(component);
            auto const gradient = heads::GaussianAxisLogDensityGradient(
                static_cast<double>(row.action[gaussianAxes_[component]]), GaussianMean(component), logStd);
            // Mean over the minibatch: each sample contributes derivative / minibatch size.
            gaussianGradient_[component] =
                static_cast<float>(logProbabilityGradient * gradient.mean / minibatchSize);
            // The entropy bonus adds d entropy / d log-std = 1 per component.
            logStdGradient[component] += static_cast<float>(
                (logProbabilityGradient * gradient.logStd - settings_.entropyCoefficient) / minibatchSize);
            entropy += heads::GaussianAxisEntropy(logStd);
        }
        return entropy;
    }

    /**
     * @brief Writes d loss / d logit into the Bernoulli head gradient (zero when inactive).
     * @return Entropy of the active Bernoulli components.
     */
    template<typename Row>
    double AccumulateBernoulliGradients(Row const& row, double logProbabilityGradient,
        double minibatchSize)
    {
        double entropy = 0;
        for (std::size_t component = 0; component < bernoulliAxes_.size(); ++component)
        {
            if (!active_[bernoulliAxes_[component]])
            {
                bernoulliGradient_[component] = 0.F;
                continue;
            }
            double const logit = BernoulliLogit(component);
            double const surrogatePart =
                logProbabilityGradient * heads::BernoulliLogProbabilityGradient(logit, Drawn(row, component));
            // The loss subtracts the entropy bonus, hence minus its gradient.
            double const entropyPart = settings_.entropyCoefficient * heads::BernoulliEntropyGradient(logit);
            double const logitGradient = surrogatePart - entropyPart;
            bernoulliGradient_[component] = static_cast<float>(logitGradient / minibatchSize);
            entropy += heads::BernoulliEntropy(logit);
        }
        return entropy;
    }

    template<typename Row>
    bool Drawn(Row const& row, std::size_t component) const noexcept
    {
        return row.action[bernoulliAxes_[component]] > 0.5F;
    }

    /** @brief Gradient of (V(s) - target)^2, averaged over the minibatch. */
    template<typename Row>
    void AccumulateCriticSample(Row const& row, double minibatchSize,
        core::UpdateStatistics& statistics)
    {
        double const error = Value(row.observation.data()) - row.valueTarget;
        if (!std::isfinite(error))
        {
            throw std::runtime_error("PPOKit graph learner: non-finite value error");
        }
        if (probing_ && criticProbes_ != nullptr)
        {
            criticProbes_->ObserveForward();
        }
        // With shared blocks the value loss is weighted in the one loss; alone it has its own optimizer.
        double const weight = Joint() ? settings_.valueCoefficient : 1.0;
        valueGradient_[0] = static_cast<float>(weight * 2.0 * error / minibatchSize);
        critic_.Backward(valueGradient_.data());
        statistics.valueLoss += error * error;
    }

    void Write(std::filesystem::path const& path, artifacts::ArtifactIdentity const& identity,
        artifacts::ArtifactKind kind)
    {
        artifacts::ArtifactHeader header;
        header.kind = kind;
        header.identity = identity;
        header.updates = updates_;
        header.seed = settings_.seed;
        header.created = artifacts::UtcNow();
        bool const training = kind == artifacts::ArtifactKind::Training;
        artifacts::WriteFileAtomically(
            path,
            [&](std::ostream& out) {
                out << artifacts::HeaderLine(header);
                checkpoint::WriteTensors(out, "actor", actor_.Parameters());
                if (!training)
                {
                    return;
                }
                checkpoint::WriteTensors(out, "critic", critic_.Parameters());
                if (Joint())
                {
                    checkpoint::WriteAdam(out, "adam", jointOptimizer_->Save());
                    checkpoint::WriteRandomState(out, rng_);
                    return;
                }
                checkpoint::WriteAdam(out, "actor_adam", actorOptimizer_->Save());
                checkpoint::WriteAdam(out, "critic_adam", criticOptimizer_->Save());
                checkpoint::WriteRandomState(out, rng_);
            },
            artifacts::AtomicWriteOptions{}, "PPOKit graph learner checkpoint");
    }

    void Read(std::filesystem::path const& path, artifacts::ArtifactIdentity const& expected, bool training)
    {
        std::string const source = path.filename().string();
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error(source + ": cannot open the checkpoint");
        }
        artifacts::ArtifactHeader const header = artifacts::ReadCompatibleHeader(in, expected, source, training);
        // Everything is read before anything is applied: a damaged file leaves the learner as it was.
        std::vector<Parameter*> const actorParameters = actor_.Parameters();
        checkpoint::Tensors const actor =
            checkpoint::ReadTensors(in, "actor", checkpoint::SizesOf(actorParameters), source);
        if (!training)
        {
            if (header.kind == artifacts::ArtifactKind::Policy)
            {
                artifacts::RequirePayloadEnd(in, source);
            }
            checkpoint::Apply(actor, actorParameters);
            CopySharedValues();
            return;
        }
        std::vector<Parameter*> const criticParameters = critic_.Parameters();
        checkpoint::Tensors const critic =
            checkpoint::ReadTensors(in, "critic", checkpoint::SizesOf(criticParameters), source);
        if (Joint())
        {
            Adam::State state = checkpoint::ReadAdam(in, "adam", JointParameters(), source);
            std::mt19937 const rng = checkpoint::ReadRandomState(in, source);
            artifacts::RequirePayloadEnd(in, source);
            jointOptimizer_->ValidateState(state);
            checkpoint::Apply(actor, actorParameters);
            checkpoint::Apply(critic, criticParameters);
            CopySharedValues();
            jointOptimizer_->Restore(std::move(state));
            rng_ = rng;
            settings_.seed = header.seed;
            updates_ = header.updates;
            return;
        }
        Adam::State actorState = checkpoint::ReadAdam(in, "actor_adam", actorParameters, source);
        Adam::State criticState = checkpoint::ReadAdam(in, "critic_adam", criticParameters, source);
        std::mt19937 const rng = checkpoint::ReadRandomState(in, source);
        artifacts::RequirePayloadEnd(in, source);
        actorOptimizer_->ValidateState(actorState);
        criticOptimizer_->ValidateState(criticState);
        checkpoint::Apply(actor, actorParameters);
        checkpoint::Apply(critic, criticParameters);
        actorOptimizer_->Restore(std::move(actorState));
        criticOptimizer_->Restore(std::move(criticState));
        rng_ = rng;
        settings_.seed = header.seed;
        updates_ = header.updates;
    }

    /** @brief Parameters of one block, in the order the graph lists them. */
    static std::vector<Parameter*> BlockParameters(Graph& network, std::string const& name)
    {
        std::vector<Parameter*> parameters;
        network.At(network.Find(name)).CollectParameters(parameters);
        return parameters;
    }

    /**
     * @brief Pairs each parameter of a sharing critic block with its actor block's.
     * @throws std::invalid_argument when the two blocks do not have the same tensors, or the
     *         settings ask for two learning rates.
     */
    void TieSharedBlocks(std::vector<SharedBlock> const& shared)
    {
        for (SharedBlock const& block : shared)
        {
            std::vector<Parameter*> const critic = BlockParameters(critic_, block.critic);
            std::vector<Parameter*> const actor = BlockParameters(actor_, block.actor);
            bool same = critic.size() == actor.size() && !critic.empty();
            for (std::size_t index = 0; same && index < critic.size(); ++index)
            {
                same = critic[index]->value.size() == actor[index]->value.size();
            }
            if (!same)
            {
                throw std::invalid_argument("PPOKit graph learner: critic block '" + block.critic
                    + "' cannot share the parameters of actor block '" + block.actor + "' (other tensors)");
            }
            for (std::size_t index = 0; index < critic.size(); ++index)
            {
                ties_.emplace_back(actor[index], critic[index]);
            }
        }
        if (!ties_.empty() && settings_.criticLearningRate != settings_.actorLearningRate)
        {
            throw std::invalid_argument("PPOKit graph learner: a critic that shares actor blocks is optimized with "
                                        "the actor (one Adam): set critic_learning_rate to actor_learning_rate");
        }
    }

    /** @brief Actor parameters, then the critic's own (its shared copies are the actor's). */
    std::vector<Parameter*> JointParameters()
    {
        std::vector<Parameter*> parameters = actor_.Parameters();
        for (Parameter* parameter : critic_.Parameters())
        {
            bool const shared = std::any_of(ties_.begin(), ties_.end(),
                [parameter](std::pair<Parameter*, Parameter*> const& tie) { return tie.second == parameter; });
            if (!shared)
            {
                parameters.push_back(parameter);
            }
        }
        return parameters;
    }

    /** @brief The critic's copies of the shared blocks take the actor's values. */
    void CopySharedValues()
    {
        for (auto const& [actorParameter, criticParameter] : ties_)
        {
            criticParameter->value = actorParameter->value;
        }
    }

    void ResetProbes() noexcept
    {
        if (actorProbes_ != nullptr)
        {
            actorProbes_->Reset();
        }
        if (criticProbes_ != nullptr)
        {
            criticProbes_->Reset();
        }
    }

    /** @brief Gradient sizes of this minibatch, before the optimizer steps consume them. */
    void ObserveProbeGradients() noexcept
    {
        if (actorProbes_ != nullptr)
        {
            actorProbes_->ObserveGradients();
        }
        if (criticProbes_ != nullptr)
        {
            criticProbes_->ObserveGradients();
        }
    }

    /** @brief Keeps a learned log-std in its training bounds (a computed one is clamped by its head). */
    void ClampLogStd() noexcept
    {
        if (gaussianHead_ == nullptr || gaussianHead_->StateDependentLogStd())
        {
            return;
        }
        for (float& logStd : gaussianHead_->LogStd().value)
        {
            logStd = std::clamp(logStd, static_cast<float>(settings_.minimumLogStd),
                static_cast<float>(settings_.maximumLogStd));
        }
    }

    std::size_t observationSize_;
    std::size_t actionSize_;
    core::PpoSettings settings_;
    Graph actor_;
    Graph critic_;
    std::vector<std::size_t> gaussianAxes_;
    std::vector<std::size_t> bernoulliAxes_;
    InputBinder actorInputs_;
    InputBinder criticInputs_;
    std::vector<ComponentActivity> activities_;
    /** @brief Active components for the observation last given to FindActive. */
    std::vector<std::uint8_t> active_;
    std::vector<std::uint8_t> oneOnTie_;
    HeadNode* gaussianHead_ = nullptr;
    std::size_t bernoulliOutput_ = 0;
    std::unique_ptr<Adam> actorOptimizer_;
    std::unique_ptr<Adam> criticOptimizer_;
    /** @brief With shared blocks: (actor parameter, critic copy) pairs and the one optimizer. */
    std::vector<std::pair<Parameter*, Parameter*>> ties_;
    std::unique_ptr<Adam> jointOptimizer_;
    Adam::UpdateBackup policyBackup_;
    Adam::UpdateBackup criticBackup_;
    std::vector<float> gaussianGradient_;
    std::vector<float> bernoulliGradient_;
    std::vector<float const*> outputGradients_;
    std::array<float, 1> valueGradient_{};
    runtime::PpoUpdateEngine updateEngine_;
    std::mt19937 rng_;
    std::uint64_t updates_ = 0;
    GraphProbes* actorProbes_ = nullptr;
    GraphProbes* criticProbes_ = nullptr;
    /** @brief True during the epoch the probes measure. */
    bool probing_ = false;
    /** @brief Sampled KL of the minibatch being accumulated, for the KL guard. */
    double minibatchSampledKl_ = 0.0;
};

/**
 * @brief Source-compatible adapter for fixed-dimension collectors.
 * @details It delegates all learning, checkpoint and policy operations to the runtime-sized engine.
 */
template<std::size_t ObservationSize, std::size_t ActionSize>
class GraphLearner final : private GraphLearnerCore
{
public:
    using Observation = std::array<float, ObservationSize>;
    using Action = std::array<float, ActionSize>;
    using Buffer = runtime::RolloutBuffer<double, ObservationSize, ActionSize>;

    GraphLearner(Graph actor, Graph critic, PolicyLayout layout, core::PpoSettings const& settings)
        : GraphLearnerCore(std::move(actor), std::move(critic), std::move(layout), settings, ObservationSize, ActionSize) {}

    runtime::PolicyDecision<double, ActionSize> Decide(Observation const& observation)
    {
        runtime::PolicyDecision<double, ActionSize> decision;
        GraphLearnerCore::Decide(observation.data(), decision.action.data(), decision.logProbability, decision.value);
        return decision;
    }

    Action Mean(Observation const& observation)
    {
        Action action{};
        GraphLearnerCore::Mean(observation.data(), action.data());
        return action;
    }

    double Value(Observation const& observation) { return GraphLearnerCore::Value(observation.data()); }
    core::UpdateStatistics Update(Buffer& buffer) { return GraphLearnerCore::Update(buffer); }

    using GraphLearnerCore::Actor;
    using GraphLearnerCore::Critic;
    using GraphLearnerCore::Joint;
    using GraphLearnerCore::LogStd;
    using GraphLearnerCore::SetProbes;
    using GraphLearnerCore::SaveTraining;
    using GraphLearnerCore::SavePolicy;
    using GraphLearnerCore::LoadTraining;
    using GraphLearnerCore::LoadPolicy;
    using GraphLearnerCore::Updates;
};

} // namespace ppokit::graph
