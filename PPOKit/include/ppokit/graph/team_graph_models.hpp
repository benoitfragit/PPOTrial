/**
 * @file PPOKit/include/ppokit/graph/team_graph_models.hpp
 * @brief Actor and critic block graphs for the hosted MAPPO trainer.
 * @details Implements the Models interface of ppokit::runtime::HostedTrainer with PPOKit
 * graphs (ppokit.graph/2): the actor reads the observation and ends in a Gaussian head,
 * the critic reads the centralized critic input of task.json and ends in a value. Each
 * network has its own Adam; the log-std of the Gaussian head is clamped after each step.
 *
 * @code
 * auto const task = ppokit::schema::LoadTaskSchema("task.json");
 * auto const graphs = ppokit::schema::LoadBlockGraph("graph.json", task);
 * ppokit::graph::TeamGraphModels models(ppokit::schema::BuildGraph(graphs.actor),
 *     ppokit::schema::BuildGraph(graphs.critic), ppokit::schema::MakeInputSlices(graphs.actor, "actor", task),
 *     ppokit::schema::MakeCriticInputSlices(graphs.critic, task), settings, observationSize, criticSize,
 *     actionSize);
 * @endcode
 *
 * The graphs keep the state of one row: a training backward runs each row forward again
 * before differentiating it. Not thread-safe.
 */
#pragma once

#include "ppokit/core/ppo_settings.hpp"
#include "ppokit/graph/adam.hpp"
#include "ppokit/graph/graph.hpp"
#include "ppokit/graph/policy_layout.hpp"
#include "ppokit/graph/probes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::graph {

class TeamGraphModels final
{
public:
    /**
     * @param settings Learning rates, Adam arithmetic, initial log-std and its bounds, and the
     *        seed of the weights.
     * @param observationSize, criticSize, actionSize Values of one agent's observation,
     *        centralized critic input and action (task.json).
     * @throws std::invalid_argument when the graphs do not fit the sizes.
     */
    TeamGraphModels(Graph actor, Graph critic, std::vector<InputSlice> actorInputs,
        std::vector<InputSlice> criticInputs, core::PpoSettings const& settings, std::size_t observationSize,
        std::size_t criticSize, std::size_t actionSize)
        : settings_(core::ValidatePpoSettings(settings)), observationSize_(observationSize),
          criticSize_(criticSize), actionSize_(actionSize), actor_(std::move(actor)), critic_(std::move(critic)),
          actorInputs_(std::move(actorInputs), observationSize), criticInputs_(std::move(criticInputs), criticSize)
    {
        head_ = actor_.OutputCount() == 1 ? dynamic_cast<HeadNode*>(&actor_.At(actor_.OutputId(0))) : nullptr;
        if (head_ == nullptr || head_->Kind() != HeadKind::Gaussian
            || head_->OutputShape() != Shape::Vector(actionSize_))
        {
            throw std::invalid_argument("PPOKit team models: the actor must end in one Gaussian head of "
                + std::to_string(actionSize_) + " actions");
        }
        if (head_->StateDependentLogStd())
        {
            throw std::invalid_argument("PPOKit team models: the log-std of the Gaussian head is learned "
                                        "(\"log_std\": \"state\" is for single-agent learners)");
        }
        if (critic_.OutputCount() != 1 || critic_.OutputShape() != Shape::Vector(1))
        {
            throw std::invalid_argument("PPOKit team models: the critic outputs one value");
        }
        std::mt19937 rng(settings_.seed);
        actor_.Initialize(rng);
        critic_.Initialize(rng);
        std::fill(head_->LogStd().value.begin(), head_->LogStd().value.end(),
            static_cast<float>(settings_.initialLogStd));
        actorOptimizer_ = std::make_unique<Adam>(actor_.Parameters(), AdamFor(settings_, settings_.actorLearningRate));
        criticOptimizer_ =
            std::make_unique<Adam>(critic_.Parameters(), AdamFor(settings_, settings_.criticLearningRate));
    }

    std::size_t ObservationSize() const noexcept { return observationSize_; }
    std::size_t CriticSize() const noexcept { return criticSize_; }
    std::size_t ActionSize() const noexcept { return actionSize_; }

    float LogStd(std::size_t action) const { return head_->LogStd().value.at(action); }

    void ActorMeans(float const* const* observations, std::size_t rows, float* means)
    {
        Run(actor_, actorInputs_, observations, rows, means, actionSize_);
    }

    /**
     * @brief ActorMeans, also copying the output of actor node `node` (a vector of `width`
     *        values) of each row into `features`, row after row; a null row gives zeros.
     */
    void ActorMeansAndFeatures(float const* const* observations, std::size_t rows, float* means, NodeId node,
        std::size_t width, float* features)
    {
        for (std::size_t row = 0; row < rows; ++row)
        {
            float* const mean = means + row * actionSize_;
            float* const feature = features + row * width;
            if (observations[row] == nullptr)
            {
                std::fill(mean, mean + actionSize_, 0.F);
                std::fill(feature, feature + width, 0.F);
                continue;
            }
            Signal const result = actor_.Forward(actorInputs_.Bind(observations[row]));
            std::copy(result.values, result.values + actionSize_, mean);
            Signal const captured = actor_.OutputOf(node);
            std::copy(captured.values, captured.values + width, feature);
        }
    }

    void CriticValues(float const* const* inputs, std::size_t rows, float* values)
    {
        Run(critic_, criticInputs_, inputs, rows, values, 1);
    }

    void ActorForward(float const* const* observations, std::size_t rows, float* means)
    {
        Remember(actorRows_, observations, rows);
        ActorMeans(observations, rows, means);
    }

    /** @brief Backpropagates d loss / d mean of every remembered row, adds d loss / d log-std, one Adam step. */
    void ActorBackwardAndStep(float const* meanGradients, float const* logStdGradients)
    {
        actor_.ZeroGradients();
        Backpropagate(actor_, actorInputs_, actorRows_, meanGradients, actionSize_, actorProbes_);
        std::vector<float>& logStd = head_->LogStd().gradient;
        for (std::size_t action = 0; action < actionSize_; ++action)
        {
            logStd[action] += logStdGradients[action];
        }
        if (actorProbes_ != nullptr)
        {
            actorProbes_->ObserveGradients();
        }
        if (settings_.maxGradientNorm > 0.0)
        {
            actorOptimizer_->ClipGradientNorm(settings_.maxGradientNorm);
        }
        actorOptimizer_->Step();
        for (float& value : head_->LogStd().value)
        {
            value = std::clamp(value, static_cast<float>(settings_.minimumLogStd),
                static_cast<float>(settings_.maximumLogStd));
        }
    }

    void CriticForward(float const* const* inputs, std::size_t rows, float* values)
    {
        Remember(criticRows_, inputs, rows);
        CriticValues(inputs, rows, values);
    }

    void CriticBackwardAndStep(float const* valueGradients)
    {
        critic_.ZeroGradients();
        Backpropagate(critic_, criticInputs_, criticRows_, valueGradients, 1, criticProbes_);
        if (criticProbes_ != nullptr)
        {
            criticProbes_->ObserveGradients();
        }
        if (settings_.maxGradientNorm > 0.0)
        {
            criticOptimizer_->ClipGradientNorm(settings_.maxGradientNorm);
        }
        criticOptimizer_->Step();
    }

    /**
     * @brief Probes fed by every training row and step (null: off, the default).
     * @details The trainer does not tell the models about epochs, so the probes see every
     * epoch of an update; reset them before the update to probe.
     */
    void SetProbes(GraphProbes* actor, GraphProbes* critic) noexcept
    {
        actorProbes_ = actor;
        criticProbes_ = critic;
    }

    Graph& Actor() noexcept { return actor_; }
    Graph& Critic() noexcept { return critic_; }
    /** @brief Optimizers, for checkpoints. */
    Adam& ActorOptimizer() noexcept { return *actorOptimizer_; }
    Adam& CriticOptimizer() noexcept { return *criticOptimizer_; }

private:
    /** @brief Forward of each row; a null row (padding) gives zeros. */
    static void Run(Graph& network, InputBinder& inputs, float const* const* rows, std::size_t count, float* outputs,
        std::size_t width)
    {
        for (std::size_t row = 0; row < count; ++row)
        {
            float* const output = outputs + row * width;
            if (rows[row] == nullptr)
            {
                std::fill(output, output + width, 0.F);
                continue;
            }
            Signal const result = network.Forward(inputs.Bind(rows[row]));
            std::copy(result.values, result.values + width, output);
        }
    }

    static void Remember(std::vector<float const*>& remembered, float const* const* rows, std::size_t count)
    {
        remembered.assign(rows, rows + count);
    }

    /** @brief Runs each remembered row again, then backward with its output gradient. */
    static void Backpropagate(Graph& network, InputBinder& inputs, std::vector<float const*> const& rows,
        float const* gradients, std::size_t width, GraphProbes* probes)
    {
        for (std::size_t row = 0; row < rows.size(); ++row)
        {
            if (rows[row] == nullptr)
            {
                continue;
            }
            network.Forward(inputs.Bind(rows[row]));
            if (probes != nullptr)
            {
                probes->ObserveForward();
            }
            network.Backward(gradients + row * width);
        }
    }

    core::PpoSettings settings_;
    std::size_t observationSize_;
    std::size_t criticSize_;
    std::size_t actionSize_;
    Graph actor_;
    Graph critic_;
    InputBinder actorInputs_;
    InputBinder criticInputs_;
    HeadNode* head_ = nullptr;
    std::unique_ptr<Adam> actorOptimizer_;
    std::unique_ptr<Adam> criticOptimizer_;
    std::vector<float const*> actorRows_;
    std::vector<float const*> criticRows_;
    GraphProbes* actorProbes_ = nullptr;
    GraphProbes* criticProbes_ = nullptr;
};

/** @brief Random interface of HostedTrainer on std::mt19937. */
class StandardTeamRandom final
{
public:
    explicit StandardTeamRandom(std::uint32_t seed) : engine_(seed) {}

    /** @brief The generator, for checkpoints. */
    std::mt19937& Engine() noexcept { return engine_; }

    void Shuffle(std::vector<std::size_t>& order) { std::shuffle(order.begin(), order.end(), engine_); }

    float Normal(float mean, float deviation)
    {
        std::normal_distribution<float> distribution(mean, deviation);
        return distribution(engine_);
    }

private:
    std::mt19937 engine_;
};

} // namespace ppokit::graph
