/**
 * @file ppokit/runtime/ppo_update_engine.hpp
 * @brief Shared PPO/MAPPO update orchestration, independent of models and simulators.
 */
#pragma once

#include "ppokit/runtime/advantages.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ppokit::runtime {

/** @brief Counts of samples passed to the actor and critic across an update. */
struct PpoSampleCounts final
{
    std::size_t actor = 0;
    std::size_t critic = 0;
};

/**
 * @brief Common update engine for single-stream PPO and synchronized MAPPO.
 * @details Rollouts provide trajectory-specific GAE. Model adapters provide forward,
 * backward and optimizer operations for one minibatch; the engine owns the sample
 * order, epoch schedule and statistics reduction. The scalar and random generator
 * remain those of the caller to preserve checkpoint continuation and rounding.
 */
class PpoUpdateEngine final
{
public:
    explicit PpoUpdateEngine(std::size_t rows = 0) { ResetOrder(rows); }

    /** @brief Reset to collection order before baselines and the first shuffled epoch. */
    void ResetOrder(std::size_t rows)
    {
        order_.resize(rows);
        std::iota(order_.begin(), order_.end(), std::size_t{0});
    }

    std::size_t Size() const noexcept { return order_.size(); }
    std::size_t IndexAt(std::size_t position) const noexcept { return order_[position]; }

    /**
     * @brief Compute GAE, report pre-normalization diagnostics and normalize actor advantages.
     * @param beforeNormalize Callback receiving the number of learnable rows, after GAE.
     * @return Explained variance before normalization, in the rollout's scalar type.
     */
    template<typename Rollout, typename Scalar, typename Diagnostics>
    static Scalar Prepare(Rollout& rollout, Scalar gamma, Scalar lambda, Diagnostics&& beforeNormalize)
    {
        if (!rollout.Full() || rollout.Size() == 0)
        {
            throw std::invalid_argument("PPOKit update: a full non-empty rollout is required");
        }
        std::size_t const learnable = rollout.ComputeAdvantages(gamma, lambda);
        beforeNormalize(learnable);
        Scalar const explained = rollout.ExplainedVariance();
        rollout.NormalizeAdvantages();
        return explained;
    }

    /**
     * @brief Shuffle once per epoch and visit each row exactly once in non-overlapping batches.
     * @param shuffle Mutates the order using the learner's checkpointed generator.
     * @param beginEpoch Called with zero-based epoch index before shuffling.
     * @param trainBatch Receives a half-open interval of positions; returns its actor sample count.
     * @param endEpoch Called with the completed, one-based epoch number.
     */
    template<typename Shuffle, typename BeginEpoch, typename TrainBatch, typename EndEpoch>
    PpoSampleCounts RunEpochs(std::size_t epochs, std::size_t minibatch, Shuffle&& shuffle,
        BeginEpoch&& beginEpoch, TrainBatch&& trainBatch, EndEpoch&& endEpoch)
    {
        if (epochs == 0 || minibatch == 0 || order_.empty())
        {
            throw std::invalid_argument("PPOKit update: positive epochs, minibatch and rows required");
        }
        PpoSampleCounts counts;
        for (std::size_t epoch = 0; epoch < epochs; ++epoch)
        {
            beginEpoch(epoch);
            shuffle(order_);
            for (std::size_t start = 0; start < order_.size();)
            {
                std::size_t const end = start + std::min(minibatch, order_.size() - start);
                counts.actor += trainBatch(start, end);
                counts.critic += end - start;
                start = end;
            }
            endEpoch(epoch + 1);
        }
        return counts;
    }

    /** @brief Reduce accumulated losses and diagnostics over their respective sample populations. */
    template<typename Statistics>
    static void Finalize(Statistics& statistics, PpoSampleCounts counts) noexcept
    {
        using Scalar = decltype(statistics.policyLoss);
        if (counts.actor != 0)
        {
            Scalar const inverse = Scalar(1) / static_cast<Scalar>(counts.actor);
            statistics.policyLoss *= inverse;
            statistics.entropy *= inverse;
            statistics.approximateKl *= inverse;
            statistics.clipFraction *= inverse;
        }
        if (counts.critic != 0)
        {
            statistics.valueLoss /= static_cast<Scalar>(counts.critic);
        }
    }

private:
    std::vector<std::size_t> order_;
};

} // namespace ppokit::runtime
