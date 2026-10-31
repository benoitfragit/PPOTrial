/**
 * @file generalized_patrol_production_policy.hpp
 * @brief Lightweight deterministic runner for the promoted blue MAPPO actor.
 */
#pragma once

#include "generalized_attention_actor.hpp"
#include "generalized_checkpoint.hpp"
#include "generalized_observation.hpp"
#include "team_capacity.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Owns only the actor and fixed inference workspaces used in production.
 *
 * Unlike GeneralizedPatrolLearner, this class has no rollout, critic,
 * optimizer, reward, checkpoint-writing, or evaluation-campaign capability.
 * It therefore makes the production execution path structurally unable to
 * train or to trigger periodic evaluations.
 */
class GeneralizedPatrolProductionPolicy final
{
public:
    using Observations = std::array<GeneralizedObservation,
                                    TeamCapacity::MaximumBlueMobiles>;
    using Actions = std::array<Action, TeamCapacity::MaximumBlueMobiles>;
    using AvailabilityMask = std::array<bool,
                                        TeamCapacity::MaximumBlueMobiles>;

    /**
     * @brief Loads a promoted generalized actor for deterministic inference.
     * @param counts Active P/N composition of the current scenario.
     * @param checkpoint Strict generalized V8 production checkpoint.
     * @throws std::invalid_argument If @p counts or @p checkpoint is invalid.
     * @throws std::runtime_error If the checkpoint cannot be read.
     */
    GeneralizedPatrolProductionPolicy(
        ActiveTeamCounts const counts,
        std::filesystem::path const& checkpoint)
        : counts_{counts}
    {
        if (!Valid(counts_) || checkpoint.empty())
            throw std::invalid_argument(
                "Invalid generalized production policy configuration");

        try
        {
            Allocate();
            std::ifstream stream{checkpoint};
            if (!stream)
            {
                throw std::runtime_error(
                    "Cannot open generalized blue production checkpoint: "
                    + checkpoint.string());
            }
            (void)ReadGeneralizedPolicy(stream, actor_);
        }
        catch (...)
        {
            Release();
            throw;
        }
    }

    /** @brief Releases the actor and its fixed inference workspaces. */
    ~GeneralizedPatrolProductionPolicy() { Release(); }

    GeneralizedPatrolProductionPolicy(
        GeneralizedPatrolProductionPolicy const&) = delete;
    GeneralizedPatrolProductionPolicy& operator=(
        GeneralizedPatrolProductionPolicy const&) = delete;
    GeneralizedPatrolProductionPolicy(
        GeneralizedPatrolProductionPolicy&&) = delete;
    GeneralizedPatrolProductionPolicy& operator=(
        GeneralizedPatrolProductionPolicy&&) = delete;

    /**
     * @brief Computes deterministic actor means for every available blue slot.
     * @param observations First P entries contain current blue observations.
     * @param availability True for each living blue slot.
     * @return Fixed-capacity actions; unavailable slots remain zero.
     */
    Actions Evaluate(
        Observations const& observations,
        AvailabilityMask const& availability)
    {
        ValidateBatch(observations, availability);
        rlt::set_all(device_, inferenceInput_, 0.F);
        for (Index agent = 0;
             agent < static_cast<Index>(counts_.blueMobiles); ++agent)
        {
            for (Index feature = 0;
                 feature < static_cast<Index>(
                               GeneralizedObservationLayout::Size);
                 ++feature)
            {
                rlt::set(inferenceInput_, agent, feature,
                         observations[static_cast<std::size_t>(agent)]
                             .values[static_cast<std::size_t>(feature)]);
            }
        }

        rlt::evaluate(device_, actor_, inferenceInput_, inferenceOutput_,
                      inferenceBuffer_);

        Actions actions{};
        for (std::size_t agent = 0U;
             agent < counts_.blueMobiles; ++agent)
        {
            if (!availability[agent])
                continue;
            for (std::size_t action = 0U;
                 action < GeneralizedObservationLayout::ActionSize; ++action)
            {
                float const value = rlt::get(
                    inferenceOutput_, static_cast<Index>(agent),
                    static_cast<Index>(action));
                if (!std::isfinite(value))
                {
                    throw std::runtime_error(
                        "Nonfinite generalized production action");
                }
                actions[agent][action] = value;
            }
        }
        return actions;
    }

private:
    using ActorSpecification = TrainingConfig::NN::ACTOR_SPEC;
    using Actor = GeneralizedAttentionActor<ActorSpecification>;
    using InputSpecification = rlt::matrix::Specification<
        float, Index, static_cast<Index>(TeamCapacity::MaximumBlueMobiles),
        static_cast<Index>(GeneralizedObservationLayout::Size)>;
    using OutputSpecification = rlt::matrix::Specification<
        float, Index, static_cast<Index>(TeamCapacity::MaximumBlueMobiles),
        static_cast<Index>(GeneralizedObservationLayout::ActionSize)>;

    /**
     * @brief Validates observation metadata and the fixed availability mask.
     * @param observations Candidate P/N observation batch.
     * @param availability Candidate active-blue mask.
     */
    void ValidateBatch(
        Observations const& observations,
        AvailabilityMask const& availability) const
    {
        for (std::size_t agent = 0U;
             agent < counts_.blueMobiles; ++agent)
        {
            auto const& observation = observations[agent];
            if (observation.metadata.activeBlueMobiles
                    != counts_.blueMobiles
                || observation.metadata.activeRedMobiles
                    != counts_.redMobiles
                || !Valid(observation))
            {
                throw std::invalid_argument(
                    "Invalid generalized production observation");
            }
        }
        for (std::size_t agent = counts_.blueMobiles;
             agent < TeamCapacity::MaximumBlueMobiles; ++agent)
        {
            if (availability[agent])
            {
                throw std::invalid_argument(
                    "Absent generalized production slot is available");
            }
        }
    }

    /** @brief Allocates the actor and fixed-capacity inference workspaces. */
    void Allocate()
    {
        rlt::malloc(device_, actor_);
        actorAllocated_ = true;
        rlt::malloc(device_, inferenceInput_);
        inputAllocated_ = true;
        rlt::malloc(device_, inferenceOutput_);
        outputAllocated_ = true;
        rlt::malloc(device_, inferenceBuffer_);
        bufferAllocated_ = true;
    }

    /** @brief Releases every successfully allocated RLtools resource. */
    void Release() noexcept
    {
        if (bufferAllocated_)
        {
            rlt::free(device_, inferenceBuffer_);
            bufferAllocated_ = false;
        }
        if (outputAllocated_)
        {
            rlt::free(device_, inferenceOutput_);
            outputAllocated_ = false;
        }
        if (inputAllocated_)
        {
            rlt::free(device_, inferenceInput_);
            inputAllocated_ = false;
        }
        if (actorAllocated_)
        {
            rlt::free(device_, actor_);
            actorAllocated_ = false;
        }
    }

    ActiveTeamCounts counts_{};
    Device device_{};
    Actor actor_{};
    rlt::MatrixDynamic<InputSpecification> inferenceInput_{};
    rlt::MatrixDynamic<OutputSpecification> inferenceOutput_{};
    typename Actor::template Buffer<
        static_cast<Index>(TeamCapacity::MaximumBlueMobiles)>
        inferenceBuffer_{};
    bool actorAllocated_{};
    bool inputAllocated_{};
    bool outputAllocated_{};
    bool bufferAllocated_{};
};

} // namespace ppo::examples::bluedefense::defenders
