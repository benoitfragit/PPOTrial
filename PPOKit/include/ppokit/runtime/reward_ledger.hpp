/**
 * @file PPOKit/include/ppokit/runtime/reward_ledger.hpp
 * @brief Per-channel reward accounting for the run journal and the studio.
 * @details A task reports named reward channels (task.json "rewards") and may bound their weighted sum.
 * The ledger receives the raw value of every channel at every step of every stream (an
 * agent, or an environment) and answers, once per update:
 *
 * - the mean per step of each channel, raw and weighted;
 * - the mean learner return per completed episode, and each channel before any final reward clamp;
 * - the running total of each weighted channel since the start of the run.
 *
 * Weights may change between updates (SetWeights); a step is weighted with the weights
 * in force when it is recorded. Storage is allocated at construction: Record never
 * allocates.
 */
#pragma once

#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::runtime {

/** @brief One channel over the last update window. */
struct ChannelSummary final
{
    /** @brief Mean raw value per step. */
    double stepMean = 0.0;
    /** @brief Mean weighted value per step (contribution to the reward). */
    double weightedStepMean = 0.0;
    /** @brief Mean weighted channel sum per completed episode, before any final reward clamp. */
    double episodeMean = 0.0;
    /** @brief Weighted sum since the start of the run. */
    double cumulative = 0.0;
};

/** @brief All channels over the last update window. */
struct RewardSummary final
{
    std::vector<ChannelSummary> channels;
    /** @brief Steps and completed episodes in the window. */
    std::size_t steps = 0;
    std::size_t episodes = 0;
    /** @brief Mean of the scalar returns received by the learner over completed streams. */
    double episodeReturnMean = 0.0;
};

class RewardLedger final
{
public:
    /**
     * @param weights Initial weight of each channel (task.json defaults).
     * @param streams Number of trajectories recorded in parallel (agents, environments).
     * @throws std::invalid_argument without channels or streams.
     */
    RewardLedger(std::vector<double> weights, std::size_t streams)
        : weights_(std::move(weights)), streams_(streams), episode_(streams * weights_.size(), 0.0),
          episodeReturns_(streams, 0.0), windowRaw_(weights_.size(), 0.0), windowWeighted_(weights_.size(), 0.0),
          windowEpisodes_(weights_.size(), 0.0), cumulative_(weights_.size(), 0.0)
    {
        if (weights_.empty() || streams == 0)
        {
            throw std::invalid_argument("PPOKit reward ledger: at least one channel and one stream are required");
        }
    }

    std::size_t Channels() const noexcept { return weights_.size(); }
    std::vector<double> const& Weights() const noexcept { return weights_; }

    /**
     * @brief Weights of the following steps (the caller applies them at an update boundary).
     * @throws std::invalid_argument for a wrong count.
     */
    void SetWeights(std::vector<double> const& weights)
    {
        if (weights.size() != weights_.size())
        {
            throw std::invalid_argument("PPOKit reward ledger: " + std::to_string(weights_.size())
                + " weights expected, " + std::to_string(weights.size()) + " given");
        }
        weights_ = weights;
    }

    /**
     * @brief Restore the recorded channel totals when starting a new collection session.
     * @param checkpoint Summary at the resumed update, or null when its prior totals are unknown.
     */
    void RestoreCumulative(RewardSummary const* checkpoint)
    {
        if (checkpoint && checkpoint->channels.size() != Channels())
            throw std::invalid_argument("PPOKit reward ledger: checkpoint channel count differs");
        for (std::size_t channel = 0; channel < Channels(); ++channel)
        {
            cumulative_[channel] = checkpoint ? checkpoint->channels[channel].cumulative
                                              : std::numeric_limits<double>::quiet_NaN();
        }
    }

    /** @brief Raw channel values of one step of `stream` (Channels() values). */
    void Record(std::size_t stream, double const* channels)
    {
        RequireStream(stream);
        double reward = 0.0;
        for (std::size_t channel = 0; channel < weights_.size(); ++channel)
        {
            double const weighted = weights_[channel] * channels[channel];
            reward += weighted;
            windowRaw_[channel] += channels[channel];
            windowWeighted_[channel] += weighted;
            episode_[stream * weights_.size() + channel] += weighted;
            cumulative_[channel] += weighted;
        }
        episodeReturns_[stream] += reward;
        ++windowSteps_;
    }

    /**
     * @brief Record the weighted channels and the scalar reward of the transition.
     * @param stream Agent or trajectory that produced the transition.
     * @param weightedChannels Contributions in task order, before any final task-level clamp.
     * @param learnerReward Reward stored in the rollout. Omit only for tasks using the sum of contributions.
     * @details The raw value is recovered as weighted / weight; it is unknown (NaN, a gap in
     * the studio) for a channel whose weight is zero.
     */
    void RecordWeighted(std::size_t stream, double const* weightedChannels,
        std::optional<double> learnerReward = std::nullopt)
    {
        RequireStream(stream);
        episodeReturns_[stream] += learnerReward ? *learnerReward
            : std::accumulate(weightedChannels, weightedChannels + weights_.size(), 0.0);
        for (std::size_t channel = 0; channel < weights_.size(); ++channel)
        {
            double const weighted = weightedChannels[channel];
            double const raw = weights_[channel] != 0.0 ? weighted / weights_[channel]
                                                        : std::numeric_limits<double>::quiet_NaN();
            windowRaw_[channel] += raw;
            windowWeighted_[channel] += weighted;
            episode_[stream * weights_.size() + channel] += weighted;
            cumulative_[channel] += weighted;
        }
        ++windowSteps_;
    }

    /** @brief Closes the episode of `stream`: its channel sums count in the episode means. */
    void EndEpisode(std::size_t stream)
    {
        RequireStream(stream);
        for (std::size_t channel = 0; channel < weights_.size(); ++channel)
        {
            double& sum = episode_[stream * weights_.size() + channel];
            windowEpisodes_[channel] += sum;
            sum = 0.0;
        }
        completedReturns_ += episodeReturns_[stream];
        episodeReturns_[stream] = 0.0;
        ++windowEpisodeCount_;
    }

    /**
     * @brief Summary of the steps and episodes since the previous call, then a new window.
     * @details Episodes still running keep accumulating into the next window; the
     * cumulative totals never reset.
     */
    RewardSummary Summarize()
    {
        RewardSummary summary;
        summary.steps = windowSteps_;
        summary.episodes = windowEpisodeCount_;
        summary.episodeReturnMean = Mean(completedReturns_, windowEpisodeCount_);
        summary.channels.resize(weights_.size());
        for (std::size_t channel = 0; channel < weights_.size(); ++channel)
        {
            ChannelSummary& entry = summary.channels[channel];
            entry.stepMean = Mean(windowRaw_[channel], windowSteps_);
            entry.weightedStepMean = Mean(windowWeighted_[channel], windowSteps_);
            entry.episodeMean = Mean(windowEpisodes_[channel], windowEpisodeCount_);
            entry.cumulative = cumulative_[channel];
            windowRaw_[channel] = 0.0;
            windowWeighted_[channel] = 0.0;
            windowEpisodes_[channel] = 0.0;
        }
        windowSteps_ = 0;
        windowEpisodeCount_ = 0;
        completedReturns_ = 0.0;
        return summary;
    }

    /** @brief Weighted channel sums of the window being collected (the live status shows them). */
    std::vector<double> const& WindowWeighted() const noexcept { return windowWeighted_; }
    std::size_t WindowSteps() const noexcept { return windowSteps_; }
    std::size_t WindowEpisodes() const noexcept { return windowEpisodeCount_; }

private:
    static double Mean(double sum, std::size_t count) noexcept
    {
        return count == 0 ? 0.0 : sum / static_cast<double>(count);
    }

    void RequireStream(std::size_t stream) const
    {
        if (stream >= streams_)
        {
            throw std::out_of_range("PPOKit reward ledger: stream " + std::to_string(stream) + " out of "
                + std::to_string(streams_));
        }
    }

    std::vector<double> weights_;
    std::size_t streams_;
    /** @brief Weighted channel sums of the running episode of each stream. */
    std::vector<double> episode_;
    std::vector<double> episodeReturns_;
    double completedReturns_ = 0.0;
    std::vector<double> windowRaw_;
    std::vector<double> windowWeighted_;
    std::vector<double> windowEpisodes_;
    std::vector<double> cumulative_;
    std::size_t windowSteps_ = 0;
    std::size_t windowEpisodeCount_ = 0;
};

} // namespace ppokit::runtime
