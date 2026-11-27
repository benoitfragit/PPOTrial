/**
 * @file PPOKit/tests/environments/team_line_environment.cpp
 * @brief A team environment DLL for the ppokit-run tests: up to three agents on a line,
 *        two of them present, each moving to the origin.
 * @details Each agent observes its offset to the origin and the elapsed share of the
 * episode; its action is a velocity. Channels: "Progress" (distance gained) and "Arrival"
 * (1 when within the radius). Each agent's reward is its channels' sum bounded to
 * +-RewardBound, so that the reward differs from the channel sum when it is large. The
 * second agent leaves the team at step LeaveStep (no longer present), so availability
 * changes during an episode. The team's episode ends when every present agent arrived
 * (terminated) or after EpisodeSteps steps (truncated). Metrics: "reached" (all arrived)
 * and "begun" (how far the BeginStep calls of the episode are from one per step after its
 * first: 0 when ppokit-run calls it before every step but the first).
 */
#include "ppokit/environment/sdk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <random>

namespace {

constexpr std::size_t Capacity = 3;
constexpr std::size_t Present = 2;
constexpr std::size_t ObservationSize = 2;
constexpr std::size_t ActionSize = 1;
constexpr std::size_t Channels = 2;
constexpr std::size_t EpisodeSteps = 40;
constexpr std::size_t LeaveStep = 6;
constexpr double Speed = 0.2;
constexpr double ArrivalRadius = 0.05;
constexpr double RewardBound = 0.15;

class TeamLineEnvironment final
{
public:
    static constexpr std::size_t Agents = Capacity;
    static constexpr std::size_t MetricCount = 2;

    TeamLineEnvironment(std::filesystem::path const& /*project*/, std::uint64_t seed)
        : random_(static_cast<std::uint32_t>(seed))
    {
    }

    void Reset(float* observations, std::uint8_t* available, float* /*world*/)
    {
        std::uniform_real_distribution<double> start(-1.0, 1.0);
        step_ = 0;
        begun_ = 0;
        reached_ = false;
        for (std::size_t agent = 0; agent < Capacity; ++agent)
        {
            present_[agent] = agent < Present;
            position_[agent] = present_[agent] ? start(random_) : 0.0;
        }
        Write(observations, available);
    }

    void ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t /*episodes*/, float* observations,
        std::uint8_t* available, float* world)
    {
        random_.seed(seed + episode);
        Reset(observations, available, world);
    }

    /** @brief Agent one leaves between decisions, changing the mask and observation used by the next action. */
    void BeginStep(float* observations, std::uint8_t* available, float* /*world*/)
    {
        ++begun_;
        if (step_ == LeaveStep)
        {
            present_[1] = false;
            Write(observations, available);
        }
    }

    ppokit::runtime::StepEnd Step(float const* actions, float* observations, std::uint8_t* available,
        float* /*world*/, double* rewards, double* rewardChannels)
    {
        std::fill(rewards, rewards + Capacity, 0.0);
        std::fill(rewardChannels, rewardChannels + Capacity * Channels, 0.0);
        for (std::size_t agent = 0; agent < Capacity; ++agent)
        {
            if (present_[agent])
            {
                MoveAgent(agent, actions[agent * ActionSize], rewards[agent], rewardChannels + agent * Channels);
            }
        }
        ++step_;
        Write(observations, available);
        reached_ = AllArrived();
        if (reached_)
        {
            return ppokit::runtime::StepEnd::Terminated;
        }
        if (step_ == EpisodeSteps)
        {
            return ppokit::runtime::StepEnd::Truncated;
        }
        return ppokit::runtime::StepEnd::Running;
    }

    void EpisodeMetrics(double* metrics) const
    {
        metrics[0] = reached_ ? 1.0 : 0.0;
        metrics[1] = std::abs(static_cast<double>(begun_) - static_cast<double>(step_ - 1));
    }

    void SetWeights(double const* weights) { std::copy(weights, weights + Channels, weights_.begin()); }

    void Weights(double* weights) const { std::copy(weights_.begin(), weights_.end(), weights); }

private:
    /** @brief Moves one present agent and writes its weighted channels and bounded reward. */
    void MoveAgent(std::size_t agent, float action, double& reward, double* channels)
    {
        double const before = std::abs(position_[agent]);
        double const velocity = std::clamp(static_cast<double>(action), -1.0, 1.0);
        position_[agent] += Speed * velocity;
        double const after = std::abs(position_[agent]);
        channels[0] = weights_[0] * (before - after);
        channels[1] = after < ArrivalRadius ? weights_[1] : 0.0;
        reward = std::clamp(channels[0] + channels[1], -RewardBound, RewardBound);
    }

    bool AllArrived() const noexcept
    {
        for (std::size_t agent = 0; agent < Capacity; ++agent)
        {
            if (present_[agent] && std::abs(position_[agent]) >= ArrivalRadius)
            {
                return false;
            }
        }
        return true;
    }

    /** @brief Observations of the present agents (absent rows zero) and their availability. */
    void Write(float* observations, std::uint8_t* available) const
    {
        double const elapsed = static_cast<double>(step_) / static_cast<double>(EpisodeSteps);
        for (std::size_t agent = 0; agent < Capacity; ++agent)
        {
            float* const row = observations + agent * ObservationSize;
            available[agent] = present_[agent] ? 1 : 0;
            row[0] = present_[agent] ? static_cast<float>(-position_[agent]) : 0.F;
            row[1] = present_[agent] ? static_cast<float>(elapsed) : 0.F;
        }
    }

    std::mt19937 random_;
    std::array<double, Capacity> position_{};
    std::array<bool, Capacity> present_{};
    std::array<double, Channels> weights_{1.0, 1.0};
    std::size_t step_ = 0;
    std::size_t begun_ = 0;
    bool reached_ = false;
};

} // namespace

PPOKIT_ENVIRONMENT(TeamLineEnvironment, "team_line", ObservationSize, ActionSize, Channels, 0)
