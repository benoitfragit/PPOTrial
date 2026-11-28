/**
 * @file environment.cpp
 * @brief reach simulation adapter for the common ppokit-run learner.
 */
#include "task.hpp"

#include "ppokit/environment/sdk.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <random>
#include <vector>

namespace {
using namespace reach_example;

class ReachEnvironment final
{
public:
    static constexpr std::size_t MetricCount = 1;

    ReachEnvironment(std::filesystem::path const& /*project*/, std::uint64_t seed)
        : task_(static_cast<std::uint32_t>(seed)), scenarioRng_(static_cast<std::uint32_t>(seed ^ 0x9e3779b9U))
    {
    }

    /** @brief Start an episode and publish its first observation. */
    void Reset(float* observation, float* /*world*/)
    {
        task_.Configure({std::uniform_real_distribution<double>(minimumDistance_, maximumDistance_)(scenarioRng_)});
        Observation const value = task_.Reset();
        std::copy(value.begin(), value.end(), observation);
    }

    /** @brief Report the stage promotion metric declared by task.json. */
    void EpisodeMetrics(double* metrics) const { metrics[0] = task_.Succeeded() ? 1.0 : 0.0; }

    /** @brief Apply the distance range selected by PPOKit's common curriculum. */
    void ApplyCurriculumStage(std::string const& name, std::string const& payload)
    {
        auto const stage = nlohmann::json::parse(payload);
        if (!stage.contains("start_distance") || !stage.at("start_distance").is_array()
            || stage.at("start_distance").size() != 2)
            throw std::invalid_argument("curriculum stage '" + name + "' requires start_distance [min,max]");
        double const minimum = stage.at("start_distance").at(0).get<double>();
        double const maximum = stage.at("start_distance").at(1).get<double>();
        if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum <= 0.0 || minimum > maximum)
            throw std::invalid_argument("curriculum stage '" + name + "' has an invalid distance range");
        minimumDistance_ = minimum;
        maximumDistance_ = maximum;
    }

    /** @brief Apply one action through the task without owning any PPO logic. */
    ppokit::runtime::StepEnd Step(float const* action, float* observation, float* /*world*/,
        double* rewardChannels)
    {
        Action applied{};
        std::copy_n(action, applied.size(), applied.begin());
        Observation next{};
        auto const outcome = task_.Step(applied, next);
        std::copy(next.begin(), next.end(), observation);
        std::copy_n(task_.LastRewardChannels(), Layout::Reward::Count, rewardChannels);
        return outcome.end;
    }

    /** @brief Apply task.json reward-channel weights. */
    void SetWeights(double const* weights)
    {
        task_.SetRewardWeights(std::vector<double>(weights, weights + Layout::Reward::Count));
    }

    /** @brief Return task.json reward-channel weights. */
    void Weights(double* weights) const
    {
        std::vector<double> const values = task_.RewardWeights();
        std::copy(values.begin(), values.end(), weights);
    }

private:
    ReachTask task_;
    std::mt19937 scenarioRng_;
    double minimumDistance_ = ReachTask::DefaultStartDistance;
    double maximumDistance_ = ReachTask::DefaultStartDistance;
};
} // namespace

PPOKIT_ENVIRONMENT(ReachEnvironment, "reach", ObservationSize, ActionSize, Layout::Reward::Count, 0)
