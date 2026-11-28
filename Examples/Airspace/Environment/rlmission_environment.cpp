/**
 * @file Examples/Airspace/Environment/rlmission_environment.cpp
 * @brief The Airspace world as an environment DLL for ppokit-run: the RLMission training
 *        agent of a scenario, driven from outside the world.
 * @details Training episodes are those of the project's scenario (project.json "settings",
 * e.g. Playground.json), whose one agent in training mode is driven from outside
 * (training_scenario.hpp: reward, step limit, random initializer drawn from the seed); its
 * learner is in the network DLL.
 *
 * Evaluation episodes (ppokit-run's suites, reset_evaluation) are the evaluator's cases
 * (evaluation_episodes.hpp): the seed of an episode is its case number.
 *
 * The world row (tactical_world.hpp) is the scene of the world the agent plays in, for the
 * Airspace Visualiseur.
 */
#include "airspacemeta.hpp"
#include "evaluation_episodes.hpp"
#include "missionlayout.hpp"
#include "tactical_world.hpp"
#include "targetselector.hpp"
#include "training_scenario.hpp"

#include "ppokit/environment/sdk.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace {

constexpr std::size_t ObservationSize = rl::MissionObservationLayout::kStateDimension;
constexpr std::size_t ActionSize = rl::MissionObservationLayout::kActionCount;
/** @brief task.json "rewards" of rl_mission: the REWARDS of the mission model. */
constexpr std::size_t Channels = rl::StatesAndActionsForMission::REWARDS.size();

/** @brief The project's scenario, which RLMission requires. */
std::filesystem::path RequiredScenario(std::filesystem::path const& project)
{
    std::optional<std::filesystem::path> const scenario = ppo::examples::airspace::TrainingScenarioOf(project);
    if (!scenario)
    {
        throw std::runtime_error(project.string() + ": project.json names no scenario (\"settings\")");
    }
    return *scenario;
}

class RLMissionEnvironment final
{
public:
    /** @brief One agent, through the team ABI: the reward is the agent's clamped one. */
    static constexpr std::size_t Agents = 1;
    /** @brief task.json "metrics": the results of an evaluation case. */
    static constexpr std::size_t MetricCount = ppo::examples::airspace::EvaluationMetricCount;

    /**
     * @param project RLMission project folder (project.json names the scenario).
     * @param seed Seed of the training episodes: the draws of the world's random initializer.
     */
    RLMissionEnvironment(std::filesystem::path const& project, std::uint64_t seed)
    {
        ppo::examples::airspace::RegisterAirspaceMeta();
        // The opponents of the evaluation cases.
        rl::EnsureMissionEvaluationFactoriesRegistered();
        training_.emplace(RequiredScenario(project), seed, ObservationSize, ActionSize, Channels);
    }

    void Reset(float* observations, std::uint8_t* available, float* world)
    {
        evaluating_ = false;
        training_->Reset(observations, available, world);
    }

    /** @brief Evaluation case `seed` (the suite's seeds are case numbers). */
    void ResetEvaluation(std::uint32_t seed, std::uint32_t /*episode*/, std::uint32_t /*episodes*/,
        float* observations, std::uint8_t* available, float* world)
    {
        evaluating_ = true;
        evaluation_.Start(seed, observations);
        evaluation_.WriteWorld(world);
        available[0] = 1;
    }

    void BeginStep(float* observations, std::uint8_t* available, float* world)
    {
        if (evaluating_)
        {
            return; // The campaign's world has no step to begin.
        }
        training_->BeginStep(observations, available, world);
    }

    ppokit::runtime::StepEnd Step(float const* actions, float* observations, std::uint8_t* available,
        float* world, double* rewards, double* rewardChannels)
    {
        if (evaluating_)
        {
            return EvaluationStep(actions, observations, available, world, rewards, rewardChannels);
        }
        return training_->Step(actions, observations, available, world, rewards, rewardChannels);
    }

    /** @brief The last evaluation case's metrics, in task.json order. */
    void EpisodeMetrics(double* metrics) const { evaluation_.Metrics(metrics); }

    void SetWeights(double const* weights) { training_->SetWeights(weights); }

    void Weights(double* weights) const { training_->Weights(weights); }

private:
    /** @brief A step of an evaluation case: no reward, the case's end. */
    ppokit::runtime::StepEnd EvaluationStep(float const* actions, float* observations, std::uint8_t* available,
        float* world, double* rewards, double* rewardChannels)
    {
        rewards[0] = 0.0;
        std::fill(rewardChannels, rewardChannels + Channels, 0.0);
        available[0] = 1;
        ppokit::runtime::StepEnd const end = evaluation_.Step(actions, observations);
        evaluation_.WriteWorld(world);
        return end;
    }

    std::optional<ppo::examples::airspace::TrainingScenario> training_;
    ppo::examples::airspace::EvaluationEpisodes<rl::TargetSelectorProdAgent> evaluation_{"RLMission", ActionSize};
    /** @brief The current episode is an evaluation case. */
    bool evaluating_ = false;
};

} // namespace

PPOKIT_ENVIRONMENT(RLMissionEnvironment, "rl_mission", ObservationSize, ActionSize, Channels,
    ppo::examples::airspace::tactical::WorldSize)
