/**
 * @file Examples/Airspace/Environment/rlsna_environment.cpp
 * @brief RLSNA (interception) as an environment DLL for ppokit-run: training, inference and
 *        evaluation.
 * @details Training episodes are those of the project's scenario (project.json "settings",
 * Scenarios/RLSNATraining.json: the interceptor in training mode and its target), driven from
 * outside (training_scenario.hpp: reward of the SNATrainingAgent, step limit, random
 * initializer drawn from the seed); its learner is in the network DLL. Without a scenario the
 * project only plays and evaluates, and training is refused.
 *
 * Evaluation episodes (ppokit-run's suites, reset_evaluation) are the evaluator's cases
 * (evaluation_episodes.hpp, Scenarios/Evaluation/RLSNA.json): the seed of an episode is its
 * case number. A played episode (play) is the next evaluation case in turn. The world row
 * (tactical_world.hpp) is the scene, for the Airspace Visualiseur.
 */
#include "airspacemeta.hpp"
#include "evaluation_episodes.hpp"
#include "interception.hpp"
#include "tactical_world.hpp"
#include "training_scenario.hpp"

#include "ppokit/environment/sdk.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace {

using Interception = rl::StatesAndActions;
constexpr std::size_t ObservationSize = std::tuple_size<Interception::STATES>::value;
constexpr std::size_t ActionSize = std::tuple_size<Interception::ACTIONS>::value;
/** @brief task.json "rewards" of rl_sna: the REWARDS of the interception model. */
constexpr std::size_t Channels = Interception::REWARDS.size();

class RLSNAEnvironment final
{
public:
    /** @brief One agent, through the team ABI: the reward is the agent's clamped one. */
    static constexpr std::size_t Agents = 1;
    /** @brief task.json "metrics": the results of an evaluation case. */
    static constexpr std::size_t MetricCount = ppo::examples::airspace::EvaluationMetricCount;

    /**
     * @param project RLSNA project folder (project.json "settings": its training scenario).
     * @param seed Seed of the training episodes: the draws of the world's random initializer.
     */
    RLSNAEnvironment(std::filesystem::path const& project, std::uint64_t seed)
    {
        ppo::examples::airspace::RegisterAirspaceMeta();
        // The opponents of the evaluation cases.
        rl::EnsureInterceptionEvaluationFactoriesRegistered();
        if (std::optional<std::filesystem::path> const scenario = ppo::examples::airspace::TrainingScenarioOf(project))
        {
            training_.emplace(*scenario, seed, ObservationSize, ActionSize, Channels);
        }
    }

    /** @brief Training: the scenario's episode. Playing: the next evaluation case, in turn. */
    void Reset(float* observations, std::uint8_t* available, float* world)
    {
        if (learning_ && training_)
        {
            evaluating_ = false;
            training_->Reset(observations, available, world);
            return;
        }
        std::size_t const count = evaluation_.CaseCount();
        StartCase(nextCase_ % count, observations, available, world);
        nextCase_ = (nextCase_ + 1) % count;
    }

    /** @brief Evaluation case `seed` (the suite's seeds are case numbers). */
    void ResetEvaluation(std::uint32_t seed, std::uint32_t /*episode*/, std::uint32_t /*episodes*/,
        float* observations, std::uint8_t* available, float* world)
    {
        StartCase(seed, observations, available, world);
    }

    void BeginStep(float* observations, std::uint8_t* available, float* world)
    {
        if (evaluating_)
        {
            return; // An evaluation case has no step to begin.
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

    /**
     * @brief Training needs the project's scenario; playing and evaluating use the cases.
     * @throws std::runtime_error when training is asked of a project without a scenario.
     */
    void SetLearning(bool learning)
    {
        if (learning && !training_)
        {
            throw std::runtime_error("this RLSNA project names no training scenario (project.json \"settings\")");
        }
        learning_ = learning;
    }

    void EpisodeMetrics(double* metrics) const { evaluation_.Metrics(metrics); }

    void SetWeights(double const* weights)
    {
        if (training_)
        {
            training_->SetWeights(weights);
        }
    }

    void Weights(double* weights) const
    {
        if (training_)
        {
            training_->Weights(weights);
            return;
        }
        std::fill(weights, weights + Channels, 0.0);
    }

private:
    void StartCase(std::size_t index, float* observations, std::uint8_t* available, float* world)
    {
        evaluating_ = true;
        evaluation_.Start(index, observations);
        evaluation_.WriteWorld(world);
        available[0] = 1;
    }

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
    ppo::examples::airspace::EvaluationEpisodes<rl::SNAProdAgent> evaluation_{"RLSNA", ActionSize};
    std::size_t nextCase_ = 0;
    /** @brief ppokit-run trains (set_learning 1, the ABI's default) or plays. */
    bool learning_ = true;
    /** @brief The current episode is an evaluation case. */
    bool evaluating_ = false;
};

} // namespace

PPOKIT_ENVIRONMENT(RLSNAEnvironment, "rl_sna", ObservationSize, ActionSize, Channels,
    ppo::examples::airspace::tactical::WorldSize)
