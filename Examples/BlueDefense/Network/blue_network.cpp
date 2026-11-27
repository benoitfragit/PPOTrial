/**
 * @file Examples/BlueDefense/Network/blue_network.cpp
 * @brief The blue team's networks as a team network DLL for ppokit-run: the generalized
 *        MAPPO learner and the production actor of BlueDefense, unchanged.
 * @details The learner is GeneralizedPatrolLearner (attention actor, centralized critic,
 * PPOKit's HostedTrainer), created as the former PPO.exe's Training mode created it: the project's P/N
 * team (scenario.json), the training settings of blue_patrol_training.json, its seed and
 * rollout. Its checkpoints are PPOKit artifacts (generalized_artifact.hpp: training state and
 * actor, backend rltools); former V7/V10 files still load, so artifacts of earlier runs resume
 * and play. The policy is GeneralizedPatrolProductionPolicy on an actor artifact. Layer probes run every
 * ProbeEveryUpdates updates, as in the former PPO.exe's journal.
 *
 * Rows of agents are the blue slots blue_1..blue_9; the first P are the episode's blues, the
 * other rows are ignored and their actions are zero.
 */
#include "generalized_patrol_learner.hpp"
#include "generalized_patrol_production_policy.hpp"
#include "patrol_config.hpp"
#include "spawn_config.hpp"

#include "ppokit/network/team_sdk.hpp"
#include "ppokit/schema/project.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

namespace defenders = ppo::examples::bluedefense::defenders;

constexpr std::size_t Capacity = defenders::TeamCapacity::MaximumBlueMobiles;
constexpr std::size_t ObservationSize = defenders::GeneralizedObservationLayout::Size;
constexpr std::size_t ActionSize = defenders::GeneralizedObservationLayout::ActionSize;
/** @brief Updates between two probed updates, as the former PPO.exe's journal. */
constexpr std::uint64_t ProbeEveryUpdates = 10;
/** @brief The MAPPO training file of a BluePatrol project. */
constexpr char TrainingFile[] = "blue_patrol_training.json";

/** @brief The P blues and N reds of the project's scenario (its "blueDefense" spawn settings). */
defenders::ActiveTeamCounts TeamCountsOf(std::filesystem::path const& project)
{
    std::optional<std::filesystem::path> const scenario = ppokit::schema::LoadProject(project).SettingsFile();
    if (!scenario)
    {
        throw std::runtime_error(project.string() + ": project.json names no scenario (\"settings\")");
    }
    std::ifstream stream(*scenario);
    nlohmann::json const document = nlohmann::json::parse(stream, nullptr, false);
    if (document.is_discarded() || !document.contains("blueDefense"))
    {
        throw std::runtime_error(scenario->string() + ": not a BlueDefense scenario (no \"blueDefense\")");
    }
    std::string diagnostic;
    std::optional<ppo::examples::bluedefense::SpawnSettings> const spawn =
        ppo::examples::bluedefense::ParseSpawnSettings(document.at("blueDefense"), &diagnostic);
    if (!spawn || !spawn->blueTeam || !spawn->redTeam)
    {
        throw std::runtime_error(scenario->string() + ": invalid blue/red teams: " + diagnostic);
    }
    return {spawn->blueTeam->activeCount, spawn->redTeam->activeCount};
}

/** @brief The project's MAPPO training settings, read as the former PPO.exe read them. */
defenders::PatrolTrainingSettings TrainingSettingsOf(std::filesystem::path const& project)
{
    return defenders::LoadPatrolTrainingSettings(
        defenders::PatrolSettingsIn(std::filesystem::path{PPO_SHARED_SETTINGS}, project / TrainingFile));
}

/** @brief The rows of the P blues as the learner's observations, with the episode's counts. */
void ReadTeam(float const* rows, std::uint8_t const* available, defenders::ActiveTeamCounts counts,
    defenders::GeneralizedPatrolLearner::Observations& observations,
    defenders::GeneralizedPatrolLearner::AvailabilityMask& availability) noexcept
{
    for (std::size_t slot = 0; slot < counts.blueMobiles; ++slot)
    {
        float const* const row = rows + slot * ObservationSize;
        std::copy(row, row + ObservationSize, observations[slot].values.begin());
        observations[slot].metadata = {counts.blueMobiles, counts.redMobiles};
        availability[slot] = available[slot] != 0;
    }
}

/** @brief The P blues' actions as rows; the other rows zero. */
void WriteActions(defenders::GeneralizedPatrolLearner::Actions const& actions, std::size_t blues,
    float* rows) noexcept
{
    std::fill(rows, rows + Capacity * ActionSize, 0.F);
    for (std::size_t slot = 0; slot < blues; ++slot)
    {
        std::copy(actions[slot].begin(), actions[slot].end(), rows + slot * ActionSize);
    }
}

/** @brief One block probe in the C form; its strings point into `probe`. */
PpokitBlockProbe ToBlockProbe(char const* network, ppokit::runtime::BlockProbe const& probe) noexcept
{
    PpokitBlockProbe result{};
    result.network = network;
    result.block = probe.block.c_str();
    result.type = probe.type.c_str();
    result.samples = probe.samples;
    result.mean = probe.mean;
    result.deviation = probe.deviation;
    result.minimum = probe.minimum;
    result.maximum = probe.maximum;
    result.deadShare = probe.deadShare;
    result.saturatedShare = probe.saturatedShare;
    result.hasGradient = probe.hasGradient ? 1 : 0;
    result.gradientNorm = probe.gradientNorm;
    return result;
}

class BlueTeamLearner final
{
public:
    BlueTeamLearner(std::filesystem::path const& project, std::uint64_t seed)
        : counts_(TeamCountsOf(project)), training_(TrainingSettingsOf(project))
    {
        std::uint32_t const learnerSeed = seed == 0 ? training_.seed : static_cast<std::uint32_t>(seed);
        learner_ = std::make_unique<defenders::GeneralizedPatrolLearner>(counts_, learnerSeed, training_.rolloutSteps);
        ScheduleProbes();
    }

    void Act(float const* observations, std::uint8_t const* available, float* actions, bool deterministic)
    {
        ReadTeam(observations, available, counts_, observations_, availability_);
        defenders::GeneralizedPatrolLearner::Actions const chosen = deterministic
            ? learner_->EvaluateActions(observations_, availability_)
            : learner_->ChooseActions(observations_, availability_);
        WriteActions(chosen, counts_.blueMobiles, actions);
    }

    bool Observe(float const* observations, std::uint8_t const* available, double const* rewards,
        ppokit::runtime::StepEnd end)
    {
        ReadTeam(observations, available, counts_, observations_, availability_);
        // The environment's rewards are the controller's floats: the conversion is exact.
        for (std::size_t slot = 0; slot < counts_.blueMobiles; ++slot)
        {
            rewards_[slot] = static_cast<float>(rewards[slot]);
        }
        bool const updated = learner_->Observe(observations_, rewards_, availability_,
            end == ppokit::runtime::StepEnd::Terminated, end == ppokit::runtime::StepEnd::Truncated);
        if (updated)
        {
            ScheduleProbes();
        }
        return updated;
    }

    PpokitUpdateStatistics Statistics() const noexcept
    {
        ppokit::runtime::TeamUpdateStatistics const& last = learner_->LastUpdate();
        PpokitUpdateStatistics statistics{};
        statistics.updates = learner_->StepCount();
        statistics.policyLoss = last.policyLoss;
        statistics.valueLoss = last.valueLoss;
        statistics.entropy = last.entropy;
        statistics.approximateKl = last.approximateKl;
        statistics.clipFraction = last.clipFraction;
        statistics.explainedVariance = last.explainedVariance;
        statistics.advantageAbsMean = last.advantageAbsMean;
        statistics.actionStdMean = last.actionStdMean;
        statistics.learnableRows = last.learnableRows;
        statistics.criticRows = last.criticRows;
        return statistics;
    }

    PpokitTeamSettings Settings() const noexcept
    {
        ppokit::runtime::TeamPpoSettings const settings = defenders::GeneralizedPatrolLearner::PpoSettings();
        PpokitTeamSettings result{};
        result.gamma = settings.gamma;
        result.lambda = settings.lambda;
        result.clip = settings.clip;
        result.entropyCoefficient = settings.entropyCoefficient;
        result.epochs = static_cast<std::uint32_t>(settings.epochs);
        result.minibatch = static_cast<std::uint32_t>(settings.minibatch);
        return result;
    }

    /** @brief The actor's then the critic's blocks, on the updates the former PPO.exe's journal probed. */
    std::size_t ProbeCount() const noexcept
    {
        if (learner_->StepCount() % ProbeEveryUpdates != 0)
        {
            return 0;
        }
        defenders::GeneralizedLayerProbes const& probes = learner_->LayerProbes();
        if (probes.actor.empty())
        {
            return 0;
        }
        return probes.actor.size() + probes.critic.size();
    }

    PpokitBlockProbe Probe(std::size_t index) const
    {
        if (index >= ProbeCount())
        {
            throw std::out_of_range("no probe " + std::to_string(index) + " in the last update");
        }
        defenders::GeneralizedLayerProbes const& probes = learner_->LayerProbes();
        if (index < probes.actor.size())
        {
            return ToBlockProbe("actor", probes.actor[index]);
        }
        return ToBlockProbe("critic", probes.critic[index - probes.actor.size()]);
    }

    /** @brief The training artifact and the actor artifact (PPOKit format). */
    void Save(std::filesystem::path const& trainingFile, std::filesystem::path const& policyFile) const
    {
        if (!trainingFile.empty())
        {
            std::filesystem::create_directories(trainingFile.parent_path());
            learner_->SaveTrainingState(trainingFile);
        }
        if (!policyFile.empty())
        {
            std::filesystem::create_directories(policyFile.parent_path());
            learner_->SaveActor(policyFile);
        }
    }

    void Load(std::filesystem::path const& trainingFile)
    {
        learner_->LoadTrainingState(trainingFile);
        ScheduleProbes();
    }

    std::uint64_t Updates() const noexcept { return learner_->StepCount(); }

private:
    /** @brief Probes the next update when it is one of every ProbeEveryUpdates. */
    void ScheduleProbes() noexcept
    {
        learner_->ProbeLayers((learner_->StepCount() + 1U) % ProbeEveryUpdates == 0U);
    }

    defenders::ActiveTeamCounts counts_;
    defenders::PatrolTrainingSettings training_;
    std::unique_ptr<defenders::GeneralizedPatrolLearner> learner_;
    defenders::GeneralizedPatrolLearner::Observations observations_{};
    defenders::GeneralizedPatrolLearner::AvailabilityMask availability_{};
    defenders::GeneralizedPatrolLearner::Rewards rewards_{};
};

class BlueTeamPolicy final
{
public:
    BlueTeamPolicy(std::filesystem::path const& project, std::filesystem::path const& policyFile)
        : counts_(TeamCountsOf(project)), policy_(counts_, policyFile)
    {
    }

    void Act(float const* observations, std::uint8_t const* available, float* actions)
    {
        ReadTeam(observations, available, counts_, observations_, availability_);
        WriteActions(policy_.Evaluate(observations_, availability_), counts_.blueMobiles, actions);
    }

private:
    defenders::ActiveTeamCounts counts_;
    defenders::GeneralizedPatrolProductionPolicy policy_;
    defenders::GeneralizedPatrolLearner::Observations observations_{};
    defenders::GeneralizedPatrolLearner::AvailabilityMask availability_{};
};

} // namespace

PPOKIT_TEAM_NETWORK(BlueTeamLearner, BlueTeamPolicy, "blue_patrol", Capacity, ObservationSize, ActionSize, "rltools")
