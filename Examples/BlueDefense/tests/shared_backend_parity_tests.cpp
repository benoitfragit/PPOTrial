/**
 * @file Examples/BlueDefense/tests/shared_backend_parity_tests.cpp
 * @brief Blue MAPPO on the shared graph backend updates bit for bit as its own learner.
 * @details The Blue learner (GeneralizedPatrolLearner: rl_tools storage, its attention graph,
 * rl_tools Adam) trains one update, then its training state is converted to the shared
 * backend (WriteSharedBackendTraining) and loaded by TeamGraphLearner, built from
 * tests/fixtures/shared_backend (task.json, network.json with "team_critic", training.json
 * with the rl_tools Adam). Both then receive the same team steps for two more updates:
 * - every sampled action is the same float;
 * - every update statistic is the same float (losses, KL, clip fraction, explained
 *   variance, entropy, advantages, action deviation, row counts);
 * - the converted Blue state and the shared learner's own state are the same file
 *   (weights, both Adams with their step counts, generator), after the header.
 *
 * P = 3 blues, N = 2 reds: agent 2 is destroyed in the second half of each episode, rows
 * 3..8 never hold an agent. Rollouts end on episode ends, so resuming starts an episode.
 */
#include "generalized_patrol_learner.hpp"
#include "shared_backend_conversion.hpp"

#include "ppokit/graph/team_graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace blue = ppo::examples::bluedefense::defenders;

namespace {

constexpr std::size_t Capacity = blue::TeamCapacity::MaximumBlueMobiles;
constexpr std::size_t ObservationSize = blue::GeneralizedObservationLayout::Size;
constexpr std::size_t ActionSize = blue::GeneralizedObservationLayout::ActionSize;
constexpr blue::ActiveTeamCounts Counts{3U, 2U};
constexpr std::size_t RolloutTeamSteps = 256;
constexpr int EpisodeSteps = 64;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool SameBits(float a, float b) noexcept
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

/** @brief A file after its header line (which holds the writing date). */
std::string Body(std::filesystem::path const& path)
{
    std::ifstream in(path, std::ios::binary);
    std::string header;
    std::getline(in, header);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** @brief Deterministic team steps: observations, the living blues, rewards, episode ends. */
class Scenario final
{
public:
    Scenario() : roles_(*blue::BlueTeamRoles::Create(Counts.blueMobiles, 0U)) {}

    /** @brief Observations and availability at team step `clock` (episode step `clock % EpisodeSteps`). */
    void Fill(int clock, blue::GeneralizedPatrolLearner::Observations& observations,
        blue::GeneralizedPatrolLearner::AvailabilityMask& available) const
    {
        int const step = clock % EpisodeSteps;
        for (std::size_t agent = 0; agent < Counts.blueMobiles; ++agent)
        {
            auto& observation = observations[agent];
            observation.metadata = {Counts.blueMobiles, Counts.redMobiles};
            observation.values.fill(0.F);
            // Own state and lidar, then two red tracks, the other blues and the objective, masked in.
            for (std::size_t feature = 0; feature < blue::GeneralizedObservationLayout::LidarOffset
                     + blue::ObservationLayout::Lidar;
                 ++feature)
            {
                observation.values[feature] = Wave(clock, feature, agent);
            }
            FillTracks(observation.values, blue::GeneralizedObservationLayout::RedOffset, Counts.redMobiles,
                blue::GeneralizedObservationLayout::RedFeatures, clock, agent);
            FillTracks(observation.values, blue::GeneralizedObservationLayout::BlueOffset, Counts.blueMobiles - 1U,
                blue::GeneralizedObservationLayout::FriendlyFeatures, clock, agent + 5U);
            FillTracks(observation.values, blue::GeneralizedObservationLayout::ObjectiveOffset, 1U,
                blue::GeneralizedObservationLayout::FriendlyFeatures, clock, agent + 9U);
            for (std::size_t feature = 0; feature < blue::ObservationLayout::Patrol; ++feature)
            {
                observation.values[blue::GeneralizedObservationLayout::PatrolOffset + feature] =
                    0.5F * Wave(clock, feature + 200U, agent);
            }
            blue::EncodeGeneralizedRole(observation.values, *roles_.Assignment(agent));
            blue::EncodeGeneralizedTeamCounts(observation.values, Counts);
            available[agent] = agent != 2 || step < EpisodeSteps / 2;
        }
    }

    static bool EndsEpisode(int clock) noexcept { return clock % EpisodeSteps == EpisodeSteps - 1; }
    static bool Terminated(int clock) noexcept { return EndsEpisode(clock) && (clock / EpisodeSteps) % 2 == 1; }

private:
    static float Wave(int clock, std::size_t feature, std::size_t agent) noexcept
    {
        return std::sin(0.013F * static_cast<float>(clock * 7 + static_cast<int>(feature * 3 + agent * 11)));
    }

    /** @brief `tracks` present tracks of `features` values from `offset`, the mask (last value) at 1. */
    static void FillTracks(blue::GeneralizedRawObservation& values, std::size_t offset, std::size_t tracks,
        std::size_t features, int clock, std::size_t salt) noexcept
    {
        for (std::size_t track = 0; track < tracks; ++track)
        {
            for (std::size_t feature = 0; feature + 1 < features; ++feature)
            {
                values[offset + track * features + feature] = Wave(clock, feature + 20U * track, salt);
            }
            values[blue::GeneralizedTrackMaskIndex(offset, track, features)] = 1.F;
        }
    }

    blue::BlueTeamRoles roles_;
};

/** @brief The shared learner's rows: every slot of the team capacity. */
struct SharedRows final
{
    std::array<float, Capacity * ObservationSize> observations{};
    std::array<std::uint8_t, Capacity> available{};
};

SharedRows RowsOf(blue::GeneralizedPatrolLearner::Observations const& observations,
    blue::GeneralizedPatrolLearner::AvailabilityMask const& available)
{
    SharedRows rows;
    for (std::size_t agent = 0; agent < Counts.blueMobiles; ++agent)
    {
        std::copy(observations[agent].values.begin(), observations[agent].values.end(),
            rows.observations.begin() + static_cast<std::ptrdiff_t>(agent * ObservationSize));
        rows.available[agent] = available[agent] ? 1 : 0;
    }
    return rows;
}

ppokit::runtime::StepEnd EndOf(bool terminated, bool truncated) noexcept
{
    if (terminated)
    {
        return ppokit::runtime::StepEnd::Terminated;
    }
    return truncated ? ppokit::runtime::StepEnd::Truncated : ppokit::runtime::StepEnd::Running;
}

/** @brief Statistics of the two learners after one update, compared bit for bit. */
bool SameUpdate(ppokit::runtime::TeamUpdateStatistics const& a, ppokit::runtime::TeamUpdateStatistics const& b)
{
    return SameBits(a.policyLoss, b.policyLoss) && SameBits(a.valueLoss, b.valueLoss)
        && SameBits(a.entropy, b.entropy) && SameBits(a.approximateKl, b.approximateKl)
        && SameBits(a.clipFraction, b.clipFraction) && SameBits(a.explainedVariance, b.explainedVariance)
        && SameBits(a.advantageAbsMean, b.advantageAbsMean) && SameBits(a.actionStdMean, b.actionStdMean)
        && SameBits(a.actorDerivativeL1, b.actorDerivativeL1) && SameBits(a.criticDerivativeL1, b.criticDerivativeL1)
        && SameBits(a.logStdGradientL1, b.logStdGradientL1) && a.learnableRows == b.learnableRows
        && a.criticRows == b.criticRows;
}

void SharedBackendMatchesTheBlueLearner(bool paddedCritic)
{
    std::filesystem::path const fixture{BLUE_SHARED_BACKEND_FIXTURE};
    std::ifstream taskFile(fixture / "task.json");
    std::ifstream networkFile(fixture / "network.json");
    auto taskJson = nlohmann::ordered_json::parse(taskFile);
    auto networkJson = nlohmann::ordered_json::parse(networkFile);
    if (paddedCritic)
    {
        for (char const* group : {"own", "team_mean", "team_max"})
            taskJson["critic"][group]["size"] = blue::GeneralizedCriticLayout::Compact;
        networkJson["team_critic"]["pad_features"] = true;
    }
    auto const task = ppokit::schema::ParseTaskSchema(taskJson.dump(), "task.json");
    auto const graphs = ppokit::schema::ParseBlockGraph(networkJson.dump(), "network.json", task);
    ppokit::core::PpoSettings const settings = ppokit::schema::LoadTrainingPlan(fixture / "training.json", task)
                                                   .ppo.Settings();
    ppokit::runtime::TeamPpoSettings const team = ppokit::graph::TeamSettingsOf(settings);
    ppokit::runtime::TeamPpoSettings const own = blue::GeneralizedPatrolLearner::PpoSettings();
    Check(SameBits(team.gamma, own.gamma) && SameBits(team.lambda, own.lambda) && SameBits(team.clip, own.clip)
            && SameBits(team.entropyCoefficient, own.entropyCoefficient) && team.epochs == own.epochs
            && team.minibatch == own.minibatch,
        "parity: training.json holds the Blue learner's PPO settings");

    blue::GeneralizedPatrolLearner original{Counts, 731U, RolloutTeamSteps};
    Scenario const scenario;
    blue::GeneralizedPatrolLearner::Observations observations{};
    blue::GeneralizedPatrolLearner::AvailabilityMask available{};
    blue::GeneralizedPatrolLearner::Rewards rewards{};
    int clock = 0;
    // One update of the Blue learner alone, so that both Adams and the generator have moved.
    while (original.StepCount() < 1U)
    {
        scenario.Fill(clock, observations, available);
        auto const actions = original.ChooseActions(observations, available);
        for (std::size_t agent = 0; agent < Counts.blueMobiles; ++agent)
        {
            rewards[agent] = std::cos(0.05F * static_cast<float>(clock)) + 0.1F * actions[agent][0];
        }
        scenario.Fill(clock + 1, observations, available);
        original.Observe(observations, rewards, available, Scenario::Terminated(clock),
            Scenario::EndsEpisode(clock) && !Scenario::Terminated(clock));
        ++clock;
    }

    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "blue_shared_backend_parity";
    std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);
    blue::WriteSharedBackendTraining(original, task, graphs, folder / "converted.ppokit");
    ppokit::graph::TeamGraphLearner shared(task, graphs, settings, RolloutTeamSteps);
    shared.LoadTraining(folder / "converted.ppokit", ppokit::schema::ArtifactIdentityOf(task, graphs));
    Check(shared.Updates() == 1U, "parity: the converted state keeps the update count");

    bool sameActions = true;
    bool sameUpdates = true;
    std::uint64_t updates = 0;
    while (updates < 2U)
    {
        scenario.Fill(clock, observations, available);
        SharedRows const now = RowsOf(observations, available);
        auto const actions = original.ChooseActions(observations, available);
        std::array<float, Capacity * ActionSize> sharedActions{};
        shared.Act(now.observations.data(), now.available.data(), sharedActions.data(), false);
        std::array<double, Capacity> sharedRewards{};
        for (std::size_t agent = 0; agent < Counts.blueMobiles; ++agent)
        {
            for (std::size_t action = 0; action < ActionSize; ++action)
            {
                sameActions =
                    sameActions && SameBits(actions[agent][action], sharedActions[agent * ActionSize + action]);
            }
            rewards[agent] = std::cos(0.05F * static_cast<float>(clock)) + 0.1F * actions[agent][0];
            sharedRewards[agent] = rewards[agent];
        }
        scenario.Fill(clock + 1, observations, available);
        SharedRows const next = RowsOf(observations, available);
        bool const terminated = Scenario::Terminated(clock);
        bool const truncated = Scenario::EndsEpisode(clock) && !terminated;
        bool const updated = original.Observe(observations, rewards, available, terminated, truncated);
        bool const sharedUpdated = shared.Observe(next.observations.data(), next.available.data(),
            sharedRewards.data(), EndOf(terminated, truncated));
        sameUpdates = sameUpdates && updated == sharedUpdated;
        if (updated)
        {
            ++updates;
            sameUpdates = sameUpdates && SameUpdate(original.LastUpdate(), shared.LastUpdate());
        }
        ++clock;
    }
    Check(sameActions, "parity: every sampled action is the same float");
    Check(sameUpdates, "parity: every update statistic is the same float");

    blue::WriteSharedBackendTraining(original, task, graphs, folder / "original.ppokit");
    shared.SaveTraining(folder / "shared.ppokit", ppokit::schema::ArtifactIdentityOf(task, graphs));
    Check(Body(folder / "original.ppokit") == Body(folder / "shared.ppokit"),
        "parity: weights, Adam moments and steps, and generator are the same after two updates");
    std::filesystem::remove_all(folder);
}

} // namespace

int main()
{
    try
    {
        SharedBackendMatchesTheBlueLearner(false);
        SharedBackendMatchesTheBlueLearner(true);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " shared backend parity check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Blue shared backend parity tests passed\n";
    return EXIT_SUCCESS;
}
