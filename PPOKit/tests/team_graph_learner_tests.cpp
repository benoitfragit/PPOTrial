/**
 * @file PPOKit/tests/team_graph_learner_tests.cpp
 * @brief The team learner of the shared graph backend: its critic input, its schema section,
 *        and the team ABI of ppokit_graph_network on a declarative project (tests/fixtures/team_graph).
 *
 * Episodes of the fixture have three rows: agents 0 and 1 are present, agent 1 is destroyed
 * half-way, row 2 never holds an agent. The checks: the critic input of each member
 * (TeamCriticBuilder), membership and absent rows, updates, resuming bit for bit from a
 * saved training state, the policy file, and errors in words.
 */
#include "ppokit/graph/team_graph_learner.hpp"
#include "ppokit/network/network_library.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ppokit;

constexpr std::size_t Capacity = 3;
constexpr std::size_t ObservationSize = 2;
constexpr int EpisodeSteps = 6;
constexpr int DestroyedAt = 3;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool Contains(std::string const& text, std::string const& part)
{
    return text.find(part) != std::string::npos;
}

std::string TextOf(std::filesystem::path const& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::filesystem::path const Fixture{PPOKIT_TEAM_GRAPH_PROJECT};

/** @brief Own row, mean and max over the living, fractions over the capacity; absent rows zero. */
void CriticInputOfEachMember()
{
    auto const task = schema::LoadTaskSchema(Fixture / "task.json");
    auto const graphs = schema::LoadBlockGraph(Fixture / "network.json", task);
    Check(graphs.teamCritic.has_value() && graphs.teamCritic->features == "embed" && graphs.teamCritic->width == 4,
        "team_critic: the section is read with the width of its actor block");
    graph::TeamCriticBuilder builder(*graphs.teamCritic, *task.critic, Capacity);
    std::array<float, Capacity * 4> const features{1.F, -2.F, 3.F, 0.5F, 3.F, 4.F, -1.F, 0.5F, 9.F, 9.F, 9.F, 9.F};
    std::vector<float> critics(Capacity * task.critic->size, 7.F);
    builder.Build(features.data(), {true, true, false}, {true, true, false}, critics.data());
    std::vector<float> const expectedFirst{1.F, -2.F, 3.F, 0.5F, 2.F, 1.F, 1.F, 0.5F, 3.F, 4.F, 3.F, 0.5F,
        2.F / 3.F, 2.F / 3.F};
    std::vector<float> const first(critics.begin(), critics.begin() + 14);
    Check(first == expectedFirst, "team critic: own row, team mean and max, living and team fractions");
    Check(std::all_of(critics.begin() + 28, critics.end(), [](float value) { return value == 0.F; }),
        "team critic: the row of an absent agent is zero");

    builder.Build(features.data(), {true, true, false}, {true, false, false}, critics.data());
    std::vector<float> const second(critics.begin() + 14, critics.begin() + 28);
    std::vector<float> const expectedSecond{0.F, 0.F, 0.F, 0.F, 1.F, -2.F, 3.F, 0.5F, 1.F, -2.F, 3.F, 0.5F,
        1.F / 3.F, 2.F / 3.F};
    Check(second == expectedSecond, "team critic: a destroyed member keeps the team's context, not its own row");

    builder.Build(features.data(), {true, true, false}, {false, false, false}, critics.data());
    std::vector<float> const none(critics.begin(), critics.begin() + 14);
    std::vector<float> expectedNone(14, 0.F);
    expectedNone[13] = 2.F / 3.F;
    Check(none == expectedNone, "team critic: with nobody alive, mean and max are 0, not -infinity");
}

/** @brief Parses the fixture's network.json after text replacements, returning the error or "". */
std::string SchemaError(std::vector<std::pair<std::string, std::string>> const& replacements)
{
    std::string text = TextOf(Fixture / "network.json");
    for (auto const& [from, to] : replacements)
    {
        std::size_t const at = text.find(from);
        if (at == std::string::npos)
        {
            return "fixture text not found: " + from;
        }
        text.replace(at, from.size(), to);
    }
    try
    {
        auto const task = schema::LoadTaskSchema(Fixture / "task.json");
        (void)schema::ParseBlockGraph(text, "network.json", task);
    }
    catch (schema::SchemaError const& error)
    {
        return error.what();
    }
    return "";
}

void TeamCriticSchemaErrors()
{
    core::PpoSettings ppo;
    ppo.targetKl = 0.0125;
    Check(graph::TeamSettingsOf(ppo).targetKl == static_cast<float>(ppo.targetKl),
        "team learner: target KL from training settings reaches the hosted trainer");
    Check(Contains(SchemaError({{R"("features": "embed")", R"("features": "trunk")"}}), "vector of 8 values"),
        "team_critic: a features block of another width than the groups is refused");
    Check(Contains(SchemaError({{R"("features": "embed")", R"("features": "policy")"}}), "head"),
        "team_critic: a head cannot describe an agent");
    Check(Contains(SchemaError({{R"("max": "critic.team_max")", R"("max": "critic.team_mean")"}}), "already filled"),
        "team_critic: a group filled twice is refused");
    Check(Contains(SchemaError({{R"("own": "critic.own")", R"("own": "observation.line")"}}), "critic.<group>"),
        "team_critic: parts go to critic groups only");
    // A block read by the team critic alone is not an unused block.
    std::string const spare = SchemaError({{R"("embed": {)", R"("spare": { "type": "dense", "width": 4 }, "embed": {)"},
        {R"(["line", "embed"],)", R"(["line", "embed"], ["line", "spare"],)"},
        {R"("features": "embed")", R"("features": "spare")"}});
    Check(spare.empty(), "team_critic: its features block counts as read: " + spare);
    std::string const unread = SchemaError({{R"("embed": {)", R"("spare": { "type": "dense", "width": 4 }, "embed": {)"},
        {R"(["line", "embed"],)", R"(["line", "embed"], ["line", "spare"],)"}});
    Check(Contains(unread, "not used by any node"), "team_critic: another unread block is still refused");
}

/** @brief Larger critic groups are zero-padded only when the graph explicitly requests it. */
void PaddedTeamCritic()
{
    auto taskJson = nlohmann::ordered_json::parse(TextOf(Fixture / "task.json"));
    for (char const* group : {"own", "team_mean", "team_max"})
        taskJson["critic"][group]["size"] = 6;
    auto const task = schema::ParseTaskSchema(taskJson.dump(), "task.json");
    auto network = nlohmann::ordered_json::parse(TextOf(Fixture / "network.json"));
    bool refusedWithoutPadding = false;
    try { (void)schema::ParseBlockGraph(network.dump(), "network.json", task); }
    catch (schema::SchemaError const&) { refusedWithoutPadding = true; }
    Check(refusedWithoutPadding, "team critic: a wider group requires explicit padding");

    network["team_critic"]["pad_features"] = true;
    auto const graphs = schema::ParseBlockGraph(network.dump(), "network.json", task);
    Check(graphs.teamCritic && graphs.teamCritic->padFeatures, "team critic: explicit padding is parsed");
    graph::TeamCriticBuilder builder(*graphs.teamCritic, *task.critic, Capacity);
    std::array<float, Capacity * 4> const features{1.F, -2.F, 3.F, 0.5F, 3.F, 4.F, -1.F, 0.5F, 9.F, 9.F, 9.F, 9.F};
    std::vector<float> rows(Capacity * task.critic->size, 7.F);
    builder.Build(features.data(), {true, true, false}, {true, false, false}, rows.data());
    std::vector<float> const expected{0.F, 0.F, 0.F, 0.F, 0.F, 0.F,
        1.F, -2.F, 3.F, 0.5F, 0.F, 0.F, 1.F, -2.F, 3.F, 0.5F, 0.F, 0.F, 1.F / 3.F, 2.F / 3.F};
    Check(std::equal(expected.begin(), expected.end(), rows.begin() + task.critic->size),
        "team critic: destroyed member and padded pooled features retain the correct positions");
    Check(std::all_of(rows.begin() + 2 * task.critic->size, rows.end(), [](float value) { return value == 0.F; }),
        "team critic: absent rows remain zero with padding");

    taskJson["critic"]["own"]["size"] = 3;
    auto const narrow = schema::ParseTaskSchema(taskJson.dump(), "task.json");
    bool refusedNarrow = false;
    try { (void)schema::ParseBlockGraph(network.dump(), "network.json", narrow); }
    catch (schema::SchemaError const&) { refusedNarrow = true; }
    Check(refusedNarrow, "team critic: padding never truncates features");
}

/** @brief One synthetic team episode step: the present agents and their observations. */
struct TeamStep final
{
    std::array<float, Capacity * ObservationSize> observations{};
    std::array<std::uint8_t, Capacity> available{};
};

TeamStep StepOf(int episodeStep, int episode)
{
    TeamStep step;
    for (std::size_t agent = 0; agent < 2; ++agent)
    {
        step.observations[agent * ObservationSize] =
            std::sin(0.3F * static_cast<float>(episodeStep + 7 * episode) + static_cast<float>(agent));
        step.observations[agent * ObservationSize + 1] = 0.1F * static_cast<float>(episodeStep);
    }
    // Row 2 is never present: noise that must not matter.
    step.observations[2 * ObservationSize] = 1e6F;
    step.available = {1, static_cast<std::uint8_t>(episodeStep < DestroyedAt ? 1 : 0), 0};
    return step;
}

/** @brief Drives `learner` through team steps until `updates` more updates; returns every action. */
std::vector<float> Drive(network::TeamLearner& learner, int& clock, std::uint64_t updates,
    std::vector<PpokitUpdateStatistics>* statistics = nullptr)
{
    std::vector<float> actions;
    std::uint64_t const target = learner.Updates() + updates;
    while (learner.Updates() < target)
    {
        int const episode = clock / EpisodeSteps;
        int const episodeStep = clock % EpisodeSteps;
        TeamStep const now = StepOf(episodeStep, episode);
        std::array<float, Capacity> chosen{};
        chosen.fill(5.F);
        learner.Act(now.observations.data(), now.available.data(), chosen.data(), false);
        actions.insert(actions.end(), chosen.begin(), chosen.end());
        bool const last = episodeStep + 1 == EpisodeSteps;
        TeamStep const next = StepOf(episodeStep + 1, episode);
        std::array<double, Capacity> const rewards{0.5 + chosen[0], 0.5 - chosen[1], 1e9};
        bool const updated = learner.Observe(next.observations.data(), next.available.data(), rewards.data(),
            last ? PPOKIT_STEP_TRUNCATED : PPOKIT_STEP_RUNNING);
        if (updated && statistics != nullptr)
        {
            statistics->push_back(learner.Statistics());
        }
        ++clock;
    }
    return actions;
}

bool SameStatistics(PpokitUpdateStatistics const& a, PpokitUpdateStatistics const& b)
{
    return a.updates == b.updates && a.policyLoss == b.policyLoss && a.valueLoss == b.valueLoss
        && a.approximateKl == b.approximateKl && a.explainedVariance == b.explainedVariance
        && a.criticRows == b.criticRows && a.learnableRows == b.learnableRows;
}

/** @brief Scheduled probes work through the shared ABI and preserve training and resume. */
void ScheduledTeamProbes(std::filesystem::path const& folder,
    std::shared_ptr<network::NetworkLibrary const> const& library)
{
    std::filesystem::create_directories(folder);
    for (char const* name : {"task.json", "network.json"})
        std::filesystem::copy_file(Fixture / name, folder / name);
    auto training = nlohmann::ordered_json::parse(TextOf(Fixture / "training.json"));
    training["probes"] = {{"every", 4}};
    std::ofstream(folder / "training.json") << training.dump(2);
    network::TeamLearner probed(library, folder, 0);
    network::TeamLearner plain(library, Fixture, 0);
    int clock = 0;
    int plainClock = 0;
    // Three 16-step rollouts end on a six-step episode boundary before saving.
    Check(Drive(probed, clock, 3) == Drive(plain, plainClock, 3) && probed.Probes().empty(),
        "team probes: unprobed updates match before the episode boundary");
    probed.Save(folder / "resume.ppokit", {});
    network::TeamLearner resumed(library, folder, 99);
    resumed.Load(folder / "resume.ppokit");
    int resumedClock = clock;
    auto const actions = Drive(probed, clock, 1);
    auto const plainActions = Drive(plain, plainClock, 1);
    auto const resumedActions = Drive(resumed, resumedClock, 1);
    Check(actions == plainActions && actions == resumedActions,
        "team probes: sampled actions match with probes and after resume");
    bool actor = false;
    bool critic = false;
    for (auto const& probe : probed.Probes())
    {
        actor = actor || (std::string(probe.network) == "actor" && probe.samples > 0);
        critic = critic || (std::string(probe.network) == "critic" && probe.samples > 0);
    }
    Check(actor && critic && !resumed.Probes().empty(),
        "team probes: actor and critic measurements appear on update four, including after resume");
    Drive(probed, clock, 1);
    Check(probed.Probes().empty(), "team probes: results are cleared on an unprobed update");
}

void SharedBackendTrainsATeam()
{
    auto const task = schema::LoadTaskSchema(Fixture / "task.json");
    auto const library = network::NetworkLibrary::Open(PPOKIT_SHARED_GRAPH_NETWORK, {}, &task);
    Check(library->Info().agentCapacity == Capacity, "shared backend: a team task gives its agent capacity");

    network::TeamLearner learner(library, Fixture, 0);
    Check(learner.Settings().epochs == 2 && learner.Settings().minibatch == 8, "team learner: training.json settings");
    int clock = 0;
    std::vector<PpokitUpdateStatistics> statistics;
    std::vector<float> const actions = Drive(learner, clock, 2, &statistics);
    bool absentZero = true;
    for (std::size_t index = 2; index < actions.size(); index += Capacity)
    {
        absentZero = absentZero && actions[index] == 0.F;
    }
    Check(absentZero, "team learner: an absent row draws no action");
    // 16 team steps x 2 members; agent 1 lives the first 3 steps of each episode (9 of the 16).
    Check(statistics.size() == 2 && statistics[0].criticRows == 32 && statistics[0].learnableRows == 16 + 9
            && std::isfinite(statistics[0].policyLoss) && statistics[1].updates == 2,
        "team learner: members train the critic, living members the actor");

    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_team_graph_learner_tests";
    std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);
    learner.Save(folder / "training.ppokit", folder / "policy.ppokit");

    // The policy file acts as the learner's means.
    network::TeamPolicy policy(library, Fixture, folder / "policy.ppokit");
    TeamStep const probe = StepOf(1, 0);
    std::array<float, Capacity> fromLearner{};
    std::array<float, Capacity> fromPolicy{};
    learner.Act(probe.observations.data(), probe.available.data(), fromLearner.data(), true);
    policy.Act(probe.observations.data(), probe.available.data(), fromPolicy.data());
    Check(fromLearner == fromPolicy && fromPolicy[2] == 0.F, "team policy: the saved actor acts as the learner");

    // A resumed learner continues exactly as the one that saved.
    network::TeamLearner resumed(library, Fixture, 99);
    resumed.Load(folder / "training.ppokit");
    Check(resumed.Updates() == 2, "team learner: the update count is restored");
    int resumedClock = clock;
    std::vector<PpokitUpdateStatistics> continued;
    std::vector<PpokitUpdateStatistics> resumedStatistics;
    std::vector<float> const after = Drive(learner, clock, 1, &continued);
    std::vector<float> const resumedAfter = Drive(resumed, resumedClock, 1, &resumedStatistics);
    Check(after == resumedAfter && continued.size() == 1 && resumedStatistics.size() == 1
            && SameStatistics(continued[0], resumedStatistics[0]),
        "team learner: resuming a training state continues bit for bit (weights, Adam, generator)");

    bool singleRefused = false;
    try
    {
        network::Learner single(library, Fixture);
    }
    catch (std::exception const& error)
    {
        singleRefused = Contains(error.what(), "single-agent");
    }
    Check(singleRefused, "shared backend: a team task has no single-agent learner");

    network::TeamLearner joining(library, Fixture, 0);
    TeamStep start = StepOf(0, 0);
    std::array<float, Capacity> chosen{};
    joining.Act(start.observations.data(), start.available.data(), chosen.data(), false);
    TeamStep next = StepOf(1, 0);
    next.available[2] = 1;
    std::array<double, Capacity> const rewards{0.0, 0.0, 0.0};
    bool joinRefused = false;
    try
    {
        joining.Observe(next.observations.data(), next.available.data(), rewards.data(), PPOKIT_STEP_RUNNING);
    }
    catch (std::exception const& error)
    {
        joinRefused = Contains(error.what(), "appeared during an episode");
    }
    Check(joinRefused, "team learner: an agent cannot join a running episode");

    // A team project whose network.json has no team_critic says what is missing.
    std::filesystem::path const bare = folder / "bare";
    std::filesystem::create_directories(bare);
    std::filesystem::copy_file(Fixture / "task.json", bare / "task.json");
    nlohmann::json network = nlohmann::json::parse(TextOf(Fixture / "network.json"));
    network.erase("team_critic");
    std::ofstream(bare / "network.json") << network.dump(2);
    std::string message;
    try
    {
        network::TeamLearner refused(library, bare, 0);
    }
    catch (std::exception const& error)
    {
        message = error.what();
    }
    Check(Contains(message, "team_critic"), "team learner: without team_critic, the error names it: " + message);
    ScheduledTeamProbes(folder / "probed", library);
    std::filesystem::remove_all(folder);
}

} // namespace

int main()
{
    try
    {
        CriticInputOfEachMember();
        TeamCriticSchemaErrors();
        PaddedTeamCritic();
        SharedBackendTrainsATeam();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " team graph learner check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit team graph learner tests passed\n";
    return EXIT_SUCCESS;
}
