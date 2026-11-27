/**
 * @file PPOKit/tests/ppokit_run_tests.cpp
 * @brief ppokit-run on a project made of the reach environment DLL and the reach network
 *        DLL: it trains, resumes, plays, and calls the observers of network.json with the
 *        world state after every step.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <future>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/ipc/control.hpp"

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

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

Json ReadJson(fs::path const& path)
{
    std::ifstream stream(path);
    return Json::parse(stream);
}

void WriteJson(fs::path const& path, Json const& json)
{
    std::ofstream(path) << json.dump(2) << '\n';
}

/** @brief Read every byte after an artifact's JSON header, including its original line endings. */
std::string CheckpointBody(fs::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    std::string header;
    if (!std::getline(input, header)) throw std::runtime_error("missing checkpoint header: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/** @brief Output and exit code of a command. */
struct Outcome final
{
    int code = -1;
    std::string output;
};

Outcome Run(std::string const& arguments)
{
    std::string const command = "\"" + std::string(PPOKIT_RUN) + "\" " + arguments + " 2>&1";
#if defined(_WIN32)
    FILE* const pipe = _popen(("\"" + command + "\"").c_str(), "r");
#else
    FILE* const pipe = popen(command.c_str(), "r");
#endif
    Outcome outcome;
    if (pipe == nullptr)
    {
        return outcome;
    }
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        outcome.output += buffer;
    }
#if defined(_WIN32)
    outcome.code = _pclose(pipe);
#else
    int const status = pclose(pipe);
    outcome.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    return outcome;
}

void CliNumbersAreStrict()
{
    for (std::string const value : {"0.1junk", "nan", "inf", " 0.1"})
    {
        Outcome const outcome = Run("play --project missing --step-seconds \"" + value + "\"");
        Check(outcome.code == 2 && Contains(outcome.output, "--step-seconds requires one finite number"),
            "malformed --step-seconds is a usage error with complete consumption: " + value + " / " + outcome.output);
    }
}

/** @brief Pump the supervisor with a deadline while the real runner is executing in another process. */
template<class Done>
bool PollUntil(ppokit::ipc::Supervisor& supervisor, Done done, std::chrono::seconds timeout = std::chrono::seconds(6))
{
    auto const end = std::chrono::steady_clock::now() + timeout;
    do
    {
        supervisor.Tick();
        if (done()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < end);
    return false;
}

/**
 * @brief The reach project for ppokit-run: task.json with a world (agent and target
 *        positions), network.json with a "bounded" observer on the world, and project.json
 *        naming the two libraries.
 */
fs::path MakeProject(fs::path const& folder, double observerLimit)
{
    fs::path const reach(PPOKIT_REACH_DIR);
    fs::create_directories(folder);
    Json task = ReadJson(reach / "task.json");
    task["world"] = Json{{"agent", {{"fields", {"agent_x", "agent_y"}}}}, {"target", {{"fields", {"target_x", "target_y"}}}}};
    task["metrics"] = Json{{"reached", {{"kind", "flag"}}}, {"distance", {{"kind", "number"}}}};
    WriteJson(folder / "task.json", task);
    fs::copy_file(reach / "training.json", folder / "training.json", fs::copy_options::overwrite_existing);
    Json network = ReadJson(reach / "network.json");
    network["components"] = Json::array({PPOKIT_TEST_COMPONENTS});
    network["observers"] = Json{{"nodes",
                                    {{"target", {{"type", "input"}, {"from", "world.target"}}},
                                        {"watch",
                                            {{"type", "component:bounded@1"},
                                                {"config", {{"limit", observerLimit}}}}}}},
        {"edges", Json::array({Json::array({"target", "watch"})})}};
    WriteJson(folder / "network.json", network);
    Json project;
    project["schema"] = "ppokit.project/1";
    project["name"] = "reach_run";
    project["libraries"] = {{"environment", PPOKIT_REACH_ENVIRONMENT}, {"network", PPOKIT_REACH_NETWORK}};
    project["program"] = {{"executable", "ppokit-run"}, {"train", {"train", "--project", "{project}"}},
        {"evaluate", {"eval", "--project", "{project}"}}, {"inference", {"play", "--project", "{project}"}}};
    WriteJson(folder / "project.json", project);
    return folder;
}

/**
 * @brief An evaluation suite of the project, run during training every `every` updates.
 * @param goal Condition on reached.rate for the training to be finished; empty for none.
 */
void WriteSuite(fs::path const& project, std::string const& name, std::string const& gate, std::size_t every,
    std::string const& goal = {})
{
    Json suite;
    suite["schema"] = "ppokit.eval/1";
    suite["suite"] = name;
    suite["seeds"] = {{"first", 101}, {"count", 3}};
    suite["episodes"] = 2;
    suite["gates"] = {{"reached.rate", gate}};
    suite["during_training"] = {{"every_updates", every}};
    if (!goal.empty())
    {
        suite["goal"] = {{"reached.rate", goal}};
    }
    suite["best"] = Json::array({{{"metric", "reached.rate"}, {"prefer", "higher"}},
        {{"metric", "distance.mean"}, {"prefer", "lower"}}});
    fs::create_directories(project / "eval");
    WriteJson(project / "eval" / (name + ".json"), suite);
}

/** @brief Equivalent decision observations must produce the same complete PPO update across rollout boundaries. */
void DecisionBootstrap(fs::path const& root)
{
    auto const immediate = MakeProject(root / "immediate-clock", 100.0);
    auto const deferred = MakeProject(root / "deferred-clock", 100.0);
    if (!std::ofstream(deferred / "defer-clock.fixture").put('1'))
        throw std::runtime_error("cannot enable the deferred clock fixture");
    for (auto const& project : {immediate, deferred})
    {
        auto const run = Run("train --fresh --updates 2 --project \"" + project.string() + "\"");
        Check(run.code == 0, "decision-clock training completed: " + run.output);
        if (run.code != 0) return;
    }
    auto const file = fs::path("checkpoints") / "latest.ppokit";
    auto expected = ppokit::artifacts::ReadArtifactHeader(immediate / file);
    auto const actual = ppokit::artifacts::ReadArtifactHeader(deferred / file);
    expected.created = actual.created; // Independent runs may publish in different UTC seconds.
    Check(expected.updates == 2 && ppokit::artifacts::HeaderLine(expected) == ppokit::artifacts::HeaderLine(actual)
            && CheckpointBody(immediate / file) == CheckpointBody(deferred / file),
        "PPO bootstraps on the same decision clock whether Step or BeginStep supplies it");
}

/** @brief A curriculum that the environment owns ends the training: campaigns of that update, save, stop. */
void CurriculumEndsTraining(fs::path const& root)
{
    fs::path const project = MakeProject(root / "curriculum", 100.0);
    WriteSuite(project, "campaign", ">= 0", 3);
    std::ofstream(project / "progress.fixture") << "on\n";
    std::ofstream(project / "finish-at-3.fixture") << "on\n";
    Outcome const finished = Run("train --updates 10 --fresh --project \"" + project.string() + "\"");
    Check(finished.code == 0 && Contains(finished.output, "curriculum ended the training at update 3")
            && Contains(finished.output, "evaluation 'campaign' at update 3") && !Contains(finished.output, "update 4/")
            && ppokit::artifacts::ReadArtifactHeader(project / "checkpoints" / "latest.ppokit").updates == 3,
        "a curriculum that ends runs the update's campaigns, saves and stops: " + finished.output);
}

/** @brief A curriculum completed by an evaluation stops at that update and publishes its state. */
void CampaignEndsTraining(fs::path const& root)
{
    auto const project = MakeProject(root / "campaign-curriculum", 100.0);
    WriteSuite(project, "first", ">= 0", 2);
    WriteSuite(project, "second", ">= 0", 2);
    std::ofstream(project / "progress.fixture") << "on\n";
    std::ofstream(project / "finish-on-campaign.fixture") << "on\n";
    auto const finished = Run("train --updates 10 --fresh --project \"" + project.string() + "\"");
    Check(finished.code == 0 && Contains(finished.output, "curriculum ended the training at update 2")
            && Contains(finished.output, "evaluation 'first' at update 2")
            && Contains(finished.output, "evaluation 'second' at update 2")
            && !Contains(finished.output, "update 3/"),
        "a campaign ends training without another rollout, even if a later campaign returns false: " + finished.output);
    auto const checkpoints = project / "checkpoints";
    auto const generation = ReadJson(checkpoints / "generation.json");
    auto const state = generation.at("state").get<std::string>();
    Check(generation.at("update") == 2 && Contains(state, ":campaign-finished")
            && ReadJson(checkpoints / "environment_state/2.json").at("state") == state,
        "the completed campaign's curriculum state is published at the same update");
    Check(ppokit::artifacts::ReadArtifactHeader(checkpoints / "latest.ppokit").updates == 2
            && ppokit::artifacts::Sha256OfFile(checkpoints / "latest.ppokit") == generation.at("training_sha256")
            && ppokit::artifacts::Sha256OfFile(checkpoints / "policy.ppokit") == generation.at("policy_sha256"),
        "the published generation retains both complete artifacts at update 2");
    auto const first = ReadJson(checkpoints / "best/first.json");
    auto const second = ReadJson(checkpoints / "best/second.json");
    Check(first.at("sha256") == generation.at("policy_sha256")
            && second.at("sha256") == generation.at("policy_sha256")
            && first.at("environment_state") == second.at("environment_state")
            && first.at("environment_state") != generation.at("state"),
        "both BESTs retain the original evaluation context and the identical actor bytes");
}

/** @brief Campaigns during training, the BEST, --until-passed, eval of the suites and play --best. */
void Evaluations(fs::path const& root)
{
    fs::path const project = MakeProject(root / "evaluated", 100.0);
    std::string const at = " --project \"" + project.string() + "\"";
    WriteSuite(project, "campaign", ">= 0", 2);
    std::ofstream(project / "progress.fixture") << "on\n";

    Outcome const trained = Run("train --updates 4 --fresh" + at);
    Check(trained.code == 0 && Contains(trained.output, "evaluation 'campaign' at update 2: 6 episodes, passed")
            && Contains(trained.output, "evaluation 'campaign' at update 4"),
        "campaigns run every 2 updates and after the last one: " + trained.output);
    ppokit::runtime::RunLog const log = ppokit::runtime::LoadRunLog(project / "checkpoints" / "metrics.jsonl");
    Check(log.evaluations.size() == 2 && log.evaluations[0].update == 2 && log.evaluations[0].promoted
            && log.evaluations[0].episodes == 6,
        "the journal records the campaigns, the first one becoming the BEST");
    bool aggregates = false;
    for (auto const& [name, value] : log.evaluations[0].metrics)
    {
        aggregates = aggregates || (name == "distance.mean" && value >= 0.0);
    }
    Check(aggregates, "the campaign reports the aggregates of the episode metrics");
    // The training environment hears of each update, then of each campaign's result.
    std::ifstream progressFile(project / "progress.log");
    std::vector<std::string> progress;
    for (std::string line; std::getline(progressFile, line);)
    {
        progress.push_back(line);
    }
    bool const progressOrder = progress.size() == 6 && progress[0] == "update 1" && progress[1] == "update 2"
        && Contains(progress[2], "update 2 campaign {") && Contains(progress[2], "\"suite\":\"campaign\"")
        && Contains(progress[2], "\"passed\":true") && Contains(progress[2], "\"distance.mean\":")
        && progress[3] == "update 3" && progress[4] == "update 4" && Contains(progress[5], "update 4 campaign {");
    Check(progressOrder, "the environment hears of every update and campaign, in order");
    fs::remove(project / "progress.fixture");
    Json const best = ReadJson(project / "checkpoints" / "best" / "campaign.json");
    Check(best.at("report").at("policy_sha256") == best.at("sha256"),
        "the training campaign reports the digest of the actor promoted to BEST");
    Check(best.at("suite") == "campaign" && fs::exists(project / "checkpoints" / "best" / best.at("policy").get<std::string>()),
        "the BEST manifest names a policy kept beside it");

    Outcome const goal = Run("train --updates 6 --until-passed campaign" + at);
    Check(goal.code == 0 && Contains(goal.output, "Suite 'campaign' passed at update 6"),
        "--until-passed ends the training at the first passing campaign: " + goal.output);

    WriteSuite(project, "unreachable", ">= 0", 1, ">= 2");
    Outcome const unreached = Run("train --updates 1 --until-passed unreachable" + at);
    Check(unreached.code == 3 && Contains(unreached.output, "goal not reached"),
        "a campaign that passes its gates but not its goal does not end the training: " + unreached.output);
    fs::remove(project / "eval" / "unreachable.json");

    Outcome const budget = Run("train --updates 100000000 --max-seconds 1 --evaluate-every 1000000" + at);
    Check(budget.code == 0 && Contains(budget.output, "Time budget of 1 s spent")
            && Contains(budget.output, "evaluation 'campaign'"),
        "--max-seconds ends the training like a last update, campaigns included: "
            + budget.output.substr(budget.output.size() > 400 ? budget.output.size() - 400 : 0));

    WriteSuite(project, "impossible", ">= 2", 1);
    Outcome const never = Run("train --updates 1 --until-passed impossible" + at);
    Check(never.code == 3 && Contains(never.output, "did not pass"), "a goal never reached exits with 3: " + never.output);

    Outcome const evaluated = Run("eval --suite campaign" + at);
    Check(evaluated.code == 0 && fs::exists(project / "checkpoints" / "evaluation_campaign.json"),
        "eval runs a suite and writes its report: " + evaluated.output);
    Outcome const failed = Run("eval --suite impossible" + at);
    Check(failed.code == 3 && Contains(failed.output, "not passed"), "eval exits with 3 when a gate fails");
    Outcome const unknown = Run("eval --suite nothing" + at);
    Check(unknown.code == 2 && Contains(unknown.output, "no evaluation suite 'nothing'"), "an unknown suite is a usage error");

    Outcome const best_played = Run("play --episodes 1 --best --suite campaign" + at);
    Check(best_played.code == 0 && Contains(best_played.output, "BEST of 'campaign'"),
        "play --best plays the BEST policy: " + best_played.output);

    // Selecting an actor must not silently expand the declared evaluation campaign.
    auto manifest = ReadJson(project / "project.json");
    manifest["program"]["evaluate"].push_back("--suite");
    manifest["program"]["evaluate"].push_back("campaign");
    manifest["program"]["inference"].push_back("--suite");
    manifest["program"]["inference"].push_back("campaign");
    WriteJson(project / "project.json", manifest);
    auto const unwantedReport = project / "checkpoints" / "evaluation_impossible.json";
    for (std::string const arguments : {"eval", "eval --model LATEST", "eval --policy checkpoints/policy.ppokit",
            "eval --best"})
    {
        fs::remove(unwantedReport);
        auto const result = Run(arguments + at);
        Check(result.code == 0 && !fs::exists(unwantedReport),
            arguments + " retains the project's evaluation suite: " + result.output);
    }
    auto const explicitSuite = Run("eval --model LATEST --suite impossible" + at);
    Check(explicitSuite.code == 3 && fs::exists(unwantedReport),
        "an explicit suite overrides the manifest independently of the policy");
    auto const declaredBest = Run("play --episodes 1 --best" + at);
    Check(declaredBest.code == 0 && Contains(declaredBest.output, "BEST of 'campaign'"),
        "explicit BEST uses the declared inference suite: " + declaredBest.output);
}

/** @brief The environment states saved with the run, by update: their texts (episodes trained). */
std::map<std::uint64_t, std::uint64_t> SavedStates(fs::path const& project)
{
    std::map<std::uint64_t, std::uint64_t> states;
    fs::path const folder = project / "checkpoints" / "environment_state";
    if (!fs::exists(folder))
    {
        return states;
    }
    for (fs::directory_entry const& entry : fs::directory_iterator(folder))
    {
        Json const state = ReadJson(entry.path());
        states[state.at("update").get<std::uint64_t>()] = std::stoull(state.at("state").get<std::string>());
    }
    return states;
}

/** @brief The archives of former runs (checkpoints/archive/run-*). */
std::vector<fs::path> Archives(fs::path const& project)
{
    std::vector<fs::path> archives;
    fs::path const folder = project / "checkpoints" / "archive";
    if (fs::exists(folder))
    {
        for (fs::directory_entry const& entry : fs::directory_iterator(folder))
        {
            archives.push_back(entry.path());
        }
    }
    return archives;
}

/**
 * @brief The environment's training state follows the run (audit P0-03, P1-02): saved with
 *        each checkpoint, restored on resume, empty for a fresh run whose predecessor is
 *        archived, never mixed with the state of another checkpoint.
 */
void ConcurrentWriters(fs::path const& root)
{
    auto const first = MakeProject(root / "writer-a", 100.0);
    auto const other = MakeProject(root / "writer-b", 100.0);
    for (auto const& project : {first, other})
    {
        auto manifest = ReadJson(project / "project.json");
        manifest["checkpoints"] = {{"training", "../shared/latest.ppokit"},
            {"training_policy", "../shared/./policy.ppokit"}, {"policy", "deployed/eval.ppokit"}};
        WriteJson(project / "project.json", manifest);
        fs::create_directories(project / "deployed");
        std::ofstream(project / "deployed/eval.ppokit") << "protected";
    }
    auto const args = " --project \"" + first.string() + "\"";
    ppokit::ipc::Supervisor supervisor;
    auto owner = std::async(std::launch::async, [&] { return Run("train --fresh --updates 100000 --max-seconds 15 --control "
        + supervisor.Endpoint() + args); });
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fs::exists(first / "checkpoints/metrics.jsonl") && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Check(Contains(Run("train --fresh --updates 1" + args).output, "writer lock"), "same project already owned");
    Check(Contains(Run("train --fresh --updates 1 --project \"" + other.string() + "\"").output, "writer lock"),
        "two project roots cannot overwrite shared model paths");
    Check(PollUntil(supervisor, [&] { return supervisor.Connected(); }), "the training handshake completes");
    Check(supervisor.SetRewards({17, {{0, -2.0}}}), "queue a reward edit");
    std::optional<ppokit::runtime::WeightAnswer> answer;
    Check(PollUntil(supervisor, [&] { answer = supervisor.TakeAnswer(); return answer.has_value(); })
        && answer->id == 17 && answer->applied, "the learner confirms the edit at an update boundary");
    supervisor.Request(ppokit::ipc::Command::Stop);
    Check(PollUntil(supervisor, [&] { return owner.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }),
        "cooperative training stop finishes within its deadline");
    Check(owner.get().code == 0, "owner stops cleanly after refused competing writers");
    auto const stoppedAt = ReadJson(first / "checkpoints/generation.json").at("update").get<std::uint64_t>();
    Check(stoppedAt > 0 && ppokit::artifacts::ReadArtifactHeader(root / "shared/latest.ppokit").updates == stoppedAt,
        "a Stop arriving during collection publishes the completed update");

    for (auto const& project : {first, other})
    {
        std::string reference;
        std::ifstream(project / "deployed/eval.ppokit") >> reference;
        Check(reference == "protected", "fresh does not archive or overwrite deployed evaluation weights");
    }
}

void RunState(fs::path const& root)
{
    fs::path const project = MakeProject(root / "run_state", 100.0);
    std::string const at = " --project \"" + project.string() + "\"";
    WriteSuite(project, "campaign", ">= 0", 2);

    Check(Run("train --updates 3 --fresh" + at).code == 0, "a first run trains");
    auto const manifestPath = project / "checkpoints/best/campaign.json";
    auto const promotedBest = ReadJson(manifestPath);
    auto const first = SavedStates(project);
    Check(first.size() == 1 && first.count(3) == 1 && first.at(3) > 0,
        "the environment state of the checkpoint's update is saved beside it, and only it");

    // An interruption between the state of a later save and its checkpoint leaves a state ahead.
    fs::copy_file(project / "checkpoints" / "environment_state" / "3.json",
        project / "checkpoints" / "environment_state" / "4.json");
    Json ahead = ReadJson(project / "checkpoints" / "environment_state" / "4.json");
    ahead["update"] = 4;
    ahead["state"] = "1000000";
    WriteJson(project / "checkpoints" / "environment_state" / "4.json", ahead);
    Outcome const resumed = Run("train --resume --updates 1" + at);
    auto const second = SavedStates(project);
    Check(resumed.code == 0 && second.size() == 1 && second.count(4) == 1 && second.at(4) > first.at(3)
            && second.at(4) < 1000000,
        "a resumed run continues the state of its own checkpoint's update: " + resumed.output);

    // Preserve an older BEST while the current checkpoint advances. A different state is opt-in.
    WriteJson(manifestPath, promotedBest);
    Check(promotedBest.at("update").get<unsigned>() < 4, "the selected BEST predates latest");
    auto const reportPath = project / "checkpoints/evaluation_campaign.json";
    Check(Run("eval --best --suite campaign" + at).code == 0, "evaluate an older generation");
    auto report = ReadJson(reportPath);
    Check(report.at("policy_sha256") == promotedBest.at("sha256") && report.at("update") == promotedBest.at("update")
            && report.at("environment_state") == promotedBest.at("environment_state")
            && report.at("reward_weights") == promotedBest.at("reward_weights")
            && report.at("evaluation_suite_sha256").get<std::string>().size() == 64, "BEST carries its own state and digest");
    auto const overridePath = project / "override.json";
    WriteJson(overridePath, Json{{"state", "17"}});
    Check(Run("eval --best --suite campaign --environment-state \"" + overridePath.string() + "\"" + at).code == 0,
        "evaluate a BEST at another explicit difficulty");
    report = ReadJson(reportPath);
    Check(report.at("context_source") == "explicit" && report.at("environment_state") == "17"
            && report.at("policy_sha256") == promotedBest.at("sha256"), "override changes only the evaluation state");
    Outcome const fresh = Run("train --updates 1 --fresh" + at);
    auto const third = SavedStates(project);
    std::vector<fs::path> const archives = Archives(project);
    Check(fresh.code == 0 && Contains(fresh.output, "The former run is archived in") && archives.size() == 1,
        "a fresh run archives the former one: " + fresh.output);
    Check(third.size() == 1 && third.count(1) == 1 && third.at(1) < second.at(4),
        "a fresh run starts its environment state afresh");
    if (archives.size() == 1)
    {
        fs::path const archive = archives.front();
        Check(fs::exists(archive / "latest.ppokit") && fs::exists(archive / "policy.ppokit")
                && fs::exists(archive / "metrics.jsonl") && fs::exists(archive / "environment_state" / "4.json")
                && fs::exists(archive / "best" / "campaign.json"),
            "the archive keeps the former checkpoint, policy, journal, environment state and BEST");
        Check(ppokit::artifacts::ReadArtifactHeader(archive / "latest.ppokit").updates == 4,
            "the archived checkpoint is the former run's, untouched");
    }
    Json const best = ReadJson(project / "checkpoints" / "best" / "campaign.json");
    Check(best.at("update") == 1, "the BEST of the fresh run is its own, not the former run's");

    auto const generation = ReadJson(project / "checkpoints/generation.json");
    Check(generation.at("schema") == "ppokit.generation/2", "new publications include their reward coefficients");
    // The journal can be unavailable when an operator moves the complete generation for evaluation.
    fs::remove(project / "checkpoints/metrics.jsonl");
    auto const evaluation = Run("eval --suite campaign --policy checkpoints/latest.ppokit" + at);
    Check(evaluation.code == 0, "the actor in the current training checkpoint can be evaluated: " + evaluation.output);
    report = ReadJson(reportPath);
    Check(report.at("policy_sha256") == generation.at("training_sha256")
            && report.at("environment_state") == generation.at("state") && report.at("context_source") == "checkpoint",
        "the report identifies the selected training file and uses its published environment state");

    Check(report.at("reward_weights") == generation.at("reward_weights"),
        "evaluation uses the reward weights published with the loaded actor");
    auto damaged = generation;
    damaged.erase("reward_weights");
    WriteJson(project / "checkpoints/generation.json", damaged);
    Check(Run("train --resume --updates 1" + at).code != 0, "a native generation with missing weights is refused");
    WriteJson(project / "checkpoints/generation.json", generation);

    // A checkpoint and states of different runs are refused, not mixed.
    fs::rename(project / "checkpoints" / "environment_state" / "1.json",
        project / "checkpoints" / "environment_state" / "7.json");
    Json other = ReadJson(project / "checkpoints" / "environment_state" / "7.json");
    other["update"] = 7;
    WriteJson(project / "checkpoints" / "environment_state" / "7.json", other);
    Outcome const mixed = Run("train --updates 1" + at);
    Check(mixed.code != 0 && Contains(mixed.output, "no environment state of update 1"),
        "a checkpoint whose update has no saved state is not resumed: " + mixed.output);

    for (std::string const command : {"play", "eval"})
    {
        for (std::string const file : {"policy.ppokit", "latest.ppokit"})
        {
            Outcome const mixedAlias = Run(command + " --policy checkpoints/../checkpoints/" + file + at);
            Check(mixedAlias.code == 1 && Contains(mixedAlias.output, "no environment state of update"),
                command + " checks the generation of " + file + ": " + mixedAlias.output);
        }
    }

    // A checkpoint of an earlier ppokit-run, saved without any state or generation manifest.
    fs::remove(project / "checkpoints/generation.json");
    fs::remove_all(project / "checkpoints" / "environment_state");
    Outcome const former = Run("train --updates 1" + at);
    Check(former.code == 0 && Contains(former.output, "No environment state saved with this checkpoint")
            && SavedStates(project).count(2) == 1,
        "a checkpoint saved without a state resumes, and its next save has one: " + former.output);
}

/** @brief A fresh run archives generations once, even after overlapping suites were renamed or removed. */
void FreshAfterSuiteEdits(fs::path const& root)
{
    auto const project = MakeProject(root / "fresh_suites", 100.0);
    auto const command = "train --fresh --updates 1 --project \"" + project.string() + "\"";
    for (auto const* suite : {"approach", "approach_wind", "retired"})
        WriteSuite(project, suite, ">= 0", 1);
    Check(Run(command).code == 0, "train the three original suites");
    auto const best = project / "checkpoints/best";
    auto retired = ReadJson(best / "retired.json");
    fs::path const published = retired.at("policy").get<std::string>();
    auto historic = published;
    historic.replace_extension(".policy");
    fs::rename(best / published, best / historic);
    retired["policy"] = historic.string();
    WriteJson(best / "retired.json", retired);
    fs::copy_file(best / historic, best / "retired_0.policy");
    fs::remove(project / "eval/retired.json");
    // Older, externally managed gate winners and unrelated files share the parent directory.
    fs::create_directories(best / "2v2");
    WriteJson(best / "2v2/champion.json", Json{{"owner", "native"}});
    std::ofstream(best / "approach_notes.txt") << "operator notes";
    std::ofstream(best / "approach_1.txt") << "operator notes from update 1";
    WriteJson(best / "operator.json", Json{{"owner", "operator"}});
    auto const fresh = Run(command);
    auto const archives = Archives(project);
    Check(fresh.code == 0 && archives.size() == 1,
        "fresh start succeeds with two suite names sharing a prefix: " + fresh.output);
    if (archives.size() != 1) return;
    auto const archived = archives.front() / "best";
    for (auto const* suite : {"approach", "approach_wind", "retired"})
    {
        auto const manifest = ReadJson(archived / (std::string(suite) + ".json"));
        Check(fs::is_regular_file(archived / manifest.at("policy").get<std::string>()),
            "each former BEST is archived with its exact generation");
    }
    Check(fs::is_regular_file(archived / "retired_0.policy"), "older generations retain their historical extension");
    Check(ReadJson(archived / "retired.json") == retired && !fs::exists(best / "retired.json")
            && !fs::exists(best / retired.at("policy").get<std::string>()),
        "a removed suite cannot bring the previous run's BEST back when it is reintroduced");
    Check(fs::is_regular_file(best / "2v2/champion.json") && fs::is_regular_file(best / "approach_notes.txt")
            && fs::is_regular_file(best / "operator.json") && fs::is_regular_file(best / "approach_1.txt"),
        "native champions and operator notes remain in place even if their name contains an update");
}

/** @brief Sets task.json "contract" of a project (removes it when empty). */
void SetContract(fs::path const& project, std::string const& contract)
{
    Json task = ReadJson(project / "task.json");
    if (contract.empty())
    {
        task.erase("contract");
    }
    else
    {
        task["contract"] = contract;
    }
    WriteJson(project / "task.json", task);
}

/**
 * @brief Checkpoints keep their environment (audit P1-04): the same shapes trained in
 *        another environment are refused, a transfer is explicit, an older checkpoint without
 *        a contract still loads.
 */
void Contracts(fs::path const& root)
{
    fs::path const first = MakeProject(root / "contract_a", 100.0);
    SetContract(first, "plane_a@1");
    Check(Run("train --updates 1 --fresh --project \"" + first.string() + "\"").code == 0, "a project trains");
    Check(ppokit::artifacts::ReadArtifactHeader(first / "checkpoints" / "latest.ppokit").identity.contract
            == "plane_a@1",
        "its checkpoints carry its contract");

    // The same shapes in another environment: its checkpoint copied by mistake is refused.
    fs::path const second = MakeProject(root / "contract_b", 100.0);
    SetContract(second, "plane_b@1");
    fs::create_directories(second / "checkpoints");
    fs::copy_file(first / "checkpoints" / "latest.ppokit", second / "checkpoints" / "latest.ppokit");
    fs::copy_file(first / "checkpoints" / "policy.ppokit", second / "checkpoints" / "policy.ppokit");
    std::string const atSecond = " --project \"" + second.string() + "\"";
    Outcome const resumed = Run("train --updates 1" + atSecond);
    Check(resumed.code != 0 && Contains(resumed.output, "trained in the environment 'plane_a@1'")
            && Contains(resumed.output, "ppokit-run transfer"),
        "train refuses a checkpoint of another environment: " + resumed.output);
    Outcome const played = Run("play --episodes 1" + atSecond);
    Check(played.code != 0 && Contains(played.output, "plane_a@1"), "play refuses it too: " + played.output);

    // An explicit transfer: never over an existing checkpoint, then the project's own.
    Outcome const refused = Run("transfer --from \"" + (first / "checkpoints" / "latest.ppokit").string() + "\"" + atSecond);
    Check(refused.code != 0 && Contains(refused.output, "nothing is overwritten"),
        "transfer never overwrites: " + refused.output);
    fs::remove(second / "checkpoints" / "latest.ppokit");
    fs::remove(second / "checkpoints" / "policy.ppokit");
    Outcome const transferred =
        Run("transfer --from \"" + (first / "checkpoints" / "latest.ppokit").string() + "\"" + atSecond);
    Check(transferred.code == 0 && Contains(transferred.output, "for the environment 'plane_b@1'"),
        "transfer starts the project from the checkpoint of another environment: " + transferred.output);
    Check(CheckpointBody(first / "checkpoints/latest.ppokit") == CheckpointBody(second / "checkpoints/latest.ppokit"),
        "transferring the checkpoint preserves the original payload byte for byte");
    Check(Run("train --updates 1" + atSecond).code == 0
            && ppokit::artifacts::ReadArtifactHeader(second / "checkpoints" / "latest.ppokit").identity.contract
                == "plane_b@1",
        "and the project trains on from it, under its contract");

    // A checkpoint written before contracts existed: nothing tells its environment.
    fs::path const third = MakeProject(root / "contract_none", 100.0);
    Check(Run("train --updates 1 --fresh --project \"" + third.string() + "\"").code == 0, "a task without contract");
    SetContract(third, "plane_c@1");
    Check(Run("train --updates 1 --project \"" + third.string() + "\"").code == 0,
        "a checkpoint without a contract still loads in a project with one");
}

/**
 * @brief The team_line project: three agent rows, two agents present (one leaves during
 *        each episode), its networks only in the team DLL (no network.json), one suite.
 * @param maximumAgents task.json "agents" max (3 matches the DLLs).
 */
fs::path MakeTeamProject(fs::path const& folder, std::size_t maximumAgents)
{
    fs::create_directories(folder);
    Json task = ReadJson(PPOKIT_TEAM_LINE_TASK);
    task["agents"]["max"] = maximumAgents;
    WriteJson(folder / "task.json", task);
    Json project;
    project["schema"] = "ppokit.project/1";
    project["name"] = "team_line";
    project["libraries"] = {{"environment", PPOKIT_TEAM_LINE_ENVIRONMENT}, {"network", PPOKIT_TEAM_LINE_NETWORK}};
    project["program"] = {{"executable", "ppokit-run"}, {"train", {"train", "--project", "{project}"}},
        {"evaluate", {"eval", "--project", "{project}"}}, {"inference", {"play", "--project", "{project}"}}};
    WriteJson(folder / "project.json", project);
    Json suite;
    suite["schema"] = "ppokit.eval/1";
    suite["suite"] = "line";
    suite["seeds"] = {{"first", 11}, {"count", 2}};
    suite["episodes"] = 2;
    // begin_step comes before every step but an episode's first: "begun" is 0 in every episode.
    suite["gates"] = {{"reached.rate", ">= 0"}, {"begun.mean", "<= 0"}};
    suite["during_training"] = {{"every_updates", 2}};
    suite["best"] = Json::array({{{"metric", "reached.rate"}, {"prefer", "higher"}}});
    fs::create_directories(folder / "eval");
    WriteJson(folder / "eval" / "line.json", suite);
    return folder;
}

/** @brief First word and update count of a team_line state file. */
std::pair<std::string, std::uint64_t> ReadTeamState(fs::path const& file)
{
    std::ifstream in(file);
    std::string tag;
    double gain = 0.0;
    std::uint64_t updates = 0;
    in >> tag >> gain >> updates;
    return {tag, updates};
}

/** @brief ppokit-run with a team: training (saves between rollouts only), resume, campaigns, eval, play. */
void Team(fs::path const& root)
{
    fs::path const project = MakeTeamProject(root / "team", 3);
    std::string const at = " --project \"" + project.string() + "\"";

    Outcome const trained = Run("train --updates 3 --fresh" + at);
    Check(trained.code == 0 && Contains(trained.output, "update 3/3")
            && Contains(trained.output, "evaluation 'line' at update 2: 4 episodes, passed"),
        "ppokit-run trains a team and runs its campaigns: " + trained.output);
    if (trained.code != 0) return;
    auto const [trainingTag, trainingUpdates] = ReadTeamState(project / "checkpoints" / "latest.ppokit");
    auto const [policyTag, policyUpdates] = ReadTeamState(project / "checkpoints" / "policy.ppokit");
    Check(trainingTag == "team_line_training" && trainingUpdates == 3 && policyTag == "team_line_policy"
            && policyUpdates == 3,
        "the team DLL saved its training state and policy in its own format at update 3");
    ppokit::runtime::RunLog const log = ppokit::runtime::LoadRunLog(project / "checkpoints" / "metrics.jsonl");
    std::size_t episodes = 0;
    bool sawBoundedReturn = false;
    for (auto const& update : log.updates)
    {
        episodes += update.rewards.episodes;
        if (update.rewards.episodes && update.rewards.channels.size() == 2)
        {
            auto const channelSum = update.rewards.channels[0].episodeMean + update.rewards.channels[1].episodeMean;
            sawBoundedReturn = sawBoundedReturn || std::abs(channelSum - update.rewards.episodeReturnMean) > 1e-6;
        }
    }
    Check(sawBoundedReturn, "the CLI journals the bounded team-line reward fed to the learner");
    Check(log.updates.size() == 3 && log.header.channels.size() == 2 && episodes > 0 && log.evaluations.size() == 2
            && log.evaluations[0].promoted,
        "the journal has the team's updates, episodes and campaigns, the first one the BEST");
    bool teamMetrics = false;
    for (auto const& [name, value] : log.updates[0].ppo)
    {
        teamMetrics = teamMetrics || (name == "learnable_rows" && value == 16.0);
    }
    bool teamSettings = false;
    for (auto const& [name, value] : log.header.settings)
    {
        teamSettings = teamSettings || (name == "gamma" && value == 0.5);
    }
    Check(teamMetrics && teamSettings, "the journal has the team learner's own metrics and settings");
    Check(log.probes.size() == 2 && log.probes[0].update == 2 && log.probes[0].network == "actor"
            && log.probes[1].network == "critic" && log.probes[0].blocks.size() == 1,
        "the journal has the probes of the updates the team learner probed");

    Outcome const resumed = Run("train --resume --updates 1" + at);
    Check(resumed.code == 0 && Contains(resumed.output, "Resuming from") && Contains(resumed.output, "update 4/4"),
        "a team's training resumes: " + resumed.output);
    Outcome const until = Run("train --until-update 5" + at);
    Check(until.code == 0 && Contains(until.output, "update 5/5") && !Contains(until.output, "update 6"),
        "--until-update trains up to an absolute update: " + until.output);
    Outcome const reached = Run("train --until-update 5" + at);
    Check(reached.code == 0 && !Contains(reached.output, "update 6"), "--until-update past the target trains no more");

    Outcome const evaluated = Run("eval" + at);
    Check(evaluated.code == 0 && Contains(evaluated.output, "evaluation 'line' at update 5: 4 episodes, passed")
            && fs::exists(project / "checkpoints" / "evaluation_line.json"),
        "eval plays the team policy, whose file is the DLL's own: " + evaluated.output);

    Outcome const played = Run("play --episodes 2 --best" + at);
    Check(played.code == 0 && Contains(played.output, "BEST of 'line'") && Contains(played.output, "episode 2:"),
        "play --best plays the team's BEST policy: " + played.output);

    // A policy file of the project's own, e.g. a production policy kept apart.
    fs::copy_file(project / "checkpoints" / "policy.ppokit", project / "production_policy.txt");
    Outcome const production = Run("eval --policy production_policy.txt" + at);
    Check(production.code == 0 && Contains(production.output, "evaluation 'line': 4 episodes"),
        "eval --policy evaluates a policy file named relative to the project: " + production.output);
    Outcome const both = Run("play --best --policy production_policy.txt" + at);
    Check(both.code == 2 && Contains(both.output, "select one of"), "--best and --policy together are refused");

    fs::path const changed = MakeTeamProject(root / "team_contract", 3);
    Json changedTask = ReadJson(changed / "task.json");
    changedTask["actions"]["move"]["bounds"] = {-0.5, 1.0};
    WriteJson(changed / "task.json", changedTask);
    Outcome const wrongContract = Run("train --updates 1 --project \"" + changed.string() + "\"");
    Check(wrongContract.code == 1 && Contains(wrongContract.output, "compiled task contract differs")
        && !fs::exists(changed / "checkpoints"), "same-sized incompatible task is refused before training");

    fs::path const mismatch = MakeTeamProject(root / "team_mismatch", 2);
    Outcome const refused = Run("train --updates 1 --project \"" + mismatch.string() + "\"");
    Check(refused.code == 1 && Contains(refused.output, "same sizes or agents"),
        "a task.json with another number of agents than the DLLs is refused: " + refused.output);
}

/** @brief An explicit manifest keeps its checkpoints and journal separate from the default run. */
void SeparateManifestRuns(fs::path const& root)
{
    fs::path const project = MakeProject(root / "separate_manifests", 100.0);
    Json alternate = ReadJson(project / "project.json");
    alternate["checkpoints"] = {{"root", "checkpoints/alternate"}};
    fs::path const manifest = project / "project.legacy.json";
    WriteJson(manifest, alternate);
    std::string const atAlternate = " --project \"" + manifest.string() + "\"";
    std::string const atDefault = " --project \"" + project.string() + "\"";
    Outcome const first = Run("train --fresh --updates 1" + atAlternate);
    auto const alternateCheckpoint = project / "checkpoints/alternate/latest.ppokit";
    auto const defaultCheckpoint = project / "checkpoints/latest.ppokit";
    Check(first.code == 0 && fs::exists(alternateCheckpoint)
            && fs::exists(project / "checkpoints/alternate/metrics.jsonl") && !fs::exists(defaultCheckpoint),
        "explicit manifest trains into its own checkpoint root: " + first.output);
    if (first.code != 0 || !fs::exists(alternateCheckpoint)) return;
    Outcome const normal = Run("train --fresh --updates 1" + atDefault);
    Check(normal.code == 0 && fs::exists(defaultCheckpoint), "default project trains independently: " + normal.output);
    if (normal.code != 0 || !fs::exists(defaultCheckpoint)) return;
    auto const defaultDigest = ppokit::artifacts::Sha256OfFile(defaultCheckpoint);
    Outcome const resumed = Run("train --resume --updates 1" + atAlternate);
    Check(resumed.code == 0 && ppokit::artifacts::ReadArtifactHeader(alternateCheckpoint).updates == 2
            && ppokit::artifacts::Sha256OfFile(defaultCheckpoint) == defaultDigest,
        "resuming the explicit manifest leaves the default checkpoint untouched: " + resumed.output);
}

/** @brief The same declared actor is verified by role and by explicit path; LATEST stays independent. */
void ModelRoles(fs::path const& project, std::string const& at)
{
    auto const fixed = project / "packaged.ppokit";
    fs::copy_file(project / "checkpoints/policy.ppokit", fixed);
    auto const digest = ppokit::artifacts::Sha256OfFile(fixed);
    Json const ref = {{"path", "packaged.ppokit"}, {"sha256", digest}, {"suite", "smoke"},
        {"source", {{"commit", std::string(40, 'a')}, {"path", "original"}, {"sha256", digest}, {"update", 4}}}};
    WriteJson(project / "models.json", {{"schema", "ppokit.models/1"}, {"policies", {{"PROD", ref}}}});
    auto const policy = Run("play --model PROD --episodes 1" + at);
    Check(policy.code == 0 && Contains(policy.output, "Model PROD") && Contains(policy.output, digest),
        "selected reference exposes its confirmed digest: " + policy.output);
    Check(Run("train --model PROD --updates 1" + at).code == 2, "train refuses an inference-only reference");
    Check(Run("play --model NONEXISTENT" + at).code == 2, "unknown role fails before inference");
    auto original = ReadJson(project / "project.json");
    auto manifest = original;
    manifest["program"]["inference"].push_back("--model");
    manifest["program"]["inference"].push_back("PROD");
    WriteJson(project / "project.json", manifest);
    auto const bare = Run("play --episodes 1" + at);
    Check(bare.code == 0 && Contains(bare.output, "Model PROD"), "bare CLI follows the declared inference model");
    std::ofstream(fixed, std::ios::app) << "modified";
    Check(Run("play --episodes 1" + at).code == 1, "a bad declared policy cannot silently fall back to LATEST");
    Check(Run("play --model PROD" + at).code == 1 && Run("play --policy packaged.ppokit" + at).code == 1,
        "both ways of naming a modified immutable actor are rejected");
    Check(Run("play --model LATEST --episodes 1" + at).code == 0, "mutable training actor has its own selection");
    fs::remove(project / "models.json");
    WriteJson(project / "project.json", original);
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path() / "ppokit_run_tests";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    try
    {
        CliNumbersAreStrict();
        fs::path const project = MakeProject(root / "reach", 100.0);
        std::string const at = " --project \"" + project.string() + "\"";

        Outcome const usage = Run("trian" + at);
        Check(usage.code == 2 && Contains(usage.output, "unknown command 'trian'"), "a usage error: " + usage.output);

        Outcome const absent = Run("train --resume --updates 1" + at);
        Check(absent.code == 1 && Contains(absent.output, "--resume requires")
                && !fs::exists(project / "checkpoints/latest.ppokit"),
            "explicit resume cannot create a random learner");
        Check(Run("train --fresh --resume" + at).code == 2, "contradictory training modes are refused");
        Check(Run("play --fresh" + at).code == 2, "training mode is refused for inference");

        Outcome const trained = Run("train --updates 3 --fresh" + at);
        Check(trained.code == 0 && Contains(trained.output, "update 3/3"), "ppokit-run trains: " + trained.output);
        Check(ppokit::artifacts::ReadArtifactHeader(project / "checkpoints" / "latest.ppokit").updates == 3,
            "the training checkpoint holds 3 updates");
        ppokit::runtime::RunLog const log = ppokit::runtime::LoadRunLog(project / "checkpoints" / "metrics.jsonl");
        Check(log.updates.size() == 3 && log.header.channels.size() == 3 && !log.header.acceptsWeightRequests,
            "the journal has the updates and the task's channels");
        Check(!fs::exists(project / "checkpoints" / "live.json"), "the live announcement goes with the run");

        Outcome const resumed = Run("train --resume --updates 1" + at);
        Check(resumed.code == 0 && Contains(resumed.output, "Resuming from") && Contains(resumed.output, "update 4/4"),
            "a second session resumes: " + resumed.output);
        auto const continuedLog = ppokit::runtime::LoadRunLog(project / "checkpoints/metrics.jsonl");
        auto const& currentRewards = continuedLog.updates.back().rewards;
        for (std::size_t i = 0; i < currentRewards.channels.size(); ++i)
        {
            auto const& channel = currentRewards.channels[i];
            double const expected = log.updates.back().rewards.channels[i].cumulative
                + channel.weightedStepMean * static_cast<double>(currentRewards.steps);
            Check(std::abs(channel.cumulative - expected) < 1e-9, "the CLI continues each weighted channel total after resume");
        }

        ModelRoles(project, at);
        Outcome const played = Run("play --episodes 2" + at);
        Check(played.code == 0 && Contains(played.output, "episode 2:") && !Contains(played.output, "stopped"),
            "play runs the policy and the observer sees what it expects: " + played.output);

        // A paused, real-time play must accept Stop before even the first episode ends.
        ppokit::ipc::Supervisor supervisor;
        auto child = std::async(std::launch::async, [&] {
            return Run("play --episodes 1 --realtime --step-seconds 0.1 --control " + supervisor.Endpoint() + at);
        });
        Check(supervisor.Request(ppokit::ipc::Command::Pause), "pause can be queued before HELLO");
        bool const paused = PollUntil(supervisor, [&] { return supervisor.LastState() == ppokit::ipc::State::Paused; });
        Check(paused, "the child confirms pause at a simulation boundary");
        supervisor.Request(ppokit::ipc::Command::Stop);
        Check(PollUntil(supervisor, [&] { return child.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }),
            "stop during pause exits at the next safe point");
        auto const stopped = child.get();
        Check(stopped.code == 0 && Contains(stopped.output, "Stopped on request at a simulation step")
            && !Contains(stopped.output, "episode 1:"), "a partial episode is never reported: " + stopped.output);

        // The observer receives the world: with a tiny limit, the target position exceeds it.
        fs::path const strict = MakeProject(root / "strict", 1.0e-6);
        fs::copy(project / "checkpoints", strict / "checkpoints", fs::copy_options::recursive);
        Outcome const watched = Run("play --episodes 1 --project \"" + strict.string() + "\"");
        Check(watched.code == 0 && Contains(watched.output, "observer 'watch' stopped")
                && Contains(watched.output, "above the limit"),
            "the observer is called with the world and switched off when it fails: " + watched.output);

        DecisionBootstrap(root);
        Evaluations(root);
        CurriculumEndsTraining(root);
        CampaignEndsTraining(root);
        RunState(root);
        FreshAfterSuiteEdits(root);
        ConcurrentWriters(root);
        Contracts(root);
        Team(root);
        SeparateManifestRuns(root);

        fs::remove(project / "project.json");
        Outcome const missing = Run("play" + at);
        Check(missing.code == 1 && Contains(missing.output, "not a PPOKit project"), "a folder without project.json");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " ppokit-run check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "ppokit-run tests passed\n";
    return EXIT_SUCCESS;
}
