/**
 * @file PPOKit/tests/command_line_tests.cpp
 * @brief The standard example command line on the reach example: usage errors, check,
 *        train, resume (update count, journal, curriculum stage), a live weight request,
 *        a refused resume after a network change, and eval with its exit code.
 * @details Training runs a few updates only: this checks the plumbing, not learning
 * (only an evaluation on independent seeds says a policy learned).
 */
#include "task.hpp"

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/runtime/weight_request.hpp"
#include "ppokit/live/live_channel.hpp"
#include "ppokit/tools/command_line.hpp"

#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ppokit;

int failures = 0;
/** @brief Update count of the checkpoint after the stopped run. */
std::uint64_t stoppedAt = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief What one command printed and returned. */
struct Result final
{
    int code = 0;
    std::string output;
    std::string errors;
};

/** @brief Runs the reach command line in process with `arguments`, capturing its output. */
Result Run(std::vector<std::string> arguments)
{
    arguments.insert(arguments.begin(), "reach");
    std::vector<char*> argv;
    for (std::string& argument : arguments)
    {
        argv.push_back(argument.data());
    }
    std::ostringstream output;
    std::ostringstream errors;
    std::streambuf* const savedOutput = std::cout.rdbuf(output.rdbuf());
    std::streambuf* const savedErrors = std::cerr.rdbuf(errors.rdbuf());
    Result result;
    result.code = tools::RunCommandLine<reach_example::ObservationSize, reach_example::ActionSize>(
        static_cast<int>(argv.size()), argv.data(), reach_example::MakeTask, PPOKIT_REACH_DIR);
    std::cout.rdbuf(savedOutput);
    std::cerr.rdbuf(savedErrors);
    result.output = output.str();
    result.errors = errors.str();
    return result;
}

bool Contains(std::string const& text, std::string const& part)
{
    return text.find(part) != std::string::npos;
}

std::string FirstLine(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    std::string line;
    std::getline(stream, line);
    return line;
}

void WriteText(std::filesystem::path const& path, std::string const& text)
{
    std::ofstream stream(path, std::ios::binary);
    stream << text;
}

/** @brief Updates in a journal being written (0 while a line is half written). */
std::size_t JournalUpdates(std::filesystem::path const& journal)
{
    try
    {
        return runtime::LoadRunLog(journal).updates.size();
    }
    catch (std::exception const&)
    {
        return 0;
    }
}

void Usage()
{
    Check(Run({"help"}).code == 0 && Contains(Run({"help"}).output, "train"), "help lists the commands");
    Result const unknown = Run({"trian"});
    Check(unknown.code == 2 && Contains(unknown.errors, "unknown command 'trian'"), "an unknown command is a usage error");
    Result const count = Run({"train", "--updates", "ten"});
    Check(count.code == 2 && Contains(count.errors, "--updates takes a whole number"), "a bad count names the option");
    Result const option = Run({"eval", "--updates", "3"});
    Check(option.code == 2 && Contains(option.errors, "unknown argument '--updates'"), "eval refuses train options");
    Check(Run({"train", "--project"}).code == 2, "an option without its value is a usage error");
    Result const missing = Run({"check", "--project", "/no/such/project"});
    Check(missing.code == 1 && Contains(missing.errors, "no project folder"), "a missing project is named");
}

void CheckFiles()
{
    Result const result = Run({"check"});
    Check(result.code == 0 && Contains(result.output, "All files are valid: task 'reach'")
            && Contains(result.output, "no curriculum") && Contains(result.output, "'reach_smoke'"),
        "check validates the example's files: " + result.output + result.errors);
}

void TrainAndResume(std::filesystem::path const& project)
{
    std::filesystem::path const model = project / "checkpoints";
    Result const first = Run({"train", "--legacy-control", "--updates", "2", "--project", project.string(), "--fresh"});
    Check(first.code == 0, "a fresh training runs: " + first.errors);
    Check(Contains(first.output, "open the project in ppokit-studio"), "train says how to watch the run");
    Check(artifacts::ReadArtifactHeader(model / "latest.ppokit").updates == 2, "the training checkpoint is at update 2");
    Check(std::filesystem::exists(model / "policy.ppokit"), "a policy is written for eval");
    Check(FirstLine(model / "curriculum.txt") == "near", "the curriculum stage is saved");
    runtime::RunLog const log = runtime::LoadRunLog(model / "metrics.jsonl");
    Check(log.updates.size() == 2 && log.header.acceptsWeightRequests && log.header.channels.size() == 3,
        "the journal has both updates and the three task channels, and accepts weight requests");

    // The studio asks for a new Time weight; the stage is moved on by hand to see it restored.
    runtime::WriteWeightRequest(model / "metrics.jsonl", 7, {{"Time", 2.0}});
    WriteText(model / "curriculum.txt", "far\n");
    Result const second = Run({"train", "--legacy-control", "--updates", "1", "--project", project.string()});
    Check(second.code == 0, "the second session resumes: " + second.errors);
    Check(Contains(second.output, "Resuming from") && Contains(second.output, "stage 2/2 'far'"),
        "the resume says where it continues: " + second.output);
    Check(artifacts::ReadArtifactHeader(model / "latest.ppokit").updates == 3, "the update count continues at 3");
    Check(FirstLine(model / "curriculum.txt") == "far", "the restored stage is kept");
    runtime::RunLog const resumed = runtime::LoadRunLog(model / "metrics.jsonl");
    Check(resumed.updates.size() == 3 && resumed.updates.back().update == 3, "the journal continues the same run");
    Check(resumed.weightChanges.size() == 1 && resumed.weightChanges[0].weights.at(1) == 2.0
            && resumed.weightChanges[0].fromUpdate == 4,
        "the weight change is journaled from the next update");
    auto const answer = runtime::ReadWeightAnswer(model / "metrics.jsonl");
    Check(answer && answer->id == 7 && answer->applied, "the studio is told the request was applied");

    // A third session keeps the weight the studio set during the second one.
    Result const third = Run({"train", "--legacy-control", "--updates", "1", "--project", project.string()});
    runtime::RunLog const continued = runtime::LoadRunLog(model / "metrics.jsonl");
    Check(third.code == 0 && Contains(third.output, "Reward weights changed during earlier sessions are kept"),
        "a resume keeps the weights changed from the studio: " + third.output + third.errors);
    Check(continued.weightChanges.size() == 1 && continued.updates.size() == 4,
        "keeping them adds no weight change to the journal");

    // A stop request left by an earlier run is ignored.
    WriteText(model / "stop.request", "");
    Result const stale = Run({"train", "--legacy-control", "--updates", "1", "--project", project.string()});
    Check(stale.code == 0 && !Contains(stale.output, "Stopped on request"),
        "a stale stop request does not end a new run");
    Check(artifacts::ReadArtifactHeader(model / "latest.ppokit").updates == 5, "the stale-stop run trained");
    Check(!std::filesystem::exists(model / "stop.request"), "the stale stop request is removed");

    // A stop asked while training runs (as the studio's Stop button does) ends it after the
    // current update, with a checkpoint.
    std::size_t const before = runtime::LoadRunLog(model / "metrics.jsonl").updates.size();
    std::thread stopper([&model, before] {
        for (int attempt = 0; attempt < 3000; ++attempt)
        {
            if (JournalUpdates(model / "metrics.jsonl") > before)
            {
                WriteText(model / "stop.request", "");
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    Result const stopped = Run({"train", "--legacy-control", "--updates", "500", "--project", project.string()});
    stopper.join();
    std::uint64_t const reached = artifacts::ReadArtifactHeader(model / "latest.ppokit").updates;
    Check(stopped.code == 0 && Contains(stopped.output, "Stopped on request"), "a stop request ends training cleanly");
    Check(reached > 5 && reached < 505, "the checkpoint holds the updates done before the stop");
    Check(!std::filesystem::exists(model / "stop.request"), "the stop request is consumed");
    stoppedAt = reached;
}

/** @brief What the studio saw of a live run, for the checks made after it ended. */
struct LiveObservation final
{
    bool attached = false;
    bool paused = false;
    bool stillWhilePaused = false;
    bool resumed = false;
    bool weightsApplied = false;
    bool stopping = false;
};

/** @brief Polls the client until `done` holds or ten seconds pass. */
template<typename Done>
bool WaitFor(live::LiveClient& client, Done&& done)
{
    for (int attempt = 0; attempt < 2000; ++attempt)
    {
        client.Poll();
        if (done())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

/** @brief The studio's side: attach, pause, check nothing moves, change a weight, play, stop. */
void DriveLiveRun(std::filesystem::path const& model, LiveObservation& seen)
{
    std::optional<live::LiveClient> client;
    for (int attempt = 0; attempt < 2000 && !client; ++attempt)
    {
        client = live::LiveClient::Attach(model);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    seen.attached = client.has_value();
    if (!client)
    {
        return;
    }
    std::size_t const known = client->Log().updates.size();
    WaitFor(*client, [&] { return client->HasLog() && client->Log().updates.size() > known; });
    client->Pause();
    seen.paused = WaitFor(*client, [&] { return client->Status() && client->Status()->state == live::RunState::Paused; });
    std::uint64_t const steps = client->Status()->steps;
    // Wait past a few status lines, then read the last one.
    for (int attempt = 0; attempt < 80; ++attempt)
    {
        client->Poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    seen.stillWhilePaused = client->Status()->steps == steps;
    runtime::WeightRequest request;
    request.id = 9;
    request.weights = {{1, 3.0}};
    client->SendWeights(request);
    client->Play();
    std::uint64_t const update = client->Status()->update;
    seen.resumed = WaitFor(*client, [&] { return client->Status()->update > update; });
    seen.weightsApplied = WaitFor(*client, [&] {
        std::optional<runtime::WeightAnswer> const answer = client->TakeAnswer();
        return answer && answer->id == 9 && answer->applied;
    });
    client->Stop();
    seen.stopping = WaitFor(*client, [&] {
        return client->Status()->state == live::RunState::Stopping || client->Status()->state == live::RunState::Finished
            || !client->Connected();
    });
}

/** @brief A run trains live: the studio pauses it, changes a weight, resumes and stops it. */
void LiveTraining(std::filesystem::path const& project)
{
    std::filesystem::path const model = project / "checkpoints";
    std::uint64_t const before = artifacts::ReadArtifactHeader(model / "latest.ppokit").updates;
    LiveObservation seen;
    std::thread studio([&model, &seen] { DriveLiveRun(model, seen); });
    Result const run = Run({"train", "--legacy-control", "--updates", "100000", "--project", project.string()});
    studio.join();
    Check(seen.attached, "the studio attaches to the training run");
    Check(seen.paused, "pause reaches the run, which says it is paused");
    Check(seen.stillWhilePaused, "a paused run makes no step");
    Check(seen.resumed, "play resumes the training");
    Check(seen.weightsApplied, "a weight changed live is applied at the next update");
    Check(seen.stopping, "stop is acknowledged");
    Check(run.code == 0 && Contains(run.output, "checkpoint saved"), "the stopped run saved: " + run.output + run.errors);
    Check(artifacts::ReadArtifactHeader(model / "latest.ppokit").updates > before, "the run trained before the stop");
    runtime::RunLog const log = runtime::LoadRunLog(model / "metrics.jsonl");
    Check(!log.weightChanges.empty() && log.weightChanges.back().weights.at(1) == 3.0,
        "the live weight change is in the journal");
    Check(!std::filesystem::exists(live::LiveFilePath(model)), "the run no longer announces itself once ended");
}

void RefusedResume(std::filesystem::path const& project)
{
    // Same task, wider actor: the checkpoint no longer fits.
    std::filesystem::path const network = project / "network.json";
    std::ifstream original(network);
    std::string text((std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
    original.close();
    std::string const narrow = "\"width\": 32";
    std::size_t const width = text.find(narrow);
    Check(width != std::string::npos, "the example network has a width to change");
    std::string const before = text;
    text.replace(width, narrow.size(), "\"width\": 48");
    WriteText(network, text);
    Result const result = Run({"train", "--updates", "1", "--project", project.string()});
    Check(result.code == 1 && Contains(result.errors, "train --fresh"),
        "a resume on other networks is refused and says how to start over: " + result.errors);
    Check(artifacts::ReadArtifactHeader(project / "checkpoints" / "latest.ppokit").updates == stoppedAt,
        "the refused run changed nothing");
    WriteText(network, before);
}

void Evaluate(std::filesystem::path const& project)
{
    std::filesystem::path const always = project / "eval" / "always.json";
    WriteText(always, R"({"schema": "ppokit.eval/1", "suite": "always", "seeds": [501, 502], "episodes": 2,
        "gates": {"mean_length": "<= 40"}})");
    Result const passed = Run({"eval", "--project", project.string(), "--suite", always.string()});
    Check(passed.code == 0 && Contains(passed.output, "PASSED"), "a holding gate passes: " + passed.errors);

    std::filesystem::path const never = project / "eval" / "never.json";
    WriteText(never, R"({"schema": "ppokit.eval/1", "suite": "never", "seeds": [501], "episodes": 1,
        "gates": {"mean_length": "<= 0"}})");
    Result const failed = Run({"eval", "--project", project.string(), "--suite", never.string()});
    Check(failed.code == 1 && Contains(failed.output, "FAILED"), "a failing gate gives exit code 1");

    Result const played = Run({"play", "--project", project.string(), "--episodes", "3"});
    Check(played.code == 0 && Contains(played.output, "episode seed 9001") && Contains(played.output, "episode seed 9003"),
        "play prints every episode: " + played.output + played.errors);
}

/** @brief A project without checkpoints: eval and play say to train first. */
void Untrained(std::filesystem::path const& project)
{
    Result const missing = Run({"eval", "--project", project.string()});
    Check(missing.code == 1 && Contains(missing.errors, "train first"), "no policy: eval says to train first");
    Result const play = Run({"play", "--project", project.string()});
    Check(play.code == 1 && Contains(play.errors, "train first"), "no policy: play says to train first");
}

/** @brief Copies the project files of the reach example (not its checkpoints). */
std::filesystem::path CopyProject(std::filesystem::path const& into)
{
    std::filesystem::path const source = PPOKIT_REACH_DIR;
    std::filesystem::create_directories(into / "eval");
    for (char const* file : {"project.json", "task.json", "network.json", "training.json", "eval/smoke.json"})
    {
        std::filesystem::copy_file(source / file, into / file, std::filesystem::copy_options::overwrite_existing);
    }
    // Keep the retired in-process command line as a compatibility oracle. Production
    // projects use curriculum.json through ppokit-run and must not duplicate this plan.
    std::ifstream input(into / "training.json");
    std::string training((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::size_t const closingBrace = training.rfind('}');
    if (closingBrace == std::string::npos) throw std::runtime_error("invalid copied training.json");
    training.insert(closingBrace, R"(,
  "curriculum": {
    "window": 50,
    "promote_at": 0.8,
    "stages": [
      { "name": "near", "sample": { "start_distance": { "uniform": [0.2, 0.5] } } },
      { "name": "far", "sample": { "start_distance": { "uniform": [0.5, 1.5] } } }
    ]
  }
)");
    WriteText(into / "training.json", training);
    return into;
}

} // namespace

int main()
{
    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_command_line_tests";
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    std::filesystem::create_directories(folder);
    std::filesystem::path const project = CopyProject(folder / "reach");
    std::filesystem::path const fresh = CopyProject(folder / "fresh");
    try
    {
        Usage();
        CheckFiles();
        Untrained(fresh);
        TrainAndResume(project);
        RefusedResume(project);
        Evaluate(project);
        LiveTraining(project);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(folder, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " command line check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "command line checks passed\n";
    return EXIT_SUCCESS;
}
