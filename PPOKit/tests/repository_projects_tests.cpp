/**
 * @file PPOKit/tests/repository_projects_tests.cpp
 * @brief The projects of the repository (Examples/<Name>/projects/) open, are complete, and describe the
 *        task and network their program is compiled with.
 * @details FlightControl3D and BluePatrol compile task.json and their network: a project copy
 * that drifts from them would show the user a network the program does not run. The T-6 and
 * the PC-7 run under ppokit-run: their libraries, their training.json (the PPO values of their
 * scenario), their validation suite, their Visualiseur and, for the T-6, its checkpoint are
 * checked; the PC-7 starts its training from scratch. BluePatrol trains and evaluates under
 * ppokit-run with the shared graph backend and separate historical checkpoint files; its suite is the
 * campaign of its training file. Every world group of a project reaches its Visualiseur.
 */
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/component/native_module.hpp"
#include "ppokit/eval/evaluation.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace schema = ppokit::schema;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string ReadFile(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

/** @brief Opens a project, checks it is complete and that its task and network load. */
schema::Project CheckOpens(std::filesystem::path const& folder)
{
    schema::Project const project = schema::LoadProject(folder);
    std::string const name = folder.filename().string();
    Check(schema::CheckProject(project).empty(), name + " is complete (no remark)");
    for (schema::ProjectCommand const command :
         {schema::ProjectCommand::Train, schema::ProjectCommand::Evaluate, schema::ProjectCommand::Inference})
    {
        Check(schema::CommandArguments(project, command).has_value(),
            name + " offers " + schema::CommandName(command));
    }
    schema::TaskSchema const task = schema::LoadTaskSchema(project.Task());
    schema::LoadBlockGraph(project.Network(), task);
    return project;
}

/** @brief The Flight block network has the trunk and heads of the compiled MLP (graph.json). */
void CheckFlightNetwork(schema::Project const& project, std::filesystem::path const& compiledGraph)
{
    nlohmann::json const compiled = nlohmann::json::parse(ReadFile(compiledGraph));
    nlohmann::json const blocks = nlohmann::json::parse(ReadFile(project.Network()));
    for (char const* network : {"actor", "critic"})
    {
        nlohmann::json const& hidden = compiled.at(network).at("hidden");
        nlohmann::json const& trunk = blocks.at(network).at("nodes").at("trunk");
        Check(trunk.at("layers") == hidden.at("layers") && trunk.at("width") == hidden.at("width")
                && trunk.at("activation") == compiled.at(network).at("activation"),
            project.name + " " + network + " trunk matches " + compiledGraph.string());
    }
}

/** @brief Whether the observers of network.json wire `source` to the port `port` of the watch. */
bool Wires(nlohmann::json const& observers, std::string const& source, std::string const& port)
{
    for (auto const& edge : observers.at("edges"))
    {
        std::string const& from = edge.at(0).get_ref<std::string const&>();
        if (observers.at("nodes").at(from).value("from", "") == source && edge.at(1) == "watch." + port)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief network.json names `library` and its watch `watchType`, and every group of the task's
 *        world reaches the watch's port of the same name.
 */
void CheckVisualiser(schema::Project const& project, std::string const& library, std::string const& watchType)
{
    nlohmann::json const network = nlohmann::json::parse(ReadFile(project.Network()));
    Check(network.at("components") == nlohmann::json::array({library})
            && network.at("observers").at("nodes").at("watch").at("type") == "component:" + watchType,
        project.name + "/network.json wires its Visualiseur (" + watchType + ")");
    nlohmann::json const task = nlohmann::json::parse(ReadFile(project.Task()));
    nlohmann::json const& observers = network.at("observers");
    for (auto const& [group, declaration] : task.at("world").items())
    {
        static_cast<void>(declaration);
        Check(Wires(observers, "world." + group, group),
            project.name + "/network.json wires world." + group + " to the Visualiseur");
    }
}

/**
 * @brief Check the shared BluePatrol project and its isolated historical configuration.
 */
void CheckBluePatrol(schema::Project const& project)
{
    Check(project.environmentLibrary == "blue_environment" && project.networkLibrary == "ppokit_graph_network"
            && project.networkEditable && project.SettingsFile(),
        "BluePatrol uses its editable shared graph and its environment settings");
    Check(project.Checkpoints() == project.root / "checkpoints/shared", "shared run state is isolated");
    Check(project.externalCurriculumProgress == "checkpoints/adaptive-curriculum-progress.json",
        "the external curriculum references the CI launcher's progress file");
    schema::Project const legacy = schema::LoadProject(project.root / "project.legacy.json");
    Check(legacy.networkLibrary == "blue_network" && !legacy.networkEditable
            && legacy.Checkpoints() == project.root / "checkpoints"
            && std::filesystem::is_regular_file(legacy.LatestCheckpoint())
            && std::filesystem::is_regular_file(legacy.Policy()),
        "the historical learner and its checkpoints remain available");
    Check(std::filesystem::is_regular_file(project.LatestCheckpoint()) && std::filesystem::is_regular_file(project.Policy())
            && project.LatestCheckpoint().filename() == "latest.ppokit"
            && project.Policy().filename() == "policy.ppokit",
        "BluePatrol resumes from its training artifact and plays its policy artifact, as saved");
    nlohmann::json const settings = nlohmann::json::parse(ReadFile(project.root / "blue_patrol_training.json"));
    nlohmann::json const& training = settings.at("training");
    nlohmann::json const& evaluation = settings.at("evaluation");
    std::optional<schema::ProjectCommandLine> const& train = project.program.train;
    Check(project.program.executable == "ppokit-run" && train && train->executable.empty()
            && train->arguments.size() >= 2 && train->arguments.front() == "train"
            && train->arguments[train->arguments.size() - 2] == "--updates" && train->arguments.back() == "100",
        "BluePatrol can continue from its converted checkpoint with an additional budget");
    std::optional<schema::ProjectCommandLine> const& inference = project.program.inference;
    auto const argument = [&inference](std::string const& value) {
        return inference
            && std::find(inference->arguments.begin(), inference->arguments.end(), value) != inference->arguments.end();
    };
    Check(inference && inference->executable.empty() && !inference->arguments.empty()
            && inference->arguments.front() == "play" && argument("--model") && argument("PROD")
            && argument("--realtime"),
        "the BluePatrol inference plays the production actor with ppokit-run, at a watchable pace");

    schema::TaskSchema const task = schema::LoadTaskSchema(project.Task());
    ppokit::eval::EvaluationSuite const suite =
        ppokit::eval::LoadEvaluationSuite(project.Evaluations() / "validation.json", task.metrics);
    Check(suite.seeds.size() == evaluation.at("episodes").get<std::size_t>()
            && suite.seeds.front() == evaluation.at("seed").get<std::uint32_t>() && suite.episodesPerSeed == 1
            && suite.everyUpdates == training.at("checkpointEvery").get<std::size_t>(),
        "the BluePatrol suite plays the cases of the training file's campaign, at its rhythm");
    Check(suite.gates.empty() && suite.best.size() == 1 && suite.best[0].metric == "score.mean" && suite.best[0].higher,
        "the BEST is the best campaign score, as PPO.exe promoted it");
}

/**
 * @brief A FlightControl3D project under ppokit-run: the FlightControl3D task (plus the world
 *        the Visualiseur reads), the compiled trunk sizes, the PPO values of its scenario in
 *        training.json, the validation of FlightControlHeadless as an evaluation suite, and an
 *        inference that flies at the historical desktop pace until the Studio stops it.
 */
schema::Project CheckFlightProject(std::filesystem::path const& flight, char const* name)
{
    std::filesystem::path const folder = flight / "projects" / name;
#if defined(PPOKIT_FLIGHT_LIBRARIES)
    // The network.json names flight_visualiser: it opens only where FlightControl3D is built.
    schema::Project const project = CheckOpens(folder);
#else
    schema::Project const project = schema::LoadProject(folder);
#endif
    Check(project.environmentLibrary == "flight_environment" && project.networkLibrary == "ppokit_graph_network"
            && project.program.executable == "ppokit-run" && project.environmentRunProfile,
        project.name + " runs under ppokit-run with the FlightControl3D environment and network DLLs");
    nlohmann::json task = nlohmann::json::parse(ReadFile(project.Task()));
    Check(task.contains("world"), project.name + "/task.json has the world of the Visualiseur");
    task.erase("world");
    // The aircraft and contract of the scenario: the T-6 and the PC-7 keep their checkpoints apart.
    nlohmann::json const scenario = nlohmann::json::parse(ReadFile(folder / "scenario.json"));
    Check(task.value("contract", std::string()) == scenario.at("aircraft").get<std::string>() + "@8-centreline1200",
        project.name + "/task.json names the scenario's aircraft and contract");
    task.erase("contract");
    Check(task == nlohmann::json::parse(ReadFile(flight / "task.json")),
        project.name + "/task.json is the FlightControl3D task, plus its world and contract");
    CheckFlightNetwork(project, flight / "graph.json");
    CheckVisualiser(project, "flight_visualiser", "flight_watch@2");
    nlohmann::json const ppo = nlohmann::json::parse(ReadFile(project.Training())).at("ppo");
    nlohmann::json const& training = scenario.at("training");
    std::pair<char const*, char const*> const same[] = {{"gamma", "gamma"}, {"lambda", "lambda"},
        {"rollout_steps", "rolloutSteps"}, {"minibatch", "minibatch"}, {"epochs", "epochs"},
        {"actor_learning_rate", "actorRate"}, {"critic_learning_rate", "criticRate"}, {"clip", "clip"},
        {"target_kl", "targetKl"}, {"entropy_coefficient", "entropy"}, {"seed", "seed"}};
    for (auto const& [ours, theirs] : same)
    {
        Check(ppo.at(ours) == training.at(theirs), project.name + " training.json " + ours + " is the scenario's");
    }

    // The validation of FlightSession: episodes, seeds, success threshold and rhythm of the scenario.
    schema::TaskSchema const parsed = schema::LoadTaskSchema(project.Task());
    nlohmann::json const& evaluation = scenario.at("evaluation");
    ppokit::eval::EvaluationSuite const suite =
        ppokit::eval::LoadEvaluationSuite(project.Evaluations() / "validation.json", parsed.metrics);
    auto const first = evaluation.at("seed").get<std::uint32_t>();
    Check(suite.seeds.size() == evaluation.at("episodes").get<std::size_t>() && suite.seeds.front() == first
            && suite.episodesPerSeed == 1 && suite.everyUpdates == evaluation.at("everyUpdates").get<std::size_t>(),
        project.name + " validation suite plays the scenario's validation episodes and seeds, at its rhythm");
    Check(suite.campaignEnvironment, project.name + ": the validation uses the historical simulator lifetime");
    Check(suite.gates.size() == 1 && suite.gates[0].metric == "stopped.rate"
            && suite.gates[0].threshold == evaluation.at("minimumSuccessRate").get<double>(),
        project.name + ": a campaign passes at the scenario's minimum success rate");
    Check(suite.goal.size() == 1 && suite.goal[0].metric == "difficulty.mean" && suite.goal[0].threshold == 1.0,
        project.name + ": the training is finished on the full task only");
    Check(suite.best.size() == 3 && suite.best[0].metric == "difficulty.mean" && suite.best[2].metric == "stop_m.mean"
            && !suite.best[2].higher,
        project.name + ": the BEST ranks as FlightModelStore: difficulty, stop rate, then shorter stops");

    // The desktop waited 2 ms after each action, which advances 0.1 s of simulated flight.
    nlohmann::json const& mission = scenario.at("mission");
    double const stepSeconds = mission.at("physicsDt").get<double>() * mission.at("actionSubsteps").get<double>();
    std::optional<schema::ProjectCommandLine> const& inference = project.program.inference;
    std::vector<std::string> const desktopPace = {"play", "--project", "{project}", "--realtime",
        "--step-seconds", "0.002", "--best", "--suite", "validation"};
    Check(inference && inference->executable.empty() && inference->arguments == desktopPace
            && std::abs(stepSeconds - 0.1) < 1e-12,
        project.name + " inference uses the historical desktop delay per action");
    return project;
}

/**
 * @brief The T-6 run continues from update 870 (master c8282b8, 1.2 km interception), ported once
 *        from FlightControlHeadless's checkpoint.
 */
void CheckFlightT6(std::filesystem::path const& flight)
{
    schema::Project const project = CheckFlightProject(flight, "T6");
    ppokit::artifacts::ArtifactHeader const header = ppokit::artifacts::ReadArtifactHeader(project.LatestCheckpoint());
#if defined(PPOKIT_FLIGHT_LIBRARIES)
    schema::TaskSchema const parsed = schema::LoadTaskSchema(project.Task());
    schema::BlockGraphSchema const graphs = schema::LoadBlockGraph(project.Network(), parsed);
    bool resumable = true;
    try
    {
        ppokit::artifacts::RequireCompatible(
            header.identity, schema::ArtifactIdentityOf(parsed, graphs), project.LatestCheckpoint().string());
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        resumable = false;
    }
    Check(resumable, "the T6 checkpoint resumes with these networks");
#endif
    Check(header.updates == 870 && std::filesystem::is_regular_file(project.Policy())
            && std::filesystem::is_regular_file(project.Checkpoints() / "ppokit_curriculum_level.txt"),
        "T6 continues its run at update 870 with its curriculum level");
}

/** @brief The PC-7 has the T-6's networks and suite and starts its training from scratch. */
void CheckFlightPc7(std::filesystem::path const& flight)
{
    schema::Project const project = CheckFlightProject(flight, "PC7");
    nlohmann::json const scenario = nlohmann::json::parse(ReadFile(project.root / "scenario.json"));
    Check(scenario.at("aircraft") == "pc7", "PC7 flies the PC-7");
    Check(!std::filesystem::exists(project.LatestCheckpoint()) && !std::filesystem::exists(project.Policy()),
        "PC7 has no checkpoint yet: its first training starts from scratch");
}

/**
 * @brief An Airspace project (RLMission, RLSNA) under ppokit-run: its two libraries, the task
 *        its agent is compiled with, its Visualiseur wired to the world row.
 */
void CheckAirspace(std::filesystem::path const& airspace, std::string const& family, std::string const& agent,
    std::string const& libraries)
{
    std::filesystem::path const folder = airspace / "projects" / family;
#if defined(PPOKIT_AIRSPACE_LIBRARIES)
    // The network.json names airspace_visualiser: it opens only where Airspace's DLLs are built.
    schema::Project const project = schema::LoadProject(folder);
    Check(schema::CheckProject(project).empty(), family + " is complete (no remark)");
    schema::LoadBlockGraph(project.Network(), schema::LoadTaskSchema(project.Task()));
#else
    schema::Project const project = schema::LoadProject(folder);
#endif
    Check(project.environmentLibrary == libraries + "_environment" && project.networkLibrary == "ppokit_graph_network",
        family + " runs on " + libraries + "_environment and ppokit_graph_network");
    Check(ReadFile(project.Task()) == ReadFile(airspace / "Agents" / agent / "task.json"),
        family + "/task.json is the task the " + agent + " agent is compiled with");
    CheckVisualiser(project, "airspace_visualiser", "airspace_watch@1");
}

} // namespace

int main()
{
    try
    {
        // T6 names its Visualiseur like a library of the build, as ppokit-run finds it.
        ppokit::component::SetKnownModules(schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
        std::filesystem::path const repository = PPOTRIAL_REPOSITORY;
        std::filesystem::path const flight = repository / "Examples/FlightControl3D";
        CheckFlightT6(flight);
        CheckFlightPc7(flight);
        std::filesystem::path const blue = repository / "Examples/BlueDefense/Agents/BluePatrol";
#if defined(PPOKIT_BLUE_LIBRARIES)
        // The network.json names blue_visualiser: it opens only where BlueDefense's DLLs are built.
        schema::Project const patrol = CheckOpens(repository / "Examples/BlueDefense/projects/BluePatrol");
#else
        schema::Project const patrol = schema::LoadProject(repository / "Examples/BlueDefense/projects/BluePatrol");
#endif
        CheckBluePatrol(patrol);
        Check(ReadFile(patrol.Task()) == ReadFile(blue / "task.json"),
            "BluePatrol/task.json is the task the Blue agent is compiled with");
        nlohmann::json const network = nlohmann::json::parse(ReadFile(patrol.Network()));
        nlohmann::json const shared = nlohmann::json::parse(ReadFile(
            repository / "Examples/BlueDefense/tests/fixtures/shared_backend/network.json"));
        Check(network.at("actor") == shared.at("actor") && network.at("critic") == shared.at("critic")
                && network.at("team_critic").at("pad_features") == true,
            "BluePatrol uses the parity-tested shared actor and padded team critic");
        CheckVisualiser(patrol, "blue_visualiser", "blue_watch@2");
        Check(std::filesystem::is_regular_file(patrol.root / "blue_patrol_training.json"),
            "BluePatrol keeps its MAPPO training file");
        std::filesystem::path const airspace = repository / "Examples/Airspace";
        CheckAirspace(airspace, "RLMission", "Mission", "rlmission");
        CheckAirspace(airspace, "RLSNA", "Interception", "rlsna");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " repository project check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "repository project checks passed\n";
    return EXIT_SUCCESS;
}
