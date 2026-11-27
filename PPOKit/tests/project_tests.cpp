/**
 * @file PPOKit/tests/project_tests.cpp
 * @brief Project folders: project.json round trip, commands with the folder substituted,
 *        programs found in the folder, the registry or not at all, and plain-word checks.
 */
#include "ppokit/schema/project.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace ppokit;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

void WriteText(std::filesystem::path const& path, std::string const& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

std::string LoadError(std::filesystem::path const& folder)
{
    try
    {
        schema::LoadProject(folder);
    }
    catch (std::exception const& error)
    {
        return error.what();
    }
    return {};
}

#if defined(_WIN32)
constexpr char ProgramName[] = "trainer.exe";
constexpr char ViewerName[] = "viewer.exe";
#else
constexpr char ProgramName[] = "trainer";
constexpr char ViewerName[] = "viewer";
#endif

} // namespace

int main()
{
    std::filesystem::path const root = std::filesystem::temp_directory_path() / "ppokit_project_tests";
    std::error_code removed;
    std::filesystem::remove_all(root, removed);
    std::filesystem::path const folder = root / "lander";
    std::filesystem::create_directories(folder);

    Check(!schema::IsProject(folder) && LoadError(folder).find("has no project.json") != std::string::npos,
        "a folder without project.json is not a project, and the error says why");

    schema::Project project;
    project.root = folder;
    project.name = "lander";
    project.program.executable = "trainer";
    project.program.train = schema::ProjectCommandLine{{}, {"train", "--project", "{project}"}};
    project.program.evaluate = schema::ProjectCommandLine{{}, {"eval", "--project={project}"}};
    schema::SaveProject(project);
    Check(schema::IsProject(folder), "a saved project is a project");

    schema::Project const loaded = schema::LoadProject(folder);
    Check(loaded.name == "lander" && loaded.program.executable == "trainer" && loaded.program.train
            && !loaded.program.inference,
        "project.json round-trips, and a command left out stays absent");
    Check(loaded.Network() == folder / "network.json" && loaded.Journal() == folder / "checkpoints" / "metrics.jsonl",
        "the project names its files");

    // An explicit manifest can keep an independent run beside the default project.
    schema::Project alternate = loaded;
    alternate.manifestName = "project.legacy.json";
    alternate.name = "legacy";
    alternate.checkpointRoot = "checkpoints/legacy";
    schema::SaveProject(alternate);
    auto const explicitProject = schema::LoadProject(folder / "project.legacy.json");
    Check(explicitProject.name == "legacy" && explicitProject.root == folder
            && explicitProject.Manifest() == folder / "project.legacy.json"
            && explicitProject.Checkpoints() == folder / "checkpoints/legacy"
            && explicitProject.LatestCheckpoint() == folder / "checkpoints/legacy/latest.ppokit"
            && explicitProject.Journal() == folder / "checkpoints/legacy/metrics.jsonl",
        "an explicit manifest resolves its own run files from the same project folder");
    Check(schema::LoadProject(folder).name == "lander", "saving the alternate manifest preserves the default");
    Check(LoadError(folder / "missing.json").find("manifest not found") != std::string::npos,
        "a missing explicit manifest never falls back to project.json");
    WriteText(folder / "invalid-root.json",
        R"({"schema":"ppokit.project/1","name":"invalid","checkpoints":{"root":"../outside"}})");
    Check(LoadError(folder / "invalid-root.json").find("checkpoints.root") != std::string::npos,
        "a checkpoint root cannot escape its project");
    WriteText(folder / "rooted.json",
        R"({"schema":"ppokit.project/1","name":"invalid","checkpoints":{"root":"/outside"}})");
    Check(LoadError(folder / "rooted.json").find("root") != std::string::npos,
        "a rooted checkpoint path cannot escape its project");
#if defined(_WIN32)
    WriteText(folder / "drive-root.json",
        R"({"schema":"ppokit.project/1","name":"invalid","checkpoints":{"root":"C:outside"}})");
    Check(LoadError(folder / "drive-root.json").find("checkpoints.root") != std::string::npos,
        "a drive-relative checkpoint root cannot escape its project");
#endif
    auto const alternateTrain = schema::CommandArguments(explicitProject, schema::ProjectCommand::Train);
    Check(alternateTrain && (*alternateTrain)[2] == (folder / "project.legacy.json").string(),
        "commands preserve an explicitly selected manifest");

    auto const train = schema::CommandArguments(loaded, schema::ProjectCommand::Train);
    Check(train && train->size() == 3 && (*train)[2] == folder.string(), "{project} becomes the folder");
    auto const evaluate = schema::CommandArguments(loaded, schema::ProjectCommand::Evaluate);
    Check(evaluate && (*evaluate)[1] == "--project=" + folder.string(), "{project} is replaced inside an argument");
    Check(!schema::CommandArguments(loaded, schema::ProjectCommand::Inference), "a missing command is not offered");

    Check(schema::FindProgram(loaded, schema::ProjectCommand::Train, {}).empty(), "a program found nowhere gives an empty path");
    std::filesystem::path const registered = root / "build" / ProgramName;
    WriteText(registered, "");
    WriteText(root / "registry.txt", (root / "other").string() + "\r\n" + registered.string() + "\n\n");
    auto const known = schema::ReadProgramRegistry(root / "registry.txt");
    Check(known.size() == 2, "the registry lists one program per line");
    Check(schema::FindProgram(loaded, schema::ProjectCommand::Train, known) == registered, "a registered program is found by name");
    WriteText(folder / ProgramName, "");
    Check(schema::FindProgram(loaded, schema::ProjectCommand::Train, known) == folder / ProgramName, "a program in the project folder comes first");
    Check(schema::ReadProgramRegistry(root / "missing.txt").empty(), "no registry, no program");

    auto const remarks = schema::CheckProject(loaded);
    Check(remarks.size() == 4, "missing task, network, training and evaluation are reported");
    WriteText(folder / "task.json", "{}");
    WriteText(folder / "network.json", "{}");
    WriteText(folder / "training.json", "{}");
    WriteText(folder / "eval" / "smoke.json", "{}");
    Check(schema::CheckProject(loaded).empty(), "a complete project has no remark");

    // A program of its own: settings file, compiled network, a windowed inference program.
    schema::Project legacy = loaded;
    legacy.settings = "scenario.json";
    legacy.networkEditable = false;
    legacy.journal = "checkpoints/latest/metrics.jsonl";
    legacy.program.inference = schema::ProjectCommandLine{"viewer", {"--project", "{project}"}};
    schema::SaveProject(legacy);
    schema::Project const reloaded = schema::LoadProject(folder);
    Check(reloaded.settings == "scenario.json" && !reloaded.networkEditable
            && reloaded.SettingsFile() == folder / "scenario.json"
            && reloaded.Journal() == folder / "checkpoints/latest/metrics.jsonl",
        "settings, networkEditable and journal round-trip");
    Check(reloaded.LatestCheckpoint() == folder / "checkpoints" / "latest.ppokit"
            && reloaded.Policy() == folder / "checkpoints" / "policy.ppokit",
        "without \"checkpoints\", the checkpoints are latest.ppokit and policy.ppokit");
    // Networks that keep their own file names (a team DLL's text checkpoints).
    schema::Project named = reloaded;
    named.trainingCheckpoint = "checkpoints/latest/training_v7.txt";
    named.policyCheckpoint = "checkpoints/latest/actor_v10.txt";
    schema::SaveProject(named);
    schema::Project const renamed = schema::LoadProject(folder);
    Check(renamed.LatestCheckpoint() == folder / "checkpoints/latest/training_v7.txt"
            && renamed.Policy() == folder / "checkpoints/latest/actor_v10.txt",
        "\"checkpoints\" names the training state and the policy, and round-trips");
    schema::SaveProject(reloaded);
    Check(schema::CommandExecutable(reloaded, schema::ProjectCommand::Inference) == "viewer"
            && schema::CommandExecutable(reloaded, schema::ProjectCommand::Train) == "trainer",
        "a command can run its own program; the others run the project's");
    auto const inference = schema::CommandArguments(reloaded, schema::ProjectCommand::Inference);
    Check(inference && inference->size() == 2 && (*inference)[1] == folder.string(), "its arguments are replaced too");
    WriteText(folder / ViewerName, "");
    Check(schema::FindProgram(reloaded, schema::ProjectCommand::Inference, known) == folder / ViewerName,
        "the program of a command is found by its own name");
    Check(schema::ProjectFolderOf(folder / "scenario.json") == folder, "a settings file in a project names it");
    Check(!schema::ProjectFolderOf(root / "scenario.json"), "a settings file elsewhere has no project");
    Check(schema::CheckProject(reloaded).size() == 1 && schema::CheckProject(reloaded)[0].find("scenario.json") == 0,
        "a missing settings file is reported, and training.json and eval/ are not asked for");
    WriteText(folder / "scenario.json", "{}");
    Check(schema::CheckProject(reloaded).empty(), "with its settings file, such a project is complete");

    // Old exported projects still refer to the retired per-project graph libraries.
    struct LegacyNetwork
    {
        char const* name;
        char const* environment;
        char const* network;
    };
    for (auto const& old : {LegacyNetwork{"FlightControl3D_T6", "flight_environment", "flight_network"},
             LegacyNetwork{"FlightControl3D_PC7", "flight_environment", "flight_network"},
             LegacyNetwork{"Airspace_RLMission", "rlmission_environment", "rlmission_network"},
             LegacyNetwork{"Airspace_RLSNA", "rlsna_environment", "rlsna_network"}})
    {
        schema::Project exported = loaded;
        exported.name = old.name;
        exported.environmentLibrary = old.environment;
        exported.networkLibrary = old.network;
        exported.trainingCheckpoint = "models/existing.ppokit";
        exported.policyCheckpoint = "models/production.ppokit";
        schema::SaveProject(exported);
        auto const migrated = schema::LoadProject(folder);
        Check(migrated.networkLibrary == "ppokit_graph_network"
                && migrated.LatestCheckpoint() == exported.LatestCheckpoint()
                && migrated.Policy() == exported.Policy(),
            "retired library is migrated without changing checkpoint paths");
        exported.networkLibrary = std::string("custom/") + old.network;
        schema::SaveProject(exported);
        Check(schema::LoadProject(folder).networkLibrary == exported.networkLibrary, "explicit custom path stays intact");
        exported.name = "third_party";
        exported.networkLibrary = old.network;
        schema::SaveProject(exported);
        Check(schema::LoadProject(folder).networkLibrary == exported.networkLibrary, "unrelated project stays intact");
    }

    schema::Project external = loaded;
    external.externalCurriculum = "../curriculum.json";
    external.externalCurriculumProgress = "checkpoints/progress.json";
    schema::SaveProject(external);
    auto const roundTrip = schema::LoadProject(folder);
    Check(roundTrip.HasDeclaredCurriculum() && roundTrip.externalCurriculum == external.externalCurriculum
        && roundTrip.externalCurriculumProgress == external.externalCurriculumProgress,
        "external curriculum paths round-trip without copying the source");
    external.curriculumPointer = "/curriculum";
    external.settings = "scenario.json";
    schema::SaveProject(external);
    Check(LoadError(folder).find("cannot also declare nativeCurriculum") != std::string::npos,
        "a project cannot ambiguously select both curriculum sources");

    WriteText(folder / "project.json", R"({"schema": "ppokit.project/1", "name": "x", "program": {"executable": 3}})");
    Check(LoadError(folder).find("program.executable") != std::string::npos, "a wrong field is named");
    WriteText(folder / "project.json", R"({"schema": "ppokit.project/1", "name": "x", "colour": "blue"})");
    Check(LoadError(folder).find("colour") != std::string::npos, "an unknown key is refused");
    WriteText(folder / "project.json",
        R"({"schema": "ppokit.project/1", "name": "x", "program": {"executable": "t", "train": {"arguments": []}}})");
    Check(LoadError(folder).find("program.train.executable") != std::string::npos,
        "a command written as an object names its program");

    std::filesystem::remove_all(root, removed);
    if (failures != 0)
    {
        std::cerr << failures << " project check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "project checks passed\n";
    return EXIT_SUCCESS;
}
