/**
 * @file PPOKit/src/schema/project.cpp
 * @brief Reading, writing and checking project folders.
 */
#include "ppokit/schema/project.hpp"

#include "json_validation.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace ppokit::schema {
namespace {

using detail::ErrorReporter;
using detail::Json;

constexpr char ProjectSchema[] = "ppokit.project/1";

std::vector<std::string> ReadArgumentList(Json const& value, std::string const& where, ErrorReporter const& errors)
{
    if (!value.is_array())
    {
        errors.Fail(where, "must be a list of arguments, e.g. [\"train\", \"--project\", \"{project}\"]");
    }
    std::vector<std::string> arguments;
    for (Json const& argument : value)
    {
        if (!argument.is_string())
        {
            errors.Fail(where, "every argument must be a string");
        }
        arguments.push_back(argument.get<std::string>());
    }
    return arguments;
}

/** @brief A command: a list of arguments, or {"executable": ..., "arguments": [...]}; nothing when absent. */
std::optional<ProjectCommandLine> ReadCommand(Json const& program, std::string const& key, ErrorReporter const& errors)
{
    if (!program.contains(key))
    {
        return std::nullopt;
    }
    Json const& value = program.at(key);
    std::string const where = "program." + key;
    ProjectCommandLine command;
    if (!value.is_object())
    {
        command.arguments = ReadArgumentList(value, where, errors);
        return command;
    }
    detail::RequireKnownKeys(value, {"executable", "arguments"}, where, errors);
    if (!value.contains("executable") || !value.at("executable").is_string() || value.at("executable").get<std::string>().empty())
    {
        errors.Fail(where + ".executable", "must name the program of this command");
    }
    command.executable = value.at("executable").get<std::string>();
    if (value.contains("arguments"))
    {
        command.arguments = ReadArgumentList(value.at("arguments"), where + ".arguments", errors);
    }
    return command;
}

void WriteCommand(Json& program, char const* key, std::optional<ProjectCommandLine> const& command)
{
    if (!command)
    {
        return;
    }
    if (command->executable.empty())
    {
        program[key] = command->arguments;
        return;
    }
    Json entry;
    entry["executable"] = command->executable;
    entry["arguments"] = command->arguments;
    program[key] = entry;
}

/** @brief A file of the project folder named by `key` ("" when absent); an absolute path is refused. */
std::string RelativeFile(Json const& document, char const* key, char const* example, ErrorReporter const& errors)
{
    if (!document.contains(key))
    {
        return {};
    }
    Json const& value = document.at(key);
    if (!value.is_string() || value.get<std::string>().empty()
        || std::filesystem::path(value.get<std::string>()).is_absolute())
    {
        errors.Fail(key, std::string("must be a file of the project folder, e.g. \"") + example + "\"");
    }
    return value.get<std::string>();
}

/** @brief Upgrade only the retired built-in module names; never redirect custom libraries. */
void MigrateBuiltInNetwork(Project& project)
{
    bool const flight = project.environmentLibrary == "flight_environment"
        && project.networkLibrary == "flight_network"
        && (project.name == "FlightControl3D_T6" || project.name == "FlightControl3D_PC7");
    bool const mission = project.name == "Airspace_RLMission"
        && project.environmentLibrary == "rlmission_environment" && project.networkLibrary == "rlmission_network";
    bool const sna = project.name == "Airspace_RLSNA"
        && project.environmentLibrary == "rlsna_environment" && project.networkLibrary == "rlsna_network";
    if (flight || mission || sna)
    {
        project.networkLibrary = "ppokit_graph_network";
    }
}

/** @brief The command line of `command`, or null when the project does not offer it. */
std::optional<ProjectCommandLine> const& CommandOf(Project const& project, ProjectCommand command) noexcept
{
    if (command == ProjectCommand::Evaluate)
    {
        return project.program.evaluate;
    }
    if (command == ProjectCommand::Inference)
    {
        return project.program.inference;
    }
    return project.program.train;
}

std::string ReplacePlaceholder(std::string argument, std::string const& folder)
{
    std::string_view const placeholder = ProjectPlaceholder;
    for (std::size_t at = argument.find(placeholder); at != std::string::npos;
         at = argument.find(placeholder, at + folder.size()))
    {
        argument.replace(at, placeholder.size(), folder);
    }
    return argument;
}

/** @brief `name` as an executable file name on this platform. */
std::filesystem::path ExecutableName(std::string const& name)
{
    std::filesystem::path path = name;
#if defined(_WIN32)
    if (path.extension().empty())
    {
        path += ".exe";
    }
#endif
    return path;
}

bool IsFile(std::filesystem::path const& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

/** @brief The executable in one of the PATH folders, empty when it is in none. */
std::filesystem::path FindOnPath(std::filesystem::path const& executable)
{
    char const* const variable = std::getenv("PATH");
    if (variable == nullptr)
    {
        return {};
    }
#if defined(_WIN32)
    constexpr char Separator = ';';
#else
    constexpr char Separator = ':';
#endif
    std::string_view folders = variable;
    while (!folders.empty())
    {
        std::size_t const end = folders.find(Separator);
        std::string_view const folder = folders.substr(0, end);
        if (!folder.empty() && IsFile(std::filesystem::path(std::string(folder)) / executable))
        {
            return std::filesystem::path(std::string(folder)) / executable;
        }
        if (end == std::string_view::npos)
        {
            break;
        }
        folders.remove_prefix(end + 1);
    }
    return {};
}

} // namespace

Project LoadProject(std::filesystem::path const& input)
{
    std::error_code error;
    bool const explicitManifest = input.extension() == ".json" && !std::filesystem::is_directory(input, error);
    std::filesystem::path const manifest = explicitManifest ? input : input / "project.json";
    std::filesystem::path root = explicitManifest ? input.parent_path() : input;
    if (root.empty()) root = ".";
    if (!IsFile(manifest))
    {
        if (explicitManifest)
            throw std::runtime_error("project manifest not found: " + manifest.string());
        throw std::runtime_error(root.string() + " is not a PPOKit project: it has no project.json");
    }
    ErrorReporter const errors(manifest.string());
    Json const document = detail::ParseRootObject(detail::ReadTextFile(manifest), errors);
    detail::RequireKnownKeys(
        document, {"schema", "name", "program", "settings", "networkEditable", "journal", "libraries", "checkpoints", "runProfile",
            "nativeCurriculum", "learnerSettings", "controlProtocol", "externalCurriculum", "externalCurriculumProgress"},
        "", errors);
    if (document.value("schema", std::string()) != ProjectSchema)
    {
        errors.Fail("schema", std::string("must be \"") + ProjectSchema + "\"");
    }
    Project project;
    if (document.contains("runProfile"))
    {
        if (document.at("runProfile") != "environment/1")
            errors.Fail("runProfile", "expected environment/1");
        project.environmentRunProfile = true;
    }
    if (document.contains("controlProtocol"))
    {
        auto const& control = document.at("controlProtocol");
        if (control != "udp/1" && control != "legacy-files") errors.Fail("controlProtocol", "expected udp/1 or legacy-files");
        project.controlProtocol = control.get<std::string>();
    }
    project.root = root;
    if (explicitManifest) project.manifestName = manifest.filename().string();
    if (!document.contains("name") || !document.at("name").is_string() || document.at("name").get<std::string>().empty())
    {
        errors.Fail("name", "must name the project");
    }
    project.name = document.at("name").get<std::string>();
    if (document.contains("program"))
    {
        Json const& program = document.at("program");
        detail::RequireObject(program, "program", errors);
        detail::RequireKnownKeys(program, {"executable", "train", "evaluate", "inference"}, "program", errors);
        if (!program.contains("executable") || !program.at("executable").is_string())
        {
            errors.Fail("program.executable", "must name the program that trains the project");
        }
        project.program.executable = program.at("executable").get<std::string>();
        project.program.train = ReadCommand(program, "train", errors);
        project.program.evaluate = ReadCommand(program, "evaluate", errors);
        project.program.inference = ReadCommand(program, "inference", errors);
    }
    if (document.contains("libraries"))
    {
        Json const& libraries = document.at("libraries");
        detail::RequireObject(libraries, "libraries", errors);
        detail::RequireKnownKeys(libraries, {"environment", "network"}, "libraries", errors);
        project.environmentLibrary = libraries.value("environment", std::string());
        project.networkLibrary = libraries.value("network", std::string());
        MigrateBuiltInNetwork(project);
    }
    if (document.contains("checkpoints"))
    {
        Json const& checkpoints = document.at("checkpoints");
        detail::RequireObject(checkpoints, "checkpoints", errors);
        detail::RequireKnownKeys(checkpoints, {"root", "training", "policy", "training_policy"}, "checkpoints", errors);
        project.checkpointRoot = RelativeFile(checkpoints, "root", "checkpoints/shared", errors);
        std::filesystem::path const checkpointRoot(project.checkpointRoot);
        if (!project.checkpointRoot.empty()
            && (checkpointRoot.has_root_name() || checkpointRoot.has_root_directory() || checkpointRoot == "."))
            errors.Fail("checkpoints.root", "must name a folder inside the project");
        for (auto const& part : checkpointRoot)
        {
            if (part == "..") errors.Fail("checkpoints.root", "must stay inside the project folder");
        }
        project.trainingCheckpoint =
            RelativeFile(checkpoints, "training", "checkpoints/latest/training_state.txt", errors);
        project.trainingPolicy = RelativeFile(checkpoints, "training_policy", "checkpoints/latest/policy.ppokit", errors);
        project.policyCheckpoint = RelativeFile(checkpoints, "policy", "checkpoints/latest/policy.txt", errors);
    }
    project.settings = RelativeFile(document, "settings", "scenario.json", errors);
    project.learnerSettings = RelativeFile(document, "learnerSettings", "learner.json", errors);
    project.externalCurriculum = RelativeFile(document, "externalCurriculum", "curriculum.json", errors);
    project.externalCurriculumProgress = RelativeFile(document, "externalCurriculumProgress", "checkpoints/progress.json", errors);
    if (!project.externalCurriculumProgress.empty() && project.externalCurriculum.empty())
        errors.Fail("externalCurriculumProgress", "requires externalCurriculum");
    if (!project.externalCurriculum.empty() && document.contains("nativeCurriculum"))
        errors.Fail("externalCurriculum", "cannot also declare nativeCurriculum");
    if (document.contains("nativeCurriculum"))
    {
        project.curriculumPointer = document.at("nativeCurriculum").get<std::string>();
        if (project.settings.empty() || project.curriculumPointer.empty())
            errors.Fail("nativeCurriculum", "requires a settings file and a nonempty JSON pointer");
        try { (void)Json::json_pointer(project.curriculumPointer); }
        catch (Json::exception const& error) { errors.Fail("nativeCurriculum", error.what()); }
    }
    project.journal = RelativeFile(document, "journal", "checkpoints/latest/metrics.jsonl", errors);
    if (document.contains("networkEditable"))
    {
        if (!document.at("networkEditable").is_boolean())
        {
            errors.Fail("networkEditable", "must be true or false");
        }
        project.networkEditable = document.at("networkEditable").get<bool>();
    }
    return project;
}

void SaveProject(Project const& project)
{
    Json document;
    document["schema"] = ProjectSchema;
    document["name"] = project.name;
    if (!project.curriculumPointer.empty()) document["nativeCurriculum"] = project.curriculumPointer;
    if (!project.externalCurriculum.empty()) document["externalCurriculum"] = project.externalCurriculum;
    if (!project.externalCurriculumProgress.empty()) document["externalCurriculumProgress"] = project.externalCurriculumProgress;
    if (!project.learnerSettings.empty()) document["learnerSettings"] = project.learnerSettings;
    if (project.environmentRunProfile) document["runProfile"] = "environment/1";
    if (!project.controlProtocol.empty()) document["controlProtocol"] = project.controlProtocol;
    if (!project.program.executable.empty())
    {
        Json program;
        program["executable"] = project.program.executable;
        WriteCommand(program, "train", project.program.train);
        WriteCommand(program, "evaluate", project.program.evaluate);
        WriteCommand(program, "inference", project.program.inference);
        document["program"] = program;
    }
    if (!project.settings.empty())
    {
        document["settings"] = project.settings;
    }
    if (!project.networkEditable)
    {
        document["networkEditable"] = false;
    }
    if (!project.journal.empty())
    {
        document["journal"] = project.journal;
    }
    if (!project.checkpointRoot.empty() || !project.trainingCheckpoint.empty() || !project.policyCheckpoint.empty()
        || !project.trainingPolicy.empty())
    {
        Json checkpoints = Json::object();
        if (!project.checkpointRoot.empty()) checkpoints["root"] = project.checkpointRoot;
        if (!project.trainingCheckpoint.empty())
        {
            checkpoints["training"] = project.trainingCheckpoint;
        }
        if (!project.trainingPolicy.empty()) checkpoints["training_policy"] = project.trainingPolicy;
        if (!project.policyCheckpoint.empty())
        {
            checkpoints["policy"] = project.policyCheckpoint;
        }
        document["checkpoints"] = checkpoints;
    }
    if (!project.environmentLibrary.empty() || !project.networkLibrary.empty())
    {
        Json libraries = Json::object();
        if (!project.environmentLibrary.empty())
        {
            libraries["environment"] = project.environmentLibrary;
        }
        if (!project.networkLibrary.empty())
        {
            libraries["network"] = project.networkLibrary;
        }
        document["libraries"] = libraries;
    }
    artifacts::WriteFileAtomically(
        project.Manifest(), [&document](std::ostream& stream) { stream << document.dump(2) << '\n'; },
        artifacts::AtomicWriteOptions{}, "PPOKit project");
}

bool IsProject(std::filesystem::path const& folder)
{
    return IsFile(folder / "project.json");
}

std::optional<std::filesystem::path> Project::SettingsFile() const
{
    if (settings.empty())
    {
        return std::nullopt;
    }
    return root / settings;
}

std::optional<std::filesystem::path> ProjectFolderOf(std::filesystem::path const& settingsFile)
{
    std::filesystem::path const folder = settingsFile.parent_path();
    if (folder.empty() || !IsProject(folder))
    {
        return std::nullopt;
    }
    return folder;
}

std::optional<std::vector<std::string>> CommandArguments(Project const& project, ProjectCommand command)
{
    std::optional<ProjectCommandLine> const& line = CommandOf(project, command);
    if (!line || CommandExecutable(project, command).empty())
    {
        return std::nullopt;
    }
    std::string const folder = (project.manifestName.empty() ? project.root : project.Manifest()).string();
    std::vector<std::string> replaced;
    for (std::string const& argument : line->arguments)
    {
        replaced.push_back(ReplacePlaceholder(argument, folder));
    }
    return replaced;
}

CommandPolicy DeclaredPolicy(Project const& project, ProjectCommand command)
{
    CommandPolicy policy;
    if (command == ProjectCommand::Train || std::filesystem::path(CommandExecutable(project, command)).stem() != "ppokit-run")
        return policy;
    auto const args = CommandArguments(project, command);
    if (!args) return policy;
    for (std::size_t index = 1; index < args->size(); ++index)
    {
        auto const& flag = args->at(index);
        if (flag != "--best" && flag != "--model" && flag != "--policy" && flag != "--suite") continue;
        if (flag != "--suite" && (!policy.role.empty() || !policy.file.empty()))
            throw std::runtime_error("project.json: conflicting policy selections");
        if (flag == "--best") { policy.role = "BEST"; continue; }
        if (++index == args->size() || args->at(index).empty())
            throw std::runtime_error("project.json: missing value for " + flag);
        auto const& value = args->at(index);
        if (flag == "--suite") policy.suite = value;
        else if (flag == "--model") policy.role = value;
        else policy.file = value;
    }
    return policy;
}

std::string CommandExecutable(Project const& project, ProjectCommand command)
{
    std::optional<ProjectCommandLine> const& line = CommandOf(project, command);
    if (!line)
    {
        return {};
    }
    if (!line->executable.empty())
    {
        return line->executable;
    }
    return project.program.executable;
}

char const* CommandName(ProjectCommand command) noexcept
{
    switch (command)
    {
    case ProjectCommand::Train:
        return "Train";
    case ProjectCommand::Evaluate:
        return "Evaluate";
    case ProjectCommand::Inference:
        return "Inference";
    }
    return "?";
}

std::filesystem::path FindProgram(Project const& project, ProjectCommand command,
    std::vector<std::filesystem::path> const& knownPrograms)
{
    std::string const name = CommandExecutable(project, command);
    if (name.empty())
    {
        return {};
    }
    std::filesystem::path const executable = ExecutableName(name);
    if (executable.is_absolute())
    {
        return IsFile(executable) ? executable : std::filesystem::path{};
    }
    if (IsFile(project.root / executable))
    {
        return project.root / executable;
    }
    for (std::filesystem::path const& known : knownPrograms)
    {
        if (known.filename() == executable.filename() && IsFile(known))
        {
            return known;
        }
    }
    return executable.has_parent_path() ? std::filesystem::path{} : FindOnPath(executable);
}

namespace {

/** @brief Name of a library file without its "lib" prefix and its extension. */
std::string LibraryName(std::filesystem::path const& file)
{
    std::string name = file.stem().string();
    if (name.rfind("lib", 0) == 0)
    {
        name.erase(0, 3);
    }
    return name;
}

} // namespace

std::filesystem::path FindLibrary(Project const& project, std::string const& name,
    std::vector<std::filesystem::path> const& knownPrograms)
{
    std::filesystem::path const written(name);
    if (written.is_absolute())
    {
        return IsFile(written) ? written : std::filesystem::path{};
    }
    if (IsFile(project.root / written))
    {
        return project.root / written;
    }
    for (std::filesystem::path const& known : knownPrograms)
    {
        if (LibraryName(known) == name && IsFile(known))
        {
            return known;
        }
    }
    return {};
}

std::vector<std::filesystem::path> ReadProgramRegistry(std::filesystem::path const& registry)
{
    std::vector<std::filesystem::path> programs;
    std::ifstream stream(registry);
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            programs.emplace_back(line);
        }
    }
    return programs;
}

std::vector<std::string> CheckProject(Project const& project)
{
    std::vector<std::string> remarks;
    if (!IsFile(project.Task()))
    {
        remarks.emplace_back("task.json is missing: the observation, actions and rewards are not described.");
    }
    if (!IsFile(project.Network()))
    {
        remarks.emplace_back("network.json is missing: there is no actor and critic to train.");
    }
    std::optional<std::filesystem::path> const settings = project.SettingsFile();
    if (settings && !IsFile(*settings))
    {
        remarks.emplace_back(project.settings + " is missing: the program has no settings to start from.");
    }
    if (!project.externalCurriculum.empty() && !IsFile(project.root / project.externalCurriculum))
        remarks.emplace_back(project.externalCurriculum + " is missing: the external curriculum cannot be inspected.");
    if (!project.learnerSettings.empty() && !IsFile(project.root / project.learnerSettings))
        remarks.emplace_back(project.learnerSettings + " is missing: the native learner settings cannot be read.");
    // A program with its own settings file takes its PPO parameters and evaluation from there.
    if (!settings && !IsFile(project.Training()))
    {
        remarks.emplace_back("training.json is missing: training uses the default PPO parameters.");
    }
    std::error_code error;
    bool suite = false;
    for (auto const& entry : std::filesystem::directory_iterator(project.Evaluations(), error))
    {
        suite = suite || entry.path().extension() == ".json";
    }
    if (!settings && !suite)
    {
        remarks.emplace_back("eval/ has no evaluation suite: Evaluate has nothing to check.");
    }
    if (project.program.executable.empty())
    {
        remarks.emplace_back("project.json names no program: the project can be edited, not trained.");
    }
    return remarks;
}

} // namespace ppokit::schema
