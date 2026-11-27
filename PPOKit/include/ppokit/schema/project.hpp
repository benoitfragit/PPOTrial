/**
 * @file PPOKit/include/ppokit/schema/project.hpp
 * @brief A PPOKit project: one folder that holds everything of a trained agent.
 * @details
 * @verbatim
 *   my_project/
 *     project.json          name, and the program that trains, evaluates and runs it
 *     task.json             observation, actions, reward channels, scenario parameters
 *     network.json          actor and critic (network.layout.json: where the blocks are drawn)
 *     training.json         PPO and GAE parameters, probes, curriculum
 *     eval/                 evaluation suites (*.json)
 *     components/           component DLLs the network uses
 *     checkpoints/          latest.ppokit (resume), policy.ppokit (evaluate), metrics.jsonl (journal)
 * @endverbatim
 *
 * project.json:
 * @code{.json}
 * {
 *   "schema": "ppokit.project/1",
 *   "name": "reach",
 *   "program": {
 *     "executable": "ppokit_reach",
 *     "train": ["train", "--project", "{project}"],
 *     "evaluate": ["eval", "--project", "{project}"],
 *     "inference": ["play", "--project", "{project}"]
 *   }
 * }
 * @endcode
 * "{project}" stands for the project folder. The executable is looked up in the project
 * folder, then among the programs the build registered (ppokit_register_program), then on
 * the PATH. A command left out is not offered. A command may run another program, e.g. a
 * windowed one for inference: `"inference": {"executable": "FlightControl3D", "arguments": [...]}`.
 *
 * Programs that existed before PPOKit keep their own settings file in the project:
 * `"settings": "scenario.json"`. Their network is compiled into them: `"networkEditable": false`
 * shows network.json read-only; `"journal": "checkpoints/latest/metrics.jsonl"` says where it
 * writes its run journal. A program finds its project from its settings file
 * (ProjectFolderOf) and then keeps its checkpoints in checkpoints/.
 */
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::schema {

/** @brief Placeholder of the project folder in program arguments. */
inline constexpr char ProjectPlaceholder[] = "{project}";

/** @brief What the studio can launch for a project. */
enum class ProjectCommand
{
    Train,
    Evaluate,
    Inference,
};

/** @brief One command: its arguments, and its own executable when it is not the project's. */
struct ProjectCommandLine final
{
    /** @brief Empty: the program's executable. */
    std::string executable;
    std::vector<std::string> arguments;
};

/** @brief The program of a project and its command lines. */
struct ProjectProgram final
{
    /** @brief Executable name or path ("ppokit_reach", "bin/FlightControlHeadless"). */
    std::string executable;
    std::optional<ProjectCommandLine> train;
    std::optional<ProjectCommandLine> evaluate;
    std::optional<ProjectCommandLine> inference;
};

/** @brief A project folder and its project.json. */
struct Project final
{
    std::filesystem::path root;
    /** @brief Explicit manifest file name, empty for project.json. */
    std::string manifestName;
    std::string name;
    ProjectProgram program;
    /** @brief The program's own settings file, relative to the folder (e.g. "scenario.json"); empty when none. */
    std::string settings;
    /** @brief False when the program compiles its network: network.json is then shown read-only. */
    bool networkEditable = true;
    /** @brief Run journal relative to the folder when the program writes it elsewhere than checkpoints/. */
    std::string journal;
    /**
     * @brief Libraries run by ppokit-run ("libraries" of project.json): the example's
     *        environment DLL and the networks' DLL, by name or path; empty when unused.
     */
    std::string environmentLibrary;
    std::string networkLibrary;
    /** @brief The environment supplies the versioned native training and inference profile. */
    bool environmentRunProfile = false;
    /** @brief Optional process supervision for a custom executable: udp/1 or legacy-files. */
    std::string controlProtocol;
    /** @brief JSON pointer into SettingsFile() for a provider-owned curriculum, if declared. */
    std::string curriculumPointer;
    /** @brief Original external curriculum JSON, relative to the project; owned by its launcher. */
    std::string externalCurriculum;
    /** @brief Optional external launcher's progress JSON, relative to the project. */
    std::string externalCurriculumProgress;
    /** @brief Config file used by a native learner in place of training.json. */
    std::string learnerSettings;
    /**
     * @brief Checkpoint files relative to the folder ("checkpoints" of project.json), when
     *        the networks keep their own file names; empty for checkpoints/latest.ppokit and
     *        checkpoints/policy.ppokit.
     */
    std::string trainingCheckpoint;
    std::string policyCheckpoint;
    /** @brief Run-state directory relative to root, empty for checkpoints/. */
    std::string checkpointRoot;
    /** @brief A separate writable policy when Policy() is a deployed evaluation reference. */
    std::string trainingPolicy;

    std::filesystem::path Manifest() const { return root / (manifestName.empty() ? "project.json" : manifestName); }
    std::filesystem::path Task() const { return root / "task.json"; }
    std::filesystem::path Network() const { return root / "network.json"; }
    std::filesystem::path NetworkLayout() const { return root / "network.layout.json"; }
    std::filesystem::path Training() const { return root / "training.json"; }
    std::filesystem::path Evaluations() const { return root / "eval"; }
    std::filesystem::path Components() const { return root / "components"; }
    std::filesystem::path Checkpoints() const { return root / (checkpointRoot.empty() ? "checkpoints" : checkpointRoot); }
    /** @brief The resumable training state. */
    std::filesystem::path LatestCheckpoint() const
    {
        return trainingCheckpoint.empty() ? Checkpoints() / "latest.ppokit" : root / trainingCheckpoint;
    }
    /** @brief The policy of the last save. */
    std::filesystem::path Policy() const
    {
        return policyCheckpoint.empty() ? Checkpoints() / "policy.ppokit" : root / policyCheckpoint;
    }
    std::filesystem::path Journal() const
    {
        return journal.empty() ? Checkpoints() / "metrics.jsonl" : root / journal;
    }
    std::filesystem::path TrainingPolicy() const { return trainingPolicy.empty() ? Policy() : root / trainingPolicy; }
    /** @brief checkpoints/stop.request: the program saves a checkpoint and stops when it appears. */
    std::filesystem::path StopRequest() const { return Checkpoints() / "stop.request"; }
    /** @brief The program's own settings file, or nothing. */
    std::optional<std::filesystem::path> SettingsFile() const;
    /** @brief A provider-owned subtree or an external curriculum source was declared. */
    bool HasDeclaredCurriculum() const noexcept
    {
        return !curriculumPointer.empty() || !externalCurriculum.empty();
    }
};

/**
 * @brief Reads a project folder or an explicitly named JSON manifest in that folder.
 * @details Retired bare network module names of the four built-in graph projects are
 *          normalized to ppokit_graph_network. Explicit custom paths are never rewritten.
 * @throws std::runtime_error naming the folder when it is not a project, or the field that is wrong.
 */
Project LoadProject(std::filesystem::path const& root);

/** @brief Writes project.json of `project` (atomically). */
void SaveProject(Project const& project);

/** @brief Whether `folder` holds a project.json. */
bool IsProject(std::filesystem::path const& folder);

/**
 * @brief The project folder that holds `settingsFile` (a program's settings file), or nothing.
 * @details Lets a program started with `--config <project>/scenario.json` keep its checkpoints
 *          in the project instead of its usual model folder.
 */
std::optional<std::filesystem::path> ProjectFolderOf(std::filesystem::path const& settingsFile);

/** @brief Arguments of `command` with {project} replaced, or nothing when the project does not offer it. */
std::optional<std::vector<std::string>> CommandArguments(Project const& project, ProjectCommand command);

struct CommandPolicy final
{
    std::string role;
    std::filesystem::path file;
    std::string suite;
};

/** @brief The policy selector on a declared ppokit-run command. Empty fields select the project actor. */
CommandPolicy DeclaredPolicy(Project const& project, ProjectCommand command);

/** @brief "Train", "Evaluate", "Inference". */
char const* CommandName(ProjectCommand command) noexcept;

/** @brief Executable that runs `command`: its own, else the program's (empty when not offered). */
std::string CommandExecutable(Project const& project, ProjectCommand command);

/**
 * @brief Resolves the executable of `command` to an absolute path.
 * @details Explicit paths are only resolved from the project. Bare names are searched in
 *          the project, then the registry, then PATH. An ambiguous registry or PATH lookup
 *          is rejected instead of selecting an arbitrary old binary.
 * @return Absolute executable path, or an empty path when absent or ambiguous.
 */
std::filesystem::path FindProgram(Project const& project, ProjectCommand command,
    std::vector<std::filesystem::path> const& knownPrograms);

/** @brief Program paths of a registry file, relative entries resolved beside it. */
std::vector<std::filesystem::path> ReadProgramRegistry(std::filesystem::path const& registry);

/**
 * @brief The file of a library named in project.json: a path, a file of the project folder,
 *        or a library the build registered whose name is `name` (prefix "lib" and extension
 *        ignored).
 * @return Absolute path, or empty when none is found or a bare name is ambiguous.
 */
std::filesystem::path FindLibrary(Project const& project, std::string const& name,
    std::vector<std::filesystem::path> const& knownPrograms);

/**
 * @brief What is missing or unusual in a project, in plain words (empty when complete).
 * @details Missing task.json, network.json or training.json; no evaluation suite; a
 * program that cannot be found is reported by the caller, which knows the studio folder.
 */
std::vector<std::string> CheckProject(Project const& project);

} // namespace ppokit::schema
