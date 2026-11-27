/**
 * @file PPOKit/studio/tests/project_window_tests.cpp
 * @brief The studio on a copy of the reach project, headless: every section draws, overview
 *        clicks open what they summarize, Train / Evaluate / Inference start the real program
 *        and its console arrives, Stop ends training cleanly, changes are saved before a
 *        launch, a missing program is explained, a project run by its own program keeps its
 *        settings, a new project is created and opened.
 */
#include "studio_app.hpp"
#include "launch_settings.hpp"
#include "studio_theme.hpp"

#include "ppokit/runtime/publication.hpp"
#include "ppokit/schema/project.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <imnodes.h>
#include <implot.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

/** @brief A JSON file, its keys in the order of the file. */
nlohmann::ordered_json ReadOrderedJson(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    return nlohmann::ordered_json::parse(stream);
}

void WriteOrderedJson(std::filesystem::path const& path, nlohmann::ordered_json const& json)
{
    std::ofstream(path) << json.dump(2) << '\n';
}

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

template<typename View>
void Frame(View& view, std::string* renderedText = nullptr)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.F, 0.F));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("studio", nullptr, ImGuiWindowFlags_NoDecoration);
    if (renderedText) ImGui::LogToBuffer(2);
    view.Draw();
    if (renderedText)
    {
        *renderedText = ImGui::GetCurrentContext()->LogBuffer.c_str();
        ImGui::LogFinish();
    }
    ImGui::End();
    ImGui::Render();
}

/** @brief A left click at `point`: press in one frame, release in the next. */
template<typename View>
void Click(View& view, ImVec2 point)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(point.x, point.y);
    Frame(view);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    Frame(view);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    Frame(view);
    io.AddMousePosEvent(-1.F, -1.F);
    Frame(view);
}

bool ConsoleHas(studio::ProjectWindow const& window, std::string const& part)
{
    for (std::string const& line : window.ConsoleLines())
    {
        if (line.find(part) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

/** @brief Draws frames until the program ends (at most 60 s). */
void WaitForProgram(studio::ProjectWindow& window)
{
    for (int frame = 0; frame < 6000 && window.Running(); ++frame)
    {
        Frame(window);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Frame(window);
}

/** @brief A copy of the reach project whose commands are short (3 updates, 2 episodes). */
std::filesystem::path CopyReach(std::filesystem::path const& into)
{
    std::filesystem::path const source = PPOKIT_REACH_PROJECT;
    std::filesystem::create_directories(into / "eval");
    for (char const* file : {"task.json", "network.json", "training.json", "eval/smoke.json"})
    {
        std::filesystem::copy_file(source / file, into / file, std::filesystem::copy_options::overwrite_existing);
    }
    schema::Project project;
    project.root = into;
    project.name = "reach";
    project.controlProtocol = "udp/1";
    project.program.executable = "ppokit_reach";
    project.program.train = schema::ProjectCommandLine{{}, {"train", "--project", "{project}", "--updates", "3"}};
    project.program.evaluate = schema::ProjectCommandLine{{}, {"eval", "--project", "{project}"}};
    project.program.inference = schema::ProjectCommandLine{{}, {"play", "--project", "{project}", "--episodes", "2"}};
    schema::SaveProject(project);
    return into;
}

/** @brief The settings that the user chooses reach the actual program arguments. */
void LaunchChoices()
{
    using Command = schema::ProjectCommand;
    studio::LaunchSettings choices;
    std::vector<std::string> const train{"train", "--fresh", "--updates", "60", "--save-every", "3"};
    Check(choices.Arguments(Command::Train, train) == train, "project defaults preserve the declared command");
    choices.start = studio::LaunchSettings::Start::Resume;
    choices.budget = studio::LaunchSettings::Budget::Absolute;
    choices.updates = 75;
    Check(choices.Arguments(Command::Train, train) ==
        std::vector<std::string>{"train", "--save-every", "3", "--resume", "--until-update", "75"},
        "explicit resume and absolute budget replace only inherited fields");
    choices.inference.role = "EVAL";
    Check(choices.Arguments(Command::Inference, {"play", "--best", "--realtime", "--suite", "validation"}) ==
        std::vector<std::string>{"play", "--realtime", "--suite", "validation", "--model", "EVAL"},
        "an explicit policy retains the pacing and suite");
    Check(choices.Arguments(Command::Evaluate, {"eval", "--best"}) == std::vector<std::string>{"eval", "--best"},
        "overriding inference preserves the independent evaluation choice");
    choices.evaluation.role = "CUSTOM";
    bool rejected = false;
    try { (void)choices.Arguments(Command::Evaluate, {"eval"}); }
    catch (std::runtime_error const&) { rejected = true; }
    Check(rejected, "an empty custom file cannot launch a different policy by accident");
}

/** @brief Headless host exposing the launch button's position for interaction tests. */
struct LaunchSettingsHost
{
    studio::LaunchSettings settings;
    schema::Project project;
    ImVec2 buttonCenter{};

    void Draw()
    {
        ImVec2 const topLeft = ImGui::GetCursorScreenPos();
        buttonCenter = ImVec2(topLeft.x + ImGui::CalcTextSize("Launch settings...").x * 0.5F
                + ImGui::GetStyle().FramePadding.x,
            topLeft.y + ImGui::GetFrameHeight() * 0.5F);
        settings.Draw(project, {});
    }
};

/** @brief The launch panel fits a small window, closes, and retains its selections. */
void LaunchSettingsPanel(std::filesystem::path const& folder)
{
    LaunchSettingsHost host;
    host.project.root = folder;
    host.settings.start = studio::LaunchSettings::Start::Resume;
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 const previousSize = io.DisplaySize;
    io.DisplaySize = ImVec2(640.F, 480.F);
    Frame(host);
    Click(host, host.buttonCenter);
    Frame(host);
    ImGuiWindow const* popup = nullptr;
    for (ImGuiWindow const* window : ImGui::GetCurrentContext()->Windows)
    {
        if (window->Active && !window->Hidden && (window->Flags & ImGuiWindowFlags_Popup) != 0)
            popup = window;
    }
    Check(popup != nullptr, "launch settings open as a popup");
    if (popup)
    {
        Check(popup->Size.x <= io.DisplaySize.x * 0.9F + 1.F
                && popup->Size.y <= io.DisplaySize.y * 0.8F + 1.F,
            "launch settings fit the small viewport");
        Check(popup->ScrollMax.y > 0.F, "long launch settings remain accessible by scrolling");
    }
    Click(host, ImVec2(630.F, 470.F));
    Frame(host);
    Check(ImGui::GetCurrentContext()->OpenPopupStack.empty(), "clicking outside closes the launch panel");
    Check(host.settings.start == studio::LaunchSettings::Start::Resume,
        "closing the launch panel retains its choices");
    io.DisplaySize = previousSize;
}

/** @brief A real ppokit-run receives Stop even while its Studio session is negotiating. */
void SupervisedProject(std::filesystem::path const& folder)
{
    CopyReach(folder);
    auto task = ReadOrderedJson(folder / "task.json");
    task["world"] = {{"agent", {{"fields", {"agent_x", "agent_y"}}}}, {"target", {{"fields", {"target_x", "target_y"}}}}};
    task["metrics"] = {{"reached", {{"kind", "flag"}}}, {"distance", {{"kind", "number"}}}};
    WriteOrderedJson(folder / "task.json", task);
    auto project = schema::LoadProject(folder);
    project.environmentLibrary = PPOKIT_REACH_ENVIRONMENT;
    project.networkLibrary = PPOKIT_REACH_NETWORK;
    project.program.executable = "ppokit-run";
    project.program.train = schema::ProjectCommandLine{{}, {"train", "--project", "{project}", "--updates", "100000"}};
    schema::SaveProject(project);
    studio::ProjectWindow window(folder, {PPOKIT_RUN_PROGRAM});
    Check(window.Launch(schema::ProjectCommand::Train), "Studio starts its UDP-supervised generic runner");
    Frame(window);
    Click(window, window.StopButtonCenter());
    WaitForProgram(window);
    Check(!window.Running() && ConsoleHas(window, "Stopped on request") && ConsoleHas(window, "< exit code 0"),
        "UDP stop is confirmed by a successful process exit");
    Check(!ConsoleHas(window, "--control") && !std::filesystem::exists(project.StopRequest()) &&
        !std::filesystem::exists(project.Checkpoints() / "live.json"),
        "launch and shutdown use a private session without control files");

    project.program.train = schema::ProjectCommandLine{{}, {"train", "--project", "{project}", "--updates", "1"}};
    project.program.inference = schema::ProjectCommandLine{{}, {"play", "--project", "{project}", "--episodes", "1000000",
        "--realtime", "--step-seconds", "0.1"}};
    schema::SaveProject(project);
    studio::ProjectWindow inference(folder, {PPOKIT_RUN_PROGRAM});
    Check(inference.Launch(schema::ProjectCommand::Train), "prepare a local actor for inference");
    WaitForProgram(inference);
    Check(inference.Launch(schema::ProjectCommand::Inference), "launch supervised inference");
    auto const previousSize = ImGui::GetIO().DisplaySize;
    ImGui::GetIO().DisplaySize = ImVec2(640.F, 720.F);
    Frame(inference);
    Frame(inference);
    auto const button = inference.StopButtonCenter();
    Check(button.x >= 0.F && button.x < 640.F && button.y >= 0.F && button.y < 720.F,
        "Stop remains inside a compact window");
    Check(inference.Running(), "long inference is still running before the click");
    auto const trainingBefore = ReadOrderedJson(project.Training());
    auto const evaluationBefore = ReadOrderedJson(project.Evaluations() / "smoke.json");
    inference.Training()->Document().Set(core::PpoParameter::Epochs, 3.0);
    inference.Evaluation()->Document().Suite(0).episodesPerSeed += 1;
    Check(inference.Unsaved() && !inference.SaveAll() && !inference.Training()->Save()
            && !inference.Network()->Save() && !inference.Evaluation()->Save()
            && ReadOrderedJson(project.Training()) == trainingBefore
            && ReadOrderedJson(project.Evaluations() / "smoke.json") == evaluationBefore,
        "active inference protects every saved configuration even from direct save calls");
    Click(inference, button);
    WaitForProgram(inference);
    Check(!inference.Running() && inference.Status() == "Inference finished.",
        "the real inference Stop button reaches successful OS process exit: " + inference.Status());
    Check(inference.SaveAll() && !inference.Unsaved(), "configuration saving is available after process exit");
    ImGui::GetIO().DisplaySize = previousSize;
}

void Sections(studio::ProjectWindow& window)
{
    using Section = studio::ProjectWindow::Section;
    Check(window.Remarks().empty(), "a complete project has nothing to look at");
    for (Section const section : {Section::Overview, Section::Network, Section::Training, Section::Evaluation,
             Section::Task, Section::Run, Section::Console})
    {
        window.Show(section);
        Frame(window);
        Frame(window);
    }
    Check(window.Network() != nullptr && window.Training() != nullptr, "network and training settings are open");
    Check(window.Evaluation() != nullptr && window.Evaluation()->Document().Suites().size() == 1
            && !window.Evaluation()->Dirty(),
        "the evaluation suites of eval/ are open, unchanged");
}

/** @brief A narrow project window can scroll to the rightmost overview cards. */
void NarrowOverview(studio::ProjectWindow& window)
{
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 const previousSize = io.DisplaySize;
    io.DisplaySize = ImVec2(640.F, 720.F);
    window.Show(studio::ProjectWindow::Section::Overview);
    Frame(window);
    Frame(window);
    bool scrollable = false;
    for (ImGuiWindow const* child : ImGui::GetCurrentContext()->Windows)
    {
        if (child->Active && std::string(child->Name).find("/section_") != std::string::npos)
            scrollable = child->ScrollMax.x > 0.F;
    }
    Check(scrollable, "the overview exposes horizontal scrolling on a narrow window");
    io.DisplaySize = previousSize;
}

/** @brief Whether a tooltip is visible in the current rendered headless ImGui frame. */
bool TooltipVisible()
{
    for (ImGuiWindow const* window : ImGui::GetCurrentContext()->Windows)
    {
        if (window->Active && !window->Hidden && (window->Flags & ImGuiWindowFlags_Tooltip) != 0)
        {
            return true;
        }
    }
    return false;
}

/** @brief Every diagram card opens its help immediately and leaving the diagram hides it. */
void OverviewTooltips(studio::ProjectWindow& window)
{
    using Box = studio::OverviewView::Box;
    window.Show(studio::ProjectWindow::Section::Overview);
    Frame(window);
    ImGuiIO& io = ImGui::GetIO();
    for (std::size_t index = 0; index < static_cast<std::size_t>(Box::Count); ++index)
    {
        ImVec2 const center = window.Overview().BoxCenter(static_cast<Box>(index));
        io.AddMousePosEvent(center.x, center.y);
        // Allow ImGui to size a newly opened tooltip, but not to pass the normal hover delay.
        Frame(window);
        Frame(window);
        Frame(window);
        Check(TooltipVisible(), "overview card " + std::to_string(index) + " displays immediate help");
        io.AddMousePosEvent(-1.F, -1.F);
        Frame(window);
        Frame(window);
        Check(!TooltipVisible(), "overview help disappears when the pointer leaves");
    }
}

void OverviewClicks(studio::ProjectWindow& window)
{
    using Box = studio::OverviewView::Box;
    using Section = studio::ProjectWindow::Section;
    window.Show(Section::Overview);
    Frame(window);
    Click(window, window.Overview().BoxCenter(Box::Critic));
    Check(window.Shown() == Section::Network, "the critic box opens the network");
    window.Show(Section::Overview);
    Frame(window);
    Click(window, window.Overview().BoxCenter(Box::Gae));
    Check(window.Shown() == Section::Training, "the GAE box opens the training settings");
    window.Show(Section::Overview);
    Frame(window);
    Click(window, window.Overview().BoxCenter(Box::Environment));
    Check(window.Shown() == Section::Environment, "the environment card opens the actual simulation settings");
    window.Show(Section::Overview);
    Frame(window);
    Click(window, window.Overview().BoxCenter(Box::Observation));
    Check(window.Shown() == Section::Task, "the observation box opens the task");
    window.Show(Section::Overview);
    Frame(window);
    Click(window, window.Overview().BoxCenter(Box::Curriculum));
    Check(window.Shown() == Section::NativeCurriculum, "the generic curriculum card opens the editor directly");
    std::string rendered;
    Frame(window, &rendered);
    Check(rendered.find("near") != std::string::npos && rendered.find("far") != std::string::npos,
        "the dedicated curriculum page renders the real stages without another navigation step");
}

void Launches(studio::ProjectWindow& window)
{
    window.Training()->Document().Set(core::PpoParameter::Epochs, 3.0);
    Check(window.Unsaved(), "a changed parameter is unsaved");
    Check(window.Launch(schema::ProjectCommand::Train), "Train starts the program: " + window.Status());
    Check(window.Status().find("changes saved first") != std::string::npos && !window.Unsaved(),
        "changes are saved before training");
    Check(window.Shown() == studio::ProjectWindow::Section::Run, "training shows its curves");
    Check(!window.Launch(schema::ProjectCommand::Evaluate), "a second program waits for the first");
    WaitForProgram(window);
    Check(ConsoleHas(window, "update 3/3") && window.Status() == "Train finished.",
        "the console shows the training, and its end: " + window.Status());
    Check(std::filesystem::exists(window.Project().Journal()), "the journal is in the project's checkpoints");

    Check(window.Launch(schema::ProjectCommand::Inference), "Inference starts: " + window.Status());
    WaitForProgram(window);
    Check(ConsoleHas(window, "episode seed 9001") && ConsoleHas(window, "episode seed 9002"),
        "inference prints its episodes in the console");

    Check(window.Launch(schema::ProjectCommand::Evaluate), "Evaluate starts: " + window.Status());
    WaitForProgram(window);
    Check(ConsoleHas(window, "reach_smoke"), "the evaluation report is in the console");
}

void StopTraining(std::filesystem::path const& folder, std::vector<std::filesystem::path> const& programs)
{
    schema::Project project = schema::LoadProject(folder);
    project.program.train = schema::ProjectCommandLine{{}, {"train", "--project", "{project}", "--updates", "100000"}};
    schema::SaveProject(project);
    studio::ProjectWindow window(folder, programs);
    Check(window.Launch(schema::ProjectCommand::Train), "a long training starts");
    for (int frame = 0; frame < 6000 && !ConsoleHas(window, "update "); ++frame)
    {
        Frame(window);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Frame(window);
    Click(window, window.StopButtonCenter());
    WaitForProgram(window);
    Check(ConsoleHas(window, "Stopped on request") && window.Status() == "Train finished.",
        "Stop ends training after an update, with a checkpoint: " + window.Status());
}

void MissingProgram(std::filesystem::path const& folder)
{
    studio::ProjectWindow window(folder, {});
    schema::Project const& project = window.Project();
    Check(project.program.executable == "ppokit_reach", "the copy names its program");
    if (schema::FindProgram(project, schema::ProjectCommand::Train, {}).empty())
    {
        Check(!window.Launch(schema::ProjectCommand::Train)
                && window.Status().find("not found: build it") != std::string::npos,
            "a program that is not built is explained: " + window.Status());
    }
}

void StartScreen(std::filesystem::path const& root, std::vector<std::filesystem::path> const& programs)
{
    studio::StudioApp app(root / "recent.txt", programs);
    Frame(app);
    Check(!app.OpenProject(root / "not-a-project") && app.Project() == nullptr, "a folder without project.json is refused");
    Frame(app);
    Check(app.CreateProject("lander", root) && app.Project() != nullptr, "a new project is created and opened");
    Check(std::filesystem::exists(root / "lander" / "project.json") && std::filesystem::exists(root / "lander" / "components"),
        "the new project has its files and folders");
    Frame(app);
    Frame(app);
    Check(app.Start().Recent().size() == 1, "the opened project is remembered");
    Check(app.RequestQuit() && app.QuitConfirmed(), "quitting with nothing unsaved needs no question");
}

/**
 * @brief A project run by its own program (here the PC-7 project made so: its own executable,
 *        no training.json, a compiled network): the network opens read-only, the training
 *        settings stay in the program's settings file (no training.json is written), and
 *        nothing is saved.
 */
void ProgramSettingsProject(std::filesystem::path const& into)
{
    std::filesystem::copy(PPOKIT_FLIGHT_PC7_PROJECT, into,
        std::filesystem::copy_options::recursive | std::filesystem::copy_options::skip_existing);
    std::filesystem::remove_all(into / "checkpoints");
    std::filesystem::remove(into / "training.json");
    nlohmann::ordered_json project = ReadOrderedJson(into / "project.json");
    project.erase("libraries");
    project["program"] = {{"executable", "flight_own_program"}, {"train", {"--project", "{project}"}},
        {"evaluate", {"--project", "{project}", "--evaluate"}}, {"inference", {"--project", "{project}"}}};
    project["networkEditable"] = false;
    WriteOrderedJson(into / "project.json", project);
    nlohmann::ordered_json network = ReadOrderedJson(into / "network.json");
    network.erase("components");
    network.erase("observers");
    WriteOrderedJson(into / "network.json", network);

    auto const native = schema::LoadProject(into);
    studio::ScenarioView scenario(native);
    auto const summary = scenario.CurriculumSummary();
    Check(std::find(summary.begin(), summary.end(), "runway-heading-offset") != summary.end(),
        "the native scenario exposes its last named stage without a training.json");
    auto const savedScenario = ReadOrderedJson(into / "scenario.json");
    WriteOrderedJson(into / "scenario.json", {{"aircraft", "PC7-visible-scenario"}});
    scenario.Reload();
    Check(scenario.CurriculumSummary().front().find("/training/curriculum") != std::string::npos,
        "a missing native curriculum identifies its configured path");
    {
        studio::ProjectWindow partial(into, {});
        partial.Show(studio::ProjectWindow::Section::Environment);
        std::string text;
        Frame(partial, &text);
        Check(text.find("PC7-visible-scenario") != std::string::npos,
            "a missing curriculum pointer preserves the other readable scenario settings");
        partial.Show(studio::ProjectWindow::Section::NativeCurriculum);
        Frame(partial, &text);
        Check(text.find("Missing curriculum at /training/curriculum") != std::string::npos
                && text.find("no PPOKit generation manifest") != std::string::npos,
            "an untrained project reports the missing curriculum source and checkpoint independently");
    }
    WriteOrderedJson(into / "scenario.json", savedScenario);

    runtime::PublishGeneration(native, 23, R"({"level":4,"window":{"episodes":7,"stops":5}})", {1.0},
        [](auto const& training, auto const& policy) {
            std::ofstream(training) << "fixture learner";
            std::ofstream(policy) << "fixture actor";
        });
    studio::ProjectWindow window(into, {});
    Check(window.Remarks().empty(), "the own-program project is complete");
    using Section = studio::ProjectWindow::Section;
    for (Section const section :
        {Section::Overview, Section::Network, Section::Training, Section::Evaluation, Section::Task, Section::Run})
    {
        window.Show(section);
        Frame(window);
        Frame(window);
    }
    window.Show(Section::Overview);
    Frame(window);
    Click(window, window.Overview().BoxCenter(studio::OverviewView::Box::Curriculum));
    std::string curriculum;
    Frame(window, &curriculum);
    Check(curriculum.find("/training/curriculum") != std::string::npos
            && curriculum.find("runway-heading-offset") != std::string::npos,
        "the native curriculum card opens the configured PC-7 stages without a training.json");
    Check(curriculum.find("update 23") != std::string::npos && curriculum.find("level  4") != std::string::npos
            && curriculum.find("stops  5") != std::string::npos,
        "the curriculum shows the update, level and window from the published training state");
    Check(window.Network() != nullptr && window.Network()->ReadOnly(), "a compiled network opens read-only");
    Check(window.Training() == nullptr, "a project with its own settings file has no PPOKit training view");
    Check(!window.Unsaved(), "reading the own-program project changes nothing");
    Check(!window.Launch(schema::ProjectCommand::Train)
            && window.Status().find("'flight_own_program' not found") != std::string::npos,
        "without its program, the launch names it: " + window.Status());
    Check(!std::filesystem::exists(into / "training.json"), "launching writes no training.json");
}

/** @brief All shipped projects must open non-empty Task, Environment and Curriculum pages.
 * This test only reads manifests and settings; it never trains or runs the models.
 */
void ShippedProjectNavigation()
{
    using Section = studio::ProjectWindow::Section;
    std::filesystem::path const root = PPOKIT_REPOSITORY_ROOT;
    auto const programs = schema::ReadProgramRegistry(PPOKIT_STUDIO_REGISTRY);
    for (char const* relative : {"Examples/FlightControl3D/projects/T6", "Examples/FlightControl3D/projects/PC7",
             "Examples/Airspace/projects/RLMission", "Examples/Airspace/projects/RLSNA",
             "Examples/BlueDefense/projects/BluePatrol"})
    {
        studio::ProjectWindow window(root / relative, programs);
        auto const& project = window.Project();
        std::string rendered;
        window.Show(Section::Task);
        Frame(window, &rendered);
        Check(rendered.find("Task contract (read-only)") != std::string::npos
            && rendered.find("Validated schema") != std::string::npos, project.name + ": task contract is visible");
        window.Show(Section::Environment);
        Frame(window, &rendered);
        Check(rendered.find("Environment module") != std::string::npos
            && project.SettingsFile() && rendered.find(project.SettingsFile()->filename().string()) != std::string::npos,
            project.name + ": real environment settings source is visible");
        window.Show(Section::NativeCurriculum);
        Frame(window, &rendered);
        if (!project.curriculumPointer.empty())
            Check(rendered.find("Configured curriculum") != std::string::npos
                && rendered.find(project.curriculumPointer) != std::string::npos,
                project.name + ": native curriculum is visible");
        else if (!project.externalCurriculum.empty())
            Check(rendered.find("Configured curriculum") != std::string::npos
                && rendered.find(std::filesystem::path(project.externalCurriculum).filename().string()) != std::string::npos
                && rendered.find("normal Train command does not execute") != std::string::npos,
                project.name + ": existing external curriculum source is visible without implying it runs");
        else
            Check(rendered.find("No curriculum stages") != std::string::npos
                || rendered.find("no native curriculum") != std::string::npos
                || rendered.find("configured stages") != std::string::npos,
                project.name + ": curriculum availability is explicit");
        Check(!window.Unsaved(), project.name + ": inspection does not change settings");
    }
}

/** @brief Explicit native settings take precedence over obsolete generic files. */
void NativeSettingsSource(std::filesystem::path const& into)
{
    CopyReach(into);
    auto project = schema::LoadProject(into);
    project.learnerSettings = "native.json";
    project.settings = "scenario.json";
    WriteOrderedJson(into / "native.json", {{"formatVersion", 1}, {"sourceMarker", "native-settings-source"}});
    WriteOrderedJson(into / "scenario.json", {{"formatVersion", 1}});
    schema::SaveProject(project);
    auto const previousTraining = ReadOrderedJson(project.Training());
    studio::ProjectWindow window(into, {});
    Check(window.Training() == nullptr, "explicit native learner settings take precedence over training.json");
    window.Show(studio::ProjectWindow::Section::Training);
    std::string rendered;
    Frame(window, &rendered);
    Check(rendered.find("native-settings-source") != std::string::npos, "native settings are actually rendered");
    Check(!window.Unsaved() && ReadOrderedJson(project.Training()) == previousTraining,
        "inspection never rewrites the obsolete generic file");

    WriteOrderedJson(into / "native.json", nlohmann::ordered_json::array());
    studio::ProjectWindow malformed(into, {});
    malformed.Show(studio::ProjectWindow::Section::Training);
    Frame(malformed, &rendered);
    Check(rendered.find("Expected a JSON object") != std::string::npos
            && rendered.find("native.json") != std::string::npos,
        "invalid settings show a source-specific diagnostic instead of an empty page");
}

#if defined(PPOKIT_FLIGHT_ENVIRONMENT)
/** @brief Exercise the actual Flight provider on a scratch project: accepted and rejected draft saves. */
void FlightScenarioEditing(std::filesystem::path const& into)
{
    std::filesystem::create_directories(into);
    std::filesystem::path const source = PPOKIT_FLIGHT_PC7_PROJECT;
    for (char const* file : {"project.json", "task.json", "scenario.json"})
        std::filesystem::copy_file(source / file, into / file, std::filesystem::copy_options::overwrite_existing);
    auto const project = schema::LoadProject(into);
    studio::ScenarioView scenario(project, {PPOKIT_FLIGHT_ENVIRONMENT});
    Check(scenario.Editable(), "native flight environment exposes a validated editor");
    auto const before = ReadOrderedJson(into / "scenario.json");
    Check(!scenario.SetScalar("/aircraft", "t6texan2"), "scenario editor cannot modify model contract");
    Check(scenario.SetScalar("/training/curriculum/windowEpisodes", std::uint64_t{0}) && scenario.Dirty(),
        "a native curriculum setting becomes an unsaved draft");
    Check(!scenario.Save() && scenario.Dirty() && ReadOrderedJson(into / "scenario.json") == before,
        "provider rejects invalid curriculum without modifying scenario.json: " + scenario.Status());
    scenario.Reload();
    Check(scenario.Dirty(), "refresh does not discard a pending draft");
    Check(scenario.SetScalar("/training/curriculum/windowEpisodes", std::uint64_t{31}) && scenario.Save(),
        "the real provider validates and saves a corrected curriculum: " + scenario.Status());
    auto expected = before;
    expected["training"]["curriculum"]["windowEpisodes"] = 31U;
    Check(!scenario.Dirty() && ReadOrderedJson(into / "scenario.json") == expected,
        "only the edited curriculum scalar changes on disk");
    auto const oldSeconds = before["mission"]["maxSeconds"];
    nlohmann::ordered_json seconds = oldSeconds.is_number_float() ? nlohmann::ordered_json(901.0)
        : nlohmann::ordered_json(std::uint64_t{901});
    Check(scenario.SetScalar("/mission/maxSeconds", seconds) && scenario.Save(),
        "a real environment setting can also be edited and saved");
    expected["mission"]["maxSeconds"] = seconds;
    Check(ReadOrderedJson(into / "scenario.json") == expected, "environment save preserves the remaining settings");
    // An external edit must survive a conflicting Studio draft.
    Check(scenario.SetScalar("/training/curriculum/windowEpisodes", std::uint64_t{32}), "create conflicting draft");
    auto external = expected;
    external["training"]["curriculum"]["windowEpisodes"] = 33U;
    WriteOrderedJson(into / "scenario.json", external);
    Check(!scenario.Save() && scenario.Dirty() && ReadOrderedJson(into / "scenario.json") == external,
        "save refuses to overwrite externally modified settings");
    scenario.Revert();
    Check(!scenario.Dirty() && scenario.SetScalar("/training/curriculum/windowEpisodes", std::uint64_t{33})
            && !scenario.Dirty(), "discard reloads the actual on-disk value");
    scenario.SetEditingAllowed(false);
    Check(!scenario.SetScalar("/training/curriculum/windowEpisodes", std::uint64_t{32}),
        "scenario editing is disabled during a run");
}
#endif

#if defined(PPOKIT_BLUE_ENVIRONMENT)
/** @brief A Blue training starts its configured viewer without a manual View click. */
void BlueTrainingOpensViewer(std::filesystem::path const& into)
{
    std::filesystem::create_directories(into);
    auto const source = std::filesystem::path(PPOKIT_REPOSITORY_ROOT) / "Examples/BlueDefense/projects/BluePatrol";
    for (char const* file : {"project.json", "task.json", "network.json", "training.json",
             "scenario.json", "blue_patrol_training.json"})
        std::filesystem::copy_file(source / file, into / file, std::filesystem::copy_options::overwrite_existing);
    auto network = ReadOrderedJson(into / "network.json");
    network["components"] = {PPOKIT_BLUE_VISUALISER};
    WriteOrderedJson(into / "network.json", network);
    auto project = schema::LoadProject(into);
    project.program.executable = "ppokit_reach";
    project.program.train = schema::ProjectCommandLine{{}, {"help"}};
    schema::SaveProject(project);

    // A disposable executable with the viewer's name exercises process launch without a window.
    auto const fakeViewer = into / (std::string("ppokit-viewer")
        + std::filesystem::path(PPOKIT_RUN_PROGRAM).extension().string());
    std::filesystem::copy_file(PPOKIT_RUN_PROGRAM, fakeViewer, std::filesystem::copy_options::overwrite_existing);
    studio::ProjectWindow window(into, {PPOKIT_REACH_PROGRAM, fakeViewer});
    Check(window.Launch(schema::ProjectCommand::Train), "Blue training starts");
    Check(window.Status().find("Viewer opened on 'watch'") != std::string::npos,
        "training automatically launches the configured Blue viewer: " + window.Status());
    window.Kill();
    WaitForProgram(window);
}

/** @brief Locate a reward by its schema name so the test follows the actual channel order. */
std::size_t RewardChannelIndex(schema::TaskSchema const& task, std::string const& name)
{
    for (std::size_t index = 0; index < task.rewards.size(); ++index)
        if (task.rewards[index].name == name) return index;
    throw std::runtime_error("Missing Blue reward channel: " + name);
}

/** @brief Validate Blue scenario edits through the real environment on a disposable project. */
void BlueScenarioEditing(std::filesystem::path const& into)
{
    std::filesystem::create_directories(into);
    std::filesystem::path const source = std::filesystem::path(PPOKIT_REPOSITORY_ROOT)
        / "Examples/BlueDefense/projects/BluePatrol";
    for (char const* file : {"project.json", "task.json", "scenario.json", "blue_patrol_training.json"})
        std::filesystem::copy_file(source / file, into / file, std::filesystem::copy_options::overwrite_existing);
    auto original = ReadOrderedJson(into / "scenario.json");
    original["agentProfiles"] = std::filesystem::absolute(source / original["agentProfiles"].get<std::string>())
        .lexically_normal().generic_string();
    WriteOrderedJson(into / "scenario.json", original);
    auto const project = schema::LoadProject(into);
    studio::ScenarioView view(project, {PPOKIT_BLUE_ENVIRONMENT});
    Check(view.Editable(), "Blue environment exposes a validated scenario editor");
    Check(!view.SetScalar("/formatVersion", std::uint64_t{1}), "scenario contract is read-only");
    Check(view.SetScalar("/blueDefense/blueTeam/activeCount", std::uint64_t{0}), "invalid team draft is editable");
    Check(!view.Save() && ReadOrderedJson(into / "scenario.json") == original,
        "invalid team is rejected without modifying the source");
    Check(view.SetScalar("/blueDefense/blueTeam/activeCount", std::uint64_t{1})
            && view.SetScalar("/blueDefense/redTeam/activeCount", std::uint64_t{0}) && view.Save(),
        "a valid 1v0 scenario is saved: " + view.Status());
    auto expected = original;
    expected["blueDefense"]["blueTeam"]["activeCount"] = 1U;
    expected["blueDefense"]["redTeam"]["activeCount"] = 0U;
    Check(ReadOrderedJson(into / "scenario.json") == expected, "save preserves all unrelated fields");
    // The real ABI receives a full vector even when Studio edits just one channel.
    auto const library = ppokit::environment::EnvironmentLibrary::Open(PPOKIT_BLUE_ENVIRONMENT);
    ppokit::environment::Environment environment(library, into, 42U);
    auto const task = schema::LoadTaskSchema(into / "task.json");
    Check(library->Info().rewardChannels == task.rewards.size(), "Blue task and environment channel counts match");
    std::vector<double> weights(library->Info().rewardChannels);
    environment.Weights(weights.data());
    std::vector<double> const masterWeights{
        1., 4., 2., 1., 1.5, 2., 2., 2., 1.5, 1.5, 1., 1., 0.01, 5., 24.};
    bool matchesMaster = weights.size() == masterWeights.size();
    for (std::size_t index = 0; matchesMaster && index < weights.size(); ++index)
        matchesMaster = std::abs(weights[index] - masterWeights[index]) < 1.e-6;
    Check(matchesMaster, "Blue environment exposes master's 15 mapped reward weights in task order");
    auto const ownLoss = RewardChannelIndex(task, "OwnLoss");
    auto const patrol = RewardChannelIndex(task, "PatrolTracking");
    if (ownLoss >= weights.size() || patrol >= weights.size())
        throw std::runtime_error("Blue task reward channel is outside the environment's channel range");
    Check(weights[ownLoss] == 24., "Blue environment starts with master's own-loss weight");
    weights[patrol] = 2.;
    environment.SetWeights(weights.data());
    std::vector<double> applied(weights.size());
    environment.Weights(applied.data());
    Check(applied == weights, "the environment applies a live edit while preserving own loss");
    view.SetEditingAllowed(false);
    Check(!view.SetScalar("/blueDefense/blueTeam/activeCount", std::uint64_t{2}),
        "Blue scenario cannot be edited during a run");
}
#endif

/** @brief Edit the CI gate schedule on a copy, including bounds and conflict handling. */
void ExternalCurriculumEditing(std::filesystem::path const& into)
{
    std::filesystem::create_directories(into);
    auto const source = std::filesystem::path(PPOKIT_REPOSITORY_ROOT) / "tools/ci/adaptive_curriculum/gates.json";
    std::filesystem::copy_file(source, into / "gates.json", std::filesystem::copy_options::overwrite_existing);
    schema::Project project;
    project.root = into;
    project.externalCurriculum = "gates.json";
    studio::ScenarioView view(project);
    auto const original = ReadOrderedJson(into / "gates.json");
    Check(!view.SetExternalScalar("/gates/0/blue", std::uint64_t{2}), "fixed gate dimensions are read-only");
    Check(!view.SetExternalScalar("/gates/0/thresholdPercent", std::uint64_t{101}), "threshold is bounded");
    Check(!view.SetExternalScalar("/gates/0/maxSeconds", std::uint64_t{0}), "duration must be positive");
    Check(!view.SetExternalScalar("/rehearsalEveryBlocks", std::uint64_t{1}), "rehearsal period is bounded");
    Check(view.SetExternalScalar("/gates/0/thresholdPercent", std::uint64_t{80})
            && view.SetExternalScalar("/gates/0/autoAdvance", false) && view.Dirty(),
        "gate parameters are editable");
    view.Reload();
    Check(view.Dirty() && view.Save(), "refresh retains and save writes a curriculum draft");
    auto expected = original;
    expected["gates"][0]["thresholdPercent"] = 80U;
    expected["gates"][0]["autoAdvance"] = false;
    Check(ReadOrderedJson(into / "gates.json") == expected, "save preserves the gate sequence");
    Check(view.SetExternalScalar("/gates/0/maxSeconds", std::uint64_t{100}), "prepare conflicting draft");
    expected["gates"][0]["maxSeconds"] = 200U;
    WriteOrderedJson(into / "gates.json", expected);
    Check(!view.Save() && ReadOrderedJson(into / "gates.json") == expected, "external changes are not overwritten");
    view.SetEditingAllowed(false);
    Check(!view.SetExternalScalar("/gates/0/maxSeconds", std::uint64_t{100}), "editing is disabled during a run");
}

int Run()
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImNodes::CreateContext();
    studio::ApplyStudioTheme();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1500.F, 920.F);
    io.DeltaTime = 1.F / 60.F;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    std::filesystem::path const root = std::filesystem::temp_directory_path() / "ppokit_project_window_tests";
    std::error_code removed;
    std::filesystem::remove_all(root, removed);
    std::filesystem::path const folder = CopyReach(root / "reach");
    std::vector<std::filesystem::path> const programs{PPOKIT_REACH_PROGRAM};
    {
        studio::ProjectWindow window(folder, programs);
        Sections(window);
        NarrowOverview(window);
        OverviewTooltips(window);
        OverviewClicks(window);
        Launches(window);
    }
    LaunchChoices();
    LaunchSettingsPanel(folder);
    SupervisedProject(root / "supervised");
    StopTraining(folder, programs);
    MissingProgram(folder);
    ProgramSettingsProject(root / "own_program");
    ShippedProjectNavigation();
    NativeSettingsSource(root / "native_settings");
    ExternalCurriculumEditing(root / "external_curriculum_editor");
#if defined(PPOKIT_FLIGHT_ENVIRONMENT)
    FlightScenarioEditing(root / "flight_editor");
#endif
#if defined(PPOKIT_BLUE_ENVIRONMENT)
    BlueScenarioEditing(root / "blue_editor");
    BlueTrainingOpensViewer(root / "blue_viewer");
#endif
    StartScreen(root, programs);

    std::filesystem::remove_all(root, removed);
    ImNodes::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    return failures;
}

} // namespace

int main()
{
    try
    {
        if (Run() != 0)
        {
            std::cerr << failures << " project window check(s) failed\n";
            return 1;
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "studio project window checks passed\n";
    return 0;
}
