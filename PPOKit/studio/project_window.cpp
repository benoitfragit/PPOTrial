/**
 * @file PPOKit/studio/project_window.cpp
 * @brief The window of one open project: run bar, navigation, section views.
 */
#include "project_window.hpp"

#include "studio_icons.hpp"
#include "studio_theme.hpp"

#include "ppokit/editor/run_health.hpp"
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/eval/best_store.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <random>
#include <optional>
#include <utility>

namespace ppokit::studio {
namespace {

/** @brief Width of the navigation column. */
constexpr float NavigationWidth = 210.F;
/** @brief Console lines kept in the window (the program's full history is its journal). */
constexpr std::size_t MaximumConsoleLines = editor::Process::MaximumLines;
/** @brief Read-only runs and viewers never need to wait for an optimizer checkpoint. */
constexpr auto InteractiveStopTimeout = std::chrono::seconds(2);

struct SectionInfo final
{
    ProjectWindow::Section section;
    StudioIcon icon;
    char const* label;
    char const* help;
};

constexpr SectionInfo Sections[] = {
    {ProjectWindow::Section::Overview, StudioIcon::Overview, "Overview",
        "The whole agent on one screen; click a box to open it."},
    {ProjectWindow::Section::Environment, StudioIcon::Task, "Environment", "Simulation module and scenario settings."},
    {ProjectWindow::Section::Network, StudioIcon::Network, "Network", "Actor and critic block graphs (network.json)."},
    {ProjectWindow::Section::Training, StudioIcon::Training, "Training",
        "PPO and GAE parameters, probes and curriculum (training.json)."},
    {ProjectWindow::Section::NativeCurriculum, StudioIcon::Training, "Curriculum",
        "Native stages and the last published training state."},
    {ProjectWindow::Section::Evaluation, StudioIcon::Evaluate, "Evaluation",
        "Evaluation suites (eval/): seeds, gates on the metrics, campaigns during training, the BEST policy."},
    {ProjectWindow::Section::Task, StudioIcon::Task, "Task",
        "Observation, actions, reward channels and scenario of task.json."},
    {ProjectWindow::Section::Run, StudioIcon::Run, "Run",
        "Curves of the last training, live while it runs: rewards, PPO statistics, probes, reward weights."},
    {ProjectWindow::Section::Console, StudioIcon::Console, "Console",
        "What the program writes while it trains, evaluates or plays."},
};

/** @brief Height of a navigation entry, in text lines. */
constexpr float NavigationEntryLines = 2.F;

/** @brief The action buttons of the header, right-aligned in this order. */
struct CommandButton final
{
    schema::ProjectCommand command;
    StudioIcon icon;
};

constexpr CommandButton CommandButtons[] = {
    {schema::ProjectCommand::Train, StudioIcon::Train},
    {schema::ProjectCommand::Evaluate, StudioIcon::Evaluate},
    {schema::ProjectCommand::Inference, StudioIcon::Inference},
};

char const* Verb(schema::ProjectCommand command) noexcept
{
    switch (command)
    {
    case schema::ProjectCommand::Train:
        return "Training";
    case schema::ProjectCommand::Evaluate:
        return "Evaluating";
    case schema::ProjectCommand::Inference:
        return "Running inference";
    }
    return "Running";
}

char const* CommandHelp(schema::ProjectCommand command) noexcept
{
    switch (command)
    {
    case schema::ProjectCommand::Train:
        return "Train the agent: resumes from checkpoints/latest.ppokit when it exists. The Run section shows the "
               "curves live.";
    case schema::ProjectCommand::Evaluate:
        return "Evaluate the trained policy on the seeds of the evaluation suites (eval/), which training never sees.";
    case schema::ProjectCommand::Inference:
        return "Run the trained policy (for FlightControl3D: fly it in the 3D view).";
    }
    return "";
}

std::string Elapsed(std::chrono::steady_clock::time_point start)
{
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return editor::FormatDuration(seconds);
}

} // namespace

ProjectWindow::ProjectWindow(std::filesystem::path const& folder, std::vector<std::filesystem::path> programs)
    : project_(schema::LoadProject(std::filesystem::absolute(folder))), scenario_(project_, programs), programs_(std::move(programs))
{
    Open();
}

ProjectWindow::~ProjectWindow() = default;

void ProjectWindow::Open()
{
    remarks_ = schema::CheckProject(project_);
    try { references_ = schema::LoadPolicyReferences(project_.root); }
    catch (std::exception const& error) { remarks_.push_back(error.what()); }
    if (std::filesystem::exists(project_.Task()))
    {
        try
        {
            task_ = schema::LoadTaskSchema(project_.Task());
        }
        catch (std::exception const& error)
        {
            remarks_.push_back(std::string("task.json cannot be read: ") + error.what());
        }
    }
    if (task_ && std::filesystem::exists(project_.Network()))
    {
        try
        {
            network_ = std::make_unique<GraphEditorView>(*task_,
                OpenGraphDocument(project_.Network(), project_.NetworkLayout()),
                GraphFiles{project_.Network(), project_.NetworkLayout()});
            network_->SetComponentFolder(project_.Components());
            if (!project_.networkEditable)
            {
                network_->SetReadOnly("This network is compiled into the program: it is shown here to read. "
                                      "Changing it needs a change of the program.");
            }
        }
        catch (std::exception const& error)
        {
            remarks_.push_back(std::string("network.json cannot be opened: ") + error.what());
        }
    }
    // An explicitly declared native learner owns its settings even if an old training.json
    // is still present. Never show an editor whose values are ignored by the runtime.
    bool const ownSettings = !project_.learnerSettings.empty()
        || (!project_.settings.empty() && !std::filesystem::exists(project_.Training()));
    try
    {
        if (!ownSettings)
        {
            bool const separateTeamNetworks = task_ && task_->maximumAgents > 1
                && std::filesystem::path(project_.networkLibrary).stem() == "ppokit_graph_network";
            training_ = std::make_unique<TrainSettingsView>(project_.Training(), task_, separateTeamNetworks);
        }
    }
    catch (std::exception const& error)
    {
        remarks_.push_back(std::string("training.json cannot be read: ") + error.what());
    }
    if (task_)
    {
        try
        {
            evaluation_ = std::make_unique<EvaluationView>(
                project_.Evaluations(), project_.Checkpoints(), task_->metrics, references_);
        }
        catch (std::exception const& error)
        {
            remarks_.push_back(std::string("eval/ cannot be read: ") + error.what());
        }
    }
    dashboard_ = std::make_unique<TrainingDashboardView>(project_.Journal(), project_.controlProtocol == "legacy-files");
}

bool ProjectWindow::Unsaved() const noexcept
{
    return (network_ && network_->Dirty()) || (training_ && training_->Dirty())
        || (evaluation_ && evaluation_->Dirty()) || scenario_.Dirty();
}

bool ProjectWindow::SaveAll()
{
    if (ConfigurationLocked())
    {
        Report("Stop the running process before saving project changes.", true);
        return false;
    }
    if (scenario_.Dirty() && !scenario_.Save())
    {
        Report(scenario_.Status(), true);
        return false;
    }
    if (network_ && network_->Dirty() && !network_->Save())
    {
        Report("network.json was not saved: fix the network first (see the Network section).", true);
        return false;
    }
    if (training_ && training_->Dirty() && !training_->Save())
    {
        Report("training.json was not saved: " + training_->Status(), true);
        return false;
    }
    if (evaluation_ && evaluation_->Dirty() && !evaluation_->Save())
    {
        Report("The evaluation suites were not saved: " + evaluation_->Status(), true);
        return false;
    }
    return true;
}

bool ProjectWindow::ConfigurationLocked() const noexcept
{
    if (running_)
    {
        return true;
    }
    if (!control_ || !control_->Connected())
    {
        return false;
    }
    ipc::State const state = control_->LastState();
    return state != ipc::State::Stopped && state != ipc::State::Error;
}

void ProjectWindow::ApplyEditingPolicy() noexcept
{
    bool const allowed = !ConfigurationLocked();
    scenario_.SetEditingAllowed(allowed);
    if (network_) network_->SetEditingAllowed(allowed);
    if (training_) training_->SetEditingAllowed(allowed);
    if (evaluation_) evaluation_->SetEditingAllowed(allowed);
}

void ProjectWindow::VerifyLaunchCompatibility(schema::ProjectCommand command,
    std::vector<std::string> const& arguments) const
{
    if (!task_ || !std::filesystem::is_regular_file(project_.Network())) return;
    auto has = [&arguments](std::string const& flag) {
        return std::find(arguments.begin(), arguments.end(), flag) != arguments.end();
    };
    auto value = [&arguments](std::string const& flag) -> std::string {
        auto const found = std::find(arguments.begin(), arguments.end(), flag);
        return found != arguments.end() && std::next(found) != arguments.end() ? *std::next(found) : std::string();
    };
    std::filesystem::path artifact;
    if (command == schema::ProjectCommand::Train)
    {
        if (!has("--fresh")) artifact = project_.LatestCheckpoint();
    }
    else if (!value("--policy").empty()) artifact = project_.root / value("--policy");
    else
    {
        std::string const role = value("--model");
        if (role == "LATEST") artifact = project_.TrainingPolicy();
        else if (!role.empty() && role != "BEST")
        {
            for (auto const& reference : references_)
                if (reference.role == role) artifact = reference.file;
        }
        else if (role != "BEST" && !has("--best")) artifact = project_.Policy();
    }
    if (artifact.empty() || !std::filesystem::is_regular_file(artifact)) return;
    std::ifstream stream(artifact, std::ios::binary);
    std::string line;
    if (!std::getline(stream, line)) return;
    nlohmann::json const json = nlohmann::json::parse(line, nullptr, false);
    if (!json.is_object() || json.value("schema", std::string()) != "ppokit.artifact/1") return;
    artifacts::ArtifactHeader const header = artifacts::ReadArtifactHeader(artifact);
    if (header.identity.backend != "graph") return;
    auto const graphs = schema::LoadBlockGraph(project_.Network(), *task_);
    artifacts::RequireCompatible(header.identity,
        schema::ArtifactIdentityOf(*task_, graphs, header.identity.backend), artifact.filename().u8string());
}

void ProjectWindow::Report(std::string message, bool error)
{
    status_ = std::move(message);
    statusIsError_ = error;
}

bool ProjectWindow::Running() noexcept
{
    return process_.Running();
}

bool ProjectWindow::Launch(schema::ProjectCommand command)
{
    Poll();
    if (ConfigurationLocked())
    {
        Report("A program is already running: stop it first.", true);
        return false;
    }
    if (viewerStopAt_)
    {
        Report("The previous viewer is still closing.", true);
        return false;
    }
    std::optional<std::vector<std::string>> const arguments = schema::CommandArguments(project_, command);
    if (!arguments)
    {
        Report(std::string("This project has no ") + schema::CommandName(command) + " command (project.json).", true);
        return false;
    }
    std::filesystem::path const program = schema::FindProgram(project_, command, programs_);
    if (program.empty())
    {
        std::string const executable = schema::CommandExecutable(project_, command);
        Report("Program '" + executable + "' not found: build it (cmake --build ... --target " + executable
                   + "), or put it in the project folder.",
            true);
        return false;
    }
    ApplyEditingPolicy();
    bool const unsaved = Unsaved();
    if (!SaveAll())
    {
        return false;
    }
    std::string publicCommand;
    try
    {
        auto launchArguments = *arguments;
        if (program.stem() == "ppokit-run") launchArguments = launchSettings_.Arguments(command, std::move(launchArguments));
        if (program.stem() == "ppokit-run") VerifyLaunchCompatibility(command, launchArguments);
        publicCommand = editor::QuoteCommandLine(program, launchArguments);
        std::unique_ptr<ipc::Supervisor> session;
        if (program.stem() == "ppokit-run" || project_.controlProtocol == "udp/1")
        {
            session = std::make_unique<ipc::Supervisor>();
            launchArguments.insert(launchArguments.end(), {"--control", session->Endpoint()});
        }
        process_.Start(program, launchArguments, project_.root);
        control_ = std::move(session);
        dashboard_->SetSupervisor(command == schema::ProjectCommand::Train ? control_.get() : nullptr);
    }
    catch (std::exception const& error)
    {
        Report(error.what(), true);
        return false;
    }
    running_ = command;
    ApplyEditingPolicy();
    started_ = std::chrono::steady_clock::now();
    stopRequested_ = false;
    savingAt_.reset();
    escalated_ = false;
    console_.push_back("> " + publicCommand);
    Report(std::string(Verb(command)) + (unsaved ? " (changes saved first)." : "."), false);
    section_ = command == schema::ProjectCommand::Train ? Section::Run : Section::Console;
    if (command == schema::ProjectCommand::Train)
    {
        dashboard_->SetFollow(true);
    }
    // Open the world's view for training and inference when the project has a Visualiseur.
    if ((command == schema::ProjectCommand::Train || command == schema::ProjectCommand::Inference)
        && !viewer_.Running())
    {
        OpenViewer(true);
    }
    return true;
}

bool ProjectWindow::ReadyToClose()
{
    Poll();
    if (!process_.Running())
    {
        RequestViewerStop();
        return !viewer_.Running();
    }
    if (!stopRequested_)
    {
        RequestStop();
    }
    return false;
}

void ProjectWindow::RequestStop()
{
    if (!process_.Running())
    {
        return;
    }
    if (stopRequested_) return;
    if (control_) control_->Request(ipc::Command::Stop);
    else if (project_.controlProtocol == "legacy-files" && running_ == schema::ProjectCommand::Train)
    {
        // External project programs may implement only the documented legacy stop request.
        std::error_code error;
        std::filesystem::create_directories(project_.Checkpoints(), error);
        std::ofstream request(project_.StopRequest());
        request << "stop\n";
        request.close();
        if (error || !request) { Report("Could not request a graceful stop.", true); return; }
    }
    else { Kill(); return; }
    stopRequested_ = true;
    stopAt_ = std::chrono::steady_clock::now();
    savingAt_.reset();
    if (viewerForRun_) RequestViewerStop();
    Report(running_ == schema::ProjectCommand::Train
        ? "Stop requested: waiting for a complete update and checkpoint."
        : "Stop requested: closing inference/evaluation and its viewer.", false);
}

void ProjectWindow::RequestViewerStop()
{
    if (!viewer_.Running() || viewerStopAt_) return;
    if (viewerControl_) viewerControl_->Request(ipc::Command::Stop);
    viewerStopAt_ = std::chrono::steady_clock::now();
}

void ProjectWindow::Kill()
{
    if (viewerForRun_) RequestViewerStop();
    process_.Kill();
    Poll();
}

void ProjectWindow::Poll()
{
    if (control_) control_->Tick();
    if (viewerControl_) viewerControl_->Tick();
    if (viewerStopAt_ && std::chrono::steady_clock::now() - *viewerStopAt_ >= InteractiveStopTimeout)
        viewer_.Kill();
    if (!viewer_.Running()) { viewerControl_.reset(); viewerStopAt_.reset(); }
    auto const timeout = running_ == schema::ProjectCommand::Train
        ? std::chrono::seconds(stopTimeoutSeconds_) : InteractiveStopTimeout;
    if (stopRequested_ && control_ && control_->LastState() == ipc::State::Saving && !savingAt_)
    {
        // A save is forward progress, not a hung stop. Give this bounded phase its
        // own deadline so a large durable checkpoint is not killed at the old edge.
        savingAt_ = std::chrono::steady_clock::now();
    }
    auto const stopPhaseStarted = savingAt_.value_or(stopAt_);
    if (stopRequested_ && process_.Running()
        && std::chrono::steady_clock::now() - stopPhaseStarted >= timeout)
    {
        escalated_ = true;
        process_.Kill();
    }
    for (std::string& line : process_.TakeLines())
    {
        console_.push_back(std::move(line));
    }
    if (console_.size() > MaximumConsoleLines)
    {
        console_.erase(console_.begin(), console_.begin() + static_cast<std::ptrdiff_t>(console_.size() - MaximumConsoleLines));
    }
    if (running_ && !process_.Running())
    {
        int const code = process_.ExitCode().value_or(-1);
        for (std::string& line : process_.TakeLines())
        {
            console_.push_back(std::move(line));
        }
        console_.push_back("< exit code " + std::to_string(code));
        std::string const what = schema::CommandName(*running_);
        Report(code == 0 ? what + " finished." : what + " ended with exit code " + std::to_string(code) + ": see the Console.",
            code != 0);
        if (escalated_) Report(what + " exceeded its stop timeout and was terminated.", true);
        if (viewerForRun_) RequestViewerStop();
        viewerForRun_ = false;
        dashboard_->SetSupervisor(nullptr);
        control_.reset();
        if (evaluation_ && *running_ != schema::ProjectCommand::Inference) evaluation_->RefreshResults();
        running_.reset();
        stopRequested_ = false;
        savingAt_.reset();
    }
    ApplyEditingPolicy();
}

OverviewModel ProjectWindow::Model() const
{
    OverviewModel model;
    model.task = task_;
    model.settings = project_.settings;
    model.learnerSettings = project_.learnerSettings;
    model.environment = project_.environmentLibrary;
    model.nativeCurriculum = scenario_.CurriculumSummary();
    model.networkEditable = project_.networkEditable;
    if (network_)
    {
        model.hasNetwork = true;
        model.actorBlocks = network_->Document().At(editor::Network::Actor).nodes.size();
        model.criticBlocks = network_->Document().At(editor::Network::Critic).nodes.size();
    }
    if (training_)
    {
        model.hasTraining = true;
        model.ppo = training_->Document().Values();
        model.curriculum = training_->Document().Curriculum();
        model.probeEvery = training_->Document().ProbeEvery();
    }
    return model;
}

void ProjectWindow::Draw()
{
    Poll();
    DrawRunBar();
    ImGui::Spacing();
    float const height = ImGui::GetContentRegionAvail().y;
    BeginPanel("navigation", ImVec2(NavigationWidth, height));
    DrawNavigation();
    EndPanel();
    ImGui::SameLine();
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoBackground;
    if (section_ == Section::Overview) flags |= ImGuiWindowFlags_HorizontalScrollbar;
    ImGui::BeginChild("section", ImVec2(0.F, height), ImGuiChildFlags_None, flags);
    DrawSection();
    ImGui::EndChild();
}

void ProjectWindow::DrawRunBar()
{
    if (IconButton("back", StudioIcon::Back, "Back to the start screen (unsaved changes must be saved or discarded "
                                             "first)."))
    {
        closeRequested_ = true;
    }
    ImGui::SameLine();
    ImGui::BeginGroup();
    Heading(project_.name.c_str());
    DrawStatusLine();
    ImGui::EndGroup();
    // Keep the control row separate from long paths/status text, especially on narrow windows.
    DrawActions();
    if (std::filesystem::path(schema::CommandExecutable(project_, schema::ProjectCommand::Train)).stem() == "ppokit-run")
        launchSettings_.Draw(project_, references_);
    if (control_)
    {
        ImGui::Text("Control  %s", control_->Connected() ? ipc::Name(control_->LastState()) : "waiting for peer");
        if (!control_->LastError().empty()) ImGui::TextWrapped("%s", control_->LastError().c_str());
        auto const command = control_->LastCommandResult();
        if (command && command->state == ipc::CommandState::Indeterminate)
        {
            ImGui::SameLine();
            ImGui::TextColored(HealthColor(editor::Health::Watch), "Command result indeterminate: %s",
                command->message.c_str());
        }
        if (!stopRequested_ && control_->Connected())
        {
            ImGui::SameLine();
            bool const paused = control_->LastState() == ipc::State::Paused;
            if (ImGui::SmallButton(paused ? "Resume" : "Pause"))
                control_->Request(paused ? ipc::Command::Resume : ipc::Command::Pause);
        }
    }
}

/** @brief Folder of the project, then what runs or the last message. */
void ProjectWindow::DrawStatusLine()
{
    ImGui::PushTextWrapPos(0.F);
    ImGui::TextColored(MutedText, "%s", project_.root.string().c_str());
    if (running_)
    {
        ImGui::TextColored(SecondaryAccent, "%s for %s", Verb(*running_), Elapsed(started_).c_str());
        ImGui::SameLine();
    }
    if (!status_.empty())
    {
        ImGui::TextColored(statusIsError_ ? HealthColor(editor::Health::Problem) : MutedText, "%s", status_.c_str());
    }
    else if (!running_)
    {
        ImGui::TextColored(MutedText, "Ready.");
    }
    ImGui::PopTextWrapPos();
}

/** @brief Run controls, with a second row on narrow windows to keep Stop visible. */
void ProjectWindow::DrawActions()
{
    ImGuiStyle const& style = ImGui::GetStyle();
    char const* viewerLabel = viewer_.Running() ? "Close viewer" : "View";
    float width = IconLabelButtonWidth(StopLabel()) + IconLabelButtonWidth("Save") + IconLabelButtonWidth(viewerLabel)
        + style.ItemSpacing.x * 3.F;
    for (CommandButton const& button : CommandButtons)
        width += IconLabelButtonWidth(schema::CommandName(button.command)) + style.ItemSpacing.x;
    bool const compact = ImGui::GetContentRegionAvail().x < width;
    bool const running = ConfigurationLocked();
    for (CommandButton const& button : CommandButtons)
    {
        bool const offered = schema::CommandArguments(project_, button.command).has_value();
        char const* const help = offered ? CommandHelp(button.command) : "project.json has no such command.";
        bool const primary = button.command == schema::ProjectCommand::Train;
        if (IconLabelButton(schema::CommandName(button.command), button.icon, help, offered && !running && !viewerStopAt_, primary))
        {
            Launch(button.command);
        }
        if (!compact || button.command != schema::ProjectCommand::Inference) ImGui::SameLine();
    }
    char const* stopHelp = "Stop the run and its automatically opened viewer.";
    if (stopRequested_) stopHelp = "End the program at once (the last complete checkpoint stays).";
    else if (running_ == schema::ProjectCommand::Train)
        stopHelp = "Ask training to save a checkpoint and stop after the current update.";
    bool const pressed = IconLabelButton(StopLabel(), stopRequested_ ? StudioIcon::Kill : StudioIcon::Stop, stopHelp, running);
    stopButtonCenter_ = ImGui::GetItemRectMin();
    ImVec2 const maximum = ImGui::GetItemRectMax();
    stopButtonCenter_.x = (stopButtonCenter_.x + maximum.x) * 0.5F;
    stopButtonCenter_.y = (stopButtonCenter_.y + maximum.y) * 0.5F;
    if (pressed) StopOrKill();
    ImGui::SameLine();
    if (IconLabelButton("Save", StudioIcon::Save, "Save network, training, scenario and evaluation changes.", Unsaved() && !running) && SaveAll())
    {
        Report("Saved.", false);
    }
    ImGui::SameLine();
    if (IconLabelButton(viewerLabel, StudioIcon::Inference,
            viewer_.Running() ? "Close the viewer process." : "Open ppokit-viewer on the run's world.",
            !viewerStopAt_))
    {
        if (viewer_.Running()) RequestViewerStop();
        else OpenViewer();
    }
}

namespace {

/** @brief The registered ppokit-viewer, or empty. */
std::filesystem::path FindViewer(std::vector<std::filesystem::path> const& programs)
{
    for (std::filesystem::path const& program : programs)
    {
        if (program.stem() == "ppokit-viewer")
        {
            return program;
        }
    }
    return {};
}

/** @brief The "port" an observer's configuration sends to, if it has one. */
std::optional<int> ObserverPort(schema::ObserverSpec const& observer)
{
    nlohmann::json const config = nlohmann::json::parse(observer.config, nullptr, false);
    if (!config.is_object() || !config.contains("port") || !config.at("port").is_number())
    {
        return std::nullopt;
    }
    return config.at("port").get<int>();
}

} // namespace

void ProjectWindow::OpenViewer(bool const quietWithoutVisualiseur)
{
    if (!task_)
    {
        if (!quietWithoutVisualiseur)
        {
            Report("The project's task.json must be valid to find its Visualiseur.", true);
        }
        return;
    }
    try
    {
        schema::BlockGraphSchema const graphs = schema::LoadBlockGraph(project_.Network(), *task_);
        for (schema::ObserverSpec const& observer : graphs.observers)
        {
            std::optional<int> const port = ObserverPort(observer);
            if (!port)
            {
                continue;
            }
            std::filesystem::path const viewer = FindViewer(programs_);
            if (viewer.empty())
            {
                Report("ppokit-viewer not found: build it (cmake --build ... --target ppokit-viewer).", true);
                return;
            }
            auto session = std::make_unique<ipc::Supervisor>();
            viewer_.Start(viewer, {"--view", observer.component.library->Path().string(), "--port", std::to_string(*port),
                "--control", session->Endpoint()}, project_.root);
            viewerControl_ = std::move(session);
            viewerStopAt_.reset();
            viewerForRun_ = quietWithoutVisualiseur;
            Report("Viewer opened on '" + observer.name + "' (UDP port " + std::to_string(*port) + ").", false);
            return;
        }
        if (!quietWithoutVisualiseur)
        {
            Report("No Visualiseur in the Observers canvas sends frames (an observer with a \"port\" setting).", true);
        }
    }
    catch (std::exception const& error)
    {
        if (!quietWithoutVisualiseur)
        {
            Report(error.what(), true);
        }
    }
}

char const* ProjectWindow::StopLabel() const noexcept
{
    bool const unmanaged = running_ && !control_ && project_.controlProtocol != "legacy-files";
    return stopRequested_ || unmanaged ? "Kill" : "Stop";
}

void ProjectWindow::StopOrKill()
{
    if (stopRequested_)
    {
        Kill();
        return;
    }
    RequestStop();
}

void ProjectWindow::DrawNavigation()
{
    for (SectionInfo const& info : Sections)
    {
        DrawNavigationEntry(info.section, info.icon, info.label, info.help);
    }
    if (!remarks_.empty())
    {
        ImGui::Spacing();
        IconText(StudioIcon::Warning, HealthColor(editor::Health::Watch));
        ImGui::SameLine();
        ImGui::TextColored(HealthColor(editor::Health::Watch), "To look at");
        for (std::string const& remark : remarks_)
        {
            ImGui::PushTextWrapPos(0.F);
            ImGui::TextColored(MutedText, "%s", remark.c_str());
            ImGui::PopTextWrapPos();
        }
    }
}

/** @brief One entry: icon and name, an accent bar when selected, a dot when its file is unsaved. */
void ProjectWindow::DrawNavigationEntry(Section section, StudioIcon icon, char const* label, char const* help)
{
    float const line = ImGui::GetTextLineHeight();
    float const height = line * NavigationEntryLines;
    float const width = ImGui::GetContentRegionAvail().x;
    ImVec2 const topLeft = ImGui::GetCursorScreenPos();
    if (ImGui::InvisibleButton(label, ImVec2(width, height)))
    {
        if (section == Section::NativeCurriculum) scenario_.Reload();
        section_ = section;
    }
    ItemTooltip(help);
    bool const selected = section_ == section;
    bool const hovered = ImGui::IsItemHovered();
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    ImVec2 const bottomRight(topLeft.x + width, topLeft.y + height);
    if (selected || hovered)
    {
        drawList.AddRectFilled(topLeft, bottomRight, ImGui::ColorConvertFloat4ToU32(RaisedSurface), height * 0.25F);
    }
    if (selected)
    {
        drawList.AddRectFilled(ImVec2(topLeft.x, topLeft.y + height * 0.22F), ImVec2(topLeft.x + 3.F, bottomRight.y - height * 0.22F),
            ImGui::ColorConvertFloat4ToU32(PrimaryAccent), 1.5F);
    }
    ImVec4 const ink = selected ? PrimaryAccent : ImGui::GetStyleColorVec4(ImGuiCol_Text);
    float const iconSize = line * 1.15F;
    ImVec2 const iconCorner(topLeft.x + line * 0.7F, topLeft.y + (height - iconSize) * 0.5F);
    PaintIcon(drawList, icon, iconCorner, iconSize, ImGui::ColorConvertFloat4ToU32(ink));
    drawList.AddText(ImVec2(iconCorner.x + iconSize + line * 0.6F, topLeft.y + (height - line) * 0.5F),
        ImGui::ColorConvertFloat4ToU32(ink), label);
    if (SectionUnsaved(section))
    {
        drawList.AddCircleFilled(ImVec2(bottomRight.x - line * 0.8F, topLeft.y + height * 0.5F), line * 0.2F,
            ImGui::ColorConvertFloat4ToU32(HealthColor(editor::Health::Watch)));
    }
}

bool ProjectWindow::SectionUnsaved(Section section) const
{
    if (section == Section::Environment)
        return scenario_.Dirty();
    if (section == Section::NativeCurriculum)
        return scenario_.Dirty() || (!project_.HasDeclaredCurriculum() && training_ && training_->Dirty());
    if (section == Section::Network)
    {
        return network_ && network_->Dirty();
    }
    if (section == Section::Training)
    {
        return training_ && training_->Dirty();
    }
    if (section == Section::Evaluation)
    {
        return evaluation_ && evaluation_->Dirty();
    }
    return false;
}

void ProjectWindow::DrawSection()
{
    ApplyEditingPolicy();
    switch (section_)
    {
    case Section::Overview:
        DrawOverview();
        break;
    case Section::Environment:
        if (ImGui::Button("Reload settings")) scenario_.Reload();
        ImGui::SetNextItemWidth(135.0F);
        ImGui::SliderInt("Graceful stop timeout (s)", &stopTimeoutSeconds_, 1, 300);
        scenario_.DrawEnvironment();
        break;
    case Section::Network:
        DrawNetwork();
        break;
    case Section::Training:
        DrawTraining();
        break;
    case Section::NativeCurriculum:
        if (project_.HasDeclaredCurriculum()) scenario_.DrawCurriculum();
        else
        {
            Heading("Curriculum", "Stages and their configuration source.");
            if (training_)
            {
                auto const& curriculum = training_->Document().Curriculum();
                if (!curriculum || curriculum->stages.empty())
                    MutedWrapped("No curriculum stages are configured in training.json. The task defaults are used.");
                else ImGui::Text("%zu configured stages", curriculum->stages.size());
                training_->DrawCurriculum();
            }
            else
            {
                MutedWrapped("This project declares no native curriculum and no PPOKit training.json. "
                    "External CI curricula are not configured by this Studio.");
                if (!project_.learnerSettings.empty() && ImGui::Button("Open native training settings"))
                    section_ = Section::Training;
            }
        }
        break;
    case Section::Evaluation:
        DrawEvaluation();
        break;
    case Section::Task:
        DrawTask();
        break;
    case Section::Run:
        DrawRun();
        break;
    case Section::Console:
        DrawConsole();
        break;
    }
}

void ProjectWindow::DrawOverview()
{
    overview_.Draw(Model());
    if (std::optional<OverviewTarget> const target = overview_.TakeChoice())
    {
        Follow(*target);
    }
}

void ProjectWindow::Follow(OverviewTarget target)
{
    switch (target)
    {
    case OverviewTarget::Environment:
        section_ = Section::Environment;
        break;
    case OverviewTarget::Task:
        section_ = Section::Task;
        break;
    case OverviewTarget::Actor:
    case OverviewTarget::Critic:
        section_ = Section::Network;
        if (network_)
        {
            network_->ShowNetwork(target == OverviewTarget::Actor ? editor::Network::Actor : editor::Network::Critic);
        }
        break;
    case OverviewTarget::Training:
        section_ = Section::Training;
        if (training_) training_->Show(TrainSettingsView::Section::Ppo);
        break;
    case OverviewTarget::Curriculum:
        if (project_.HasDeclaredCurriculum()) scenario_.Reload();
        section_ = Section::NativeCurriculum;
        break;
    case OverviewTarget::Rewards:
        // Live weights belong to a training run; without a journal, the task lists the channels.
        if (std::filesystem::exists(project_.Journal()))
        {
            section_ = Section::Run;
            dashboard_->SelectTab(TrainingDashboardView::Tab::Weights);
        }
        else
        {
            section_ = Section::Task;
        }
        break;
    }
}

void ProjectWindow::DrawNetwork()
{
    if (network_)
    {
        network_->Draw();
        return;
    }
    Heading("Network", "The actor and the critic of the project.");
    MutedWrapped(task_ ? "network.json is missing or cannot be opened: see \"To look at\" on the left."
                       : "The network needs a valid task.json first.");
}

void ProjectWindow::DrawEvaluation()
{
    if (evaluation_)
    {
        evaluation_->Draw(dashboard_->Log());
        return;
    }
    Heading("Evaluation", "The evaluation suites of the project (eval/).");
    MutedWrapped("The suites need a valid task.json first: their metrics come from it.");
}

void ProjectWindow::DrawTraining()
{
    if (training_)
    {
        training_->Draw(project_.HasDeclaredCurriculum() ? &scenario_ : nullptr);
        return;
    }
    Heading("Training", "PPO and GAE parameters, probes and curriculum.");
    if (!project_.learnerSettings.empty())
    {
        if (ImGui::Button("Reload settings")) scenario_.Reload();
        scenario_.DrawLearnerSettings();
        return;
    }
    if (!project_.settings.empty())
    {
        MutedWrapped("This program owns its training configuration. The scenario source is shown below.");
        if (ImGui::Button("Open environment settings")) section_ = Section::Environment;
        if (project_.HasDeclaredCurriculum())
        {
            ImGui::SameLine();
            if (ImGui::Button("Open curriculum")) section_ = Section::NativeCurriculum;
        }
        scenario_.DrawLearnerSettings();
        return;
    }
    MutedWrapped("training.json cannot be read: see \"To look at\" on the left.");
}

void ProjectWindow::DrawTask()
{
    Heading("Task contract (read-only)", "The environment module defines what the policy sees and controls.");
    ImGui::TextWrapped("%s", project_.Task().string().c_str());
    MutedWrapped("This is a compatibility contract, not a scenario editor. Changing observations or actions can invalidate existing checkpoints.");
    if (ImGui::Button("Environment settings")) section_ = Section::Environment;
    ImGui::SameLine();
    if (ImGui::Button("Training settings")) section_ = Section::Training;
    ImGui::SameLine();
    if (ImGui::Button("Copy task path")) ImGui::SetClipboardText(project_.Task().string().c_str());
    if (!task_)
    {
        MutedWrapped("task.json is missing or invalid: see \"To look at\" on the left.");
        return;
    }
    ImGui::TextWrapped("Validated schema  %s  |  %zu observations  |  %zu actions", task_->name.c_str(),
        task_->observation.size, task_->actionSize);
    ImGui::TextWrapped("Environment module  %s   Network module  %s", project_.environmentLibrary.c_str(),
        project_.networkLibrary.c_str());
    if (!task_->contract.empty()) ImGui::TextWrapped("Contract  %s", task_->contract.c_str());
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    ImGui::SeparatorText("Observation");
    if (ImGui::BeginTable("observation", 3, flags))
    {
        ImGui::TableSetupColumn("Group");
        ImGui::TableSetupColumn("Values");
        ImGui::TableSetupColumn("Features");
        ImGui::TableHeadersRow();
        for (schema::ObservationGroup const& group : task_->observation.groups)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(group.name.c_str());
            ImGui::TableNextColumn();
            if (group.isSet)
            {
                ImGui::Text("set of up to %zu, %zu each", group.capacity, group.features);
                ItemTooltip("A set of entities (tracks, team mates): some places may be empty.");
            }
            else
            {
                ImGui::Text("%zu", group.features);
            }
            ImGui::TableNextColumn();
            std::string names;
            for (std::string const& name : group.featureNames)
            {
                names += (names.empty() ? "" : ", ") + name;
            }
            ImGui::TextWrapped("%s", names.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::SeparatorText("Actions");
    for (schema::ActionGroup const& group : task_->actions)
    {
        std::string components;
        for (std::string const& name : group.componentNames)
        {
            components += (components.empty() ? "" : ", ") + name;
        }
        ImGui::BulletText("%s: %s (%s), from %g to %g", group.name.c_str(),
            group.head == schema::ActionHead::Gaussian ? "continuous" : "on/off", components.c_str(), group.lowerBound,
            group.upperBound);
    }
    ImGui::SeparatorText("Reward channels");
    if (ImGui::BeginTable("rewards", 3, flags))
    {
        ImGui::TableSetupColumn("Channel");
        ImGui::TableSetupColumn("Label");
        ImGui::TableSetupColumn("Weight");
        ImGui::TableHeadersRow();
        for (schema::RewardChannel const& channel : task_->rewards)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(channel.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(channel.label.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%g", channel.weight);
        }
        ImGui::EndTable();
    }
    MutedWrapped("Weights of a running training are shown in Run > Reward weights.");
    if (ImGui::Button("Open reward weights")) { section_ = Section::Run; dashboard_->SelectTab(TrainingDashboardView::Tab::Weights); }
    if (!task_->scenario.empty())
    {
        ImGui::SeparatorText("Scenario parameters");
        for (schema::ScenarioParameter const& parameter : task_->scenario)
        {
            ImGui::BulletText("%s", parameter.name.c_str());
        }
        MutedWrapped("The curriculum chooses their values, stage by stage.");
        if (ImGui::Button("Open curriculum"))
            Follow(OverviewTarget::Curriculum);
    }
    if (!task_->metrics.empty())
    {
        ImGui::SeparatorText("Episode metrics");
        for (schema::EpisodeMetric const& metric : task_->metrics)
        {
            bool const flag = metric.kind == schema::MetricKind::Flag;
            ImGui::BulletText("%s  %s%s%s", metric.name.c_str(), metric.label.c_str(), metric.unit.empty() ? "" : ", ",
                metric.unit.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled(flag ? "(flag: %s.rate)" : "(number: %s.mean)", metric.name.c_str());
        }
        MutedWrapped("The environment reports them at the end of each episode; the evaluation suites "
                     "(Evaluation) set gates on them and rank the BEST policy with them.");
    }
}

void ProjectWindow::DrawRun()
{
    if (!running_ && std::filesystem::path(schema::CommandExecutable(project_, schema::ProjectCommand::Train)).stem() == "ppokit-run")
    {
        if (!control_ && ImGui::Button("Connect a command-line run"))
        {
            try { control_ = std::make_unique<ipc::Supervisor>(); dashboard_->SetSupervisor(control_.get()); }
            catch (std::exception const& error) { Report(error.what(), true); }
        }
        if (control_)
        {
            if (ImGui::Button("Copy --control argument"))
                ImGui::SetClipboardText(("--control " + control_->Endpoint()).c_str());
            ItemTooltip("Append this private argument when starting ppokit-run from a terminal.");
            ImGui::SameLine();
            if (ImGui::Button("Disconnect")) { dashboard_->SetSupervisor(nullptr); control_.reset(); }
            if (control_ && control_->Connected())
            {
                ImGui::SameLine();
                if (ImGui::Button("Stop attached run")) control_->Request(ipc::Command::Stop);
            }
        }
    }
    dashboard_->Draw();
}

void ProjectWindow::DrawConsole()
{
    Heading("Console", "Everything the program writes, live.");
    ImGui::Checkbox("Follow the end", &followConsole_);
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear"))
    {
        console_.clear();
    }
    ImGui::BeginChild("console-lines", ImVec2(0.F, 0.F), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(console_.size()));
    while (clipper.Step())
    {
        for (int line = clipper.DisplayStart; line < clipper.DisplayEnd; ++line)
        {
            std::string const& text = console_[static_cast<std::size_t>(line)];
            bool const command = !text.empty() && (text[0] == '>' || text[0] == '<');
            if (command)
            {
                ImGui::TextColored(SecondaryAccent, "%s", text.c_str());
            }
            else
            {
                ImGui::TextUnformatted(text.c_str());
            }
        }
    }
    if (followConsole_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.F)
    {
        ImGui::SetScrollHereY(1.F);
    }
    ImGui::EndChild();
}

} // namespace ppokit::studio
