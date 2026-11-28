/**
 * @file PPOKit/studio/project_window.hpp
 * @brief One project open in the studio: a run bar (Train, Evaluate, Inference, Stop), a
 *        navigation column (Overview, Network, Training, Evaluation, Task, Run, Console) and
 *        the view of the section chosen.
 * @details Everything is read from and written to the project folder (schema/project.hpp).
 * Train, Evaluate and Inference start the project's program in the background
 * (editor/process.hpp): its console shows live in Console, its curves in Run. Unsaved
 * network or training changes are saved before a launch, so that the program sees them.
 * Stop asks the supervised process to finish using UDP; training saves at an update boundary.
 * Kill ends it at once. Needs current ImGui, imnodes and ImPlot contexts.
 */
#pragma once

#include "evaluation_view.hpp"
#include "graph_editor_view.hpp"
#include "overview_view.hpp"
#include "scenario_view.hpp"
#include "launch_settings.hpp"
#include "studio_icons.hpp"
#include "train_settings_view.hpp"
#include "training_dashboard_view.hpp"

#include "ppokit/editor/process.hpp"
#include "ppokit/ipc/control.hpp"
#include "ppokit/schema/project.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

class ProjectWindow final
{
public:
    enum class Section
    {
        Overview,
        Network,
        Training,
        Evaluation,
        Task,
        Environment,
        NativeCurriculum,
        Run,
        Console,
    };

    /**
     * @param folder The project folder.
     * @param programs Programs the build registered (schema::ReadProgramRegistry).
     * @throws std::runtime_error when the folder holds no valid project.json.
     */
    ProjectWindow(std::filesystem::path const& folder, std::vector<std::filesystem::path> programs);
    ~ProjectWindow();
    ProjectWindow(ProjectWindow const&) = delete;
    ProjectWindow& operator=(ProjectWindow const&) = delete;

    void Draw();

    schema::Project const& Project() const noexcept { return project_; }
    Section Shown() const noexcept { return section_; }
    void Show(Section section) noexcept { section_ = section; }

    /** @brief Unsaved network, training, scenario or evaluation settings. */
    bool Unsaved() const noexcept;
    /** @brief Saves changed project files; false with a status on failure. */
    bool SaveAll();

    /** @brief Whether the user asked to go back to the start screen. */
    bool CloseRequested() const noexcept { return closeRequested_; }
    /** @brief Forgets the request to go back (it was handled, or cancelled). */
    void ClearCloseRequest() noexcept { closeRequested_ = false; }

    /**
     * @brief Starts the project's program for `command` (saving first).
     * @return False, with the reason in Status(), when it cannot start.
     */
    bool Launch(schema::ProjectCommand command);
    /** @brief Asks the program to save and stop after its current update. */
    void RequestStop();
    /** @brief Starts graceful training shutdown. False while a checkpoint is still pending. */
    bool ReadyToClose();
    /** @brief Ends the program at once. */
    void Kill();
    bool Running() noexcept;

    std::string const& Status() const noexcept { return status_; }
    std::vector<std::string> const& ConsoleLines() const noexcept { return console_; }
    /** @brief Problems found while opening the project, in plain words. */
    std::vector<std::string> const& Remarks() const noexcept { return remarks_; }
    OverviewView& Overview() noexcept { return overview_; }
    /** @brief Location of the visible stop button for headless input tests. */
    ImVec2 StopButtonCenter() const noexcept { return stopButtonCenter_; }
    GraphEditorView* Network() noexcept { return network_.get(); }
    TrainSettingsView* Training() noexcept { return training_.get(); }
    EvaluationView* Evaluation() noexcept { return evaluation_.get(); }

private:
    void Open();
    void DrawRunBar();
    void DrawStatusLine();
    void DrawActions();
    /** @brief "Stop" until a stop was asked, then "Kill". */
    char const* StopLabel() const noexcept;
    void StopOrKill();
    /** @brief Stop the viewer independently, with a short forced-termination deadline. */
    void RequestViewerStop();
    void DrawNavigation();
    void DrawNavigationEntry(Section section, StudioIcon icon, char const* label, char const* help);
    /** @brief The section edits a file with unsaved changes. */
    bool SectionUnsaved(Section section) const;
    void DrawSection();
    void DrawOverview();
    void DrawNetwork();
    void DrawTraining();
    void DrawEvaluation();
    void DrawTask();
    void DrawRun();
    void DrawConsole();
    /** @brief Follows a click on the overview. */
    void Follow(OverviewTarget target);
    /** @brief New console lines, and the end of the program. */
    void Poll();
    /** @brief Whether a local or attached active run owns the persisted configuration. */
    bool ConfigurationLocked() const noexcept;
    /** @brief Propagates the current configuration lock to every persistent editor. */
    void ApplyEditingPolicy() noexcept;
    /** @brief Refuses a selected graph artifact incompatible with the current semantic graph. */
    void VerifyLaunchCompatibility(schema::ProjectCommand command,
        std::vector<std::string> const& arguments) const;
    OverviewModel Model() const;
    void Report(std::string message, bool error);
    /**
     * @brief Opens ppokit-viewer on the first observer of network.json that sends frames
     *        (a "port" setting): the window on the world of the run.
     * @param quietWithoutVisualiseur True: a project without one says nothing (automatic opening).
     */
    void OpenViewer(bool quietWithoutVisualiseur = false);

    schema::Project project_;
    ScenarioView scenario_;
    LaunchSettings launchSettings_;
    std::vector<schema::PolicyReference> references_;
    std::vector<std::filesystem::path> programs_;
    std::optional<schema::TaskSchema> task_;
    std::unique_ptr<GraphEditorView> network_;
    std::unique_ptr<TrainSettingsView> training_;
    /** @brief The suites of eval/, once task.json reads (their metrics come from it). */
    std::unique_ptr<EvaluationView> evaluation_;
    std::unique_ptr<TrainingDashboardView> dashboard_;
    OverviewView overview_;
    std::unique_ptr<ipc::Supervisor> control_;
    std::chrono::steady_clock::time_point stopAt_{};
    std::optional<std::chrono::steady_clock::time_point> savingAt_;
    int stopTimeoutSeconds_ = 30;
    bool viewerForRun_ = false;
    std::unique_ptr<ipc::Supervisor> viewerControl_;
    std::optional<std::chrono::steady_clock::time_point> viewerStopAt_;
    bool escalated_ = false;
    ImVec2 stopButtonCenter_{};
    editor::Process process_;
    /** @brief ppokit-viewer, opened from the View button; independent of the run. */
    editor::Process viewer_;
    std::optional<schema::ProjectCommand> running_;
    std::chrono::steady_clock::time_point started_{};
    bool stopRequested_ = false;
    bool followConsole_ = true;
    std::vector<std::string> console_;
    std::vector<std::string> remarks_;
    std::string status_;
    bool statusIsError_ = false;
    Section section_ = Section::Overview;
    bool closeRequested_ = false;
};

} // namespace ppokit::studio
