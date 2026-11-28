/**
 * @file PPOKit/studio/studio_app.hpp
 * @brief The studio: the start screen, or the project the user opened; with a question
 *        before unsaved changes are lost (going back, or quitting).
 */
#pragma once

#include "project_window.hpp"
#include "start_view.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ppokit::studio {

class StudioApp final
{
public:
    /**
     * @param recentFile Where the recent projects are kept.
     * @param programs Programs the project windows may launch (schema::ReadProgramRegistry and others).
     */
    StudioApp(std::filesystem::path recentFile, std::vector<std::filesystem::path> programs);

    void Draw();

    /** @brief Opens the project in `folder`; on failure the start screen shows why. */
    bool OpenProject(std::filesystem::path const& folder);
    /** @brief Creates project `name` in `parent` and opens it. */
    bool CreateProject(std::string const& name, std::filesystem::path const& parent);

    /** @brief The window asks to close: true when it may (nothing unsaved), else a question is shown. */
    bool RequestQuit();
    /** @brief Whether the user confirmed quitting (after RequestQuit). */
    bool QuitConfirmed() const noexcept { return quitConfirmed_; }

    ProjectWindow* Project() noexcept { return project_.get(); }
    StartView& Start() noexcept { return start_; }

private:
    /** @brief What the unsaved-changes question is about. */
    enum class Pending
    {
        None,
        Back,
        Quit,
    };

    void DrawUnsavedQuestion();
    void Leave();
    void FinishClose(Pending destination);

    std::filesystem::path recentFile_;
    std::vector<std::filesystem::path> programs_;
    StartView start_;
    std::unique_ptr<ProjectWindow> project_;
    Pending pending_ = Pending::None;
    Pending stopping_ = Pending::None;
    bool quitConfirmed_ = false;
};

/** @brief Executable files in `folder` (programs beside the studio). */
std::vector<std::filesystem::path> ProgramsIn(std::filesystem::path const& folder);

} // namespace ppokit::studio
