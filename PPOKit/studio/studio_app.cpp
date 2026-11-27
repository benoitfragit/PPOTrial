/**
 * @file PPOKit/studio/studio_app.cpp
 * @brief Start screen, project window and the unsaved-changes question.
 */
#include "studio_app.hpp"

#include "studio_theme.hpp"

#include "ppokit/editor/recent_projects.hpp"
#include "ppokit/tools/scaffold.hpp"

#include <imgui.h>

#include <exception>
#include <utility>

namespace ppokit::studio {
namespace {

constexpr char UnsavedPopup[] = "Unsaved changes";

} // namespace

StudioApp::StudioApp(std::filesystem::path recentFile, std::vector<std::filesystem::path> programs)
    : recentFile_(std::move(recentFile)), programs_(std::move(programs)), start_(recentFile_)
{
}

bool StudioApp::OpenProject(std::filesystem::path const& folder)
{
    try
    {
        project_ = std::make_unique<ProjectWindow>(folder, programs_);
    }
    catch (std::exception const& error)
    {
        start_.ShowError(error.what());
        return false;
    }
    editor::RememberProject(recentFile_, folder);
    start_.Refresh();
    return true;
}

bool StudioApp::CreateProject(std::string const& name, std::filesystem::path const& parent)
{
    std::filesystem::path folder;
    try
    {
        folder = tools::WriteNewProject(name, parent);
    }
    catch (std::exception const& error)
    {
        start_.ShowError(error.what());
        return false;
    }
    return OpenProject(folder);
}

bool StudioApp::RequestQuit()
{
    if (project_ && project_->Unsaved())
    {
        pending_ = Pending::Quit;
        return false;
    }
    FinishClose(Pending::Quit);
    return quitConfirmed_;
}

void StudioApp::FinishClose(Pending destination)
{
    if (project_ && !project_->ReadyToClose())
    {
        stopping_ = destination;
        return;
    }
    stopping_ = Pending::None;
    if (destination == Pending::Quit)
    {
        quitConfirmed_ = true;
    }
    else
    {
        Leave();
    }
}

void StudioApp::Leave()
{
    project_.reset();
    start_.Refresh();
}

void StudioApp::Draw()
{
    if (!project_)
    {
        start_.Draw();
        if (std::optional<StartChoice> const choice = start_.TakeChoice())
        {
            if (choice->kind == StartChoice::Kind::Open)
            {
                OpenProject(choice->folder);
            }
            else
            {
                CreateProject(choice->name, choice->folder);
            }
        }
        return;
    }
    project_->Draw();
    if (stopping_ != Pending::None)
    {
        FinishClose(stopping_);
        return;
    }
    if (project_->CloseRequested())
    {
        project_->ClearCloseRequest();
        if (project_->Unsaved())
        {
            pending_ = Pending::Back;
        }
        else
        {
            FinishClose(Pending::Back);
            return;
        }
    }
    DrawUnsavedQuestion();
}

void StudioApp::DrawUnsavedQuestion()
{
    if (pending_ == Pending::None)
    {
        return;
    }
    if (!ImGui::IsPopupOpen(UnsavedPopup))
    {
        ImGui::OpenPopup(UnsavedPopup);
    }
    if (!ImGui::BeginPopupModal(UnsavedPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        return;
    }
    ImGui::TextUnformatted("The project has unsaved network, training, scenario or evaluation changes.");
    bool const quitting = pending_ == Pending::Quit;
    bool done = false;
    bool leave = false;
    if (AccentButton("Save"))
    {
        done = true;
        leave = project_->SaveAll();
    }
    ItemTooltip("Validate and save the changed project files, then continue.");
    ImGui::SameLine();
    if (ImGui::Button("Discard"))
    {
        done = true;
        leave = true;
    }
    ItemTooltip("Leave without writing the changes.");
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
    {
        done = true;
    }
    if (done)
    {
        ImGui::CloseCurrentPopup();
        pending_ = Pending::None;
    }
    ImGui::EndPopup();
    if (leave)
    {
        FinishClose(quitting ? Pending::Quit : Pending::Back);
    }
}

std::vector<std::filesystem::path> ProgramsIn(std::filesystem::path const& folder)
{
    std::vector<std::filesystem::path> programs;
    std::error_code error;
    for (auto const& entry : std::filesystem::directory_iterator(folder, error))
    {
        if (entry.is_regular_file(error))
        {
            programs.push_back(entry.path());
        }
    }
    return programs;
}

} // namespace ppokit::studio
