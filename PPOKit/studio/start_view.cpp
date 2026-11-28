/**
 * @file PPOKit/studio/start_view.cpp
 * @brief Drawing of the start screen.
 */
#include "start_view.hpp"

#include "file_dialogs.hpp"
#include "studio_icons.hpp"
#include "studio_theme.hpp"

#include "ppokit/editor/recent_projects.hpp"
#include "ppokit/tools/scaffold.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace ppokit::studio {
namespace {

/** @brief Width of the centered column of the start screen. */
constexpr float ColumnWidth = 680.F;
/** @brief Side of the logo and of the action tiles. */
constexpr float LogoSide = 72.F;
constexpr float TileSide = 116.F;

void CopyInto(std::array<char, 512>& buffer, std::string const& text)
{
    std::size_t const length = std::min(text.size(), buffer.size() - 1);
    std::memcpy(buffer.data(), text.data(), length);
    buffer[length] = '\0';
}

} // namespace

StartView::StartView(std::filesystem::path recentFile) : recentFile_(std::move(recentFile))
{
    Refresh();
    std::error_code error;
    CopyInto(locationBuffer_, std::filesystem::current_path(error).string());
}

void StartView::Refresh()
{
    recent_ = editor::LoadRecentProjects(recentFile_);
}

std::optional<StartChoice> StartView::TakeChoice()
{
    if (choice_)
    {
        error_.clear();
    }
    return std::exchange(choice_, std::nullopt);
}

void StartView::Draw()
{
    float const width = std::min(ColumnWidth, ImGui::GetContentRegionAvail().x);
    // The content sits a little above the middle of the window, as on MFDStudio's start page.
    float const free = ImGui::GetContentRegionAvail().y - EstimatedHeight();
    ImGui::Dummy(ImVec2(0.F, std::max(ImGui::GetFontSize(), free * 0.38F)));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.F, (ImGui::GetContentRegionAvail().x - width) * 0.5F));
    // A column of its own, so that centered text is centered in the column, not in the window.
    ImGui::BeginChild("start-column", ImVec2(width, 0.F), ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
    ImGui::PushItemWidth(width);
    DrawBrand(width);
    DrawActions(width);
    if (!error_.empty())
    {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
        ImGui::TextColored(HealthColor(editor::Health::Problem), "%s", error_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (typingPath_)
    {
        DrawOpenByPath(width);
    }
    if (creating_)
    {
        DrawCreate(width);
    }
    DrawRecent(width);
    ImGui::PopItemWidth();
    ImGui::EndChild();
}

float StartView::EstimatedHeight() const
{
    float const line = ImGui::GetFrameHeightWithSpacing();
    float const recentRows = static_cast<float>(std::max<std::size_t>(recent_.size(), 1));
    float const formLines = creating_ ? 9.F : (typingPath_ ? 2.F : 0.F);
    return LogoSide + TileSide + line * (7.F + formLines + 1.6F * recentRows);
}

void StartView::DrawBrand(float width)
{
    // The logo: the learning loop, dark on a light rounded square.
    ImVec2 const cursor = ImGui::GetCursorScreenPos();
    ImVec2 const corner(cursor.x + (width - LogoSide) * 0.5F, cursor.y);
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    drawList.AddRectFilled(corner, ImVec2(corner.x + LogoSide, corner.y + LogoSide),
        ImGui::ColorConvertFloat4ToU32(ImVec4(0.96F, 0.97F, 0.98F, 1.F)), LogoSide * 0.2F);
    PaintIcon(drawList, StudioIcon::Overview, ImVec2(corner.x + LogoSide * 0.1F, corner.y + LogoSide * 0.1F),
        LogoSide * 0.8F, ImGui::ColorConvertFloat4ToU32(BackgroundSurface));
    ImGui::Dummy(ImVec2(width, LogoSide + ImGui::GetStyle().ItemSpacing.y));
    CenteredText("P P O K I T    S T U D I O", MutedText);
    ImGui::Dummy(ImVec2(width, ImGui::GetFontSize()));
    CenteredText("Open or create a project", ImGui::GetStyleColorVec4(ImGuiCol_Text), 1.F, true);
    CenteredText("A project folder holds your task, network, training and results.", MutedText);
    ImGui::Dummy(ImVec2(width, ImGui::GetFontSize() * 0.6F));
}

void StartView::DrawActions(float width)
{
    float const spacing = ImGui::GetStyle().ItemSpacing.x * 2.F;
    float const row = TileSide * 2.F + spacing;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.F, (width - row) * 0.5F));
    if (IconTile("Open project", StudioIcon::Open, "Open the folder of a project (the one with project.json).",
            TileSide))
    {
        OpenFolder();
    }
    ImGui::SameLine(0.F, spacing);
    if (IconTile("New project", StudioIcon::New,
            "Create a project that already trains (a point that moves to a target), to change into your own task. "
            "Its program has to be built once.",
            TileSide))
    {
        creating_ = !creating_;
        typingPath_ = false;
    }
    ImGui::Spacing();
    CenteredText("Open an existing project folder, or create a new one.", MutedText);
}

/** @brief Opens the native folder picker, or the typed-path field where there is none. */
void StartView::OpenFolder()
{
    creating_ = false;
    if (!NativeDialogsAvailable())
    {
        typingPath_ = !typingPath_;
        return;
    }
    std::filesystem::path const start = recent_.empty() ? std::filesystem::path() : recent_.front().parent_path();
    if (std::optional<std::filesystem::path> const folder = PickFolder(start, L"Open a PPOKit project folder"))
    {
        choice_ = StartChoice{StartChoice::Kind::Open, *folder, {}};
    }
}

void StartView::DrawOpenByPath(float width)
{
    ImGui::Spacing();
    BeginPanel("open-path", ImVec2(width, ImGui::GetFrameHeightWithSpacing() + 2.F * ImGui::GetStyle().WindowPadding.y));
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 5.F);
    bool const entered = ImGui::InputTextWithHint("##project-path", "project folder (the one with project.json)",
        pathBuffer_.data(), pathBuffer_.size(), ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((IconLabelButton("Open", StudioIcon::Open, "Open the project in this folder.", pathBuffer_[0] != '\0', true)
            || entered)
        && pathBuffer_[0] != '\0')
    {
        choice_ = StartChoice{StartChoice::Kind::Open, std::filesystem::path(pathBuffer_.data()), {}};
    }
    EndPanel();
}

void StartView::DrawCreate(float width)
{
    ImGui::Spacing();
    float const height = ImGui::GetFrameHeightWithSpacing() * 7.F + 2.F * ImGui::GetStyle().WindowPadding.y;
    BeginPanel("create", ImVec2(width, height));
    ImGui::TextUnformatted("Create a project");
    MutedWrapped("Choose a name and the folder that will contain your new project.");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 14.F);
    ImGui::InputTextWithHint("Name", "e.g. lander", nameBuffer_.data(), nameBuffer_.size());
    HelpMark("Lowercase letters, digits and '_', starting with a letter. It names the folder and the program.");
    ImGui::TextUnformatted("Parent folder");
    ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 6.F);
    ImGui::InputText("##location", locationBuffer_.data(), locationBuffer_.size());
    ItemTooltip("Folder in which the project folder is created.");
    if (NativeDialogsAvailable())
    {
        ImGui::SameLine();
        if (IconButton("browse", StudioIcon::Open, "Choose the folder"))
        {
            if (std::optional<std::filesystem::path> const folder =
                    PickFolder(std::filesystem::path(locationBuffer_.data()), L"Where to create the project"))
            {
                CopyInto(locationBuffer_, folder->string());
            }
        }
    }
    std::string const name = nameBuffer_.data();
    bool const validName = tools::ValidProjectName(name);
    bool const valid = validName && locationBuffer_[0] != '\0';
    if (!validName)
        MutedWrapped("Name: start with a lowercase letter; use lowercase letters, digits or underscores.");
    else if (locationBuffer_[0] == '\0')
        MutedWrapped("Choose a parent folder to enable Create.");
    if (IconLabelButton("Create", StudioIcon::New,
            valid ? "Write the project folder and open it." : "Enter a valid name and parent folder first.", valid, true))
    {
        choice_ = StartChoice{StartChoice::Kind::Create, std::filesystem::path(locationBuffer_.data()), name};
        creating_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
    {
        creating_ = false;
    }
    EndPanel();
}

void StartView::DrawRecent(float width)
{
    ImGui::Dummy(ImVec2(width, ImGui::GetFontSize()));
    IconText(StudioIcon::Recent, MutedText);
    ImGui::SameLine();
    ImGui::TextColored(MutedText, "Recent projects");
    if (recent_.empty())
    {
        MutedWrapped("The projects you open appear here.");
        return;
    }
    std::optional<std::filesystem::path> forgotten;
    for (std::filesystem::path const& folder : recent_)
    {
        ImGui::PushID(folder.string().c_str());
        if (DrawRecentRow(folder, width))
        {
            forgotten = folder;
        }
        ImGui::PopID();
    }
    if (forgotten)
    {
        editor::ForgetProject(recentFile_, *forgotten);
        Refresh();
    }
}

/**
 * @brief One recent project as a card: name and folder; a click opens it, the trash icon forgets it.
 * @return True when the user asked to forget it.
 */
bool StartView::DrawRecentRow(std::filesystem::path const& folder, float width)
{
    float const line = ImGui::GetTextLineHeight();
    float const padding = ImGui::GetStyle().FramePadding.x;
    float const height = line * 2.F + padding * 1.6F;
    float const trash = ImGui::GetFrameHeight();
    ImVec2 const topLeft = ImGui::GetCursorScreenPos();
    ImVec2 const bottomRight(topLeft.x + width, topLeft.y + height);
    if (ImGui::InvisibleButton("open", ImVec2(width - trash - padding, height)))
    {
        choice_ = StartChoice{StartChoice::Kind::Open, folder, {}};
    }
    bool const hovered = ImGui::IsItemHovered();
    ItemTooltip("Open this project.");
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    drawList.AddRectFilled(topLeft, bottomRight, ImGui::ColorConvertFloat4ToU32(hovered ? RaisedSurface : PanelSurface),
        height * 0.2F);
    ImVec2 const text(topLeft.x + padding * 1.5F, topLeft.y + padding * 0.8F);
    drawList.AddText(text, ImGui::GetColorU32(ImGuiCol_Text), folder.filename().string().c_str());
    drawList.AddText(ImVec2(text.x, text.y + line), ImGui::ColorConvertFloat4ToU32(MutedText),
        folder.parent_path().string().c_str());
    ImGui::SameLine(0.F, 0.F);
    ImGui::SetCursorScreenPos(ImVec2(bottomRight.x - trash - padding, topLeft.y + (height - trash) * 0.5F));
    bool const forget = IconButton("forget", StudioIcon::Remove, "Remove from this list (the folder is not touched).");
    ImGui::SetCursorScreenPos(ImVec2(topLeft.x, bottomRight.y + ImGui::GetStyle().ItemSpacing.y));
    ImGui::Dummy(ImVec2(0.F, 0.F));
    return forget;
}

} // namespace ppokit::studio
