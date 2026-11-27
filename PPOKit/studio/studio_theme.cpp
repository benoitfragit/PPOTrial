/**
 * @file PPOKit/studio/studio_theme.cpp
 * @brief Studio fonts, colors and shared widgets (the MFDStudio editor look).
 */
#include "studio_theme.hpp"

#include <imgui.h>
#include <imnodes.h>
#include <implot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ppokit::studio {
namespace {

constexpr float MinimumDisplayScale = 0.75F;
constexpr float MaximumDisplayScale = 3.F;
/** @brief Headings are this much larger than body text. */
constexpr float HeadingScale = 1.12F;
/** @brief Tooltips wrap after this many font sizes. */
constexpr float TooltipWidthInFonts = 32.F;

ImFont* headingFont = nullptr;

/** @brief Path of a font in the Windows font folder, empty elsewhere or when missing. */
std::filesystem::path WindowsFont([[maybe_unused]] wchar_t const* fileName)
{
#if defined(_WIN32)
    std::array<wchar_t, MAX_PATH + 1U> windows{};
    UINT const length = GetWindowsDirectoryW(windows.data(), static_cast<UINT>(windows.size()));
    if (length == 0U || length >= windows.size())
    {
        return {};
    }
    std::filesystem::path const path = std::filesystem::path(std::wstring(windows.data(), length)) / "Fonts" / fileName;
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) ? path : std::filesystem::path{};
#else
    return {};
#endif
}

/** @brief The first of `candidates` that exists (fonts of Linux distributions), empty when none. */
std::filesystem::path FirstFont(std::initializer_list<char const*> candidates)
{
    for (char const* candidate : candidates)
    {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error))
        {
            return candidate;
        }
    }
    return {};
}

/** @brief Body font: Segoe UI on Windows, Inter or DejaVu Sans elsewhere. */
std::filesystem::path BodyFont()
{
    std::filesystem::path const windows = WindowsFont(L"segoeui.ttf");
    if (!windows.empty())
    {
        return windows;
    }
    return FirstFont({"/usr/share/fonts/opentype/inter/Inter-Regular.otf", "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"});
}

/** @brief Heading font: Segoe UI Semibold on Windows, Inter SemiBold or DejaVu Sans Bold elsewhere. */
std::filesystem::path HeadingFont()
{
    std::filesystem::path const windows = WindowsFont(L"seguisb.ttf");
    if (!windows.empty())
    {
        return windows;
    }
    return FirstFont({"/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
        "/usr/share/fonts/truetype/inter/Inter-SemiBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"});
}

ImFont* AddFont(std::filesystem::path const& path, float size, ImFontConfig const& configuration)
{
    ImGuiIO& io = ImGui::GetIO();
    if (!path.empty())
    {
        ImFontConfig copy = configuration;
        if (ImFont* const font = io.Fonts->AddFontFromFileTTF(path.string().c_str(), size, &copy))
        {
            return font;
        }
    }
    ImFontConfig fallback = configuration;
    fallback.SizePixels = size;
    return io.Fonts->AddFontDefault(&fallback);
}

void StyleImPlot()
{
    if (ImPlot::GetCurrentContext() == nullptr)
    {
        return;
    }
    ImPlotStyle& plot = ImPlot::GetStyle();
    plot.LineWeight = 1.8F;
    plot.PlotPadding = ImVec2(10.F, 10.F);
    plot.PlotBorderSize = 0.F;
    plot.Colors[ImPlotCol_PlotBg] = BackgroundSurface;
    plot.Colors[ImPlotCol_FrameBg] = PanelSurface;
    plot.Colors[ImPlotCol_LegendBg] = ImVec4(PanelSurface.x, PanelSurface.y, PanelSurface.z, 0.92F);
    plot.Colors[ImPlotCol_LegendBorder] = ImVec4(0.F, 0.F, 0.F, 0.F);
    plot.Colors[ImPlotCol_AxisGrid] = ImVec4(0.337F, 0.4F, 0.494F, 0.18F);
}

void StyleImNodes()
{
    if (ImNodes::GetCurrentContext() == nullptr)
    {
        return;
    }
    ImNodesStyle& nodes = ImNodes::GetStyle();
    nodes.NodeCornerRounding = 9.F;
    nodes.NodePadding = ImVec2(10.F, 7.F);
    nodes.NodeBorderThickness = 1.F;
    nodes.LinkThickness = 2.4F;
    nodes.PinCircleRadius = 4.5F;
    nodes.GridSpacing = 28.F;
    unsigned int* colors = nodes.Colors;
    colors[ImNodesCol_GridBackground] = ImGui::ColorConvertFloat4ToU32(BackgroundSurface);
    colors[ImNodesCol_GridLine] = IM_COL32(86, 102, 126, 38);
    colors[ImNodesCol_NodeBackground] = ImGui::ColorConvertFloat4ToU32(RaisedSurface);
    colors[ImNodesCol_NodeBackgroundHovered] = IM_COL32(32, 44, 62, 255);
    colors[ImNodesCol_NodeBackgroundSelected] = IM_COL32(30, 52, 78, 255);
    colors[ImNodesCol_NodeOutline] = IM_COL32(86, 102, 126, 90);
    colors[ImNodesCol_TitleBar] = IM_COL32(43, 55, 69, 255);
    colors[ImNodesCol_TitleBarHovered] = IM_COL32(55, 70, 86, 255);
    colors[ImNodesCol_TitleBarSelected] = ImGui::ColorConvertFloat4ToU32(
        ImVec4(PrimaryAccent.x * 0.7F, PrimaryAccent.y * 0.7F, PrimaryAccent.z * 0.7F, 1.F));
    colors[ImNodesCol_Link] = ImGui::ColorConvertFloat4ToU32(
        ImVec4(PrimaryAccent.x, PrimaryAccent.y, PrimaryAccent.z, 0.75F));
    colors[ImNodesCol_LinkHovered] = ImGui::ColorConvertFloat4ToU32(PrimaryAccent);
    colors[ImNodesCol_LinkSelected] = ImGui::ColorConvertFloat4ToU32(SecondaryAccent);
    colors[ImNodesCol_Pin] = ImGui::ColorConvertFloat4ToU32(SecondaryAccent);
    colors[ImNodesCol_PinHovered] = IM_COL32(255, 255, 255, 255);
    colors[ImNodesCol_BoxSelector] = IM_COL32(109, 183, 255, 40);
    colors[ImNodesCol_BoxSelectorOutline] = IM_COL32(109, 183, 255, 160);
}

} // namespace

void LoadStudioFonts(float displayScale)
{
    float const scale = std::isfinite(displayScale) ? std::clamp(displayScale, MinimumDisplayScale, MaximumDisplayScale)
                                                    : 1.F;
    float const size = std::ceil(StudioFontSize * scale);
    ImFontConfig configuration;
    configuration.OversampleH = 2;
    configuration.OversampleV = 1;
    configuration.PixelSnapH = false;
    ImFont* const body = AddFont(BodyFont(), size, configuration);
    ImGui::GetIO().FontDefault = body;
    std::filesystem::path const semibold = HeadingFont();
    headingFont = semibold.empty() ? body : AddFont(semibold, std::ceil(size * HeadingScale), configuration);
}

void ApplyStudioTheme()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 12.F;
    style.ChildRounding = 12.F;
    style.FrameRounding = 9.F;
    style.PopupRounding = 12.F;
    style.ScrollbarRounding = 10.F;
    style.GrabRounding = 9.F;
    style.TabRounding = 9.F;
    style.WindowPadding = ImVec2(14.F, 14.F);
    style.FramePadding = ImVec2(10.F, 7.F);
    style.ItemSpacing = ImVec2(10.F, 8.F);
    style.ItemInnerSpacing = ImVec2(8.F, 6.F);
    style.IndentSpacing = 20.F;
    style.ScrollbarSize = 11.F;
    style.GrabMinSize = 14.F;
    style.DisabledAlpha = 0.62F;
    style.WindowBorderSize = 0.F;
    style.ChildBorderSize = 0.F;
    style.FrameBorderSize = 0.F;
    style.PopupBorderSize = 1.F;
    style.TabBorderSize = 0.F;
    style.TabBarBorderSize = 0.F;
    style.SeparatorTextBorderSize = 1.F;

    // Hierarchy comes from tone and elevation, not outlines.
    ImVec4 const border(0.337F, 0.4F, 0.494F, 0.28F);
    ImVec4 const primary = PrimaryAccent;
    ImVec4 const secondary = SecondaryAccent;
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = ImVec4(0.925F, 0.953F, 0.992F, 1.F);
    colors[ImGuiCol_TextDisabled] = MutedText;
    colors[ImGuiCol_WindowBg] = BackgroundSurface;
    colors[ImGuiCol_ChildBg] = PanelSurface;
    colors[ImGuiCol_PopupBg] = ImVec4(PanelSurface.x, PanelSurface.y, PanelSurface.z, 0.98F);
    colors[ImGuiCol_Border] = border;
    colors[ImGuiCol_BorderShadow] = ImVec4(0.F, 0.F, 0.F, 0.F);
    colors[ImGuiCol_FrameBg] = RaisedSurface;
    colors[ImGuiCol_FrameBgHovered] = ImVec4(primary.x, primary.y, primary.z, 0.18F);
    colors[ImGuiCol_FrameBgActive] = ImVec4(primary.x, primary.y, primary.z, 0.27F);
    colors[ImGuiCol_TitleBg] = BackgroundSurface;
    colors[ImGuiCol_TitleBgActive] = PanelSurface;
    colors[ImGuiCol_MenuBarBg] = BackgroundSurface;
    colors[ImGuiCol_ScrollbarBg] = ImVec4(BackgroundSurface.x, BackgroundSurface.y, BackgroundSurface.z, 0.18F);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.337F, 0.4F, 0.494F, 0.42F);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(primary.x, primary.y, primary.z, 0.5F);
    colors[ImGuiCol_ScrollbarGrabActive] = primary;
    colors[ImGuiCol_CheckMark] = secondary;
    colors[ImGuiCol_SliderGrab] = primary;
    colors[ImGuiCol_SliderGrabActive] = secondary;
    colors[ImGuiCol_Button] = RaisedSurface;
    colors[ImGuiCol_ButtonHovered] = ImVec4(primary.x, primary.y, primary.z, 0.2F);
    colors[ImGuiCol_ButtonActive] = ImVec4(primary.x, primary.y, primary.z, 0.3F);
    colors[ImGuiCol_Header] = ImVec4(primary.x, primary.y, primary.z, 0.2F);
    colors[ImGuiCol_HeaderHovered] = ImVec4(primary.x, primary.y, primary.z, 0.26F);
    colors[ImGuiCol_HeaderActive] = ImVec4(primary.x, primary.y, primary.z, 0.34F);
    colors[ImGuiCol_Separator] = border;
    colors[ImGuiCol_SeparatorHovered] = primary;
    colors[ImGuiCol_SeparatorActive] = secondary;
    colors[ImGuiCol_ResizeGrip] = ImVec4(primary.x, primary.y, primary.z, 0.24F);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(primary.x, primary.y, primary.z, 0.62F);
    colors[ImGuiCol_ResizeGripActive] = primary;
    colors[ImGuiCol_Tab] = PanelSurface;
    colors[ImGuiCol_TabHovered] = ImVec4(primary.x, primary.y, primary.z, 0.34F);
    colors[ImGuiCol_TabSelected] = RaisedSurface;
    colors[ImGuiCol_TabSelectedOverline] = primary;
    colors[ImGuiCol_TabDimmed] = PanelSurface;
    colors[ImGuiCol_TabDimmedSelected] = RaisedSurface;
    colors[ImGuiCol_TableHeaderBg] = RaisedSurface;
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.F, 1.F, 1.F, 0.025F);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.F, 0.F, 0.F, 0.F);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.F, 0.F, 0.F, 0.F);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(primary.x, primary.y, primary.z, 0.3F);
    colors[ImGuiCol_NavCursor] = primary;
    colors[ImGuiCol_DragDropTarget] = ImVec4(0.957F, 0.765F, 0.42F, 0.92F);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.F, 0.F, 0.F, 0.58F);
    StyleImPlot();
    StyleImNodes();
}

ImVec4 HealthColor(editor::Health health) noexcept
{
    switch (health)
    {
    case editor::Health::Good:
        return ImVec4(0.51F, 0.75F, 0.61F, 1.F);
    case editor::Health::Watch:
        return ImVec4(0.84F, 0.72F, 0.49F, 1.F);
    case editor::Health::Problem:
        return ImVec4(0.86F, 0.53F, 0.56F, 1.F);
    case editor::Health::Unknown:
        break;
    }
    return MutedText;
}

void Heading(char const* title, char const* description)
{
    if (headingFont != nullptr)
    {
        ImGui::PushFont(headingFont);
    }
    ImGui::TextUnformatted(title);
    if (headingFont != nullptr)
    {
        ImGui::PopFont();
    }
    if (description != nullptr && description[0] != '\0')
    {
        MutedWrapped(description);
    }
    ImGui::Spacing();
}

bool AccentButton(char const* label)
{
    ImVec4 const accent = PrimaryAccent;
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(accent.x * 0.72F, accent.y * 0.72F, accent.z * 0.72F, 1.F));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(accent.x * 0.88F, accent.y * 0.88F, accent.z * 0.88F, 1.F));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, accent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.02F, 0.05F, 0.1F, 1.F));
    bool const pressed = ImGui::Button(label);
    ImGui::PopStyleColor(4);
    return pressed;
}

void ItemTooltip(char const* text, ImGuiHoveredFlags hoverFlags)
{
    if (text == nullptr || text[0] == '\0'
        || !ImGui::IsItemHovered(hoverFlags | ImGuiHoveredFlags_AllowWhenDisabled))
    {
        return;
    }
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * TooltipWidthInFonts);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void HelpMark(char const* text)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    ItemTooltip(text);
}

void MutedWrapped(char const* text)
{
    ImGui::PushStyleColor(ImGuiCol_Text, MutedText);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

bool ToggleChip(char const* label, bool selected, char const* tooltip)
{
    ImVec4 const fill = selected ? ImVec4(PrimaryAccent.x, PrimaryAccent.y, PrimaryAccent.z, 0.22F) : PanelSurface;
    ImGui::PushStyleColor(ImGuiCol_Button, fill);
    ImGui::PushStyleColor(ImGuiCol_Text, selected ? PrimaryAccent : MutedText);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ImGui::GetFrameHeight() * 0.5F);
    bool const pressed = ImGui::Button(label);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    ItemTooltip(tooltip);
    return pressed;
}

bool BeginPanel(char const* id, ImVec2 size)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.F);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, PanelSurface);
    bool const open = ImGui::BeginChild(id, size, ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    return open;
}

void EndPanel()
{
    ImGui::EndChild();
}

void CenteredText(char const* text, ImVec4 color, float scale, bool heading)
{
    ImFont* const font = heading && headingFont != nullptr ? headingFont : ImGui::GetFont();
    ImGui::PushFont(font, ImGui::GetStyle().FontSizeBase * scale * (heading ? HeadingScale : 1.F));
    float const width = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.F, (ImGui::GetContentRegionAvail().x - width) * 0.5F));
    ImGui::TextColored(color, "%s", text);
    ImGui::PopFont();
}

} // namespace ppokit::studio
