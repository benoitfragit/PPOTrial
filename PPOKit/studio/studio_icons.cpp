/**
 * @file PPOKit/studio/studio_icons.cpp
 * @brief ImDrawList painters of the studio icons and the buttons that show them.
 */
#include "studio_icons.hpp"

#include "studio_theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ppokit::studio {
namespace {

constexpr float Pi = 3.14159265358979323846F;
/** @brief Resting glyph color: a quiet blue-white, so that accent colors keep their meaning. */
constexpr ImU32 GlyphRest = IM_COL32(211, 225, 242, 255);
/** @brief Icon of a pill button, relative to the frame height. */
constexpr float PillIconScale = 0.8F;

/** @brief Center, radius and stroke width of one icon. */
struct Glyph final
{
    ImVec2 center;
    float radius = 0.F;
    float thickness = 0.F;
};

ImVec2 At(Glyph const& g, float x, float y) noexcept
{
    return ImVec2(g.center.x + x * g.radius, g.center.y + y * g.radius);
}

void Line(ImDrawList& d, Glyph const& g, float x0, float y0, float x1, float y1, ImU32 c)
{
    d.AddLine(At(g, x0, y0), At(g, x1, y1), c, g.thickness);
}

void PaintOpen(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // Folder: a tab on the top left, then the body.
    ImVec2 const points[] = {At(g, -1.F, 0.8F), At(g, -1.F, -0.75F), At(g, -0.35F, -0.75F), At(g, -0.15F, -0.5F),
        At(g, 1.F, -0.5F), At(g, 1.F, 0.8F)};
    d.AddPolyline(points, 6, c, ImDrawFlags_Closed, g.thickness);
    Line(d, g, -1.F, -0.2F, 1.F, -0.2F, c);
}

void PaintNew(ImDrawList& d, Glyph const& g, ImU32 c)
{
    Line(d, g, -1.F, 0.F, 1.F, 0.F, c);
    Line(d, g, 0.F, -1.F, 0.F, 1.F, c);
}

void PaintRecent(ImDrawList& d, Glyph const& g, ImU32 c)
{
    d.AddCircle(g.center, g.radius, c, 0, g.thickness);
    Line(d, g, 0.F, 0.F, 0.F, -0.6F, c);
    Line(d, g, 0.F, 0.F, 0.45F, 0.F, c);
}

void PaintBack(ImDrawList& d, Glyph const& g, ImU32 c)
{
    Line(d, g, 0.35F, -0.8F, -0.45F, 0.F, c);
    Line(d, g, -0.45F, 0.F, 0.35F, 0.8F, c);
}

void PaintTrain(ImDrawList& d, Glyph const& g, ImU32 c)
{
    d.AddTriangleFilled(At(g, -0.6F, -0.9F), At(g, 0.9F, 0.F), At(g, -0.6F, 0.9F), c);
}

void PaintEvaluate(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // A check mark in a ring: "does the policy pass on new seeds?"
    d.AddCircle(g.center, g.radius, c, 0, g.thickness);
    ImVec2 const check[] = {At(g, -0.45F, 0.F), At(g, -0.1F, 0.38F), At(g, 0.5F, -0.35F)};
    d.AddPolyline(check, 3, c, ImDrawFlags_None, g.thickness);
}

void PaintInference(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // An eye: two arcs and the pupil, "watch the policy act".
    float const lid = g.radius * 1.25F;
    d.PathArcTo(At(g, 0.F, 0.75F), lid, Pi * 1.25F, Pi * 1.75F, 16);
    d.PathStroke(c, ImDrawFlags_None, g.thickness);
    d.PathArcTo(At(g, 0.F, -0.75F), lid, Pi * 0.25F, Pi * 0.75F, 16);
    d.PathStroke(c, ImDrawFlags_None, g.thickness);
    d.AddCircleFilled(g.center, g.radius * 0.3F, c);
}

void PaintStop(ImDrawList& d, Glyph const& g, ImU32 c)
{
    d.AddRectFilled(At(g, -0.75F, -0.75F), At(g, 0.75F, 0.75F), c, g.radius * 0.18F);
}

void PaintKill(ImDrawList& d, Glyph const& g, ImU32 c)
{
    d.AddCircle(g.center, g.radius, c, 0, g.thickness);
    Line(d, g, -0.45F, -0.45F, 0.45F, 0.45F, c);
    Line(d, g, 0.45F, -0.45F, -0.45F, 0.45F, c);
}

void PaintSave(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // Floppy disk: body with a cut corner, shutter on top, label below.
    ImVec2 const body[] = {At(g, -0.9F, -0.9F), At(g, 0.55F, -0.9F), At(g, 0.9F, -0.55F), At(g, 0.9F, 0.9F),
        At(g, -0.9F, 0.9F)};
    d.AddPolyline(body, 5, c, ImDrawFlags_Closed, g.thickness);
    d.AddRect(At(g, -0.45F, -0.9F), At(g, 0.35F, -0.35F), c, 0.F, ImDrawFlags_None, g.thickness);
    d.AddRectFilled(At(g, -0.5F, 0.2F), At(g, 0.5F, 0.9F), c, g.radius * 0.1F);
}

void PaintOverview(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // The learning loop: a ring with an arrow head.
    d.PathArcTo(g.center, g.radius * 0.85F, Pi * -0.35F, Pi * 1.35F, 24);
    d.PathStroke(c, ImDrawFlags_None, g.thickness);
    ImVec2 const tip = At(g, 0.85F * std::cos(-0.35F * Pi), 0.85F * std::sin(-0.35F * Pi));
    d.AddTriangleFilled(tip, ImVec2(tip.x - g.radius * 0.55F, tip.y - g.radius * 0.1F),
        ImVec2(tip.x - g.radius * 0.05F, tip.y + g.radius * 0.5F), c);
}

void PaintNetwork(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // Two inputs wired into one block.
    Line(d, g, -0.65F, -0.6F, 0.6F, 0.F, c);
    Line(d, g, -0.65F, 0.6F, 0.6F, 0.F, c);
    float const node = g.radius * 0.32F;
    d.AddCircleFilled(At(g, -0.65F, -0.6F), node, c);
    d.AddCircleFilled(At(g, -0.65F, 0.6F), node, c);
    d.AddCircleFilled(At(g, 0.6F, 0.F), node * 1.2F, c);
}

void PaintTraining(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // Three sliders.
    float const knob = g.radius * 0.22F;
    float const positions[] = {0.35F, -0.4F, 0.1F};
    for (int row = 0; row < 3; ++row)
    {
        float const y = -0.7F + 0.7F * static_cast<float>(row);
        Line(d, g, -1.F, y, 1.F, y, c);
        d.AddCircleFilled(At(g, positions[row], y), knob * 1.3F, c);
    }
}

void PaintTask(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // A sheet with three lines.
    d.AddRect(At(g, -0.8F, -1.F), At(g, 0.8F, 1.F), c, g.radius * 0.15F, ImDrawFlags_None, g.thickness);
    for (int row = 0; row < 3; ++row)
    {
        float const y = -0.45F + 0.45F * static_cast<float>(row);
        Line(d, g, -0.4F, y, 0.4F, y, c);
    }
}

void PaintRun(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // A rising curve on axes.
    ImVec2 const axes[] = {At(g, -1.F, -1.F), At(g, -1.F, 1.F), At(g, 1.F, 1.F)};
    d.AddPolyline(axes, 3, c, ImDrawFlags_None, g.thickness);
    ImVec2 const curve[] = {At(g, -0.65F, 0.45F), At(g, -0.2F, -0.05F), At(g, 0.15F, 0.2F), At(g, 0.8F, -0.65F)};
    d.AddPolyline(curve, 4, c, ImDrawFlags_None, g.thickness);
}

void PaintConsole(ImDrawList& d, Glyph const& g, ImU32 c)
{
    d.AddRect(At(g, -1.F, -0.85F), At(g, 1.F, 0.85F), c, g.radius * 0.15F, ImDrawFlags_None, g.thickness);
    Line(d, g, -0.6F, -0.35F, -0.2F, 0.F, c);
    Line(d, g, -0.2F, 0.F, -0.6F, 0.35F, c);
    Line(d, g, 0.05F, 0.4F, 0.55F, 0.4F, c);
}

void PaintMagnifier(ImDrawList& d, Glyph const& g, ImU32 c, bool plus)
{
    ImVec2 const lens = At(g, -0.2F, -0.2F);
    float const radius = g.radius * 0.7F;
    d.AddCircle(lens, radius, c, 0, g.thickness);
    Line(d, g, 0.3F, 0.3F, 0.95F, 0.95F, c);
    float const arm = radius * 0.5F;
    d.AddLine(ImVec2(lens.x - arm, lens.y), ImVec2(lens.x + arm, lens.y), c, g.thickness);
    if (plus)
    {
        d.AddLine(ImVec2(lens.x, lens.y - arm), ImVec2(lens.x, lens.y + arm), c, g.thickness);
    }
}

void PaintFit(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // Four corner brackets around the content.
    float const corners[4][2] = {{-1.F, -1.F}, {1.F, -1.F}, {1.F, 1.F}, {-1.F, 1.F}};
    for (auto const& corner : corners)
    {
        float const x = corner[0];
        float const y = corner[1];
        ImVec2 const bracket[] = {At(g, x, y * 0.45F), At(g, x, y), At(g, x * 0.45F, y)};
        d.AddPolyline(bracket, 3, c, ImDrawFlags_None, g.thickness);
    }
    d.AddRectFilled(At(g, -0.35F, -0.35F), At(g, 0.35F, 0.35F), c, g.radius * 0.08F);
}

void PaintTurn(ImDrawList& d, Glyph const& g, ImU32 c, float side)
{
    // A curved arrow: side -1 for undo (arrow head on the left), +1 for redo.
    float const start = side < 0.F ? Pi * 1.1F : Pi * -0.1F;
    float const end = side < 0.F ? Pi * 2.1F : Pi * -1.1F;
    d.PathArcTo(At(g, 0.F, 0.25F), g.radius * 0.75F, start, end, 20);
    d.PathStroke(c, ImDrawFlags_None, g.thickness);
    ImVec2 const tip = At(g, 0.8F * side, 0.05F);
    d.AddTriangleFilled(ImVec2(tip.x, tip.y - g.radius * 0.5F), ImVec2(tip.x - side * g.radius * 0.45F, tip.y + g.radius * 0.15F),
        ImVec2(tip.x + side * g.radius * 0.45F, tip.y + g.radius * 0.15F), c);
}

void PaintRemove(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // Trash can: lid, handle, body with two ribs.
    Line(d, g, -0.95F, -0.6F, 0.95F, -0.6F, c);
    Line(d, g, -0.3F, -0.6F, -0.3F, -0.9F, c);
    Line(d, g, -0.3F, -0.9F, 0.3F, -0.9F, c);
    Line(d, g, 0.3F, -0.9F, 0.3F, -0.6F, c);
    ImVec2 const body[] = {At(g, -0.7F, -0.6F), At(g, -0.55F, 0.95F), At(g, 0.55F, 0.95F), At(g, 0.7F, -0.6F)};
    d.AddPolyline(body, 4, c, ImDrawFlags_None, g.thickness);
    Line(d, g, -0.18F, -0.25F, -0.18F, 0.6F, c);
    Line(d, g, 0.18F, -0.25F, 0.18F, 0.6F, c);
}

void PaintLibrary(ImDrawList& d, Glyph const& g, ImU32 c)
{
    // A cube: an external component plugged into the network.
    ImVec2 const outline[] = {At(g, 0.F, -1.F), At(g, 0.9F, -0.5F), At(g, 0.9F, 0.5F), At(g, 0.F, 1.F),
        At(g, -0.9F, 0.5F), At(g, -0.9F, -0.5F)};
    d.AddPolyline(outline, 6, c, ImDrawFlags_Closed, g.thickness);
    Line(d, g, -0.9F, -0.5F, 0.F, 0.F, c);
    Line(d, g, 0.9F, -0.5F, 0.F, 0.F, c);
    Line(d, g, 0.F, 0.F, 0.F, 1.F, c);
}

void PaintWarning(ImDrawList& d, Glyph const& g, ImU32 c)
{
    ImVec2 const triangle[] = {At(g, 0.F, -1.F), At(g, 1.F, 0.85F), At(g, -1.F, 0.85F)};
    d.AddPolyline(triangle, 3, c, ImDrawFlags_Closed, g.thickness);
    Line(d, g, 0.F, -0.4F, 0.F, 0.25F, c);
    d.AddCircleFilled(At(g, 0.F, 0.55F), g.thickness * 0.7F, c);
}

/** @brief End of the visible part of an ImGui label: text after "##" is only the identifier. */
char const* VisibleEnd(char const* label) noexcept
{
    char const* const hidden = std::strstr(label, "##");
    return hidden != nullptr ? hidden : label + std::strlen(label);
}

ImU32 Color(ImVec4 color)
{
    return ImGui::ColorConvertFloat4ToU32(color);
}

/** @brief The shared circle-or-pill hover background of the icon buttons. */
ImU32 HoverBackground(bool active)
{
    return ImGui::GetColorU32(active ? ImGuiCol_ButtonActive : ImGuiCol_ButtonHovered);
}

} // namespace

void PaintIcon(ImDrawList& drawList, StudioIcon icon, ImVec2 topLeft, float size, ImU32 color)
{
    Glyph const g{ImVec2(topLeft.x + size * 0.5F, topLeft.y + size * 0.5F), size * 0.3F, std::max(1.4F, size * 0.072F)};
    switch (icon)
    {
    case StudioIcon::Open: PaintOpen(drawList, g, color); return;
    case StudioIcon::New: PaintNew(drawList, g, color); return;
    case StudioIcon::Recent: PaintRecent(drawList, g, color); return;
    case StudioIcon::Back: PaintBack(drawList, g, color); return;
    case StudioIcon::Train: PaintTrain(drawList, g, color); return;
    case StudioIcon::Evaluate: PaintEvaluate(drawList, g, color); return;
    case StudioIcon::Inference: PaintInference(drawList, g, color); return;
    case StudioIcon::Stop: PaintStop(drawList, g, color); return;
    case StudioIcon::Kill: PaintKill(drawList, g, color); return;
    case StudioIcon::Save: PaintSave(drawList, g, color); return;
    case StudioIcon::Overview: PaintOverview(drawList, g, color); return;
    case StudioIcon::Network: PaintNetwork(drawList, g, color); return;
    case StudioIcon::Training: PaintTraining(drawList, g, color); return;
    case StudioIcon::Task: PaintTask(drawList, g, color); return;
    case StudioIcon::Run: PaintRun(drawList, g, color); return;
    case StudioIcon::Console: PaintConsole(drawList, g, color); return;
    case StudioIcon::ZoomIn: PaintMagnifier(drawList, g, color, true); return;
    case StudioIcon::ZoomOut: PaintMagnifier(drawList, g, color, false); return;
    case StudioIcon::Fit: PaintFit(drawList, g, color); return;
    case StudioIcon::Undo: PaintTurn(drawList, g, color, -1.F); return;
    case StudioIcon::Redo: PaintTurn(drawList, g, color, 1.F); return;
    case StudioIcon::Remove: PaintRemove(drawList, g, color); return;
    case StudioIcon::Library: PaintLibrary(drawList, g, color); return;
    case StudioIcon::Warning: PaintWarning(drawList, g, color); return;
    }
}

bool IconButton(char const* id, StudioIcon icon, char const* tooltip, bool enabled)
{
    float const size = ImGui::GetFrameHeight();
    ImVec2 const topLeft = ImGui::GetCursorScreenPos();
    ImGui::BeginDisabled(!enabled);
    bool const pressed = ImGui::InvisibleButton(id, ImVec2(size, size));
    bool const hovered = ImGui::IsItemHovered();
    bool const active = ImGui::IsItemActive();
    ImGui::EndDisabled();
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    if (enabled && (hovered || active))
    {
        drawList.AddCircleFilled(ImVec2(topLeft.x + size * 0.5F, topLeft.y + size * 0.5F), size * 0.5F,
            HoverBackground(active));
    }
    ImU32 glyph = GlyphRest;
    if (!enabled)
    {
        glyph = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    }
    else if (hovered || active)
    {
        glyph = Color(PrimaryAccent);
    }
    PaintIcon(drawList, icon, topLeft, size, glyph);
    ItemTooltip(tooltip);
    return pressed && enabled;
}

float IconLabelButtonWidth(char const* label)
{
    ImGuiStyle const& style = ImGui::GetStyle();
    float const iconSize = ImGui::GetFrameHeight() * PillIconScale;
    return style.FramePadding.x * 2.F + iconSize + style.ItemInnerSpacing.x
        + ImGui::CalcTextSize(label, VisibleEnd(label)).x;
}

bool IconLabelButton(char const* label, StudioIcon icon, char const* tooltip, bool enabled, bool primary)
{
    ImGuiStyle const& style = ImGui::GetStyle();
    float const height = ImGui::GetFrameHeight();
    float const iconSize = height * PillIconScale;
    char const* const labelEnd = VisibleEnd(label);
    ImVec2 const text = ImGui::CalcTextSize(label, labelEnd);
    float const width = IconLabelButtonWidth(label);
    ImVec2 const topLeft = ImGui::GetCursorScreenPos();
    ImGui::BeginDisabled(!enabled);
    bool const pressed = ImGui::InvisibleButton(label, ImVec2(width, height));
    bool const hovered = ImGui::IsItemHovered();
    bool const active = ImGui::IsItemActive();
    ImGui::EndDisabled();

    ImVec4 fill = ImGui::GetStyleColorVec4(ImGuiCol_Button);
    ImVec4 ink = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    if (primary && enabled)
    {
        fill = PrimaryAccent;
        ink = BackgroundSurface;
    }
    if (enabled && (hovered || active))
    {
        fill = primary ? SecondaryAccent : ImGui::GetStyleColorVec4(active ? ImGuiCol_ButtonActive : ImGuiCol_ButtonHovered);
    }
    if (!enabled)
    {
        ink = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
    }
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    ImVec2 const bottomRight(topLeft.x + width, topLeft.y + height);
    drawList.AddRectFilled(topLeft, bottomRight, Color(fill), height * 0.5F);
    ImVec2 const iconCorner(topLeft.x + style.FramePadding.x, topLeft.y + (height - iconSize) * 0.5F);
    PaintIcon(drawList, icon, iconCorner, iconSize, Color(ink));
    drawList.AddText(ImVec2(iconCorner.x + iconSize + style.ItemInnerSpacing.x, topLeft.y + (height - text.y) * 0.5F),
        Color(ink), label, labelEnd);
    ItemTooltip(tooltip);
    return pressed && enabled;
}

bool IconTile(char const* label, StudioIcon icon, char const* tooltip, float side)
{
    ImVec2 const topLeft = ImGui::GetCursorScreenPos();
    bool const pressed = ImGui::InvisibleButton(label, ImVec2(side, side));
    bool const hovered = ImGui::IsItemHovered();
    bool const active = ImGui::IsItemActive();
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    ImVec2 const bottomRight(topLeft.x + side, topLeft.y + side);
    float const rounding = side * 0.12F;
    drawList.AddRectFilled(topLeft, bottomRight, Color(hovered ? RaisedSurface : PanelSurface), rounding);
    if (hovered || active)
    {
        drawList.AddRect(topLeft, bottomRight, Color(PrimaryAccent), rounding, ImDrawFlags_None, 1.5F);
    }
    float const iconSize = side * 0.46F;
    ImVec2 const iconCorner(topLeft.x + (side - iconSize) * 0.5F, topLeft.y + side * 0.14F);
    PaintIcon(drawList, icon, iconCorner, iconSize, hovered ? Color(PrimaryAccent) : GlyphRest);
    char const* const labelEnd = VisibleEnd(label);
    ImVec2 const text = ImGui::CalcTextSize(label, labelEnd);
    drawList.AddText(ImVec2(topLeft.x + (side - text.x) * 0.5F, topLeft.y + side * 0.72F),
        Color(hovered ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : MutedText), label, labelEnd);
    ItemTooltip(tooltip);
    return pressed;
}

void IconText(StudioIcon icon, ImVec4 color)
{
    float const size = ImGui::GetTextLineHeight();
    ImVec2 const topLeft = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, size));
    PaintIcon(*ImGui::GetWindowDrawList(), icon, topLeft, size, Color(color));
}

} // namespace ppokit::studio
