/**
 * @file PPOKit/studio/studio_icons.hpp
 * @brief Line icons drawn with ImDrawList (no icon font), in the style of MFDStudio's editor.
 */
#pragma once

#include <imgui.h>

namespace ppokit::studio {

/** @brief Every icon of the studio. */
enum class StudioIcon
{
    Open,
    New,
    Recent,
    Back,
    Train,
    Evaluate,
    Inference,
    Stop,
    Kill,
    Save,
    Overview,
    Network,
    Training,
    Task,
    Run,
    Console,
    ZoomIn,
    ZoomOut,
    Fit,
    Undo,
    Redo,
    Remove,
    Library,
    Warning,
};

/**
 * @brief Paints `icon` in the square at `topLeft` of side `size`.
 * @param drawList Where to paint.
 * @param icon Icon to paint.
 * @param topLeft Top left corner of the square, in screen pixels.
 * @param size Side of the square.
 * @param color Stroke and fill color.
 */
void PaintIcon(ImDrawList& drawList, StudioIcon icon, ImVec2 topLeft, float size, ImU32 color);

/**
 * @brief Round icon-only button with a tooltip.
 * @param id ImGui identifier.
 * @param icon Icon shown.
 * @param tooltip Shown when hovered (also when disabled, to say why).
 * @param enabled False draws it dimmed and ignores clicks.
 * @return True when clicked.
 */
bool IconButton(char const* id, StudioIcon icon, char const* tooltip, bool enabled = true);

/**
 * @brief Pill button with an icon and a label, for the main actions.
 * @param label Visible text (also the ImGui identifier).
 * @param icon Icon left of the text.
 * @param tooltip Shown when hovered.
 * @param enabled False draws it dimmed and ignores clicks.
 * @param primary True fills it with the primary accent (the main action of the bar).
 * @return True when clicked.
 */
bool IconLabelButton(char const* label, StudioIcon icon, char const* tooltip, bool enabled = true,
    bool primary = false);

/** @brief Width IconLabelButton takes for `label`, to align a group of them. */
float IconLabelButtonWidth(char const* label);

/**
 * @brief Large square tile with a big icon and a caption, as on MFDStudio's start page.
 * @param label Caption (also the ImGui identifier).
 * @param icon Icon drawn in the tile.
 * @param tooltip Shown when hovered.
 * @param side Side of the tile in pixels.
 * @return True when clicked.
 */
bool IconTile(char const* label, StudioIcon icon, char const* tooltip, float side);

/** @brief Icon at the cursor, as a line item of the text height, in `color`. */
void IconText(StudioIcon icon, ImVec4 color);

} // namespace ppokit::studio
