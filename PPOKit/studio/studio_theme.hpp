/**
 * @file PPOKit/studio/studio_theme.hpp
 * @brief Look of the PPOKit studio, shared with MFDStudio's editor: Segoe UI, tonal dark
 *        surfaces without outlines, rounded frames, a blue primary and a teal secondary
 *        accent; plus the small widgets every view uses (headings, help markers, buttons).
 */
#pragma once

#include "ppokit/editor/run_health.hpp"

#include <imgui.h>

namespace ppokit::studio {

/**
 * @brief Loads the studio fonts: Segoe UI (body) and Segoe UI Semibold (headings) on
 *        Windows, ImGui's default font elsewhere. Call between font-atlas setup calls
 *        (rlImGuiBeginInitImGui / rlImGuiEndInitImGui), once.
 * @param displayScale Monitor scale (1 = 96 dpi), clamped to [0.75, 3].
 */
void LoadStudioFonts(float displayScale);

/** @brief Colors, rounding and spacing of the studio; also styles ImPlot and imnodes when present. */
void ApplyStudioTheme();

/** @brief Pixel size of body text at scale 1. */
inline constexpr float StudioFontSize = 17.F;

/** @brief Primary accent (blue) and secondary accent (teal). */
inline constexpr ImVec4 PrimaryAccent{0.58F, 0.70F, 0.82F, 1.F};
inline constexpr ImVec4 SecondaryAccent{0.53F, 0.72F, 0.68F, 1.F};
/** @brief Muted text for secondary information. */
inline constexpr ImVec4 MutedText{0.65F, 0.714F, 0.804F, 1.F};
/** @brief Raised surface: cards, frames, buttons. */
inline constexpr ImVec4 RaisedSurface{0.105F, 0.125F, 0.153F, 1.F};
/** @brief Panel surface: child windows, side bars. */
inline constexpr ImVec4 PanelSurface{0.067F, 0.080F, 0.102F, 1.F};
/** @brief Background of the window. */
inline constexpr ImVec4 BackgroundSurface{0.035F, 0.043F, 0.057F, 1.F};

/** @brief Accent of a verdict: green, amber, red, or muted when unknown. */
ImVec4 HealthColor(editor::Health health) noexcept;

/** @brief Title in the heading font, with an optional muted description under it. */
void Heading(char const* title, char const* description = nullptr);

/** @brief Filled primary button for the main action of a panel. */
bool AccentButton(char const* label);

/**
 * @brief One choice of a segmented control: filled and accented when selected.
 * @param label Visible text (also the ImGui identifier).
 * @param selected True for the current choice.
 * @param tooltip Explanation on hover.
 * @return True when clicked.
 */
bool ToggleChip(char const* label, bool selected, char const* tooltip);

/**
 * @brief Wrapped tooltip on the last item, with a short hover delay by default.
 * @param text Explanation, or nullptr for no tooltip.
 * @param hoverFlags Hover policy; ImGuiHoveredFlags_None shows help immediately.
 */
void ItemTooltip(char const* text,
    ImGuiHoveredFlags hoverFlags = ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_NoSharedDelay);

/** @brief A muted "(?)" after the last item, explaining it on hover. */
void HelpMark(char const* text);

/** @brief Muted wrapped text. */
void MutedWrapped(char const* text);

/**
 * @brief Opens a framed panel: panel surface, rounded corners, a thin border and inner padding.
 * @param id ImGui identifier.
 * @param size Size as for ImGui::BeginChild (0 = fill, negative = leave that much).
 * @return What ImGui::BeginChild returned; EndPanel must be called in every case.
 */
bool BeginPanel(char const* id, ImVec2 size);

/** @brief Closes a panel opened by BeginPanel. */
void EndPanel();

/**
 * @brief Text centered in the available width, at `scale` times the current font size.
 * @param text Text, one line.
 * @param color Text color.
 * @param scale Size factor (1 = current font size).
 * @param heading True uses the heading font.
 */
void CenteredText(char const* text, ImVec4 color, float scale = 1.F, bool heading = false);

} // namespace ppokit::studio
