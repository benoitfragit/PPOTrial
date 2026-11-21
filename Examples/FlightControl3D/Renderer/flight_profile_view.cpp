/**
 * @file flight_profile_view.cpp
 * @brief Vertical approach profile: glide plane, threshold, aim point and aircraft, runway-relative.
 *
 * Role in the chain: snapshot -> **profile panel**. The horizontal axis is the distance
 * past the threshold, the vertical axis the height above the deck; both are in metres and
 * mapped to the panel by ProfileCanvas. The glide plane is drawn with the same
 * AimPointAlongMeters / ApproachPathHeight functions the guidance uses.
 *
 * Read next: Core/flight_guidance.hpp, flight_view.cpp.
 */
#include "flight_view.hpp"
#include "flight_guidance.hpp"
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ppo::examples::flight {
namespace {
/** @brief The profile always shows at least the last 2.5 km before the threshold. */
constexpr double MinimumViewStartMeters = -2500.0;
/** @brief Horizon of the drawn velocity vector, in seconds. */
constexpr double VelocityProjectionSeconds = 4;
/** @brief Largest profile height, in pixels. */
constexpr float MaximumProfileHeightPixels = 200.F;
constexpr ImU32 GlidePlaneColor = IM_COL32(92, 208, 166, 255);
constexpr ImU32 ThresholdColor = IM_COL32(245, 192, 85, 255);

/** @brief Maps runway-relative metres to a bounded profile canvas. */
struct ProfileCanvas final
{
    ImVec2 minimum{};
    ImVec2 maximum{};
    double alongMin{};
    double alongMax{};
    double heightMin{};
    double heightMax{};

    /** @brief Pixel position of a runway-relative point (along, height), in metres. */
    ImVec2 Point(double along, double height) const noexcept
    {
        double const alongFraction = (along - alongMin) / (alongMax - alongMin);
        double const heightFraction = (height - heightMin) / (heightMax - heightMin);
        return {minimum.x + static_cast<float>(alongFraction) * (maximum.x - minimum.x),
            maximum.y - static_cast<float>(heightFraction) * (maximum.y - minimum.y)};
    }
};

/** @brief Draw a labelled tick inside the profile's plotting area. */
void Tick(ImDrawList& draw, ProfileCanvas const& canvas, double along) noexcept
{
    auto const point = canvas.Point(along, canvas.heightMin);
    if (point.x < canvas.minimum.x || point.x > canvas.maximum.x)
    {
        return;
    }
    draw.AddLine({point.x, canvas.minimum.y}, {point.x, canvas.maximum.y}, IM_COL32(73, 101, 123, 100));
    char label[32]{};
    std::snprintf(label, sizeof(label), "%.0f m", along);
    draw.AddText({point.x + 3, canvas.maximum.y - 16}, IM_COL32(167, 188, 199, 255), label);
}

/** @brief Labelled point marker. */
void Marker(ImDrawList& draw, ImVec2 point, ImU32 color, char const* label)
{
    draw.AddCircleFilled(point, 4.F, color);
    draw.AddText({point.x + 5, point.y - 15}, color, label);
}
} // namespace

void DrawFlightApproachProfile(FlightSnapshot const& state)
{
    ImGui::SeparatorText("Approach profile | runway-relative");
    if (!state.approachProfileValid)
    {
        ImGui::TextDisabled("Approach profile unavailable for this flight.");
        return;
    }
    // Only the runway and the glide slope are needed by the glide-plane helpers.
    Config reference{};
    reference.runway = state.runway;
    reference.mission.glideDegrees = state.glideDegrees;
    double const aimAlong = AimPointAlongMeters(reference);
    double const along = state.guidance.along;
    // Horizontal extent: from 2.5 km out (or further, around the aircraft) to past the aim point.
    double const startAlong = std::min(MinimumViewStartMeters, along - std::max(100.0, std::abs(along)
        * 0.1));
    double const endAlong = std::max(aimAlong + 400.0, along + 100.0);
    double const aircraftPathHeight = ApproachPathHeight(reference, along);
    double const startHeight = ApproachPathHeight(reference, startAlong);
    double const targetSink = -state.guidance.targetSpeed
        * Dot(state.guidance.targetDirection, RunwayNormal(state.runway));
    double const sink = -state.guidance.relativeVelocity.up;
    double const error = state.guidance.height - aircraftPathHeight;
    ImGui::Text("Height %.1f m | path error %+.1f m | sink %.2f / target %.2f m/s",
        state.guidance.height, error, sink, targetSink);
    ImGui::TextDisabled("Green: glide plane to the aim point | Blue: aircraft and 4 s velocity"
        " | Amber: altitude error");

    ImVec2 const available = ImGui::GetContentRegionAvail();
    if (available.x < 80.F || available.y < 35.F)
    {
        return;
    }
    ImVec2 const origin = ImGui::GetCursorScreenPos();
    ImVec2 const size{available.x, std::min(available.y, MaximumProfileHeightPixels)};
    ImGui::InvisibleButton("##approach-profile", size);
    ImDrawList& draw = *ImGui::GetWindowDrawList();
    draw.AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, IM_COL32(12, 24, 35, 255), 4.F);
    // Vertical extent: the deck, the aircraft and the glide plane at the left edge, with 12 % padding.
    double const maxHeight = std::max({10.0, state.guidance.height, startHeight});
    double const minHeight = std::min(0.0, state.guidance.height);
    double const heightPadding = std::max(5.0, (maxHeight - minHeight) * 0.12);
    ProfileCanvas const canvas{{origin.x + 10, origin.y + 8}, {origin.x + size.x - 10, origin.y + size.y - 8},
        startAlong, endAlong, minHeight - heightPadding, maxHeight + heightPadding};
    draw.PushClipRect(canvas.minimum, canvas.maximum, true);
    Tick(draw, canvas, 0);
    Tick(draw, canvas, aimAlong);
    ImVec2 const runwayStart = canvas.Point(0, 0);
    ImVec2 const runwayEnd = canvas.Point(std::min(endAlong, state.runway.lengthMeters), 0);
    draw.AddLine(runwayStart, runwayEnd, IM_COL32(180, 194, 194, 255), 4.F);
    // The glide plane is straight: it crosses the threshold at its crossing height and meets the aim point.
    draw.AddLine(canvas.Point(startAlong, startHeight), canvas.Point(aimAlong, 0), GlidePlaneColor, 2.F);
    Marker(draw, canvas.Point(0, state.runway.thresholdHeightMeters), ThresholdColor, "Threshold");
    Marker(draw, canvas.Point(aimAlong, 0), GlidePlaneColor, "Aim point");
    auto const aircraft = canvas.Point(state.guidance.along, state.guidance.height);
    auto const pathPoint = canvas.Point(along, aircraftPathHeight);
    draw.AddLine(pathPoint, aircraft, IM_COL32(255, 181, 93, 220), 1.5F);
    double const projectedAlong = state.guidance.along
        + state.guidance.relativeVelocity.north * VelocityProjectionSeconds;
    double const projectedHeight = state.guidance.height
        + state.guidance.relativeVelocity.up * VelocityProjectionSeconds;
    auto const projected = canvas.Point(projectedAlong, projectedHeight);
    draw.AddLine(aircraft, projected, IM_COL32(119, 190, 255, 210), 1.5F);
    draw.AddTriangleFilled({aircraft.x + 7, aircraft.y}, {aircraft.x - 6, aircraft.y - 4},
        {aircraft.x - 6, aircraft.y + 4}, IM_COL32(133, 208, 255, 255));
    draw.PopClipRect();
}

} // namespace ppo::examples::flight
