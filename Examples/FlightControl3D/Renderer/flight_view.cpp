/**
 * @file flight_view.cpp
 * @brief 3D scene (raylib) and instrument panels (Dear ImGui) of the FlightControl3D desktop.
 *
 * Role in the chain: snapshot -> **rendering**. Everything here reads a value-only
 * FlightSnapshot; nothing feeds back into the physics, the guidance or the learning.
 *
 * Render frame: raylib is y-up, so a world point (north, east, up) in metres is drawn at
 * (east, up, -north) / RenderMeters (one render unit = 100 m). Visual sizes below are
 * display choices only and have no physical meaning.
 *
 * Layout of the instrument window: a two-column table, instruments on the left (status,
 * targets, runway, controls and rewards, convergence diagnostics, landing details) and the
 * last crash reasons on the right. Each panel has its own function.
 *
 * Read next: flight_view.hpp, flight_display.hpp (pure display helpers),
 * Runtime/flight_session.hpp (FlightSnapshot).
 */
#include "flight_view.hpp"
#include "flight_display.hpp"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>

namespace ppo::examples::flight {
namespace {
// Render-only choices; all physics, guidance and learning use the JSON settings.
enum CameraMode { AircraftChase, FlightplanOverview, RunwayEnd };
int cameraMode = AircraftChase;
float chaseDistance = 80.F;
std::uint64_t shownCrashEvent{};
double crashAnimationStart{};
constexpr double CrashAnimationSeconds = 1.5;
/** @brief World metres per raylib render unit. */
constexpr double RenderMeters = 100.0;
constexpr Color GuidanceColor{220, 230, 240, 255};
constexpr Color ThresholdLabelColor{255, 220, 80, 255};
constexpr Color BackgroundColor{14, 27, 44, 255};
/** @brief Length of the drawn commanded-direction vector, in metres. */
constexpr double CommandVectorMeters = 180.0;
/** @brief Aircraft distance below which the runway camera looks at the threshold, in metres. */
constexpr double RunwayCameraMinimumDistanceMeters = 1.0;

/** @brief Converts a world point (north, east, up) in metres to raylib render units (y-up). */
Vector3 World(Vec3 point) noexcept
{
    return {static_cast<float>(point.east / RenderMeters), static_cast<float>(point.up / RenderMeters),
        static_cast<float>(-point.north / RenderMeters)};
}

void Segment(Vec3 from, Vec3 to, Color color)
{
    DrawLine3D(World(from), World(to), color);
}

/** @brief Double-sided quadrilateral (both windings, so it is visible from above and below). */
void Quad(Vec3 first, Vec3 second, Vec3 third, Vec3 fourth, Color color)
{
    DrawTriangle3D(World(first), World(second), World(third), color);
    DrawTriangle3D(World(first), World(third), World(fourth), color);
    DrawTriangle3D(World(third), World(second), World(first), color);
    DrawTriangle3D(World(fourth), World(third), World(first), color);
}

/**
 * @brief Draw a marking in runway coordinates, in physical metres.
 * @param runway Current physical runway pose.
 * @param start Beginning along the runway, in metres.
 * @param end End along the runway, in metres.
 * @param left Left edge relative to the centreline, in metres.
 * @param right Right edge relative to the centreline, in metres.
 * @param color Display color of the marking.
 */
void DrawRunwayMarking(Runway const& runway, double start, double end, double left, double right, Color color)
{
    // Slightly above the runway surface, to avoid z-fighting.
    constexpr double markingHeight = 0.1;
    Quad(FromRunway(runway, start, left, markingHeight), FromRunway(runway, end, left, markingHeight),
        FromRunway(runway, end, right, markingHeight), FromRunway(runway, start, right, markingHeight),
            color);
}

/**
 * @brief Draw the surrounding landable surface in the physical runway plane.
 * @param runway Runway pose, dimensions and surrounding margin.
 */
void DrawLandingArea(Runway const& runway)
{
    double const margin = runway.landingMarginMeters;
    if (margin <= 0)
    {
        return;
    }
    double const halfWidth = runway.widthMeters / 2;
    double const length = runway.lengthMeters;
    Color const ground{79, 112, 82, 255};
    // Left and right strips, then the strips before the threshold and after the far end.
    Quad(FromRunway(runway, -margin, -halfWidth - margin, 0),
        FromRunway(runway, length + margin, -halfWidth - margin, 0),
        FromRunway(runway, length + margin, -halfWidth, 0),
        FromRunway(runway, -margin, -halfWidth, 0), ground);
    Quad(FromRunway(runway, -margin, halfWidth, 0),
        FromRunway(runway, length + margin, halfWidth, 0),
        FromRunway(runway, length + margin, halfWidth + margin, 0),
        FromRunway(runway, -margin, halfWidth + margin, 0), ground);
    Quad(FromRunway(runway, -margin, -halfWidth, 0), FromRunway(runway, 0, -halfWidth, 0),
        FromRunway(runway, 0, halfWidth, 0), FromRunway(runway, -margin, halfWidth, 0), ground);
    Quad(FromRunway(runway, length, -halfWidth, 0), FromRunway(runway, length + margin, -halfWidth, 0),
        FromRunway(runway, length + margin, halfWidth, 0), FromRunway(runway, length, halfWidth, 0), ground);
}

/**
 * @brief Draw the edges of the elevated landing surface down to sea level.
 * @param runway Current physical runway pose and landing margin.
 * @param seaHeight Sea level in the local world frame, in metres.
 */
void DrawLandingAreaSides(Runway const& runway, double seaHeight)
{
    double const margin = runway.landingMarginMeters;
    double const halfWidth = runway.widthMeters / 2 + margin;
    std::array<Vec3, 4> const corners{
        FromRunway(runway, -margin, -halfWidth, 0),
        FromRunway(runway, runway.lengthMeters + margin, -halfWidth, 0),
        FromRunway(runway, runway.lengthMeters + margin, halfWidth, 0),
        FromRunway(runway, -margin, halfWidth, 0)};
    for (std::size_t corner = 0; corner < corners.size(); ++corner)
    {
        auto const top = corners[corner];
        auto const next = corners[(corner + 1) % corners.size()];
        Vec3 const bottom{top.north, top.east, seaHeight};
        Vec3 const nextBottom{next.north, next.east, seaHeight};
        Quad(top, next, nextBottom, bottom, {64, 81, 66, 255});
        Segment(top, next, {150, 169, 139, 255});
    }
}

/**
 * @brief Draw the physical runway and high-contrast markings at its exact dimensions.
 * @param runway Current physical runway pose and dimensions.
 */
void DrawRunway(Runway const& runway)
{
    constexpr int CentrelineDashes = 24;
    constexpr int StripesPerSide = 4;
    double const halfWidth = runway.widthMeters / 2;
    auto const thresholdLeft = FromRunway(runway, 0, -halfWidth, 0);
    auto const farLeft = FromRunway(runway, runway.lengthMeters, -halfWidth, 0);
    auto const farRight = FromRunway(runway, runway.lengthMeters, halfWidth, 0);
    auto const thresholdRight = FromRunway(runway, 0, halfWidth, 0);
    Quad(thresholdLeft, farLeft, farRight, thresholdRight, {38, 43, 50, 255});
    constexpr Color edgeColor{255, 210, 48, 255};
    double const edgeWidth = std::min(5.0, runway.widthMeters * 0.04);
    DrawRunwayMarking(runway, 0, runway.lengthMeters, -halfWidth, -halfWidth + edgeWidth, edgeColor);
    DrawRunwayMarking(runway, 0, runway.lengthMeters, halfWidth - edgeWidth, halfWidth, edgeColor);
    // Centreline: dashes of half the dash period, starting after the threshold bar.
    double const thresholdDepth = std::min(12.0, runway.lengthMeters * 0.02);
    double const dashWidth = std::min(6.0, runway.widthMeters * 0.08);
    for (int dash = 0; dash < CentrelineDashes; ++dash)
    {
        double const along = runway.lengthMeters * static_cast<double>(dash) / CentrelineDashes;
        DrawRunwayMarking(runway, std::max(along, thresholdDepth), along + runway.lengthMeters / 48,
            -dashWidth / 2, dashWidth / 2, WHITE);
    }
    // Threshold markings stay inside the physical surface and distinguish its entry end.
    DrawRunwayMarking(runway, 0, thresholdDepth, -halfWidth + edgeWidth, halfWidth - edgeWidth, edgeColor);
    double const stripeEnd = std::min(55.0, runway.lengthMeters * 0.08);
    double const stripeStart = thresholdDepth + (stripeEnd - thresholdDepth) * 0.1;
    double const stripeHalfWidth = runway.widthMeters * 0.0175;
    for (double const side : {-1.0, 1.0})
    {
        for (int stripe = 0; stripe < StripesPerSide; ++stripe)
        {
            double const centre = side * runway.widthMeters * (0.06 + 0.09 * stripe);
            double const stripeLeft = centre - stripeHalfWidth;
            double const stripeRight = centre + stripeHalfWidth;
            DrawRunwayMarking(runway, stripeStart, stripeEnd, stripeLeft, stripeRight, WHITE);
        }
    }
}

/**
 * @brief Projects a world point to the render target when it lies in front of the camera and on screen.
 * @param point Point in render units.
 * @param camera Current 3D camera.
 * @param target Render target dimensions.
 * @param screen Projected pixel position, written on success.
 * @return True when the point is visible.
 */
bool ProjectVisible(Vector3 point, Camera3D const& camera, RenderTexture2D const& target, Vector2& screen)
{
    Vector3 const view{camera.target.x - camera.position.x, camera.target.y - camera.position.y,
        camera.target.z - camera.position.z};
    double const depth = (point.x - camera.position.x) * view.x + (point.y - camera.position.y) * view.y
        + (point.z - camera.position.z) * view.z;
    if (!std::isfinite(depth) || depth <= 0 || target.texture.width <= 0 || target.texture.height <= 0)
    {
        return false;
    }
    screen = GetWorldToScreenEx(point, camera, target.texture.width, target.texture.height);
    return std::isfinite(screen.x) && std::isfinite(screen.y) && screen.x >= 0 && screen.y >= 0
        && screen.x < target.texture.width && screen.y < target.texture.height;
}

/**
 * @brief Label the physical threshold in screen space when it is visible.
 * @param runway Current physical runway pose.
 * @param camera Current 3D camera.
 * @param target Render target dimensions.
 */
void DrawThresholdLabel(Runway const& runway, Camera3D const& camera, RenderTexture2D const& target)
{
    constexpr int FontSize = 16;
    Vector2 screen{};
    if (!ProjectVisible(World(ThresholdPoint(runway)), camera, target, screen))
    {
        return;
    }
    int const pixelX = static_cast<int>(screen.x);
    int const pixelY = static_cast<int>(screen.y);
    DrawCircleLines(pixelX, pixelY, 8, ThresholdLabelColor);
    DrawLine(pixelX - 12, pixelY, pixelX - 5, pixelY, ThresholdLabelColor);
    DrawLine(pixelX + 5, pixelY, pixelX + 12, pixelY, ThresholdLabelColor);
    constexpr char label[] = "THRESHOLD";
    int const labelWidth = MeasureText(label, FontSize);
    int const textX = std::clamp(pixelX + 12, 0, std::max(0, target.texture.width - labelWidth));
    int const textY = std::clamp(pixelY - 20, 0, std::max(0, target.texture.height - FontSize));
    DrawRectangle(textX, textY, labelWidth, FontSize, {14, 27, 44, 230});
    DrawText(label, textX, textY, FontSize, ThresholdLabelColor);
}

/**
 * @brief Draws an expanding, fading impact fireball at the physical crash position.
 * @param position World position of the aircraft at the crash.
 * @param progress Animation fraction in [0,1].
 */
void DrawCrashExplosion(Vec3 position, double progress)
{
    constexpr int Particles = 8;
    double const fraction = std::clamp(progress, 0.0, 1.0);
    unsigned char const alpha = static_cast<unsigned char>(255.0 * (1.0 - fraction));
    auto const center = World(Add(position, {0, 0, 6.0 + 20.0 * fraction}));
    DrawSphereEx(center, static_cast<float>((10.0 + 35.0 * fraction) / RenderMeters), 8, 12, {238, 75, 18,
        alpha});
    DrawSphereEx(center, static_cast<float>((6.0 + 20.0 * fraction) / RenderMeters), 8, 12, {255, 196, 55,
        alpha});
    for (int particle = 0; particle < Particles; ++particle)
    {
        double const angle = 2.0 * Pi * static_cast<double>(particle) / 8.0;
        double const radius = 12.0 + 38.0 * fraction;
        auto const particlePosition = Add(position, {radius * std::cos(angle), radius * std::sin(angle), 10.0
            + 25.0 * fraction});
        DrawSphereEx(World(particlePosition), static_cast<float>((5.0 + 10.0 * fraction) / RenderMeters), 6,
            8,
            {255, 125, 20, alpha});
    }
}

/** @brief World position of a point given in aircraft body axes, in metres. */
Vec3 AircraftPoint(AircraftState const& aircraft, double forward, double right, double up)
{
    auto const axes = AircraftAxes(aircraft.roll, aircraft.pitch, aircraft.heading);
    return Add(aircraft.position, Add(Scale(axes.forward, forward), Add(Scale(axes.right, right),
        Scale(axes.up, up))));
}

char const* PhaseName(FlightSnapshot const& state) noexcept
{
    if (state.mission.phase == Phase::Approach)
    {
        return "Approach";
    }
    return "Rollout";
}

char const* OutcomeName(Outcome outcome) noexcept
{
    switch (outcome)
    {
    case Outcome::Flying:
        return "In flight";
    case Outcome::Stopped:
        return "Stopped on runway";
    case Outcome::Crash:
        return "Crash";
    case Outcome::Timeout:
        return "Time limit";
    }
    return "Time limit";
}

char const* GuidanceModeName(GuidanceMode mode) noexcept
{
    switch (mode)
    {
    case GuidanceMode::Flare:
        return "Flare";
    case GuidanceMode::Rollout:
        return "Runway rollout";
    case GuidanceMode::Approach:
        return "Approach";
    }
    return "Unknown";
}

/** @brief Side of the centreline for a lateral offset, in metres. */
char const* LateralSideName(double cross) noexcept
{
    if (cross < 0)
    {
        return "left";
    }
    if (cross > 0)
    {
        return "right";
    }
    return "center";
}

/**
 * @brief Draw a compact instrument tag connected to the projected aircraft.
 * @param state Render snapshot supplying aircraft and runway-relative measurements.
 * @param camera Active camera used to project the aircraft.
 * @param target Camera render target defining the available pixel extent.
 */
void DrawAircraftTag(FlightSnapshot const& state, Camera3D const& camera, RenderTexture2D const& target)
{
    constexpr int width = 190;
    constexpr int height = 146;
    constexpr int margin = 8;
    constexpr int fontSize = 16;
    constexpr int rowHeight = 21;
    constexpr int Rows = 5;
    if (target.texture.width < width + 2 * margin || target.texture.height < height + 2 * margin
        || !IsFinite(state.aircraft) || !std::isfinite(state.guidance.height)
        || !std::isfinite(DisplaySlopeDegrees(state.guidance))
            || !std::isfinite(state.runway.elevationMeters))
    {
        return;
    }
    Vector2 screen{};
    if (!ProjectVisible(World(state.aircraft.position), camera, target, screen))
    {
        return;
    }

    // Panel placement: right of the aircraft when it fits, else left; connector to the nearest edge.
    int const pixelX = static_cast<int>(screen.x);
    int const pixelY = static_cast<int>(screen.y);
    constexpr int TagOffset = 28;
    bool const fitsRight = pixelX + TagOffset + width + margin <= target.texture.width;
    int const preferredX = fitsRight ? pixelX + TagOffset : pixelX - TagOffset - width;
    int const panelX = std::clamp(preferredX, margin, target.texture.width - width - margin);
    int const panelY = std::clamp(pixelY - height / 2, margin, target.texture.height - height - margin);
    int const edgeX = panelX > pixelX ? panelX : panelX + width;
    int const edgeY = std::clamp(pixelY, panelY + 12, panelY + height - 12);
    Color accent{101, 217, 235, 255};
    if (state.mission.outcome == Outcome::Crash)
    {
        accent = Color{255, 112, 105, 255};
    }
    DrawLine(pixelX, pixelY, edgeX, edgeY, accent);
    DrawCircleLines(pixelX, pixelY, 5, accent);
    Rectangle const panel{static_cast<float>(panelX), static_cast<float>(panelY),
        static_cast<float>(width), static_cast<float>(height)};
    DrawRectangleRounded(panel, 0.08F, 6, {9, 20, 31, 225});
    DrawRectangle(panelX, panelY + 10, 3, 18, accent);
    DrawText("AIRCRAFT", panelX + 12, panelY + 10, 15, accent);
    DrawLine(panelX + 12, panelY + 31, panelX + width - 12, panelY + 31, {68, 93, 105, 255});

    double headingDegrees = std::fmod(state.aircraft.heading / DegreesToRadians, 360.0);
    if (headingDegrees < 0)
    {
        headingDegrees += 360.0;
    }
    int const roundedHeading = static_cast<int>(std::floor(headingDegrees + 0.5)) % 360;
    char values[Rows][40]{};
    // Position is expressed in the runway's local tangent frame, so this
    // altitude is approximate away from the geodetic reference point.
    std::snprintf(values[0], sizeof(values[0]), "~%.0f m", state.runway.elevationMeters
        + state.aircraft.position.up);
    std::snprintf(values[1], sizeof(values[1]), "%.0f m", state.guidance.height);
    std::snprintf(values[2], sizeof(values[2]), "%.1f m/s", state.aircraft.airspeed);
    std::snprintf(values[3], sizeof(values[3]), "%03d deg", roundedHeading);
    std::snprintf(values[4], sizeof(values[4]), "%+.1f deg", DisplaySlopeDegrees(state.guidance));
    constexpr char const* labels[]{"ALT", "H / RWY", "SPEED", "HEADING", "SLOPE"};
    for (int row = 0; row < Rows; ++row)
    {
        int const rowY = panelY + 39 + row * rowHeight;
        DrawText(labels[row], panelX + 12, rowY, fontSize, {155, 177, 187, 255});
        DrawText(values[row], panelX + width - 12 - MeasureText(values[row], fontSize), rowY, fontSize,
            {239, 246, 247, 255});
    }
}

/** @brief Chase camera behind and above the aircraft. */
void SetChaseCamera(FlightSnapshot const& state, Camera3D& camera)
{
    auto const behind = AircraftPoint(state.aircraft, -chaseDistance, 0, chaseDistance * 0.3);
    camera.position = World(behind);
    camera.target = World(AircraftPoint(state.aircraft, 30, 0, 0));
}

/** @brief Fixed camera near the threshold, zooming on the aircraft. */
void SetRunwayCamera(FlightSnapshot const& state, Camera3D& camera)
{
    auto const position = RunwayCameraPosition(state.runway);
    auto const delta = Subtract(state.aircraft.position, position);
    double const distance = std::sqrt(Dot(delta, delta));
    bool const aircraftAway = distance > RunwayCameraMinimumDistanceMeters;
    camera.position = World(position);
    camera.target = World(aircraftAway ? state.aircraft.position : ThresholdPoint(state.runway));
    camera.fovy = static_cast<float>(RunwayCameraFieldOfView(position, state.aircraft.position));
    // Looking straight up or down: use the runway axis as the camera up vector.
    auto const normal = RunwayNormal(state.runway);
    bool const vertical = aircraftAway && std::abs(Dot(delta, normal)) / distance > 0.99;
    camera.up = World(vertical ? RunwayForward(state.runway) : normal);
}

/** @brief Isometric camera framing the aircraft and the whole landing surface. */
void SetOverviewCamera(FlightSnapshot const& state, Camera3D& camera)
{
    Vec3 minimum = state.aircraft.position;
    Vec3 maximum = state.aircraft.position;
    auto include = [&minimum, &maximum](Vec3 point) {
        minimum = {std::min(minimum.north, point.north), std::min(minimum.east, point.east),
            std::min(minimum.up, point.up)};
        maximum = {std::max(maximum.north, point.north), std::max(maximum.east, point.east),
            std::max(maximum.up, point.up)};
    };
    include(state.runway.origin);
    include(FromRunway(state.runway, state.runway.lengthMeters, 0, 0));
    if (state.runway.landingMarginMeters > 0)
    {
        double const margin = state.runway.landingMarginMeters;
        double const halfWidth = state.runway.widthMeters / 2 + margin;
        for (double const along : {-margin, state.runway.lengthMeters + margin})
        {
            for (double const cross : {-halfWidth, halfWidth})
            {
                include(FromRunway(state.runway, along, cross, 0));
            }
        }
    }
    auto const centre = Scale(Add(minimum, maximum), 0.5);
    double const range = std::max({maximum.north - minimum.north, maximum.east - minimum.east,
        state.runway.lengthMeters});
    camera.target = World(centre);
    camera.position = World(Add(centre, {range, range, range}));
}

Camera3D MakeCamera(FlightSnapshot const& state)
{
    Camera3D camera{};
    camera.up = {0, 1, 0};
    camera.fovy = 50.F;
    camera.projection = CAMERA_PERSPECTIVE;
    if (cameraMode == AircraftChase)
    {
        SetChaseCamera(state, camera);
    }
    else if (cameraMode == RunwayEnd)
    {
        SetRunwayCamera(state, camera);
    }
    else
    {
        SetOverviewCamera(state, camera);
    }
    return camera;
}

/** @brief Simplified T-6 silhouette: fuselage, wing, tailplane, fin and gear legs. */
void DrawAircraftModel(FlightSnapshot const& state)
{
    auto const& aircraft = state.aircraft;
    Color color = SKYBLUE;
    if (state.mission.outcome == Outcome::Crash)
    {
        color = RED;
    }
    auto const nose = AircraftPoint(aircraft, 4.5, 0, 0);
    auto const tail = AircraftPoint(aircraft, -4.5, 0, 0);
    DrawCylinderEx(World(nose), World(tail), 0.006F, 0.004F, 8, color);
    Quad(AircraftPoint(aircraft, 1, 0, 0), AircraftPoint(aircraft, -1, -6, 0),
        AircraftPoint(aircraft, -1.5, 0, 0), AircraftPoint(aircraft, -1, 6, 0), color);
    Quad(AircraftPoint(aircraft, -3, 0, 0), AircraftPoint(aircraft, -4, -2, 0),
        AircraftPoint(aircraft, -4.5, 0, 0), AircraftPoint(aircraft, -4, 2, 0), color);
    Segment(tail, AircraftPoint(aircraft, -4, 0, 2), WHITE);
    if (aircraft.gearPosition > 0.01)
    {
        for (double const right : {-2.0, 2.0})
        {
            auto const wheel = AircraftPoint(aircraft, -1, right, -1.7 * aircraft.gearPosition);
            Segment(AircraftPoint(aircraft, -1, right, 0), wheel, WHITE);
            DrawSphere(World(wheel), 0.003F, DARKGRAY);
        }
    }
    if (cameraMode == FlightplanOverview)
    {
        DrawSphere(World(aircraft.position), 0.8F, color);
    }
}

/** @brief Target point and commanded direction, the same reference the policy and rewards use. */
void DrawGuidance(FlightSnapshot const& state)
{
    if (!state.guidance.targetValid)
    {
        return;
    }
    Segment(state.aircraft.position, state.guidance.target, GuidanceColor);
    DrawSphere(World(state.guidance.target), 0.035F, GuidanceColor);
    // The short vector displays the same 3D command used by the policy and rewards.
    Segment(state.aircraft.position,
        Add(state.aircraft.position, Scale(state.guidance.targetDirection, CommandVectorMeters)), GREEN);
}
} // namespace

void DrawFlightScene(FlightSnapshot const& snapshot, RenderTexture2D const& target)
{
    // A new crash event freezes the frame on the crash pose for a short animation.
    double const now = GetTime();
    if (snapshot.crashEvent != shownCrashEvent)
    {
        shownCrashEvent = snapshot.crashEvent;
        crashAnimationStart = snapshot.crashEvent ? now : 0;
    }
    double const crashAge = now - crashAnimationStart;
    bool const showCrash = snapshot.crashEvent != 0 && crashAge >= 0 && crashAge < CrashAnimationSeconds;
    FlightSnapshot frame = snapshot;
    if (showCrash)
    {
        frame.aircraft = snapshot.crashAircraft;
        frame.runway = snapshot.crashRunway;
        frame.mission.outcome = Outcome::Crash;
        frame.guidance.targetValid = false;
    }
    auto const camera = MakeCamera(frame);
    BeginTextureMode(target);
    ClearBackground(BackgroundColor);
    BeginMode3D(camera);
    // The sea is a visual backdrop; the actual finite moving surface is in JSBSim.
    DrawPlane(World({frame.aircraft.position.north, frame.aircraft.position.east, frame.seaHeightMeters}),
        {1000.F, 1000.F}, {18, 53, 75, 255});
    DrawLandingAreaSides(frame.runway, frame.seaHeightMeters);
    DrawLandingArea(frame.runway);
    DrawRunway(frame.runway);
    DrawGuidance(frame);
    DrawAircraftModel(frame);
    if (showCrash)
    {
        DrawCrashExplosion(frame.aircraft.position, crashAge / CrashAnimationSeconds);
    }
    EndMode3D();
    DrawThresholdLabel(frame.runway, camera, target);
    if (!showCrash)
    {
        DrawAircraftTag(frame, camera, target);
    }
    if (showCrash)
    {
        DrawText("CRASH", 16, 16, 26, {255, 140, 54, 255});
    }
    EndTextureMode();
}

void DrawFlightDisplayControls()
{
    ImGui::Combo("Camera", &cameraMode, "Aircraft chase\0Runway / flightplan\0Runway\0");
    if (cameraMode == AircraftChase)
    {
        ImGui::SliderFloat("Distance (m)", &chaseDistance, 25.F, 300.F);
    }
}

namespace {
/**
 * @brief Display the most recent crash conditions and their measured limits.
 * @param state Render snapshot, including crashes retained across resets.
 */
void DrawCrashDetails(FlightSnapshot const& state)
{
    ImGui::TextUnformatted("Last crash reasons");
    ImGui::Separator();
    if (!state.hasCrashDiagnostic)
    {
        ImGui::TextDisabled("No crash recorded");
        return;
    }
    constexpr std::array<char const*, CrashReasonCount> labels{
        "Invalid aircraft state", "Distance", "Altitude",
        "Outside landing area", "Wheel outside landing area", "Gear extension", "Structural contact",
        "Sink rate (deck-relative)", "Speed (deck-relative)", "Roll", "Pitch", "Pitch", "Heading",
            "No wheel contact"};
    for (std::size_t index = 0; index < CrashReasonCount; ++index)
    {
        auto const& condition = state.crashDiagnostic.conditions[index];
        if (!condition.failed)
        {
            continue;
        }
        auto const reason = static_cast<CrashReason>(index);
        ImGui::TextColored({1.F, 0.45F, 0.4F, 1.F}, "%s", labels[index]);
        switch (reason)
        {
        case CrashReason::Distance:
        case CrashReason::Altitude:
            ImGui::Text("%.1f / max %.1f m", condition.value, condition.limit);
            break;
        case CrashReason::Sink:
        case CrashReason::Speed:
            ImGui::Text("%.1f / max %.1f m/s", condition.value, condition.limit);
            break;
        case CrashReason::Gear:
            ImGui::Text("%.1f%% / min %.1f%%", 100 * condition.value, 100 * condition.limit);
            break;
        case CrashReason::PitchLow:
            ImGui::Text("%.1f / min %.1f deg", condition.value, condition.limit);
            break;
        case CrashReason::Roll:
        case CrashReason::PitchHigh:
        case CrashReason::Heading:
            ImGui::Text("%.1f / max %.1f deg", condition.value, condition.limit);
            break;
        default:
            break;
        }
    }
}

/** @brief Phase, outcome, curriculum, validation, BEST and guidance summary. */
void DrawStatusPanel(FlightSnapshot const& state)
{
    if (state.failed)
    {
        ImGui::TextColored({1, 0.3F, 0.3F, 1}, "Simulation stopped: see FLIGHT_ERROR in the log.");
    }
    ImGui::Text("%s | %s%s", PhaseName(state), OutcomeName(state.mission.outcome), state.finished
        ? " | Complete" : "");
    ImGui::TextUnformatted("Aircraft dynamics: T-6 Texan II");
    if (state.training && state.curriculumLevels > 0)
    {
        ImGui::Text("Curriculum level %zu / %zu | window %zu / %zu stops", state.curriculumLevel,
            state.curriculumLevels - 1, state.curriculumWindowStops, state.curriculumWindowEpisodes);
    }
    if (state.gateEvaluated > 0)
    {
        ImGui::Text("Last independent validation: %zu / %zu stops at difficulty %.2f", state.gateSuccesses,
            state.gateEvaluated, state.lastValidation.difficulty);
    }
    if (state.hasBest)
    {
        ImGui::Text("BEST: %.0f%% stops at difficulty %.2f (update %llu)", state.bestScore,
            state.bestDifficulty,
            static_cast<unsigned long long>(state.bestUpdate));
    }
    ImGui::Text("Guidance: %s", GuidanceModeName(state.guidance.mode));
    ImGui::Text("Aim point: %.0f m past the threshold", state.aimPointAlongMeters);
    double const targetSink = -state.guidance.targetSpeed * Dot(state.guidance.targetDirection,
        RunwayNormal(state.runway));
    ImGui::Text("Sink rate target / actual: %.2f / %.2f m/s", targetSink,
        -state.guidance.relativeVelocity.up);
    ImGui::Text("Episode time %.1f / %.0f simulated s", state.elapsedSeconds, state.maximumEpisodeSeconds);
    if (state.lastEpisode.outcome != Outcome::Flying)
    {
        ImGui::Text("Last episode: %s after %.1f simulated s", OutcomeName(state.lastEpisode.outcome),
            state.lastEpisodeSeconds);
    }
}

/** @brief Target, actual and error of the airspeed, the slope and the 3D track. */
void DrawTargetsTable(FlightSnapshot const& state)
{
    if (!ImGui::BeginTable("##flight-targets", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
    {
        return;
    }
    auto const& guidance = state.guidance;
    ImGui::TableSetupColumn("Measure");
    ImGui::TableSetupColumn("Target");
    ImGui::TableSetupColumn("Actual");
    ImGui::TableSetupColumn("Error");
    ImGui::TableHeadersRow();
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted("CAS (m/s)");
    ImGui::TableNextColumn();
    ImGui::Text("%.1f", guidance.targetSpeed);
    ImGui::TableNextColumn();
    ImGui::Text("%.1f", state.aircraft.airspeed);
    ImGui::TableNextColumn();
    ImGui::Text("%+.1f", state.aircraft.airspeed - guidance.targetSpeed);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted("Slope (deg)");
    ImGui::TableNextColumn();
    ImGui::Text("%+.2f", guidance.targetSlope / DegreesToRadians);
    ImGui::TableNextColumn();
    ImGui::Text("%+.2f", DisplaySlopeDegrees(guidance));
    ImGui::TableNextColumn();
    ImGui::Text("%+.2f", DisplaySlopeDegrees(guidance) - guidance.targetSlope / DegreesToRadians);
    double const track = std::atan2(guidance.velocityDirection.east, guidance.velocityDirection.north)
        / DegreesToRadians;
    double const directionDot = std::clamp(Dot(guidance.targetDirection, guidance.velocityDirection), -1.0,
        1.0);
    double const directionError = std::acos(directionDot) / DegreesToRadians;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted("Track azimuth (deg)");
    ImGui::TableNextColumn();
    ImGui::Text("%+.1f", guidance.targetHeading / DegreesToRadians);
    ImGui::TableNextColumn();
    ImGui::Text("%+.1f", track);
    ImGui::TableNextColumn();
    ImGui::Text("%.1f 3D", directionError);
    ImGui::EndTable();
}

/** @brief Runway dimensions and the aircraft position relative to it. */
void DrawRunwayPanel(FlightSnapshot const& state)
{
    ImGui::TextDisabled("Slope: + climb / - descent (deck-relative)");
    ImGui::Text("Runway dimensions: %.0f x %.0f m", state.runway.lengthMeters, state.runway.widthMeters);
    ImGui::Text("Runway threshold above sea: %.1f m", state.runway.origin.up - state.seaHeightMeters);
    if (state.runway.landingMarginMeters > 0)
    {
        ImGui::Text("Landing area margin: %.0f m", state.runway.landingMarginMeters);
    }
    ImGui::Text("Runway %.2f km | Height %.0f m", state.guidance.distance / 1000, state.guidance.height);
    ImGui::Text("Runway lateral: %+.0f m (%s)", state.guidance.cross, LateralSideName(state.guidance.cross));
}

/** @brief Network mean, sampled raw and applied value of each control. */
void DrawControlsTable(FlightSnapshot const& state)
{
    if (!ImGui::BeginTable("##flight-controls", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
    {
        return;
    }
    ImGui::TableSetupColumn("Control");
    ImGui::TableSetupColumn("Network mean");
    ImGui::TableSetupColumn("Sampled raw");
    ImGui::TableSetupColumn("Applied");
    ImGui::TableHeadersRow();
    std::array<double, ContinuousActions> const applied{state.controls.elevator, state.controls.aileron,
        state.controls.rudder, state.controls.throttle, state.controls.brake};
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(ActionNames[axis]);
        ImGui::TableNextColumn();
        if (state.diagnostics.enabled)
        {
            ImGui::Text("%+.3f", state.diagnostics.policy[axis]);
        }
        else
        {
            ImGui::TextUnformatted("-");
        }
        ImGui::TableNextColumn();
        if (state.diagnostics.enabled)
        {
            ImGui::Text("%+.3f", state.diagnostics.rawAction[axis]);
        }
        else
        {
            ImGui::TextUnformatted("-");
        }
        ImGui::TableNextColumn();
        ImGui::Text("%+.3f", applied[axis]);
    }
    ImGui::EndTable();
}

/** @brief Last-step and window-mean value of each reward channel. */
void DrawRewardsTable(FlightSnapshot const& state)
{
    if (!ImGui::BeginTable("##flight-rewards", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
    {
        return;
    }
    ImGui::TableSetupColumn("Reward");
    ImGui::TableSetupColumn("Last step");
    ImGui::TableSetupColumn("Last window mean");
    ImGui::TableHeadersRow();
    double const samples = static_cast<double>(std::max<std::uint64_t>(state.lastDiagnostics.samples, 1));
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(RewardNames[channel]);
        ImGui::TableNextColumn();
        ImGui::Text("%+.5f", state.lastStep.rewards[channel]);
        ImGui::TableNextColumn();
        if (state.diagnostics.enabled && state.lastDiagnostics.samples > 0)
        {
            ImGui::Text("%+.5f", state.lastDiagnostics.rewardSums[channel] / samples);
        }
        else
        {
            ImGui::TextUnformatted("-");
        }
    }
    ImGui::EndTable();
}

void DrawControlsAndRewards(FlightSnapshot const& state)
{
    if (!ImGui::CollapsingHeader("Controls and rewards", ImGuiTreeNodeFlags_DefaultOpen))
    {
        return;
    }
    if (!state.diagnostics.enabled)
    {
        ImGui::TextDisabled("Network diagnostics disabled in the scenario configuration.");
    }
    DrawControlsTable(state);
    DrawRewardsTable(state);
}

/** @brief Raw and applied magnitudes and saturations of each action over the last window. */
void DrawActionDiagnosticsTable(FlightDiagnostics const& window)
{
    if (window.samples == 0
        || !ImGui::BeginTable("##action-diagnostics", 5,
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
    {
        return;
    }
    ImGui::TableSetupColumn("Action");
    ImGui::TableSetupColumn("Raw mean |x|");
    ImGui::TableSetupColumn("Applied mean |x|");
    ImGui::TableSetupColumn("Raw sat");
    ImGui::TableSetupColumn("Applied sat");
    ImGui::TableHeadersRow();
    double const samples = static_cast<double>(window.samples);
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(ActionNames[axis]);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", window.rawAbsoluteSums[axis] / samples);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", window.appliedAbsoluteSums[axis] / samples);
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(window.rawSaturations[axis]));
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(window.appliedSaturations[axis]));
    }
    ImGui::EndTable();
}

/** @brief Live value, window mean, maximum and saturation count of each observation slot. */
void DrawObservationDiagnosticsTable(FlightSnapshot const& state)
{
    constexpr float TableHeight = 220;
    if (!ImGui::BeginTable("##observations", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH
        | ImGuiTableFlags_ScrollY, ImVec2(0, TableHeight)))
    {
        return;
    }
    auto const& window = state.lastDiagnostics;
    ImGui::TableSetupColumn("Observation");
    ImGui::TableSetupColumn("Now");
    ImGui::TableSetupColumn("Mean |x|");
    ImGui::TableSetupColumn("Max |x| / sat");
    ImGui::TableHeadersRow();
    double const samples = static_cast<double>(std::max<std::uint64_t>(window.samples, 1));
    for (std::size_t slot = 0; slot < ObservationSize; ++slot)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(ObservationNames[slot]);
        ImGui::TableNextColumn();
        ImGui::Text("%+.3f", state.diagnostics.observation[slot]);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", window.observationAbsoluteSums[slot] / samples);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f / %llu", window.observationMaximumAbsolute[slot],
            static_cast<unsigned long long>(window.observationSaturations[slot]));
    }
    ImGui::EndTable();
}

void DrawConvergenceDiagnostics(FlightSnapshot const& state)
{
    if (!state.diagnostics.enabled || !ImGui::CollapsingHeader("Convergence diagnostics"))
    {
        return;
    }
    auto const& window = state.lastDiagnostics;
    auto const saturatedObservations = std::count_if(window.observationSaturations.begin(),
        window.observationSaturations.end(), [](std::uint64_t count) { return count > 0; });
    auto const saturatedActions = std::accumulate(window.appliedSaturations.begin(),
        window.appliedSaturations.end(), std::uint64_t{});
    ImGui::Text("Window samples %llu | saturated inputs %zu/%zu | applied saturation hits %llu",
        static_cast<unsigned long long>(window.samples), static_cast<std::size_t>(saturatedObservations),
        ObservationSize, static_cast<unsigned long long>(saturatedActions));
    ImGui::Text("Minibatch approximate KL max %.6f | actor steps %zu",
        state.learning.maximumApproximateKl, state.learning.actorSteps);
    ImGui::Text("Sampled minibatch KL max %.6f | skipped actor batches %zu",
        state.learning.maximumSampledKl, state.learning.actorKlSkips);
    ImGui::Text("Critic steps %zu", state.learning.criticSteps);
    DrawActionDiagnosticsTable(window);
    DrawObservationDiagnosticsTable(state);
}

/** @brief Attitude, gear and the touchdown of the current (or last) episode. */
void DrawLandingDetails(FlightSnapshot const& state)
{
    auto const& result = state.mission.touchedDown ? state.mission : state.lastEpisode;
    if (!ImGui::CollapsingHeader("Aircraft / landing details"))
    {
        return;
    }
    ImGui::Text("Gear extension %.0f%% | WOW %s",
        state.aircraft.gearPosition * 100, state.aircraft.weightOnWheels ? "YES" : "NO");
    ImGui::Text("Roll %.1f | Pitch %.1f | Heading %.1f deg", state.aircraft.roll / DegreesToRadians,
        state.aircraft.pitch / DegreesToRadians, state.aircraft.heading / DegreesToRadians);
    ImGui::Text("Relative sink %.2f m/s", -state.guidance.relativeVelocity.up);
    if (result.touchedDown)
    {
        ImGui::Text("Touchdown %.1f m | Ground roll %.1f m", result.touchdownFromThreshold,
            result.groundRoll);
        ImGui::Text("Sink %.2f m/s | Speed %.1f m/s", result.touchdownSinkMps, result.touchdownSpeedMps);
        ImGui::Text("Touchdown quality %.2f | Landing area factor %.2f", result.touchdownQuality,
            result.landingAreaFactor);
    }
    if (result.outcome == Outcome::Stopped)
    {
        ImGui::Text("Stop %.1f m from threshold", result.stopFromThreshold);
    }
}
} // namespace

void DrawFlightInstruments(FlightSnapshot const& state)
{
    if (!ImGui::Begin("FlightControl3D"))
    {
        ImGui::End();
        return;
    }
    if (!ImGui::BeginTable("##flight-layout", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV))
    {
        ImGui::End();
        return;
    }
    ImGui::TableSetupColumn("Instruments", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Crash reasons", ImGuiTableColumnFlags_WidthFixed, 260.F);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    DrawStatusPanel(state);
    DrawTargetsTable(state);
    DrawRunwayPanel(state);
    DrawControlsAndRewards(state);
    DrawConvergenceDiagnostics(state);
    DrawLandingDetails(state);
    ImGui::TableNextColumn();
    DrawCrashDetails(state);
    ImGui::EndTable();
    ImGui::End();
}
} // namespace ppo::examples::flight
