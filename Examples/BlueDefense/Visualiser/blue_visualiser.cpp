/**
 * @file Examples/BlueDefense/Visualiser/blue_visualiser.cpp
 * @brief The BlueDefense Visualiseur: an observer that sends the blue team's world after
 *        each step, and a top-down 2D view of it for ppokit-viewer.
 * @details The view is what the former PPO.exe's tactical minimap (Airspace renderer) drew for
 * BlueDefense, plus its 3D pseudo-lidars and explosions, seen from above: the defended area,
 * the outer failure rectangle, the restricted zones, the objective; the leader's patrol track
 * with its direction, the racetrack's waypoints (active, validated, heading); each blue with
 * its heading, trail, role badge (L, W1...), engagement cone, pseudo-lidar beams and its
 * patrol reference (projected point, guidance point and heading); each red with its heading,
 * trail and cone (a cross once destroyed); an explosion where a mobile or the objective is
 * destroyed; a few lines of data. North is up; the view first fits the outer rectangle;
 * dragging moves it, the wheel zooms; a click on a blue shows its pseudo-lidar (the
 * leader's until then), as the former PPO.exe showed the selected mobile's.
 *
 * The observer's ports are rows of blue slots (task.json "world" of BluePatrol, one row per
 * slot): "blue" (present, alive, leader, north, east, heading, speed, reference north/east),
 * "reds" (10 x north, east, heading, alive, present), "objective" (north, east, alive,
 * defended area north, east, half extents), "waypoints" (26 x north, east, active index or
 * -1), "blue_view" (role: -1 none, 0 leader, k teammate slot W k; cone radius and half
 * angle; projected reference north, east; guidance heading; reference shown), "red_cones"
 * (10 x radius, half angle), "lidar" (61 normalized ranges, 61 distances, span; span 0: none),
 * "zones" (16 x north, east, radius, hostile; the world's zone count), "bounds" (outer
 * failure rectangle half extents), "patrol" (97 x track north, east; sample count; 26
 * waypoint headings; validated waypoints), "rewards" (the 15 weighted channels of the step).
 */
#include "ppokit/view/sdk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

using ppokit::sdk::Call;
using ppokit::sdk::Config;
using ppokit::sdk::Manifest;
using ppokit::sdk::Shapes;

constexpr std::uint32_t BlueValues = 9;
constexpr std::uint32_t Reds = 10;
constexpr std::uint32_t RedValues = 5;
constexpr std::uint32_t ObjectiveValues = 7;
constexpr std::uint32_t Waypoints = 26;
constexpr std::uint32_t WaypointValues = Waypoints * 2 + 1;
constexpr std::uint32_t RewardChannels = 15;
constexpr std::uint32_t BlueSlots = 9;
constexpr std::uint32_t BlueViewValues = 7;
constexpr std::uint32_t RedConeValues = 2;
constexpr std::uint32_t LidarBins = 61;
constexpr std::uint32_t LidarValues = 2 * LidarBins + 1;
constexpr std::uint32_t Zones = 16;
constexpr std::uint32_t ZoneValues = 4;
constexpr std::uint32_t ZonesSize = Zones * ZoneValues + 1;
constexpr std::uint32_t BoundsValues = 2;
constexpr std::uint32_t TrackSamples = 97;
constexpr std::uint32_t PatrolValues = 2 * TrackSamples + 1 + Waypoints + 1;
/** @brief Wall-clock life of an explosion, as in the former PPO.exe's renderer. */
constexpr double ExplosionSeconds = 2.0;
/** @brief Mobiles and the objective whose destruction the view animates. */
constexpr std::size_t Destructibles = BlueSlots + Reds + 1;
/** @brief Positions kept per mobile for its trail. */
constexpr std::size_t TrailLength = 240;

/** @brief Sends what it is wired to: blue, reds, objective, waypoints and rewards rows. */
class BlueWatch final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("blue_watch", 2);
        manifest.Input("blue", {BlueValues})
            .Input("reds", {Reds * RedValues})
            .Input("objective", {ObjectiveValues})
            .Input("waypoints", {WaypointValues})
            .Input("blue_view", {BlueViewValues})
            .Input("red_cones", {Reds * RedConeValues})
            .Input("lidar", {LidarValues})
            .Input("zones", {ZonesSize})
            .Input("bounds", {BoundsValues})
            .Input("patrol", {PatrolValues})
            .Input("rewards", {RewardChannels});
        manifest.Observer();
        manifest.Integer("port", 47001, 1024, 65535).Number("rate", 30.0, 1.0, 240.0);
        return manifest;
    }
    BlueWatch(Config const& config, Shapes const&, std::uint64_t)
        : sender_(config,
              {{"blue", BlueValues}, {"reds", Reds * RedValues}, {"objective", ObjectiveValues},
                  {"waypoints", WaypointValues}, {"blue_view", BlueViewValues},
                  {"red_cones", Reds * RedConeValues}, {"lidar", LidarValues}, {"zones", ZonesSize},
                  {"bounds", BoundsValues}, {"patrol", PatrolValues}, {"rewards", RewardChannels}})
    {
    }
    void Forward(Call const& call) override { sender_.Send(call); }

private:
    ppokit::view::FrameSender sender_;
};

/** @brief A point of the plane, north and east in metres. */
struct GroundPoint final
{
    float north = 0.F;
    float east = 0.F;
};

/** @brief The last positions of a mobile, oldest first once full. */
class Trail final
{
public:
    void Clear() noexcept { count_ = 0; }

    /** @brief Adds a position, unless it is where the trail already ends. */
    void Add(GroundPoint point) noexcept
    {
        if (count_ > 0)
        {
            GroundPoint const& last = At(count_ - 1);
            if (last.north == point.north && last.east == point.east)
            {
                return;
            }
        }
        points_[(start_ + count_) % TrailLength] = point;
        if (count_ < TrailLength)
        {
            ++count_;
        }
        else
        {
            start_ = (start_ + 1) % TrailLength;
        }
    }

    std::size_t Count() const noexcept { return count_; }
    GroundPoint const& At(std::size_t index) const noexcept { return points_[(start_ + index) % TrailLength]; }

private:
    std::array<GroundPoint, TrailLength> points_{};
    std::size_t start_ = 0;
    std::size_t count_ = 0;
};

/** @brief From the plane to the screen: north up, a scale in pixels per metre, a centre. */
struct ScreenFrame final
{
    float centreNorth;
    float centreEast;
    float scale;
    float width;
    float height;

    float X(GroundPoint point) const noexcept { return width * 0.5F + (point.east - centreEast) * scale; }
    float Y(GroundPoint point) const noexcept { return height * 0.5F - (point.north - centreNorth) * scale; }
};

class BlueView final
{
public:
    static char const* Name() { return "blue"; }

    explicit BlueView(std::string const& /*configJson*/) {}

    void Receive(ppokit::view::Frame const& frame)
    {
        // A sequence that goes back is a new run: its trails start over.
        if (frame.sequence < frame_.sequence)
        {
            ClearTrails();
        }
        bool const newRun = frame.sequence < frame_.sequence;
        if (newRun)
        {
            fitted_ = false;
            selected_ = NoSelection;
        }
        frame_ = frame;
        RecordTrails();
        RecordExplosions(newRun);
    }

    void Input(PpokitViewInput const& input)
    {
        if (input.dragX != 0.F || input.dragY != 0.F || input.wheel != 0.F)
        {
            moved_ = true;
        }
        panEast_ -= input.dragX / scale_;
        panNorth_ += input.dragY / scale_;
        scale_ = std::fmax(0.001F, std::fmin(20.F, scale_ * (1.F + 0.1F * input.wheel)));
        bool const pressed = (input.buttons & 1U) != 0U;
        if (pressed && !pressed_ && input.dragX == 0.F && input.dragY == 0.F)
        {
            Select(input.mouseX, input.mouseY);
        }
        pressed_ = pressed;
    }

    void Render(ppokit::view::Canvas const& canvas)
    {
        canvas.Clear(0x0E151EFFU);
        if (frame_.Find("blue") == nullptr || frame_.agents == 0)
        {
            canvas.Text("waiting for the BlueDefense Visualiseur", 20.F, 20.F, 22.F, 0xC8D2DCFFU);
            return;
        }
        Fit(canvas);
        ScreenFrame const screen = Screen(canvas);
        last_ = screen;
        DrawDefendedArea(canvas, screen);
        DrawZones(canvas, screen);
        DrawTrack(canvas, screen);
        DrawWaypoints(canvas, screen);
        DrawReds(canvas, screen);
        DrawBlues(canvas, screen);
        DrawExplosions(canvas, screen);
        DrawCompass(canvas);
        DrawData(canvas);
    }

private:
    float Blue(std::uint32_t slot, std::uint32_t index) const noexcept { return frame_.Value("blue", slot, index); }
    /** @brief The scene repeated in every row: read from row 0. */
    float Red(std::uint32_t red, std::uint32_t index) const noexcept
    {
        return frame_.Value("reds", 0, red * RedValues + index);
    }
    float Objective(std::uint32_t index) const noexcept { return frame_.Value("objective", 0, index); }
    float Waypoint(std::uint32_t index) const noexcept { return frame_.Value("waypoints", 0, index); }
    float ViewOf(std::uint32_t slot, std::uint32_t index) const noexcept
    {
        return frame_.Value("blue_view", slot, index);
    }
    float Lidar(std::uint32_t slot, std::uint32_t index) const noexcept { return frame_.Value("lidar", slot, index); }
    float RedCone(std::uint32_t red, std::uint32_t index) const noexcept
    {
        return frame_.Value("red_cones", 0, red * RedConeValues + index);
    }
    float Zone(std::uint32_t index) const noexcept { return frame_.Value("zones", 0, index); }
    float Bound(std::uint32_t index) const noexcept { return frame_.Value("bounds", 0, index); }
    float Patrol(std::uint32_t index) const noexcept { return frame_.Value("patrol", 0, index); }

    std::uint32_t Slots() const noexcept { return std::min(frame_.agents, BlueSlots); }

    /** @brief No blue clicked: the leader's lidar is shown. */
    static constexpr std::uint32_t NoSelection = BlueSlots;

    /** @brief Centred on the defended area (the world's centre), panned by the user. */
    ScreenFrame Screen(ppokit::view::Canvas const& canvas) const noexcept
    {
        return {Objective(3) + panNorth_, Objective(4) + panEast_, scale_, canvas.Width(), canvas.Height()};
    }

    /** @brief Until the user moves the view, it fits the outer rectangle (else the defended area). */
    void Fit(ppokit::view::Canvas const& canvas) noexcept
    {
        if (moved_ || fitted_)
        {
            return;
        }
        float const halfNorth = Bound(0) > 0.F ? Bound(0) : Objective(5);
        float const halfEast = Bound(1) > 0.F ? Bound(1) : Objective(6);
        if (halfNorth <= 0.F || halfEast <= 0.F)
        {
            return;
        }
        scale_ = 0.95F * std::fmin(canvas.Width() / (2.F * halfEast), canvas.Height() / (2.F * halfNorth));
        panNorth_ = 0.F;
        panEast_ = 0.F;
        fitted_ = true;
    }

    /** @brief The living blue under the cursor (within 14 pixels) becomes the selected one. */
    void Select(float x, float y) noexcept
    {
        float best = 14.F * 14.F;
        for (std::uint32_t slot = 0; slot < Slots(); ++slot)
        {
            if (Blue(slot, 0) < 0.5F || Blue(slot, 1) < 0.5F)
            {
                continue;
            }
            GroundPoint const position{Blue(slot, 3), Blue(slot, 4)};
            float const dx = last_.X(position) - x;
            float const dy = last_.Y(position) - y;
            if (dx * dx + dy * dy < best)
            {
                best = dx * dx + dy * dy;
                selected_ = slot;
            }
        }
    }

    /** @brief The blue whose lidar is drawn: the selected one while alive, else the leader. */
    std::uint32_t LidarSlot() const noexcept
    {
        if (selected_ < Slots() && Blue(selected_, 0) > 0.5F && Blue(selected_, 1) > 0.5F)
        {
            return selected_;
        }
        for (std::uint32_t slot = 0; slot < Slots(); ++slot)
        {
            if (Blue(slot, 0) > 0.5F && Blue(slot, 2) > 0.5F)
            {
                return slot;
            }
        }
        return NoSelection;
    }

    void ClearTrails() noexcept
    {
        for (Trail& trail : blueTrails_)
        {
            trail.Clear();
        }
        for (Trail& trail : redTrails_)
        {
            trail.Clear();
        }
    }

    /** @brief Adds the living mobiles' positions to their trails; a destroyed or absent one's trail stops. */
    void RecordTrails() noexcept
    {
        for (std::uint32_t slot = 0; slot < Slots(); ++slot)
        {
            if (Blue(slot, 0) > 0.5F && Blue(slot, 1) > 0.5F)
            {
                blueTrails_[slot].Add({Blue(slot, 3), Blue(slot, 4)});
            }
        }
        for (std::uint32_t red = 0; red < Reds; ++red)
        {
            if (Red(red, 4) > 0.5F && Red(red, 3) > 0.5F)
            {
                redTrails_[red].Add({Red(red, 0), Red(red, 1)});
            }
        }
    }

    void DrawDefendedArea(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        GroundPoint const areaCentre{Objective(3), Objective(4)};
        float const halfNorth = Objective(5);
        float const halfEast = Objective(6);
        if (halfNorth > 0.F && halfEast > 0.F)
        {
            GroundPoint const corner{areaCentre.north + halfNorth, areaCentre.east - halfEast};
            canvas.Rectangle(screen.X(corner), screen.Y(corner), 2.F * halfEast * screen.scale,
                2.F * halfNorth * screen.scale, true, 0x3264B418U);
            canvas.Rectangle(screen.X(corner), screen.Y(corner), 2.F * halfEast * screen.scale,
                2.F * halfNorth * screen.scale, false, 0x4BA0FFEBU);
            canvas.Text("DEFENDED AREA", screen.X(corner) + 6.F, screen.Y(corner) + 6.F, 14.F, 0x91C3FFFFU);
        }
        float const hardNorth = Bound(0);
        float const hardEast = Bound(1);
        if (hardNorth > 0.F && hardEast > 0.F)
        {
            GroundPoint const corner{hardNorth, -hardEast};
            canvas.Rectangle(screen.X(corner), screen.Y(corner), 2.F * hardEast * screen.scale,
                2.F * hardNorth * screen.scale, false, 0xFFA64DEBU);
        }
        GroundPoint const objective{Objective(0), Objective(1)};
        bool const alive = Objective(2) > 0.5F;
        canvas.Circle(screen.X(objective), screen.Y(objective), 7.F, true, alive ? 0x50C878FFU : 0x5A5A5AFFU);
    }

    /** @brief The racetrack's waypoints while followed: active, validated or ahead, heading, label. */
    void DrawWaypoints(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        float const active = Waypoint(2 * Waypoints);
        if (active < 0.F)
        {
            return;
        }
        std::uint32_t const activeIndex = static_cast<std::uint32_t>(active) % Waypoints;
        float const validated = Patrol(2 * TrackSamples + 1 + Waypoints);
        for (std::uint32_t index = 0; index < Waypoints; ++index)
        {
            GroundPoint const point{Waypoint(2 * index), Waypoint(2 * index + 1)};
            bool const isActive = index == activeIndex;
            PpokitColor const color = isActive ? 0xFFCD48FFU
                : static_cast<float>(index) < validated ? 0x697D8CCDU : 0x5AE1AAEBU;
            float const x = screen.X(point);
            float const y = screen.Y(point);
            canvas.Circle(x, y, isActive ? 6.F : 4.F, false, color);
            float const heading = Patrol(2 * TrackSamples + 1 + index);
            canvas.Line(x, y, x + 8.F * std::sin(heading), y - 8.F * std::cos(heading), isActive ? 2.F : 1.2F,
                color);
            char label[8];
            std::snprintf(label, sizeof(label), "WP%u", static_cast<unsigned>(index + 1));
            canvas.Text(label, x + 6.F, y - 7.F, 12.F, color);
        }
    }

    /** @brief The leader's analytical patrol, with its direction every 16 samples. */
    void DrawTrack(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        std::uint32_t const count = std::min(static_cast<std::uint32_t>(Patrol(2 * TrackSamples)), TrackSamples);
        constexpr PpokitColor Color = 0x46BEFFEBU;
        for (std::uint32_t sample = 1; sample < count; ++sample)
        {
            GroundPoint const from{Patrol(2 * (sample - 1)), Patrol(2 * (sample - 1) + 1)};
            GroundPoint const to{Patrol(2 * sample), Patrol(2 * sample + 1)};
            canvas.Line(screen.X(from), screen.Y(from), screen.X(to), screen.Y(to), 2.5F, Color);
            if (sample % 16 == 0 && sample + 1 < count)
            {
                DrawArrowHead(canvas, screen.X(from), screen.Y(from), screen.X(to), screen.Y(to), 8.F, Color);
            }
        }
    }

    /** @brief The restricted zones: their no-escape circle and centre, by side. */
    void DrawZones(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        std::uint32_t const count = std::min(static_cast<std::uint32_t>(Zone(Zones * ZoneValues)), Zones);
        for (std::uint32_t zone = 0; zone < count; ++zone)
        {
            GroundPoint const centre{Zone(zone * ZoneValues), Zone(zone * ZoneValues + 1)};
            bool const hostile = Zone(zone * ZoneValues + 3) > 0.5F;
            PpokitColor const side = hostile ? 0xE6504600U : 0x4682F000U;
            canvas.Circle(screen.X(centre), screen.Y(centre), Zone(zone * ZoneValues + 2) * screen.scale, true,
                side | 0x1CU);
            canvas.Circle(screen.X(centre), screen.Y(centre), Zone(zone * ZoneValues + 2) * screen.scale, false,
                side | 0x6EU);
            canvas.Circle(screen.X(centre), screen.Y(centre), 5.F, true, side | 0xFFU);
        }
    }

    void DrawReds(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        for (std::uint32_t red = 0; red < Reds; ++red)
        {
            if (Red(red, 4) < 0.5F)
            {
                continue;
            }
            GroundPoint const position{Red(red, 0), Red(red, 1)};
            DrawTrail(canvas, screen, redTrails_[red], 0x783232FFU);
            if (Red(red, 3) > 0.5F)
            {
                DrawCone(canvas, screen, position, Red(red, 2), RedCone(red, 0), RedCone(red, 1));
                DrawMobile(canvas, screen, position, Red(red, 2), 0xE65046FFU);
            }
            else
            {
                DrawCross(canvas, screen, position, 0x8C8C8CFFU);
            }
        }
    }

    void DrawBlues(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        for (std::uint32_t slot = 0; slot < Slots(); ++slot)
        {
            if (Blue(slot, 0) < 0.5F)
            {
                continue;
            }
            GroundPoint const position{Blue(slot, 3), Blue(slot, 4)};
            DrawTrail(canvas, screen, blueTrails_[slot], 0x284B7DFFU);
            if (Blue(slot, 1) < 0.5F)
            {
                DrawCross(canvas, screen, position, 0x8C8C8CFFU);
                continue;
            }
            if (slot == LidarSlot())
            {
                DrawLidar(canvas, screen, slot, position, Blue(slot, 5));
                canvas.Circle(screen.X(position), screen.Y(position), 12.F, false, 0xFFFFFFDCU);
            }
            DrawCone(canvas, screen, position, Blue(slot, 5), ViewOf(slot, 1), ViewOf(slot, 2));
            GroundPoint const reference{Blue(slot, 7), Blue(slot, 8)};
            canvas.Line(screen.X(position), screen.Y(position), screen.X(reference), screen.Y(reference), 1.F,
                0x6496C8FFU);
            canvas.Circle(screen.X(reference), screen.Y(reference), 2.5F, true, 0x6496C8FFU);
            DrawReference(canvas, screen, slot);
            DrawMobile(canvas, screen, position, Blue(slot, 5), 0x4682F0FFU);
            if (Blue(slot, 2) > 0.5F)
            {
                canvas.Circle(screen.X(position), screen.Y(position), 10.F, false, 0xF0C850FFU);
            }
            DrawRoleBadge(canvas, screen, slot, position);
            char label[16];
            std::snprintf(label, sizeof(label), "%u", static_cast<unsigned>(slot + 1));
            canvas.Text(label, screen.X(position) + 9.F, screen.Y(position) + 2.F, 12.F, 0xC8D2DCFFU);
        }
    }

    /** @brief The role marker: L for the leader, W k for teammate slot k, none otherwise. */
    static std::string RoleMarker(float role)
    {
        if (role < -0.5F)
        {
            return {};
        }
        int const value = static_cast<int>(std::lround(role));
        return value == 0 ? std::string("L") : "W" + std::to_string(value);
    }

    /** @brief The role badge above a blue, in the role's colour (leader gold, teammates cyan). */
    void DrawRoleBadge(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, std::uint32_t slot,
        GroundPoint position) const
    {
        std::string const marker = RoleMarker(ViewOf(slot, 0));
        if (marker.empty())
        {
            return;
        }
        PpokitColor const color = marker == "L" ? 0xF0C850F5U : 0x5AD2E6F5U;
        float const x = screen.X(position);
        float const y = screen.Y(position) - 17.F;
        float const radius = marker.size() > 1 ? 10.F : 8.F;  // room for "W1"
        canvas.Circle(x + 1.5F, y + 1.5F, radius, true, 0x00000096U);
        canvas.Circle(x, y, radius, true, color);
        canvas.Text(marker, x - 3.5F * static_cast<float>(marker.size()), y - 6.F, 12.F, 0x0C141CFFU);
    }

    /** @brief The patrol reference of a blue's role: projected point, guidance point and heading. */
    void DrawReference(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, std::uint32_t slot) const
    {
        if (ViewOf(slot, 6) < 0.5F)
        {
            return;
        }
        std::string const marker = RoleMarker(ViewOf(slot, 0));
        PpokitColor const color = marker == "L" ? 0xF0C850F5U : 0x5AD2E6F5U;
        GroundPoint const projected{ViewOf(slot, 3), ViewOf(slot, 4)};
        GroundPoint const guidance{Blue(slot, 7), Blue(slot, 8)};
        float const px = screen.X(projected);
        float const py = screen.Y(projected);
        float const gx = screen.X(guidance);
        float const gy = screen.Y(guidance);
        canvas.Line(px, py, gx, gy, 1.4F, (color & 0xFFFFFF00U) | 0x73U);
        canvas.Circle(px, py, 5.2F, false, color);
        canvas.Circle(gx, gy, 2.6F, true, color);
        float const heading = ViewOf(slot, 5);
        float const tipX = gx + 11.F * std::sin(heading);
        float const tipY = gy - 11.F * std::cos(heading);
        canvas.Line(gx, gy, tipX, tipY, 2.F, color);
        DrawArrowHead(canvas, gx, gy, tipX, tipY, 5.F, color);
        canvas.Text(marker, px + 7.F, py - 8.F, 12.F, color);
    }

    /** @brief An engagement cone: the no-escape arc over the firing angle, both edges, a faint fill. */
    static void DrawCone(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, GroundPoint position,
        float heading, float radius, float halfAngle)
    {
        if (radius <= 0.F || halfAngle <= 0.F)
        {
            return;
        }
        constexpr int Segments = 28;
        float const x = screen.X(position);
        float const y = screen.Y(position);
        float const pixels = radius * screen.scale;
        float previousX = 0.F;
        float previousY = 0.F;
        for (int index = 0; index <= Segments; ++index)
        {
            float const t = static_cast<float>(index) / static_cast<float>(Segments);
            float const angle = heading - halfAngle + t * 2.F * halfAngle;
            float const ax = x + pixels * std::sin(angle);
            float const ay = y - pixels * std::cos(angle);
            canvas.Line(x, y, ax, ay, 1.F, 0xFF484814U);  // the fill: radial strokes
            if (index > 0)
            {
                canvas.Line(previousX, previousY, ax, ay, 1.5F, 0xFFC4AAAAU);
            }
            if (index == 0 || index == Segments)
            {
                canvas.Line(x, y, ax, ay, 1.1F, 0xFFC4AA87U);
            }
            previousX = ax;
            previousY = ay;
        }
    }

    /**
     * @brief A blue's pseudo-lidar: a beam per pair of bins that sees something, to the decoded
     *        distance, brighter for a nearer threat (the former PPO.exe's DrawPseudoLidar3D seen from above).
     */
    void DrawLidar(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, std::uint32_t slot,
        GroundPoint position, float heading) const
    {
        float const span = Lidar(slot, 2 * LidarBins);
        if (span <= 0.F)
        {
            return;
        }
        constexpr float NoIntercept = 0.999F;
        float const step = span / static_cast<float>(LidarBins - 1);
        float const noEscape = std::fmax(ViewOf(slot, 1), 1.F);
        float const x = screen.X(position);
        float const y = screen.Y(position);
        for (std::uint32_t bin = 0; bin < LidarBins; ++bin)
        {
            float const ratio = Lidar(slot, bin);
            if (ratio >= NoIntercept)
            {
                continue;
            }
            float distance = Lidar(slot, LidarBins + bin);
            if (distance <= 0.F)
            {
                distance = ratio * noEscape / std::fmax(1.0e-4F, 1.F - ratio);
            }
            float const angle = heading - 0.5F * span + static_cast<float>(bin) * step;
            float const threat = 1.F - std::fmin(std::fmax(ratio, 0.F), 1.F);
            float const intensity = 0.35F + 0.65F * threat;
            auto const channel = [](float value) { return static_cast<PpokitColor>(std::lround(value)); };
            PpokitColor const color = channel(50.F + 35.F * intensity) << 24U | channel(180.F + 50.F * intensity) << 16U
                | channel(90.F + 40.F * intensity) << 8U | channel(50.F + 70.F * intensity);
            float const pixels = distance * screen.scale;
            canvas.Line(x, y, x + pixels * std::sin(angle), y - pixels * std::cos(angle), 1.2F, color);
        }
    }

    /** @brief An arrow head at (x1, y1) pointing from (x0, y0). */
    static void DrawArrowHead(ppokit::view::Canvas const& canvas, float x0, float y0, float x1, float y1,
        float size, PpokitColor color)
    {
        float const dx = x1 - x0;
        float const dy = y1 - y0;
        float const length = std::fmax(std::hypot(dx, dy), 0.001F);
        float const fx = dx / length;
        float const fy = dy / length;
        float const bx = x1 - size * fx;
        float const by = y1 - size * fy;
        canvas.Line(x1, y1, bx - 0.45F * size * fy, by + 0.45F * size * fx, 2.F, color);
        canvas.Line(x1, y1, bx + 0.45F * size * fy, by - 0.45F * size * fx, 2.F, color);
    }

    /** @brief North and east arrows, bottom left. */
    static void DrawCompass(ppokit::view::Canvas const& canvas)
    {
        float const x = 18.F;
        float const y = canvas.Height() - 28.F;
        canvas.Line(x, y, x, y - 20.F, 2.F, 0xFFFFFFFFU);
        canvas.Text("N", x + 6.F, y - 26.F, 14.F, 0xFFFFFFFFU);
        canvas.Line(x, y, x + 20.F, y, 2.F, 0xFFFFFFFFU);
        canvas.Text("E", x + 24.F, y - 8.F, 14.F, 0xFFFFFFFFU);
    }

    /** @brief Index of a destructible: blue slots, then reds, then the objective. */
    static constexpr std::size_t ObjectiveDestructible = Destructibles - 1;

    /** @brief Position and alive of a destructible in the current frame (present ones only). */
    bool Destructible(std::size_t index, GroundPoint& position, bool& alive) const noexcept
    {
        if (index < BlueSlots)
        {
            auto const slot = static_cast<std::uint32_t>(index);
            if (slot >= Slots() || Blue(slot, 0) < 0.5F)
            {
                return false;
            }
            position = {Blue(slot, 3), Blue(slot, 4)};
            alive = Blue(slot, 1) > 0.5F;
            return true;
        }
        if (index < ObjectiveDestructible)
        {
            auto const red = static_cast<std::uint32_t>(index - BlueSlots);
            if (Red(red, 4) < 0.5F)
            {
                return false;
            }
            position = {Red(red, 0), Red(red, 1)};
            alive = Red(red, 3) > 0.5F;
            return true;
        }
        position = {Objective(0), Objective(1)};
        alive = Objective(2) > 0.5F;
        return true;
    }

    /** @brief Starts an explosion where something alive in the previous frame is now destroyed. */
    void RecordExplosions(bool newRun) noexcept
    {
        double const now = Seconds();
        for (std::size_t index = 0; index < Destructibles; ++index)
        {
            GroundPoint position{};
            bool alive = false;
            bool const present = Destructible(index, position, alive);
            if (newRun || !present)
            {
                wasAlive_[index] = present && alive;
                continue;
            }
            if (wasAlive_[index] && !alive)
            {
                explosions_[index] = {position, now, true};
            }
            wasAlive_[index] = alive;
        }
    }

    /** @brief The explosions of the last two seconds: a growing, fading fireball and shock ring. */
    void DrawExplosions(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        double const now = Seconds();
        for (Explosion const& explosion : explosions_)
        {
            double const age = now - explosion.start;
            if (!explosion.active || age < 0.0 || age > ExplosionSeconds)
            {
                continue;
            }
            float const t = static_cast<float>(age / ExplosionSeconds);
            float const fade = 1.F - t;
            auto const alpha = static_cast<PpokitColor>(std::lround(230.F * fade));
            float const x = screen.X(explosion.position);
            float const y = screen.Y(explosion.position);
            canvas.Circle(x, y, 6.F + 14.F * t, true, 0xFF8C2800U | alpha);
            canvas.Circle(x, y, 3.F + 8.F * t, true, 0xFFE6A000U | alpha);
            canvas.Circle(x, y, 10.F + 30.F * t, false, 0xFFDCB400U | (alpha / 2U));
        }
    }

    static double Seconds() noexcept
    {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    /** @brief A mobile: a disc and its heading (clockwise from north). */
    static void DrawMobile(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, GroundPoint position,
        float heading, PpokitColor color)
    {
        float const x = screen.X(position);
        float const y = screen.Y(position);
        constexpr float Nose = 14.F;
        canvas.Circle(x, y, 5.F, true, color);
        canvas.Line(x, y, x + Nose * std::sin(heading), y - Nose * std::cos(heading), 2.F, color);
    }

    static void DrawCross(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, GroundPoint position,
        PpokitColor color)
    {
        float const x = screen.X(position);
        float const y = screen.Y(position);
        constexpr float Half = 5.F;
        canvas.Line(x - Half, y - Half, x + Half, y + Half, 2.F, color);
        canvas.Line(x - Half, y + Half, x + Half, y - Half, 2.F, color);
    }

    static void DrawTrail(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, Trail const& trail,
        PpokitColor color)
    {
        for (std::size_t index = 1; index < trail.Count(); ++index)
        {
            GroundPoint const& from = trail.At(index - 1);
            GroundPoint const& to = trail.At(index);
            canvas.Line(screen.X(from), screen.Y(from), screen.X(to), screen.Y(to), 1.F, color);
        }
    }

    /** @brief Blues and reds alive, and the step's weighted channels summed, per living blue. */
    void DrawData(ppokit::view::Canvas const& canvas) const
    {
        std::uint32_t bluesPresent = 0;
        std::uint32_t bluesAlive = 0;
        double rewardSum = 0.0;
        for (std::uint32_t slot = 0; slot < Slots(); ++slot)
        {
            if (Blue(slot, 0) < 0.5F)
            {
                continue;
            }
            ++bluesPresent;
            if (Blue(slot, 1) > 0.5F)
            {
                ++bluesAlive;
                for (std::uint32_t channel = 0; channel < RewardChannels; ++channel)
                {
                    rewardSum += frame_.Value("rewards", slot, channel);
                }
            }
        }
        std::uint32_t redsPresent = 0;
        std::uint32_t redsAlive = 0;
        for (std::uint32_t red = 0; red < Reds; ++red)
        {
            redsPresent += Red(red, 4) > 0.5F ? 1U : 0U;
            redsAlive += Red(red, 4) > 0.5F && Red(red, 3) > 0.5F ? 1U : 0U;
        }
        double const channelsPerBlue = bluesAlive == 0 ? 0.0 : rewardSum / bluesAlive;
        char line[200];
        auto const zones = static_cast<unsigned>(Zone(Zones * ZoneValues));
        std::snprintf(line, sizeof(line), "blues %u/%u   reds %u/%u   objective %s   zones %u%s", bluesAlive,
            bluesPresent, redsAlive, redsPresent, Objective(2) > 0.5F ? "safe" : "lost", zones,
            zones > Zones ? " (16 drawn)" : "");
        canvas.Text(line, 16.F, 16.F, 20.F, 0xC8D2DCFFU);
        std::snprintf(line, sizeof(line), "reward channels of the step, per living blue %+.4f   frame %llu",
            channelsPerBlue,
            static_cast<unsigned long long>(frame_.sequence));
        canvas.Text(line, 16.F, 42.F, 18.F, 0xC8D2DCFFU);
    }

    ppokit::view::Frame frame_;
    std::array<Trail, BlueSlots> blueTrails_{};
    std::array<Trail, Reds> redTrails_{};
    /** @brief An explosion: where and when (steady clock, seconds). */
    struct Explosion final
    {
        GroundPoint position{};
        double start = 0.0;
        bool active = false;
    };
    std::array<Explosion, Destructibles> explosions_{};
    std::array<bool, Destructibles> wasAlive_{};
    float panNorth_ = 0.F;
    float panEast_ = 0.F;
    /** @brief The user moved or zoomed the view: it no longer fits itself. */
    bool moved_ = false;
    bool fitted_ = false;
    bool pressed_ = false;
    std::uint32_t selected_ = NoSelection;
    /** @brief The frame of the last drawing, to find what a click is on. */
    ScreenFrame last_{0.F, 0.F, 1.F, 1.F, 1.F};
    /** @brief Pixels per metre. */
    float scale_ = 0.08F;
};

} // namespace

PPOKIT_COMPONENTS(BlueWatch)
PPOKIT_VIEWS(BlueView)
