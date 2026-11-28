/**
 * @file Examples/Airspace/Visualiser/airspace_visualiser.cpp
 * @brief The Airspace Visualiseur: an observer that sends the world row of rl_mission or
 *        rl_sna after each step, and a top-down 2D tactical view of it for ppokit-viewer.
 * @details The view is the former PPO.exe's tactical minimap (Airspace renderer): the work area and its
 * range rings, each entity by type and side (aircraft triangles, objectives, restricted zones
 * with their no-escape circle), the engagement cones of the aircraft, the trails, a cross on
 * what is destroyed and an explosion when it happens, the controlled aircraft ringed and its
 * primary target's pulsing halo; a line of data (the controlled aircraft's Pk and missiles).
 * North is up; the view first fits the work area; dragging moves it, the wheel zooms.
 *
 * The observer's ports are the world row (Environment/tactical_world.hpp): "entities"
 * (16 x present, side, type, north, east, heading, destroyed, no-escape radius, half
 * shooting angle, role) and "scene" (work area half extents, entity count, controlled Pk,
 * missiles left).
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

constexpr std::uint32_t Entities = 16;
constexpr std::uint32_t EntityValues = 10;
constexpr std::uint32_t EntitiesSize = Entities * EntityValues;
constexpr std::uint32_t SceneValues = 5;

/** @brief Values of an entity in "entities". */
enum EntityValue : std::uint32_t
{
    Present = 0,
    Side = 1,
    Type = 2,
    North = 3,
    East = 4,
    Heading = 5,
    Destroyed = 6,
    NoEscape = 7,
    HalfAngle = 8,
    RoleValue = 9
};

/** @brief Sides of eEntitySide. */
constexpr int HostileSide = 1;
constexpr int FriendSide = 2;
/** @brief Types of eEntityType. */
constexpr int ObjectiveType = 1;
constexpr int MissileType = 2;
constexpr int MobileType = 3;
constexpr int ZoneType = 4;
/** @brief Roles of the world row. */
constexpr int ControlledRole = 1;
constexpr int TargetRole = 2;

/** @brief Wall-clock life of an explosion, as in the former PPO.exe's renderer. */
constexpr double ExplosionSeconds = 2.0;
/** @brief Positions kept per entity for its trail. */
constexpr std::size_t TrailLength = 240;

/** @brief Sends what it is wired to: the entities and scene rows. */
class AirspaceWatch final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("airspace_watch", 1);
        manifest.Input("entities", {EntitiesSize}).Input("scene", {SceneValues});
        manifest.Observer();
        manifest.Integer("port", 47002, 1024, 65535).Number("rate", 30.0, 1.0, 240.0);
        return manifest;
    }
    AirspaceWatch(Config const& config, Shapes const&, std::uint64_t)
        : sender_(config, {{"entities", EntitiesSize}, {"scene", SceneValues}})
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

/** @brief The last positions of an entity, oldest first once full. */
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

/** @brief The colour of a side, as the former PPO.exe's ColorForSide (alpha in the low byte). */
PpokitColor SideColor(int side, PpokitColor alpha) noexcept
{
    if (side == HostileSide)
    {
        return 0xEB403400U | alpha;
    }
    if (side == FriendSide)
    {
        return 0x4074FF00U | alpha;
    }
    return 0xFFFFFF00U | alpha;
}

class AirspaceView final
{
public:
    static char const* Name() { return "airspace"; }

    explicit AirspaceView(std::string const& /*configJson*/) {}

    void Receive(ppokit::view::Frame const& frame)
    {
        // A sequence that goes back is a new run: its trails start over, the view fits again.
        bool const newRun = frame.sequence < frame_.sequence;
        if (newRun)
        {
            for (Trail& trail : trails_)
            {
                trail.Clear();
            }
            fitted_ = false;
        }
        bool const newEpisode = !newRun && Restarted(frame);
        if (newEpisode)
        {
            for (Trail& trail : trails_)
            {
                trail.Clear();
            }
        }
        frame_ = frame;
        RecordTrails();
        RecordExplosions(newRun || newEpisode);
    }

    void Input(PpokitViewInput const& input)
    {
        if (input.dragX != 0.F || input.dragY != 0.F || input.wheel != 0.F)
        {
            moved_ = true;
        }
        panEast_ -= input.dragX / scale_;
        panNorth_ += input.dragY / scale_;
        scale_ = std::fmax(1.0e-5F, std::fmin(20.F, scale_ * (1.F + 0.1F * input.wheel)));
    }

    void Render(ppokit::view::Canvas const& canvas)
    {
        canvas.Clear(0x0C1218FFU);
        if (frame_.Find("entities") == nullptr || frame_.agents == 0)
        {
            canvas.Text("waiting for the Airspace Visualiseur", 20.F, 20.F, 22.F, 0xC8D2DCFFU);
            return;
        }
        Fit(canvas);
        ScreenFrame const screen{panNorth_, panEast_, scale_, canvas.Width(), canvas.Height()};
        DrawWorkArea(canvas, screen);
        DrawZones(canvas, screen);
        for (std::uint32_t entity = 0; entity < Entities; ++entity)
        {
            if (Value(entity, Present) > 0.5F)
            {
                DrawEntity(canvas, screen, entity);
            }
        }
        DrawExplosions(canvas, screen);
        DrawCompass(canvas);
        DrawData(canvas);
    }

private:
    float Value(std::uint32_t entity, EntityValue value) const noexcept
    {
        return frame_.Value("entities", 0, entity * EntityValues + value);
    }
    float Scene(std::uint32_t index) const noexcept { return frame_.Value("scene", 0, index); }
    int IntegerValue(std::uint32_t entity, EntityValue value) const noexcept
    {
        return static_cast<int>(std::lround(Value(entity, value)));
    }
    GroundPoint Position(std::uint32_t entity) const noexcept { return {Value(entity, North), Value(entity, East)}; }
    bool Alive(std::uint32_t entity) const noexcept { return Value(entity, Destroyed) < 0.5F; }

    /**
     * @brief Whether `frame` starts a new episode: an entity destroyed in the current frame is
     *        alive again (the world re-initialized its entities).
     */
    bool Restarted(ppokit::view::Frame const& frame) const noexcept
    {
        if (frame_.Find("entities") == nullptr || frame.Find("entities") == nullptr)
        {
            return false;
        }
        for (std::uint32_t entity = 0; entity < Entities; ++entity)
        {
            std::uint32_t const destroyed = entity * EntityValues + Destroyed;
            if (frame_.Value("entities", 0, destroyed) > 0.5F && frame.Value("entities", 0, destroyed) < 0.5F)
            {
                return true;
            }
        }
        return false;
    }

    /** @brief Until the user moves the view, it fits the work area. */
    void Fit(ppokit::view::Canvas const& canvas) noexcept
    {
        if (moved_ || fitted_)
        {
            return;
        }
        float const halfNorth = Scene(0);
        float const halfEast = Scene(1);
        if (halfNorth <= 0.F || halfEast <= 0.F)
        {
            return;
        }
        scale_ = 0.95F * std::fmin(canvas.Width() / (2.F * halfEast), canvas.Height() / (2.F * halfNorth));
        panNorth_ = 0.F;
        panEast_ = 0.F;
        fitted_ = true;
    }

    /** @brief Adds the living entities' positions to their trails. */
    void RecordTrails() noexcept
    {
        for (std::uint32_t entity = 0; entity < Entities; ++entity)
        {
            if (Value(entity, Present) > 0.5F && Alive(entity) && IntegerValue(entity, Type) == MobileType)
            {
                trails_[entity].Add(Position(entity));
            }
        }
    }

    /** @brief Starts an explosion where something alive in the previous frame is now destroyed. */
    void RecordExplosions(bool restart) noexcept
    {
        double const now = Seconds();
        for (std::uint32_t entity = 0; entity < Entities; ++entity)
        {
            bool const present = Value(entity, Present) > 0.5F;
            bool const alive = present && Alive(entity);
            if (!restart && present && wasAlive_[entity] && !alive)
            {
                explosions_[entity] = {Position(entity), now, true};
            }
            wasAlive_[entity] = alive;
        }
    }

    /** @brief The work area, its range rings and axes. */
    void DrawWorkArea(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        float const halfNorth = Scene(0);
        float const halfEast = Scene(1);
        if (halfNorth <= 0.F || halfEast <= 0.F)
        {
            return;
        }
        GroundPoint const corner{halfNorth, -halfEast};
        float const x = screen.X(corner);
        float const y = screen.Y(corner);
        float const width = 2.F * halfEast * screen.scale;
        float const height = 2.F * halfNorth * screen.scale;
        canvas.Rectangle(x, y, width, height, true, 0x141E2855U);
        canvas.Rectangle(x, y, width, height, false, 0xEBD278DCU);
        canvas.Text("Work Area", x + 8.F, y + 6.F, 14.F, 0xEBD278DCU);
        GroundPoint const centre{0.F, 0.F};
        float const cx = screen.X(centre);
        float const cy = screen.Y(centre);
        float const ring = 0.2F * std::fmin(halfNorth, halfEast) * screen.scale;
        for (int index = 1; index <= 4; ++index)
        {
            canvas.Circle(cx, cy, ring * static_cast<float>(index), false, 0x50667A5AU);
        }
        canvas.Line(x, cy, x + width, cy, 1.F, 0x465E6E6EU);
        canvas.Line(cx, y, cx, y + height, 1.F, 0x465E6E6EU);
    }

    /** @brief The restricted zones' no-escape circles, under everything else. */
    void DrawZones(ppokit::view::Canvas const& canvas, ScreenFrame const& screen) const
    {
        for (std::uint32_t entity = 0; entity < Entities; ++entity)
        {
            if (Value(entity, Present) < 0.5F || IntegerValue(entity, Type) != ZoneType)
            {
                continue;
            }
            float const radius = Value(entity, NoEscape) * screen.scale;
            if (radius <= 0.F)
            {
                continue;
            }
            GroundPoint const position = Position(entity);
            int const side = IntegerValue(entity, Side);
            canvas.Circle(screen.X(position), screen.Y(position), radius, true, SideColor(side, 0x16U));
            canvas.Circle(screen.X(position), screen.Y(position), radius, false, SideColor(side, 0x6EU));
        }
    }

    /** @brief One entity: its trail, cone, glyph, rings and destruction cross. */
    void DrawEntity(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, std::uint32_t entity) const
    {
        GroundPoint const position = Position(entity);
        int const side = IntegerValue(entity, Side);
        int const type = IntegerValue(entity, Type);
        float const x = screen.X(position);
        float const y = screen.Y(position);
        DrawTrail(canvas, screen, trails_[entity], SideColor(side, 0x8CU));
        if (type == MobileType && Alive(entity))
        {
            DrawCone(canvas, screen, position, Value(entity, Heading), Value(entity, NoEscape),
                Value(entity, HalfAngle));
        }
        DrawGlyph(canvas, x, y, type, Value(entity, Heading), SideColor(side, 0xFFU));
        int const role = IntegerValue(entity, RoleValue);
        if (role == ControlledRole)
        {
            canvas.Circle(x, y, 10.F, false, 0x60B4FFB4U);
            canvas.Circle(x, y, 12.F, false, 0xFFFFFFDCU);
        }
        if (role == TargetRole && Alive(entity))
        {
            DrawTargetHalo(canvas, x, y);
        }
        if (!Alive(entity))
        {
            DrawCross(canvas, x, y);
        }
    }

    /** @brief The glyph of a type: aircraft triangle, objective disc and ring, zone or missile disc. */
    static void DrawGlyph(ppokit::view::Canvas const& canvas, float x, float y, int type, float heading,
        PpokitColor color)
    {
        if (type == MobileType)
        {
            constexpr float Length = 12.F;
            constexpr float Wing = 2.55F;
            float const noseX = x + Length * std::sin(heading);
            float const noseY = y - Length * std::cos(heading);
            float const leftX = x + 0.75F * Length * std::sin(heading + Wing);
            float const leftY = y - 0.75F * Length * std::cos(heading + Wing);
            float const rightX = x + 0.75F * Length * std::sin(heading - Wing);
            float const rightY = y - 0.75F * Length * std::cos(heading - Wing);
            canvas.Triangle(noseX + 1.F, noseY + 1.F, leftX + 1.F, leftY + 1.F, rightX + 1.F, rightY + 1.F,
                0x00000078U);
            canvas.Triangle(noseX, noseY, leftX, leftY, rightX, rightY, color);
            canvas.Line(noseX, noseY, leftX, leftY, 1.4F, 0xFFFFFFB4U);
            canvas.Line(leftX, leftY, rightX, rightY, 1.4F, 0xFFFFFFB4U);
            canvas.Line(rightX, rightY, noseX, noseY, 1.4F, 0xFFFFFFB4U);
            return;
        }
        if (type == ObjectiveType)
        {
            canvas.Circle(x, y, 6.F, true, color);
            canvas.Circle(x, y, 9.F, false, 0xFFFFFFA0U);
            return;
        }
        if (type == ZoneType)
        {
            canvas.Circle(x, y, 5.F, true, color);
            return;
        }
        if (type == MissileType)
        {
            canvas.Circle(x, y, 2.5F, true, color);
        }
    }

    /** @brief The primary target's pulsing halo and cross-hair. */
    static void DrawTargetHalo(ppokit::view::Canvas const& canvas, float x, float y)
    {
        float const pulse = 0.5F + 0.5F * static_cast<float>(std::sin(Seconds() * 2.8));
        float const radius = 14.F + 3.F * pulse;
        canvas.Circle(x, y, radius, false, 0xFFD65CEBU);
        canvas.Circle(x, y, radius + 5.F, false, 0xFFF4BE78U);
        canvas.Line(x - 8.F, y, x + 8.F, y, 1.4F, 0xFFE68CB4U);
        canvas.Line(x, y - 8.F, x, y + 8.F, 1.4F, 0xFFE68CB4U);
    }

    /** @brief An engagement cone: the no-escape arc over the shooting angle, both edges, a faint fill. */
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

    static void DrawCross(ppokit::view::Canvas const& canvas, float x, float y)
    {
        constexpr float Half = 6.F;
        canvas.Line(x - Half, y - Half, x + Half, y + Half, 1.5F, 0xFFFFFFFFU);
        canvas.Line(x - Half, y + Half, x + Half, y - Half, 1.5F, 0xFFFFFFFFU);
    }

    static void DrawTrail(ppokit::view::Canvas const& canvas, ScreenFrame const& screen, Trail const& trail,
        PpokitColor color)
    {
        for (std::size_t index = 1; index < trail.Count(); ++index)
        {
            GroundPoint const& from = trail.At(index - 1);
            GroundPoint const& to = trail.At(index);
            canvas.Line(screen.X(from), screen.Y(from), screen.X(to), screen.Y(to), 1.6F, color);
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
            auto const alpha = static_cast<PpokitColor>(std::lround(230.F * (1.F - t)));
            float const x = screen.X(explosion.position);
            float const y = screen.Y(explosion.position);
            canvas.Circle(x, y, 6.F + 14.F * t, true, 0xFF8C2800U | alpha);
            canvas.Circle(x, y, 3.F + 8.F * t, true, 0xFFE6A000U | alpha);
            canvas.Circle(x, y, 10.F + 30.F * t, false, 0xFFDCB400U | (alpha / 2U));
        }
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

    /** @brief Aircraft alive by side, the controlled aircraft's Pk and missiles. */
    void DrawData(ppokit::view::Canvas const& canvas) const
    {
        unsigned friends = 0;
        unsigned hostiles = 0;
        for (std::uint32_t entity = 0; entity < Entities; ++entity)
        {
            if (Value(entity, Present) < 0.5F || IntegerValue(entity, Type) != MobileType || !Alive(entity))
            {
                continue;
            }
            friends += IntegerValue(entity, Side) == FriendSide ? 1U : 0U;
            hostiles += IntegerValue(entity, Side) == HostileSide ? 1U : 0U;
        }
        auto const count = static_cast<unsigned>(std::lround(Scene(2)));
        char line[200];
        std::snprintf(line, sizeof(line), "aircraft alive: friend %u, hostile %u   entities %u%s", friends, hostiles,
            count, count > Entities ? " (16 drawn)" : "");
        canvas.Text(line, 16.F, 16.F, 20.F, 0xC8D2DCFFU);
        std::snprintf(line, sizeof(line), "controlled: Pk %.2f   missiles %u   frame %llu", Scene(3),
            static_cast<unsigned>(std::lround(Scene(4))), static_cast<unsigned long long>(frame_.sequence));
        canvas.Text(line, 16.F, 42.F, 18.F, 0xC8D2DCFFU);
    }

    static double Seconds() noexcept
    {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    /** @brief An explosion: where and when (steady clock, seconds). */
    struct Explosion final
    {
        GroundPoint position{};
        double start = 0.0;
        bool active = false;
    };

    ppokit::view::Frame frame_;
    std::array<Trail, Entities> trails_{};
    std::array<Explosion, Entities> explosions_{};
    std::array<bool, Entities> wasAlive_{};
    float panNorth_ = 0.F;
    float panEast_ = 0.F;
    /** @brief The user moved or zoomed the view: it no longer fits itself. */
    bool moved_ = false;
    bool fitted_ = false;
    /** @brief Pixels per metre. */
    float scale_ = 0.004F;
};

} // namespace

PPOKIT_COMPONENTS(AirspaceWatch)
PPOKIT_VIEWS(AirspaceView)
