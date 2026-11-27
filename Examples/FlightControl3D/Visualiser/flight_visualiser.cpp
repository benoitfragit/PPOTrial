/**
 * @file Examples/FlightControl3D/Visualiser/flight_visualiser.cpp
 * @brief The FlightControl3D Visualiseur: an observer that sends the world row of each step
 *        (Core/flight_world.hpp), the action and the rewards, and the view of ppokit-viewer
 *        that draws them as the desktop renderer of FlightControl3D did.
 * @details The view draws, through the viewer's drawing table:
 * - the 3D scene: sea, swell crests and Kelvin wake of a moving deck, the elevated landing
 *   area, the runway and its markings, the guidance target with the commanded direction,
 *   the aircraft (model, gear legs) and, for 1.5 s after a crash, the crash pose frozen with
 *   its explosion; three cameras (aircraft chase, runway / flight-plan overview, runway end);
 * - the labels of the scene: the threshold, the aircraft tag (altitude, height above the
 *   runway, speed, heading, slope), the deck banner and CRASH;
 * - the approach profile, runway-relative, below the scene;
 * - the instrument panel, in tabs: flight (status, targets, runway and deck, landing),
 *   controls and rewards (with the episode's reward bars), convergence diagnostics
 *   (actions and observations over the last complete episode) and the last crash reasons.
 *
 * What the desktop showed of the learner (PPO statistics, validation, BEST) is in the
 * PPOKit Studio, which reads the run journal; the panel says so.
 *
 * World frame of the simulation: north, east, up in metres. Render frame: x east, y up,
 * z south, in units of RenderMeters (the viewer's far plane is 1000 units: 100 km).
 */
#include "flight_aircraft_performance.hpp"
#include "flight_display.hpp"
#include "flight_guidance.hpp"
#include "flight_observation.hpp"
#include "flight_policy.hpp"
#include "flight_world.hpp"
#include "runway_geometry.hpp"

#include "ppokit/view/sdk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

namespace flight = ppo::examples::flight;
namespace world = flight::world;
using flight::Vec3;
using ppokit::sdk::Call;
using ppokit::sdk::Config;
using ppokit::sdk::Manifest;
using ppokit::sdk::Shapes;

/** @brief Values of the action port: the task's six (five controls, then the gear draw). */
constexpr std::size_t ActionValues = flight::ContinuousActions + 1;

/** @brief Sends what it is wired to: the world groups of the task, the action and the rewards. */
class FlightWatch final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("flight_watch", 2);
        manifest.Input("aircraft", {world::AircraftSize})
            .Input("airfield", {world::AirfieldSize})
            .Input("deck", {world::DeckSize})
            .Input("guidance", {world::GuidanceSize})
            .Input("mission", {world::MissionSize})
            .Input("landing", {world::LandingSize})
            .Input("crash", {world::CrashSize})
            .Input("episode_rewards", {world::EpisodeRewardsSize})
            .Input("diagnostics", {world::DiagnosticsSize})
            .Input("action", {ActionValues})
            .Input("rewards", {flight::RewardCount});
        manifest.Observer();
        manifest.Integer("port", 47000, 1024, 65535).Number("rate", 30.0, 1.0, 240.0);
        return manifest;
    }
    FlightWatch(Config const& config, Shapes const&, std::uint64_t)
        : sender_(config, {{"aircraft", world::AircraftSize}, {"airfield", world::AirfieldSize},
                              {"deck", world::DeckSize}, {"guidance", world::GuidanceSize},
                              {"mission", world::MissionSize}, {"landing", world::LandingSize},
                              {"crash", world::CrashSize}, {"episode_rewards", world::EpisodeRewardsSize},
                              {"diagnostics", world::DiagnosticsSize}, {"action", ActionValues},
                              {"rewards", flight::RewardCount}})
    {
    }
    void Forward(Call const& call) override { sender_.Send(call); }

private:
    ppokit::view::FrameSender sender_;
};

// ---------------------------------------------------------------------------------------
// What a frame holds, decoded into the example's own types.
// ---------------------------------------------------------------------------------------

/** @brief Run mode, phase and clock of the episode, the last episode and the curriculum. */
struct MissionView final
{
    flight::Phase phase{flight::Phase::Approach};
    flight::Outcome outcome{flight::Outcome::Flying};
    double elapsedSeconds{};
    double maximumSeconds{};
    flight::Outcome lastOutcome{flight::Outcome::Flying};
    double lastSeconds{};
    world::AircraftKind aircraft{world::T6Texan2};
    world::RunMode mode{world::PlayMode};
    std::size_t curriculumLevel{};
    std::size_t curriculumLevels{};
    std::size_t windowStops{};
    std::size_t windowEpisodes{};
    double difficulty{};
    /** @brief Filtered physical controls of the last action. */
    flight::Controls applied{};
    /** @brief Episodes finished by the environment of the run, and how they ended. */
    std::size_t episodes{};
    std::size_t touchdowns{};
    std::size_t stops{};
    std::size_t crashes{};
    std::size_t timeouts{};
};

/** @brief Touchdown and stop of the current episode once it touched down, else of the last one. */
struct LandingView final
{
    bool touchedDown{};
    double touchdownFromThreshold{};
    double groundRoll{};
    double touchdownSink{};
    double touchdownSpeed{};
    double touchdownQuality{};
    double landingAreaFactor{};
    flight::Outcome outcome{flight::Outcome::Flying};
    double stopFromThreshold{};
};

/** @brief The latest crash, kept across episodes. */
struct CrashView final
{
    bool has{};
    std::uint64_t event{};
    flight::CrashDiagnostic diagnostic{};
    flight::AircraftState aircraft{};
    flight::Runway runway{};
};

/** @brief Convergence diagnostics over the last complete episode, and the live values. */
struct DiagnosticsView final
{
    bool enabled{};
    double saturationThreshold{};
    std::uint64_t samples{};
    std::array<double, flight::ObservationSize> liveObservation{};
    std::array<double, flight::ContinuousActions> liveRaw{};
    std::array<double, flight::ObservationSize> observationMean{};
    std::array<double, flight::ObservationSize> observationMaximum{};
    std::array<std::uint64_t, flight::ObservationSize> observationSaturations{};
    std::array<double, flight::ContinuousActions> rawMean{};
    std::array<double, flight::ContinuousActions> appliedMean{};
    std::array<std::uint64_t, flight::ContinuousActions> rawSaturations{};
    std::array<std::uint64_t, flight::ContinuousActions> appliedSaturations{};
    std::array<double, flight::RewardCount> rewardMeans{};
};

/** @brief One step of the run as the view draws it (the desktop's FlightSnapshot, from the world row). */
struct Scene final
{
    flight::Runway runway{};
    flight::AircraftState aircraft{};
    flight::Guidance guidance{};
    flight::DeckState deck{};
    double seaHeight{};
    double aimAlong{};
    double glideDegrees{};
    MissionView mission{};
    LandingView landing{};
    CrashView crash{};
    std::array<double, flight::RewardCount> episodeSums{};
    std::array<double, flight::RewardCount> weights{};
    DiagnosticsView diagnostics{};
    std::array<double, ActionValues> action{};
    std::array<double, flight::RewardCount> stepRewards{};
};

/** @brief Reads one port of a frame, agent 0. */
class PortReader final
{
public:
    PortReader(ppokit::view::Frame const& frame, char const* port) noexcept : frame_(frame), port_(port) {}

    double operator()(std::size_t index) const noexcept
    {
        return frame_.Value(port_, 0, static_cast<std::uint32_t>(index), 0.F);
    }
    bool Flag(std::size_t index) const noexcept { return (*this)(index) > 0.5; }
    std::size_t Count(std::size_t index) const noexcept
    {
        double const value = (*this)(index);
        if (!(value > 0.0))
        {
            return 0;
        }
        return static_cast<std::size_t>(std::lround(value));
    }
    Vec3 Vector(std::size_t first) const noexcept
    {
        return {(*this)(first), (*this)(first + 1), (*this)(first + 2)};
    }

private:
    ppokit::view::Frame const& frame_;
    char const* port_;
};

flight::Outcome OutcomeOf(std::size_t value) noexcept
{
    switch (value)
    {
    case 1:
        return flight::Outcome::Stopped;
    case 2:
        return flight::Outcome::Crash;
    case 3:
        return flight::Outcome::Timeout;
    default:
        return flight::Outcome::Flying;
    }
}

flight::GuidanceMode GuidanceModeOf(std::size_t value) noexcept
{
    switch (value)
    {
    case 1:
        return flight::GuidanceMode::Flare;
    case 2:
        return flight::GuidanceMode::Rollout;
    default:
        return flight::GuidanceMode::Approach;
    }
}

world::RunMode RunModeOf(std::size_t value) noexcept
{
    switch (value)
    {
    case world::TrainMode:
        return world::TrainMode;
    case world::EvaluationMode:
        return world::EvaluationMode;
    default:
        return world::PlayMode;
    }
}

double Degrees(double radians) noexcept
{
    return radians / flight::DegreesToRadians;
}

void ReadAircraft(PortReader const& values, flight::AircraftState& aircraft) noexcept
{
    aircraft.position = values.Vector(world::AircraftNorth);
    aircraft.roll = values(world::AircraftRoll);
    aircraft.pitch = values(world::AircraftPitch);
    aircraft.heading = values(world::AircraftHeading);
    aircraft.gearPosition = values(world::AircraftGear);
    aircraft.airspeed = values(world::AircraftAirspeed);
}

void ReadRunway(PortReader const& airfield, PortReader const& deck, Scene& scene) noexcept
{
    flight::Runway& runway = scene.runway;
    runway.origin = airfield.Vector(world::RunwayNorth);
    runway.headingDegrees = Degrees(airfield(world::RunwayHeading));
    runway.lengthMeters = airfield(world::RunwayLength);
    runway.widthMeters = airfield(world::RunwayWidth);
    runway.thresholdHeightMeters = deck(world::ThresholdHeight);
    runway.landingMarginMeters = deck(world::LandingMargin);
    runway.elevationMeters = deck(world::Elevation);
    runway.velocity.up = deck(world::DeckVerticalSpeed);
    scene.seaHeight = deck(world::SeaHeight);
    scene.deck.motion.speedMps = deck(world::DeckSpeed);
    scene.deck.motion.swellAmplitudeMeters = deck(world::SwellAmplitude);
    scene.deck.motion.swellPeriodSeconds = deck(world::SwellPeriod);
    scene.deck.heaveMeters = deck(world::Heave);
    scene.deck.swellAngle = deck(world::SwellAngle);
}

void ReadGuidance(PortReader const& values, Scene& scene) noexcept
{
    flight::Guidance& guidance = scene.guidance;
    guidance.mode = GuidanceModeOf(values.Count(world::GuidanceMode));
    guidance.targetValid = values.Flag(world::TargetValid);
    guidance.targetSpeed = values(world::TargetSpeed);
    guidance.targetSlope = values(world::TargetSlope);
    guidance.targetHeading = values(world::TargetHeading);
    guidance.target = values.Vector(world::TargetNorth);
    guidance.targetDirection = values.Vector(world::DirectionNorth);
    guidance.velocityDirection = values.Vector(world::VelocityDirectionNorth);
    guidance.relativeVelocity = values.Vector(world::RelativeAlong);
    guidance.along = values(world::Along);
    guidance.cross = values(world::Cross);
    guidance.distance = values(world::Distance);
    guidance.height = values(world::Height);
    scene.aimAlong = values(world::AimAlong);
    scene.glideDegrees = values(world::GlideDegrees);
}

void ReadMission(PortReader const& values, Scene& scene) noexcept
{
    MissionView& mission = scene.mission;
    mission.phase = values.Count(world::Phase) == 1 ? flight::Phase::Rollout : flight::Phase::Approach;
    mission.outcome = OutcomeOf(values.Count(world::Outcome));
    mission.elapsedSeconds = values(world::ElapsedSeconds);
    mission.maximumSeconds = values(world::MaximumSeconds);
    mission.lastOutcome = OutcomeOf(values.Count(world::LastOutcome));
    mission.lastSeconds = values(world::LastSeconds);
    scene.aircraft.weightOnWheels = values.Flag(world::WeightOnWheels);
    mission.aircraft = values.Count(world::AircraftType) == world::Pc7 ? world::Pc7 : world::T6Texan2;
    mission.mode = RunModeOf(values.Count(world::Mode));
    mission.curriculumLevel = values.Count(world::CurriculumLevel);
    mission.curriculumLevels = values.Count(world::CurriculumLevels);
    mission.windowStops = values.Count(world::WindowStops);
    mission.windowEpisodes = values.Count(world::WindowEpisodes);
    mission.difficulty = values(world::Difficulty);
    mission.applied.elevator = values(world::AppliedElevator);
    mission.applied.aileron = values(world::AppliedAileron);
    mission.applied.rudder = values(world::AppliedRudder);
    mission.applied.throttle = values(world::AppliedThrottle);
    mission.applied.brake = values(world::AppliedBrake);
    mission.applied.flap = values(world::AppliedFlap);
    mission.applied.gear = values(world::AppliedGear);
    mission.episodes = values.Count(world::Episodes);
    mission.touchdowns = values.Count(world::Touchdowns);
    mission.stops = values.Count(world::Stops);
    mission.crashes = values.Count(world::Crashes);
    mission.timeouts = values.Count(world::Timeouts);
}

void ReadLanding(PortReader const& values, LandingView& landing) noexcept
{
    landing.touchedDown = values.Flag(world::TouchedDown);
    landing.touchdownFromThreshold = values(world::TouchdownFromThreshold);
    landing.groundRoll = values(world::GroundRoll);
    landing.touchdownSink = values(world::TouchdownSink);
    landing.touchdownSpeed = values(world::TouchdownSpeed);
    landing.touchdownQuality = values(world::TouchdownQuality);
    landing.landingAreaFactor = values(world::LandingAreaFactor);
    landing.outcome = OutcomeOf(values.Count(world::ResultOutcome));
    landing.stopFromThreshold = values(world::StopFromThreshold);
}

void ReadCrash(PortReader const& values, Scene& scene) noexcept
{
    CrashView& crash = scene.crash;
    crash.has = values.Flag(world::HasCrash);
    crash.event = values.Count(world::CrashEvent);
    for (std::size_t reason = 0; reason < flight::CrashReasonCount; ++reason)
    {
        std::size_t const at = world::CrashConditions + reason * world::CrashConditionValues;
        flight::CrashCondition& condition = crash.diagnostic.conditions[reason];
        condition.failed = values.Flag(at);
        condition.value = values(at + 1);
        condition.limit = values(at + 2);
    }
    crash.aircraft = scene.aircraft;
    crash.aircraft.position = values.Vector(world::CrashAircraftNorth);
    crash.aircraft.roll = values(world::CrashAircraftRoll);
    crash.aircraft.pitch = values(world::CrashAircraftPitch);
    crash.aircraft.heading = values(world::CrashAircraftHeading);
    crash.aircraft.gearPosition = values(world::CrashAircraftGear);
    // The runway's dimensions do not change between episodes; its pose does.
    crash.runway = scene.runway;
    crash.runway.origin = values.Vector(world::CrashRunwayNorth);
    crash.runway.headingDegrees = Degrees(values(world::CrashRunwayHeading));
}

void ReadRewards(PortReader const& values, PortReader const& rewards, PortReader const& action, Scene& scene) noexcept
{
    for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
    {
        scene.episodeSums[channel] = values(world::EpisodeSums + channel);
        scene.weights[channel] = values(world::ActiveWeights + channel);
        scene.stepRewards[channel] = rewards(channel);
    }
    for (std::size_t index = 0; index < ActionValues; ++index)
    {
        scene.action[index] = action(index);
    }
}

void ReadDiagnostics(PortReader const& values, DiagnosticsView& diagnostics) noexcept
{
    diagnostics.enabled = values.Flag(world::DiagnosticsEnabled);
    diagnostics.saturationThreshold = values(world::SaturationThreshold);
    diagnostics.samples = values.Count(world::WindowSamples);
    for (std::size_t slot = 0; slot < flight::ObservationSize; ++slot)
    {
        diagnostics.liveObservation[slot] = values(world::LiveObservation + slot);
        diagnostics.observationMean[slot] = values(world::ObservationMeanAbsolute + slot);
        diagnostics.observationMaximum[slot] = values(world::ObservationMaximumAbsolute + slot);
        diagnostics.observationSaturations[slot] = values.Count(world::ObservationSaturations + slot);
    }
    for (std::size_t axis = 0; axis < flight::ContinuousActions; ++axis)
    {
        diagnostics.liveRaw[axis] = values(world::LiveRawAction + axis);
        diagnostics.rawMean[axis] = values(world::RawMeanAbsolute + axis);
        diagnostics.appliedMean[axis] = values(world::AppliedMeanAbsolute + axis);
        diagnostics.rawSaturations[axis] = values.Count(world::RawSaturations + axis);
        diagnostics.appliedSaturations[axis] = values.Count(world::AppliedSaturations + axis);
    }
    for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
    {
        diagnostics.rewardMeans[channel] = values(world::RewardMeans + channel);
    }
}

/** @brief Decodes a frame of flight_watch@2. */
Scene Decode(ppokit::view::Frame const& frame)
{
    Scene scene;
    ReadAircraft(PortReader(frame, "aircraft"), scene.aircraft);
    ReadRunway(PortReader(frame, "airfield"), PortReader(frame, "deck"), scene);
    ReadGuidance(PortReader(frame, "guidance"), scene);
    ReadMission(PortReader(frame, "mission"), scene);
    ReadLanding(PortReader(frame, "landing"), scene.landing);
    ReadCrash(PortReader(frame, "crash"), scene);
    ReadRewards(PortReader(frame, "episode_rewards"), PortReader(frame, "rewards"), PortReader(frame, "action"), scene);
    ReadDiagnostics(PortReader(frame, "diagnostics"), scene.diagnostics);
    return scene;
}

// ---------------------------------------------------------------------------------------
// Colors and drawing helpers.
// ---------------------------------------------------------------------------------------

constexpr PpokitColor Rgba(unsigned red, unsigned green, unsigned blue, unsigned alpha = 255U) noexcept
{
    return (red << 24U) | (green << 16U) | (blue << 8U) | alpha;
}

constexpr PpokitColor BackgroundColor = Rgba(14, 27, 44);
constexpr PpokitColor GuidanceColor = Rgba(220, 230, 240);
constexpr PpokitColor CommandColor = Rgba(0, 228, 48);
constexpr PpokitColor ThresholdLabelColor = Rgba(255, 220, 80);
constexpr PpokitColor SwellCrestColor = Rgba(96, 150, 182);
constexpr PpokitColor WakeColor = Rgba(214, 232, 240);
constexpr PpokitColor PanelColor = Rgba(9, 20, 31, 228);
/** @brief The panel and the profile hide the scene behind them. */
constexpr PpokitColor OpaquePanelColor = Rgba(9, 20, 31);
constexpr PpokitColor TextColor = Rgba(239, 246, 247);
constexpr PpokitColor DimTextColor = Rgba(155, 177, 187);
constexpr PpokitColor HeaderColor = Rgba(101, 217, 235);
constexpr PpokitColor CrashTextColor = Rgba(255, 115, 102);
constexpr PpokitColor SeparatorColor = Rgba(68, 93, 105);

/** @brief The same color with another alpha. */
constexpr PpokitColor WithAlpha(PpokitColor color, unsigned alpha) noexcept
{
    return (color & 0xFFFFFF00U) | (alpha & 0xFFU);
}

/** @brief World metres per render unit: the viewer's far plane (1000 units) is then 100 km away. */
constexpr double RenderMeters = 100.0;

using Point3 = std::array<float, 3>;

/** @brief A world point (north, east, up) in metres, in render units (x east, y up, z south). */
Point3 Render(Vec3 point) noexcept
{
    return {static_cast<float>(point.east / RenderMeters), static_cast<float>(point.up / RenderMeters),
        static_cast<float>(-point.north / RenderMeters)};
}

/** @brief The 3D half of the drawing table, in world metres. */
class Draw3d final
{
public:
    explicit Draw3d(PpokitDraw const& draw) noexcept : draw_(draw) {}

    void Segment(Vec3 from, Vec3 to, PpokitColor color) const
    {
        Point3 const start = Render(from);
        Point3 const end = Render(to);
        draw_.line3d(draw_.context, start.data(), end.data(), color);
    }
    void Triangle(Vec3 first, Vec3 second, Vec3 third, PpokitColor color) const
    {
        Point3 const a = Render(first);
        Point3 const b = Render(second);
        Point3 const c = Render(third);
        draw_.triangle3d(draw_.context, a.data(), b.data(), c.data(), color);
    }
    /** @brief A quadrilateral seen from both sides. */
    void Quad(Vec3 first, Vec3 second, Vec3 third, Vec3 fourth, PpokitColor color) const
    {
        Triangle(first, second, third, color);
        Triangle(first, third, fourth, color);
    }
    /** @brief A sphere of `radius` metres. */
    void Sphere(Vec3 center, double radius, PpokitColor color) const
    {
        Point3 const at = Render(center);
        draw_.sphere3d(draw_.context, at.data(), static_cast<float>(radius / RenderMeters), color);
    }

private:
    PpokitDraw const& draw_;
};

/** @brief A pixel rectangle. */
struct Box final
{
    float x{};
    float y{};
    float width{};
    float height{};

    bool Contains(float pointX, float pointY) const noexcept
    {
        return pointX >= x && pointX < x + width && pointY >= y && pointY < y + height;
    }
    float Right() const noexcept { return x + width; }
    float Bottom() const noexcept { return y + height; }
};

/** @brief The 2D half of the drawing table, with C strings (no allocation per text). */
class Draw2d final
{
public:
    explicit Draw2d(PpokitDraw const& draw) noexcept : draw_(draw) {}

    void Text(char const* text, float x, float y, float size, PpokitColor color) const
    {
        draw_.text2d(draw_.context, text, x, y, size, color);
    }
    float Width(char const* text, float size) const { return draw_.measureText(draw_.context, text, size); }
    void Line(float x0, float y0, float x1, float y1, float thickness, PpokitColor color) const
    {
        draw_.line2d(draw_.context, x0, y0, x1, y1, thickness, color);
    }
    void Fill(Box const& box, PpokitColor color) const
    {
        draw_.rectangle2d(draw_.context, box.x, box.y, box.width, box.height, 1, color);
    }
    void Outline(Box const& box, PpokitColor color) const
    {
        draw_.rectangle2d(draw_.context, box.x, box.y, box.width, box.height, 0, color);
    }
    void Circle(float x, float y, float radius, bool filled, PpokitColor color) const
    {
        draw_.circle2d(draw_.context, x, y, radius, filled ? 1 : 0, color);
    }
    void Triangle(float x0, float y0, float x1, float y1, float x2, float y2, PpokitColor color) const
    {
        draw_.triangle2d(draw_.context, x0, y0, x1, y1, x2, y2, color);
    }
    /** @brief A bar filled to `fraction` with `overlay` centred on it. */
    void ProgressBar(
        Box const& box, double fraction, PpokitColor background, PpokitColor bar, char const* overlay) const
    {
        Fill(box, background);
        double const filled = std::clamp(fraction, 0.0, 1.0);
        Fill(Box{box.x, box.y, static_cast<float>(box.width * filled), box.height}, bar);
        float const size = 14.F;
        float const textWidth = Width(overlay, size);
        Text(overlay, box.x + (box.width - textWidth) / 2.F, box.y + (box.height - size) / 2.F, size, TextColor);
    }

private:
    PpokitDraw const& draw_;
};

/** @brief Clips the segment (x0, y0)-(x1, y1) to `box` (Liang-Barsky); false when nothing is left. */
bool ClipSegment(Box const& box, float& x0, float& y0, float& x1, float& y1) noexcept
{
    float const dx = x1 - x0;
    float const dy = y1 - y0;
    float enter = 0.F;
    float leave = 1.F;
    std::array<float, 4> const p = {-dx, dx, -dy, dy};
    std::array<float, 4> const q = {x0 - box.x, box.Right() - x0, y0 - box.y, box.Bottom() - y0};
    for (std::size_t edge = 0; edge < p.size(); ++edge)
    {
        if (p[edge] == 0.F)
        {
            if (q[edge] < 0.F)
            {
                return false;
            }
            continue;
        }
        float const t = q[edge] / p[edge];
        if (p[edge] < 0.F)
        {
            enter = std::max(enter, t);
        }
        else
        {
            leave = std::min(leave, t);
        }
    }
    if (enter > leave)
    {
        return false;
    }
    float const startX = x0 + enter * dx;
    float const startY = y0 + enter * dy;
    x1 = x0 + leave * dx;
    y1 = y0 + leave * dy;
    x0 = startX;
    y0 = startY;
    return true;
}

// ---------------------------------------------------------------------------------------
// Cameras and projection.
// ---------------------------------------------------------------------------------------

enum CameraMode
{
    AircraftChase,
    FlightplanOverview,
    RunwayEnd,
    CameraModes
};

constexpr std::array<char const*, CameraModes> CameraNames = {"Aircraft chase", "Runway / flightplan", "Runway"};

/** @brief A camera in render units, as begin3d takes it. */
struct ViewCamera final
{
    Point3 position{};
    Point3 target{};
    Point3 up{0.F, 1.F, 0.F};
    float fovy = 50.F;
};

/** @brief World position of a point given in aircraft body axes, in metres. */
Vec3 AircraftPoint(flight::AircraftState const& aircraft, double forward, double right, double up) noexcept
{
    flight::BodyAxes const axes = flight::AircraftAxes(aircraft.roll, aircraft.pitch, aircraft.heading);
    return flight::Add(
        aircraft.position, flight::Add(flight::Scale(axes.forward, forward),
                               flight::Add(flight::Scale(axes.right, right), flight::Scale(axes.up, up))));
}

/** @brief Turns `offset` (world metres) about the vertical by `yaw` and raises it by `pitch`, both radians. */
Vec3 Orbit(Vec3 offset, double yaw, double pitch) noexcept
{
    double const cosine = std::cos(yaw);
    double const sine = std::sin(yaw);
    Vec3 const turned{
        offset.north * cosine - offset.east * sine, offset.north * sine + offset.east * cosine, offset.up};
    double const horizontal = std::hypot(turned.north, turned.east);
    double const elevation = std::atan2(turned.up, horizontal) + pitch;
    double const length = flight::Length(turned);
    double const limited = std::clamp(elevation, -0.2, 1.45);
    double const scale = horizontal > 0.0 ? length * std::cos(limited) / horizontal : 0.0;
    return {turned.north * scale, turned.east * scale, length * std::sin(limited)};
}

/** @brief Chase camera behind and above the aircraft, turned by the mouse (no turn: the desktop's). */
ViewCamera ChaseCamera(Scene const& scene, double distance, double yaw, double pitch) noexcept
{
    Vec3 const behind = AircraftPoint(scene.aircraft, -distance, 0, distance * 0.3);
    Vec3 const offset = Orbit(flight::Subtract(behind, scene.aircraft.position), yaw, pitch);
    ViewCamera camera;
    camera.position = Render(flight::Add(scene.aircraft.position, offset));
    camera.target = Render(AircraftPoint(scene.aircraft, 30, 0, 0));
    return camera;
}

/** @brief Fixed camera near the threshold, zooming on the aircraft. */
ViewCamera RunwayCamera(Scene const& scene) noexcept
{
    constexpr double MinimumDistanceMeters = 1.0;
    Vec3 const position = flight::RunwayCameraPosition(scene.runway);
    Vec3 const delta = flight::Subtract(scene.aircraft.position, position);
    double const distance = flight::Length(delta);
    bool const aircraftAway = distance > MinimumDistanceMeters;
    ViewCamera camera;
    camera.position = Render(position);
    camera.target = Render(aircraftAway ? scene.aircraft.position : flight::ThresholdPoint(scene.runway));
    camera.fovy = static_cast<float>(flight::RunwayCameraFieldOfView(position, scene.aircraft.position));
    // Looking straight up or down: the runway axis is the camera's up.
    Vec3 const normal = flight::RunwayNormal(scene.runway);
    bool const vertical = aircraftAway && std::abs(flight::Dot(delta, normal)) / distance > 0.99;
    Vec3 const up = vertical ? flight::RunwayForward(scene.runway) : normal;
    camera.up = {static_cast<float>(up.east), static_cast<float>(up.up), static_cast<float>(-up.north)};
    return camera;
}

/** @brief Grows the box [minimum, maximum] to hold `point`. */
void Include(Vec3& minimum, Vec3& maximum, Vec3 point) noexcept
{
    minimum = {
        std::min(minimum.north, point.north), std::min(minimum.east, point.east), std::min(minimum.up, point.up)};
    maximum = {
        std::max(maximum.north, point.north), std::max(maximum.east, point.east), std::max(maximum.up, point.up)};
}

/** @brief Isometric camera framing the aircraft and the whole landing surface. */
ViewCamera OverviewCamera(Scene const& scene) noexcept
{
    flight::Runway const& runway = scene.runway;
    Vec3 minimum = scene.aircraft.position;
    Vec3 maximum = scene.aircraft.position;
    Include(minimum, maximum, runway.origin);
    Include(minimum, maximum, flight::FromRunway(runway, runway.lengthMeters, 0, 0));
    if (runway.landingMarginMeters > 0)
    {
        double const margin = runway.landingMarginMeters;
        double const halfWidth = runway.widthMeters / 2 + margin;
        for (double const along : {-margin, runway.lengthMeters + margin})
        {
            for (double const cross : {-halfWidth, halfWidth})
            {
                Include(minimum, maximum, flight::FromRunway(runway, along, cross, 0));
            }
        }
    }
    Vec3 const centre = flight::Scale(flight::Add(minimum, maximum), 0.5);
    double const range =
        std::max({maximum.north - minimum.north, maximum.east - minimum.east, runway.lengthMeters});
    ViewCamera camera;
    camera.target = Render(centre);
    camera.position = Render(flight::Add(centre, {range, range, range}));
    return camera;
}

using Axis3 = std::array<double, 3>;

double Dot3(Axis3 const& first, Axis3 const& second) noexcept
{
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

Axis3 Cross3(Axis3 const& first, Axis3 const& second) noexcept
{
    return {first[1] * second[2] - first[2] * second[1], first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0]};
}

/** @brief `vector` scaled to length 1 (unchanged when null). */
Axis3 Unit3(Axis3 vector) noexcept
{
    double const length = std::sqrt(Dot3(vector, vector));
    if (length > 0.0)
    {
        for (double& value : vector)
        {
            value /= length;
        }
    }
    return vector;
}

/**
 * @brief Projects a world point to the window as the viewer's perspective camera does
 *        (raylib: vertical field of view, the window's aspect ratio).
 * @return True when the point is in front of the camera and inside `area`.
 */
bool Project(
    ViewCamera const& camera, Vec3 point, float width, float height, Box const& area, float& x, float& y) noexcept
{
    Point3 const at = Render(point);
    Axis3 forward{};
    Axis3 offset{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        forward[axis] = camera.target[axis] - camera.position[axis];
        offset[axis] = at[axis] - camera.position[axis];
    }
    forward = Unit3(forward);
    Axis3 const up{camera.up[0], camera.up[1], camera.up[2]};
    Axis3 const right = Unit3(Cross3(forward, up));
    Axis3 const trueUp = Cross3(right, forward);
    double const depth = Dot3(offset, forward);
    if (!std::isfinite(depth) || depth <= 0.0 || width <= 0.F || height <= 0.F)
    {
        return false;
    }
    double const halfHeight = std::tan(camera.fovy * flight::DegreesToRadians / 2.0) * depth;
    double const halfWidth = halfHeight * width / height;
    double const ndcX = Dot3(offset, right) / halfWidth;
    double const ndcY = Dot3(offset, trueUp) / halfHeight;
    x = static_cast<float>((ndcX + 1.0) / 2.0 * width);
    y = static_cast<float>((1.0 - ndcY) / 2.0 * height);
    return std::isfinite(x) && std::isfinite(y) && area.Contains(x, y);
}

// ---------------------------------------------------------------------------------------
// The 3D scene (ported from the desktop renderer of FlightControl3D.exe, flight_view.cpp).
// ---------------------------------------------------------------------------------------

/** @brief A marking in runway coordinates, in metres, slightly above the surface. */
void RunwayMarking(Draw3d const& draw, flight::Runway const& runway, double start, double end, double left,
    double right, PpokitColor color)
{
    constexpr double MarkingHeight = 0.1;
    draw.Quad(flight::FromRunway(runway, start, left, MarkingHeight),
        flight::FromRunway(runway, end, left, MarkingHeight), flight::FromRunway(runway, end, right, MarkingHeight),
        flight::FromRunway(runway, start, right, MarkingHeight), color);
}

/** @brief The landable surface around the runway, in the runway plane. */
void DrawLandingArea(Draw3d const& draw, flight::Runway const& runway)
{
    double const margin = runway.landingMarginMeters;
    if (margin <= 0)
    {
        return;
    }
    double const halfWidth = runway.widthMeters / 2;
    double const length = runway.lengthMeters;
    PpokitColor const ground = Rgba(79, 112, 82);
    using flight::FromRunway;
    // Left and right strips, then the strips before the threshold and after the far end.
    draw.Quad(FromRunway(runway, -margin, -halfWidth - margin, 0),
        FromRunway(runway, length + margin, -halfWidth - margin, 0), FromRunway(runway, length + margin, -halfWidth, 0),
        FromRunway(runway, -margin, -halfWidth, 0), ground);
    draw.Quad(FromRunway(runway, -margin, halfWidth, 0), FromRunway(runway, length + margin, halfWidth, 0),
        FromRunway(runway, length + margin, halfWidth + margin, 0), FromRunway(runway, -margin, halfWidth + margin, 0),
        ground);
    draw.Quad(FromRunway(runway, -margin, -halfWidth, 0), FromRunway(runway, 0, -halfWidth, 0),
        FromRunway(runway, 0, halfWidth, 0), FromRunway(runway, -margin, halfWidth, 0), ground);
    draw.Quad(FromRunway(runway, length, -halfWidth, 0), FromRunway(runway, length + margin, -halfWidth, 0),
        FromRunway(runway, length + margin, halfWidth, 0), FromRunway(runway, length, halfWidth, 0), ground);
}

/** @brief The sides of the elevated landing surface, down to the sea. */
void DrawLandingAreaSides(Draw3d const& draw, flight::Runway const& runway, double seaHeight)
{
    double const margin = runway.landingMarginMeters;
    double const halfWidth = runway.widthMeters / 2 + margin;
    std::array<Vec3, 4> const corners{flight::FromRunway(runway, -margin, -halfWidth, 0),
        flight::FromRunway(runway, runway.lengthMeters + margin, -halfWidth, 0),
        flight::FromRunway(runway, runway.lengthMeters + margin, halfWidth, 0),
        flight::FromRunway(runway, -margin, halfWidth, 0)};
    for (std::size_t corner = 0; corner < corners.size(); ++corner)
    {
        Vec3 const top = corners[corner];
        Vec3 const next = corners[(corner + 1) % corners.size()];
        Vec3 const bottom{top.north, top.east, seaHeight};
        Vec3 const nextBottom{next.north, next.east, seaHeight};
        draw.Quad(top, next, nextBottom, bottom, Rgba(64, 81, 66));
        draw.Segment(top, next, Rgba(150, 169, 139));
    }
}

// Sea state: visual only, JSBSim sees the level heaving deck.
/** @brief Half extent of the drawn swell around the deck, in metres. */
constexpr double SwellDrawHalfExtentMeters = 2000.0;
/** @brief Width of a drawn swell crest band as a fraction of the wavelength. */
constexpr double SwellCrestBandFraction = 0.1;
/** @brief Height of the drawn sea-state marks above the sea, in metres (avoids z-fighting). */
constexpr double SeaMarkHeightMeters = 1.0;
/** @brief Half angle of a ship's Kelvin wake, in degrees. */
constexpr double KelvinHalfAngleDegrees = 19.47;
/** @brief Drawn wake length per metre per second of deck speed, in metres (one minute of travel). */
constexpr double WakeSecondsOfTravel = 60.0;
/** @brief Segments of the fading wake arms. */
constexpr int WakeSegments = 6;

/** @brief Deep-water wavelength of a swell period, g T^2 / (2 pi), in metres. */
double SwellWavelengthMeters(double periodSeconds) noexcept
{
    return flight::StandardGravity * periodSeconds * periodSeconds / (2.0 * flight::Pi);
}

/**
 * @brief The swell crests travelling along the runway axis around the deck: the sea height
 *        along the runway is amplitude * sin(swellAngle - k * along).
 */
void DrawSwell(Draw3d const& draw, flight::Runway const& runway, flight::DeckState const& deck, double seaHeight)
{
    flight::DeckMotionSettings const& motion = deck.motion;
    if (motion.swellAmplitudeMeters <= 0 || motion.swellPeriodSeconds <= 0)
    {
        return;
    }
    double const wavelength = SwellWavelengthMeters(motion.swellPeriodSeconds);
    double const waveNumber = 2.0 * flight::Pi / wavelength;
    double const centreAlong = runway.lengthMeters / 2;
    double const firstAlong = centreAlong - SwellDrawHalfExtentMeters;
    double const lastAlong = centreAlong + SwellDrawHalfExtentMeters;
    double const bandHalfWidth = SwellCrestBandFraction * wavelength / 2;
    double const crestHeight = seaHeight - runway.origin.up + SeaMarkHeightMeters;
    // Crest n lies where swellAngle - k * along = pi / 2 + 2 pi n; the first is just ahead of firstAlong.
    double const crestZero = (deck.swellAngle - flight::Pi / 2) / waveNumber;
    double const firstCrest = crestZero + std::ceil((firstAlong - crestZero) / wavelength) * wavelength;
    for (double along = firstCrest; along <= lastAlong; along += wavelength)
    {
        draw.Quad(flight::FromRunway(runway, along - bandHalfWidth, -SwellDrawHalfExtentMeters, crestHeight),
            flight::FromRunway(runway, along + bandHalfWidth, -SwellDrawHalfExtentMeters, crestHeight),
            flight::FromRunway(runway, along + bandHalfWidth, SwellDrawHalfExtentMeters, crestHeight),
            flight::FromRunway(runway, along - bandHalfWidth, SwellDrawHalfExtentMeters, crestHeight), SwellCrestColor);
    }
}

/** @brief The Kelvin wake that a deck under way leaves behind its stern. */
void DrawWake(Draw3d const& draw, flight::Runway const& runway, flight::DeckState const& deck, double seaHeight)
{
    double const speed = deck.motion.speedMps;
    if (speed <= 0)
    {
        return;
    }
    double const stern = -runway.landingMarginMeters;
    double const halfBeam = runway.widthMeters / 2 + runway.landingMarginMeters;
    double const length = speed * WakeSecondsOfTravel;
    double const spread = std::tan(KelvinHalfAngleDegrees * flight::DegreesToRadians);
    double const height = seaHeight - runway.origin.up + SeaMarkHeightMeters;
    using flight::FromRunway;
    // Each arm and the turbulent centre fade in segments away from the stern.
    for (int segment = 0; segment < WakeSegments; ++segment)
    {
        double const near = length * segment / WakeSegments;
        double const far = length * (segment + 1) / WakeSegments;
        unsigned const alpha = static_cast<unsigned>(200 * (WakeSegments - segment) / WakeSegments);
        PpokitColor const foam = WithAlpha(WakeColor, alpha);
        for (double const side : {-1.0, 1.0})
        {
            double const nearCross = side * (halfBeam + near * spread);
            double const farCross = side * (halfBeam + far * spread);
            double const armWidth = side * runway.widthMeters / 8;
            draw.Quad(FromRunway(runway, stern - near, nearCross, height),
                FromRunway(runway, stern - far, farCross, height),
                FromRunway(runway, stern - far, farCross - armWidth, height),
                FromRunway(runway, stern - near, nearCross - armWidth, height), foam);
        }
        PpokitColor const centre = WithAlpha(WakeColor, alpha / 2);
        double const nearHalf = halfBeam * (1.0 - 0.5 * near / length);
        double const farHalf = halfBeam * (1.0 - 0.5 * far / length);
        draw.Quad(FromRunway(runway, stern - near, -nearHalf, height),
            FromRunway(runway, stern - far, -farHalf, height), FromRunway(runway, stern - far, farHalf, height),
            FromRunway(runway, stern - near, nearHalf, height), centre);
    }
}

/** @brief The runway and its high-contrast markings at their exact dimensions. */
void DrawRunway(Draw3d const& draw, flight::Runway const& runway)
{
    constexpr int CentrelineDashes = 24;
    constexpr int StripesPerSide = 4;
    PpokitColor const white = Rgba(255, 255, 255);
    PpokitColor const edgeColor = Rgba(255, 210, 48);
    double const halfWidth = runway.widthMeters / 2;
    draw.Quad(flight::FromRunway(runway, 0, -halfWidth, 0),
        flight::FromRunway(runway, runway.lengthMeters, -halfWidth, 0),
        flight::FromRunway(runway, runway.lengthMeters, halfWidth, 0), flight::FromRunway(runway, 0, halfWidth, 0),
        Rgba(38, 43, 50));
    double const edgeWidth = std::min(5.0, runway.widthMeters * 0.04);
    RunwayMarking(draw, runway, 0, runway.lengthMeters, -halfWidth, -halfWidth + edgeWidth, edgeColor);
    RunwayMarking(draw, runway, 0, runway.lengthMeters, halfWidth - edgeWidth, halfWidth, edgeColor);
    // Centreline: dashes of half the dash period, starting after the threshold bar.
    double const thresholdDepth = std::min(12.0, runway.lengthMeters * 0.02);
    double const dashWidth = std::min(6.0, runway.widthMeters * 0.08);
    for (int dash = 0; dash < CentrelineDashes; ++dash)
    {
        double const along = runway.lengthMeters * static_cast<double>(dash) / CentrelineDashes;
        RunwayMarking(draw, runway, std::max(along, thresholdDepth), along + runway.lengthMeters / 48, -dashWidth / 2,
            dashWidth / 2, white);
    }
    // Threshold markings stay inside the physical surface and distinguish its entry end.
    RunwayMarking(draw, runway, 0, thresholdDepth, -halfWidth + edgeWidth, halfWidth - edgeWidth, edgeColor);
    double const stripeEnd = std::min(55.0, runway.lengthMeters * 0.08);
    double const stripeStart = thresholdDepth + (stripeEnd - thresholdDepth) * 0.1;
    double const stripeHalfWidth = runway.widthMeters * 0.0175;
    for (double const side : {-1.0, 1.0})
    {
        for (int stripe = 0; stripe < StripesPerSide; ++stripe)
        {
            double const centre = side * runway.widthMeters * (0.06 + 0.09 * stripe);
            RunwayMarking(
                draw, runway, stripeStart, stripeEnd, centre - stripeHalfWidth, centre + stripeHalfWidth, white);
        }
    }
}

/** @brief The target point and the commanded direction, the reference the policy and the rewards use. */
void DrawGuidance(Draw3d const& draw, Scene const& scene)
{
    constexpr double CommandVectorMeters = 180.0;
    constexpr double TargetRadiusMeters = 3.5;
    if (!scene.guidance.targetValid)
    {
        return;
    }
    draw.Segment(scene.aircraft.position, scene.guidance.target, GuidanceColor);
    draw.Sphere(scene.guidance.target, TargetRadiusMeters, GuidanceColor);
    draw.Segment(scene.aircraft.position,
        flight::Add(scene.aircraft.position, flight::Scale(scene.guidance.targetDirection, CommandVectorMeters)),
        CommandColor);
}

/** @brief Draw the eight-sided tapered fuselage of the original desktop silhouette. */
void DrawFuselage(Draw3d const& draw, flight::AircraftState const& aircraft, PpokitColor color)
{
    constexpr int Sides = 8;
    constexpr double NosePosition = 4.5;
    constexpr double TailPosition = -4.5;
    constexpr double NoseRadius = 0.6;
    constexpr double TailRadius = 0.4;
    flight::BodyAxes const axes = flight::AircraftAxes(aircraft.roll, aircraft.pitch, aircraft.heading);
    Vec3 const nose = flight::Add(aircraft.position, flight::Scale(axes.forward, NosePosition));
    Vec3 const tail = flight::Add(aircraft.position, flight::Scale(axes.forward, TailPosition));
    Vec3 previous = axes.right;
    for (int side = 0; side < Sides; ++side)
    {
        double const angle = 2.0 * flight::Pi * (side + 1) / Sides;
        Vec3 const next = flight::Add(flight::Scale(axes.right, std::cos(angle)),
            flight::Scale(axes.up, std::sin(angle)));
        Vec3 const nosePrevious = flight::Add(nose, flight::Scale(previous, NoseRadius));
        Vec3 const noseNext = flight::Add(nose, flight::Scale(next, NoseRadius));
        Vec3 const tailPrevious = flight::Add(tail, flight::Scale(previous, TailRadius));
        Vec3 const tailNext = flight::Add(tail, flight::Scale(next, TailRadius));
        draw.Quad(nosePrevious, tailPrevious, tailNext, noseNext, color);
        draw.Triangle(nose, nosePrevious, noseNext, color);
        draw.Triangle(tail, tailNext, tailPrevious, color);
        previous = next;
    }
}

/** @brief The desktop's simplified silhouette: fuselage, wing, tailplane and fin. */
void DrawSilhouette(Draw3d const& draw, flight::AircraftState const& aircraft, PpokitColor color)
{
    PpokitColor const white = Rgba(255, 255, 255);
    Vec3 const tail = AircraftPoint(aircraft, -4.5, 0, 0);
    DrawFuselage(draw, aircraft, color);
    draw.Quad(AircraftPoint(aircraft, 1, 0, 0), AircraftPoint(aircraft, -1, -6, 0), AircraftPoint(aircraft, -1.5, 0, 0),
        AircraftPoint(aircraft, -1, 6, 0), color);
    draw.Quad(AircraftPoint(aircraft, -3, 0, 0), AircraftPoint(aircraft, -4, -2, 0),
        AircraftPoint(aircraft, -4.5, 0, 0), AircraftPoint(aircraft, -4, 2, 0), color);
    draw.Segment(tail, AircraftPoint(aircraft, -4, 0, 2), white);
}

/** @brief The aircraft: its model (red after a crash), its gear legs, and a large marker in the overview. */
void DrawAircraft(Draw3d const& draw, Scene const& scene, CameraMode mode)
{
    constexpr double OverviewMarkerMeters = 80.0;
    constexpr double WheelRadiusMeters = 0.3;
    flight::AircraftState const& aircraft = scene.aircraft;
    bool const crashed = scene.mission.outcome == flight::Outcome::Crash;
    PpokitColor const color = crashed ? Rgba(230, 41, 55) : Rgba(102, 191, 255);
    DrawSilhouette(draw, aircraft, color);
    if (aircraft.gearPosition > 0.01)
    {
        for (double const right : {-2.0, 2.0})
        {
            Vec3 const wheel = AircraftPoint(aircraft, -1, right, -1.7 * aircraft.gearPosition);
            draw.Segment(AircraftPoint(aircraft, -1, right, 0), wheel, Rgba(255, 255, 255));
            draw.Sphere(wheel, WheelRadiusMeters, Rgba(80, 80, 80));
        }
    }
    if (mode == FlightplanOverview)
    {
        draw.Sphere(aircraft.position, OverviewMarkerMeters, color);
    }
}

/** @brief An expanding, fading fireball at the crash position; `progress` in [0, 1]. */
void DrawCrashExplosion(Draw3d const& draw, Vec3 position, double progress)
{
    constexpr int Particles = 8;
    double const fraction = std::clamp(progress, 0.0, 1.0);
    auto const alpha = static_cast<unsigned>(255.0 * (1.0 - fraction));
    Vec3 const center = flight::Add(position, {0, 0, 6.0 + 20.0 * fraction});
    draw.Sphere(center, 10.0 + 35.0 * fraction, Rgba(238, 75, 18, alpha));
    draw.Sphere(center, 6.0 + 20.0 * fraction, Rgba(255, 196, 55, alpha));
    for (int particle = 0; particle < Particles; ++particle)
    {
        double const angle = 2.0 * flight::Pi * static_cast<double>(particle) / Particles;
        double const radius = 12.0 + 38.0 * fraction;
        Vec3 const at =
            flight::Add(position, {radius * std::cos(angle), radius * std::sin(angle), 10.0 + 25.0 * fraction});
        draw.Sphere(at, 5.0 + 10.0 * fraction, Rgba(255, 125, 20, alpha));
    }
}

/** @brief The sea under the aircraft: a 100 km square at sea level (a backdrop). */
void DrawSea(Draw3d const& draw, Scene const& scene)
{
    constexpr double HalfSideMeters = 50000.0;
    double const north = scene.aircraft.position.north;
    double const east = scene.aircraft.position.east;
    double const up = scene.seaHeight;
    draw.Quad({north - HalfSideMeters, east - HalfSideMeters, up}, {north - HalfSideMeters, east + HalfSideMeters, up},
        {north + HalfSideMeters, east + HalfSideMeters, up}, {north + HalfSideMeters, east - HalfSideMeters, up},
        Rgba(18, 53, 75));
}

// ---------------------------------------------------------------------------------------
// Names shown by the panels.
// ---------------------------------------------------------------------------------------

char const* OutcomeName(flight::Outcome outcome) noexcept
{
    switch (outcome)
    {
    case flight::Outcome::Flying:
        return "In flight";
    case flight::Outcome::Stopped:
        return "Stopped on runway";
    case flight::Outcome::Crash:
        return "Crash";
    case flight::Outcome::Timeout:
        return "Time limit";
    }
    return "Time limit";
}

char const* GuidanceModeName(flight::GuidanceMode mode) noexcept
{
    switch (mode)
    {
    case flight::GuidanceMode::Flare:
        return "Flare";
    case flight::GuidanceMode::Rollout:
        return "Runway rollout";
    case flight::GuidanceMode::Approach:
        return "Approach";
    }
    return "Unknown";
}

char const* AircraftName(world::AircraftKind kind) noexcept
{
    if (kind == world::Pc7)
    {
        return "Pilatus PC-7";
    }
    return "T-6 Texan II";
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

constexpr std::array<char const*, flight::CrashReasonCount> CrashLabels{"Invalid aircraft state", "Distance",
    "Altitude", "Outside landing area", "Wheel outside landing area", "Gear extension", "Structural contact",
    "Sink rate (deck-relative)", "Speed (deck-relative)", "Roll", "Pitch", "Pitch", "Heading", "No wheel contact",
    "Route not completed"};

/** @brief The measured value and limit of a failed crash condition, in its unit, or empty. */
void CrashConditionText(
    flight::CrashReason reason, flight::CrashCondition const& condition, char* text, std::size_t size)
{
    text[0] = '\0';
    switch (reason)
    {
    case flight::CrashReason::Distance:
    case flight::CrashReason::Altitude:
        std::snprintf(text, size, "%.1f / max %.1f m", condition.value, condition.limit);
        break;
    case flight::CrashReason::Sink:
    case flight::CrashReason::Speed:
        std::snprintf(text, size, "%.1f / max %.1f m/s", condition.value, condition.limit);
        break;
    case flight::CrashReason::Gear:
        std::snprintf(text, size, "%.1f%% / min %.1f%%", 100 * condition.value, 100 * condition.limit);
        break;
    case flight::CrashReason::PitchLow:
        std::snprintf(text, size, "%.1f / min %.1f deg", condition.value, condition.limit);
        break;
    case flight::CrashReason::Roll:
    case flight::CrashReason::PitchHigh:
    case flight::CrashReason::Heading:
        std::snprintf(text, size, "%.1f / max %.1f deg", condition.value, condition.limit);
        break;
    default:
        break;
    }
}

// ---------------------------------------------------------------------------------------
// Panel layout: a column of lines and tables, top to bottom.
// ---------------------------------------------------------------------------------------

/** @brief Writes lines and table rows down a box, skipping the first `skip` rows (scrolling). */
class Column final
{
public:
    static constexpr float LineHeight = 18.F;
    static constexpr float FontSize = 15.F;
    /** @brief The wide tables (diagnostics) use a smaller font on shorter rows. */
    static constexpr float CompactLineHeight = 16.F;
    static constexpr float CompactFontSize = 13.F;

    Column(Draw2d const& draw, Box const& box, std::size_t skip) noexcept
        : draw_(draw), box_(box), y_(box.y), skip_(skip)
    {
    }

    /** @brief The rows that follow use the compact font. */
    void Compact() noexcept
    {
        fontSize_ = CompactFontSize;
        lineHeight_ = CompactLineHeight;
    }
    /** @brief One line of text; false when the box is full. */
    bool Line(char const* text, PpokitColor color = TextColor)
    {
        float const y = Next();
        if (y < 0.F)
        {
            return false;
        }
        draw_.Text(text, box_.x, y, fontSize_, color);
        return true;
    }
    /** @brief A formatted line. */
    template<typename... Values>
    bool Format(PpokitColor color, char const* format, Values... values)
    {
        char text[192];
        std::snprintf(text, sizeof(text), format, values...);
        return Line(text, color);
    }
    /** @brief A section title with a rule under it. */
    void Title(char const* text)
    {
        float const y = Next();
        if (y < 0.F)
        {
            return;
        }
        draw_.Text(text, box_.x, y + 1.F, fontSize_, HeaderColor);
        draw_.Line(box_.x, y + lineHeight_ - 1.F, box_.Right(), y + lineHeight_ - 1.F, 1.F, SeparatorColor);
    }
    /** @brief A table row: `cells` at the column offsets, right-aligned after the first. */
    template<std::size_t Columns>
    void Row(std::array<char const*, Columns> const& cells, std::array<float, Columns> const& offsets,
        PpokitColor color = TextColor)
    {
        float const y = Next();
        if (y < 0.F)
        {
            return;
        }
        draw_.Text(cells[0], box_.x + offsets[0], y, fontSize_, color);
        for (std::size_t column = 1; column < Columns; ++column)
        {
            float const right = box_.x + offsets[column];
            draw_.Text(cells[column], right - draw_.Width(cells[column], fontSize_), y, fontSize_, color);
        }
    }
    /** @brief A progress bar the height of a line. */
    void Bar(double fraction, PpokitColor bar, char const* overlay)
    {
        float const y = Next();
        if (y < 0.F)
        {
            return;
        }
        draw_.ProgressBar(
            Box{box_.x, y + 1.F, box_.width, lineHeight_ - 2.F}, fraction, Rgba(26, 38, 48), bar, overlay);
    }
    void Space()
    {
        Next();
    }
    /** @brief Rows written so far, the skipped ones included. */
    std::size_t Rows() const noexcept { return row_; }
    float Width() const noexcept { return box_.width; }

private:
    /** @brief Top of the next visible row, or -1 when it is skipped or the box is full. */
    float Next()
    {
        ++row_;
        if (row_ <= skip_)
        {
            return -1.F;
        }
        if (y_ + lineHeight_ > box_.Bottom())
        {
            return -1.F;
        }
        float const y = y_;
        y_ += lineHeight_;
        return y;
    }

    Draw2d const& draw_;
    Box box_;
    float y_;
    std::size_t skip_;
    std::size_t row_ = 0;
    float fontSize_ = FontSize;
    float lineHeight_ = LineHeight;
};

enum PanelTab
{
    FlightTab,
    ControlsTab,
    DiagnosticsTab,
    CrashTab,
    PanelTabs
};

constexpr std::array<char const*, PanelTabs> TabNames = {"Flight", "Controls", "Diagnostics", "Crash"};

/** @brief Deck-relative sink rate commanded by the guidance, positive down, in m/s. */
double TargetSink(Scene const& scene) noexcept
{
    return -scene.guidance.targetSpeed *
           flight::Dot(scene.guidance.targetDirection, flight::RunwayNormal(scene.runway));
}

/** @brief Phase, outcome, aircraft, run mode, curriculum and guidance summary. */
void DrawStatus(Column& column, Scene const& scene)
{
    MissionView const& mission = scene.mission;
    char const* phase = mission.phase == flight::Phase::Approach ? "Approach" : "Rollout";
    column.Format(TextColor, "%s | %s", phase, OutcomeName(mission.outcome));
    column.Format(TextColor, "Aircraft dynamics: %s", AircraftName(mission.aircraft));
    std::size_t const lastLevel = mission.curriculumLevels > 0 ? mission.curriculumLevels - 1 : 0;
    switch (mission.mode)
    {
    case world::TrainMode:
        column.Format(TextColor, "Training | level %zu / %zu | window %zu / %zu stops", mission.curriculumLevel,
            lastLevel, mission.windowStops, mission.windowEpisodes);
        break;
    case world::EvaluationMode:
        column.Format(TextColor, "Evaluation campaign | difficulty %.2f", mission.difficulty);
        break;
    case world::PlayMode:
        column.Format(TextColor, "Inference | level %zu / %zu | difficulty %.2f", mission.curriculumLevel, lastLevel,
            mission.difficulty);
        break;
    }
    column.Format(TextColor, "Episodes %zu | Touchdowns %zu | Stops %zu", mission.episodes, mission.touchdowns,
        mission.stops);
    column.Format(TextColor, "Crashes %zu | Timeouts %zu", mission.crashes, mission.timeouts);
    column.Line("Validation, BEST, PPO: see PPOKit Studio", DimTextColor);
    column.Format(TextColor, "Guidance: %s", GuidanceModeName(scene.guidance.mode));
    column.Format(TextColor, "Aim point: %.0f m past the threshold", scene.aimAlong);
    column.Format(TextColor, "Sink rate target / actual: %.2f / %.2f m/s", TargetSink(scene),
        -scene.guidance.relativeVelocity.up);
    column.Format(TextColor, "Episode time %.1f / %.0f simulated s", mission.elapsedSeconds, mission.maximumSeconds);
    if (mission.lastOutcome != flight::Outcome::Flying)
    {
        column.Format(TextColor, "Last episode: %s after %.1f simulated s", OutcomeName(mission.lastOutcome),
            mission.lastSeconds);
    }
}

/** @brief Target, actual and error of the airspeed, the slope and the 3D track. */
void DrawTargets(Column& column, Scene const& scene)
{
    flight::Guidance const& guidance = scene.guidance;
    float const width = column.Width();
    std::array<float, 4> const at = {0.F, width * 0.58F, width * 0.78F, width};
    column.Row<4>({"Measure", "Target", "Actual", "Error"}, at, DimTextColor);
    char target[32];
    char actual[32];
    char error[32];
    std::snprintf(target, sizeof(target), "%.1f", guidance.targetSpeed);
    std::snprintf(actual, sizeof(actual), "%.1f", scene.aircraft.airspeed);
    std::snprintf(error, sizeof(error), "%+.1f", scene.aircraft.airspeed - guidance.targetSpeed);
    column.Row<4>({"CAS (m/s)", target, actual, error}, at);
    double const slope = flight::DisplaySlopeDegrees(guidance);
    std::snprintf(target, sizeof(target), "%+.2f", Degrees(guidance.targetSlope));
    std::snprintf(actual, sizeof(actual), "%+.2f", slope);
    std::snprintf(error, sizeof(error), "%+.2f", slope - Degrees(guidance.targetSlope));
    column.Row<4>({"Slope (deg)", target, actual, error}, at);
    double const track = Degrees(std::atan2(guidance.velocityDirection.east, guidance.velocityDirection.north));
    double const directionDot =
        std::clamp(flight::Dot(guidance.targetDirection, guidance.velocityDirection), -1.0, 1.0);
    std::snprintf(target, sizeof(target), "%+.1f", Degrees(guidance.targetHeading));
    std::snprintf(actual, sizeof(actual), "%+.1f", track);
    std::snprintf(error, sizeof(error), "%.1f 3D", Degrees(std::acos(directionDot)));
    column.Row<4>({"Track azimuth (deg)", target, actual, error}, at);
}

/** @brief Deck speed, swell and a heave gauge from the trough to the crest of the swell. */
void DrawDeckMotion(Column& column, Scene const& scene)
{
    flight::DeckState const& deck = scene.deck;
    if (!deck.motion.Moving())
    {
        column.Line("Deck: static", DimTextColor);
        return;
    }
    column.Title("Deck motion");
    column.Format(TextColor, "Deck speed: %.1f m/s along the runway", deck.motion.speedMps);
    column.Format(TextColor, "Deck vertical speed: %+.2f m/s", scene.runway.velocity.up);
    double const amplitude = deck.motion.swellAmplitudeMeters;
    if (amplitude <= 0)
    {
        return;
    }
    column.Format(TextColor, "Swell: +/-%.2f m, period %.1f s, wavelength %.0f m", amplitude,
        deck.motion.swellPeriodSeconds, SwellWavelengthMeters(deck.motion.swellPeriodSeconds));
    // The gauge is empty at the trough and full at the crest.
    char label[32];
    std::snprintf(label, sizeof(label), "heave %+.2f m", deck.heaveMeters);
    column.Bar(0.5 + 0.5 * deck.heaveMeters / amplitude, Rgba(66, 135, 190), label);
}

/** @brief Runway dimensions and the aircraft position relative to it. */
void DrawRunwayPanel(Column& column, Scene const& scene)
{
    flight::Runway const& runway = scene.runway;
    column.Line("Slope: + climb / - descent (deck-relative)", DimTextColor);
    column.Format(TextColor, "Runway dimensions: %.0f x %.0f m", runway.lengthMeters, runway.widthMeters);
    column.Format(TextColor, "Runway threshold above sea: %.1f m", runway.origin.up - scene.seaHeight);
    if (runway.landingMarginMeters > 0)
    {
        column.Format(TextColor, "Landing area margin: %.0f m", runway.landingMarginMeters);
    }
    column.Format(TextColor, "Runway %.2f km | Height %.0f m", scene.guidance.distance / 1000, scene.guidance.height);
    column.Format(
        TextColor, "Runway lateral: %+.0f m (%s)", scene.guidance.cross, LateralSideName(scene.guidance.cross));
    DrawDeckMotion(column, scene);
}

/** @brief Attitude, gear and the touchdown of the current (or last) episode. */
void DrawLandingDetails(Column& column, Scene const& scene)
{
    flight::AircraftState const& aircraft = scene.aircraft;
    LandingView const& landing = scene.landing;
    column.Title("Aircraft / landing details");
    column.Format(TextColor, "Gear extension %.0f%% | WOW %s", aircraft.gearPosition * 100,
        aircraft.weightOnWheels ? "YES" : "NO");
    column.Format(TextColor, "Roll %.1f | Pitch %.1f | Heading %.1f deg", Degrees(aircraft.roll),
        Degrees(aircraft.pitch), Degrees(aircraft.heading));
    column.Format(TextColor, "Relative sink %.2f m/s", -scene.guidance.relativeVelocity.up);
    if (landing.touchedDown)
    {
        column.Format(TextColor, "Touchdown %.1f m | Ground roll %.1f m", landing.touchdownFromThreshold,
            landing.groundRoll);
        column.Format(TextColor, "Sink %.2f m/s | Speed %.1f m/s", landing.touchdownSink, landing.touchdownSpeed);
        column.Format(TextColor, "Touchdown quality %.2f | Landing area factor %.2f", landing.touchdownQuality,
            landing.landingAreaFactor);
    }
    if (landing.outcome == flight::Outcome::Stopped)
    {
        column.Format(TextColor, "Stop %.1f m from threshold", landing.stopFromThreshold);
    }
}

/** @brief The action each axis received and the control applied after the command filter. */
void DrawControls(Column& column, Scene const& scene)
{
    float const width = column.Width();
    std::array<float, 3> const at = {0.F, width * 0.62F, width};
    // ppokit-run samples the action while training and plays the network mean otherwise.
    char const* const received = scene.mission.mode == world::TrainMode ? "Sampled raw" : "Network mean";
    column.Row<3>({"Control", received, "Applied"}, at, DimTextColor);
    std::array<double, flight::ContinuousActions> const applied = flight::Applied(scene.mission.applied);
    for (std::size_t axis = 0; axis < flight::ContinuousActions; ++axis)
    {
        char action[32];
        char physical[32];
        std::snprintf(action, sizeof(action), "%+.3f", scene.action[axis]);
        std::snprintf(physical, sizeof(physical), "%+.3f", applied[axis]);
        column.Row<3>({flight::ActionNames[axis], action, physical}, at);
    }
    char gear[32];
    std::snprintf(gear, sizeof(gear), "%.0f", scene.action[flight::ContinuousActions]);
    char lever[32];
    std::snprintf(lever, sizeof(lever), "%.0f", scene.mission.applied.gear);
    column.Row<3>({"gear", gear, lever}, at);
    char flap[32];
    std::snprintf(flap, sizeof(flap), "%.2f", scene.mission.applied.flap);
    column.Row<3>({"flap (phase)", "-", flap}, at);
}

/** @brief Last-step and window-mean value of each reward channel, then the episode bars. */
void DrawRewards(Column& column, Scene const& scene)
{
    float const width = column.Width();
    std::array<float, 3> const at = {0.F, width * 0.6F, width};
    column.Row<3>({"Reward", "Last step", "Last episode mean"}, at, DimTextColor);
    bool const window = scene.diagnostics.enabled && scene.diagnostics.samples > 0;
    for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
    {
        char step[32];
        char mean[32] = "-";
        std::snprintf(step, sizeof(step), "%+.5f", scene.stepRewards[channel]);
        if (window)
        {
            std::snprintf(mean, sizeof(mean), "%+.5f", scene.diagnostics.rewardMeans[channel]);
        }
        column.Row<3>({flight::RewardNames[channel], step, mean}, at);
    }
    column.Title("Episode rewards");
    double const stop = scene.weights[static_cast<std::size_t>(flight::Reward::Stop)];
    double const crash = scene.weights[static_cast<std::size_t>(flight::Reward::Crash)];
    for (std::size_t channel = 0; channel < flight::RewardCount; ++channel)
    {
        double const episode = scene.episodeSums[channel];
        char overlay[96];
        std::snprintf(overlay, sizeof(overlay), "%s  %+.2f", flight::RewardNames[channel], episode);
        // Red for a net penalty, green for a net reward.
        PpokitColor const bar = episode < 0 ? Rgba(184, 64, 61) : Rgba(41, 148, 110);
        column.Bar(flight::RewardProgressFraction(episode, stop, crash), bar, overlay);
    }
}

/** @brief Raw and applied magnitudes and saturations of each action over the window. */
void DrawActionDiagnostics(Column& column, DiagnosticsView const& window)
{
    float const width = column.Width();
    std::array<float, 5> const at = {0.F, width * 0.40F, width * 0.60F, width * 0.80F, width};
    column.Row<5>({"Action", "Raw |x|", "Appl. |x|", "Raw sat", "Appl. sat"}, at, DimTextColor);
    for (std::size_t axis = 0; axis < flight::ContinuousActions; ++axis)
    {
        char raw[24];
        char applied[24];
        char rawSaturated[24];
        char appliedSaturated[24];
        std::snprintf(raw, sizeof(raw), "%.3f", window.rawMean[axis]);
        std::snprintf(applied, sizeof(applied), "%.3f", window.appliedMean[axis]);
        std::snprintf(
            rawSaturated, sizeof(rawSaturated), "%llu", static_cast<unsigned long long>(window.rawSaturations[axis]));
        std::snprintf(appliedSaturated, sizeof(appliedSaturated), "%llu",
            static_cast<unsigned long long>(window.appliedSaturations[axis]));
        column.Row<5>({flight::ActionNames[axis], raw, applied, rawSaturated, appliedSaturated}, at);
    }
}

/** @brief Live value, window mean, maximum and saturation count of each observation slot. */
void DrawObservationDiagnostics(Column& column, DiagnosticsView const& window)
{
    float const width = column.Width();
    std::array<float, 4> const at = {0.F, width * 0.70F, width * 0.79F, width};
    column.Row<4>({"Observation", "Now", "Mean |x|", "Max |x| / sat"}, at, DimTextColor);
    for (std::size_t slot = 0; slot < flight::ObservationSize; ++slot)
    {
        char now[24];
        char mean[24];
        char maximum[40];
        std::snprintf(now, sizeof(now), "%+.3f", window.liveObservation[slot]);
        std::snprintf(mean, sizeof(mean), "%.3f", window.observationMean[slot]);
        std::snprintf(maximum, sizeof(maximum), "%.3f / %llu", window.observationMaximum[slot],
            static_cast<unsigned long long>(window.observationSaturations[slot]));
        column.Row<4>({flight::ObservationNames[slot], now, mean, maximum}, at);
    }
}

void DrawDiagnostics(Column& column, Scene const& scene)
{
    DiagnosticsView const& window = scene.diagnostics;
    if (!window.enabled)
    {
        column.Line("Network diagnostics disabled in the scenario configuration.", DimTextColor);
        return;
    }
    std::size_t saturatedInputs = 0;
    for (std::uint64_t const count : window.observationSaturations)
    {
        saturatedInputs += count > 0 ? 1U : 0U;
    }
    std::uint64_t saturatedActions = 0;
    for (std::uint64_t const count : window.appliedSaturations)
    {
        saturatedActions += count;
    }
    if (window.samples == 0)
    {
        column.Line("Window: the last complete episode (none yet)", DimTextColor);
    }
    else
    {
        column.Format(DimTextColor, "Window: the last episode, %llu actions",
            static_cast<unsigned long long>(window.samples));
    }
    column.Format(TextColor, "Saturated (%.2f) inputs %zu/%zu | applied hits %llu", window.saturationThreshold,
        saturatedInputs, flight::ObservationSize, static_cast<unsigned long long>(saturatedActions));
    column.Line("KL and critic statistics: PPOKit Studio", DimTextColor);
    column.Compact();
    if (window.samples > 0)
    {
        DrawActionDiagnostics(column, window);
        column.Space();
    }
    DrawObservationDiagnostics(column, window);
}

/** @brief The most recent crash conditions and their measured limits. */
void DrawCrashDetails(Column& column, Scene const& scene)
{
    column.Title("Last crash reasons");
    if (!scene.crash.has)
    {
        column.Line("No crash recorded", DimTextColor);
        return;
    }
    for (std::size_t index = 0; index < flight::CrashReasonCount; ++index)
    {
        flight::CrashCondition const& condition = scene.crash.diagnostic.conditions[index];
        if (!condition.failed)
        {
            continue;
        }
        column.Line(CrashLabels[index], CrashTextColor);
        char text[96];
        CrashConditionText(static_cast<flight::CrashReason>(index), condition, text, sizeof(text));
        if (text[0] != '\0')
        {
            column.Line(text);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Approach profile (ported from the desktop renderer, flight_profile_view.cpp).
// ---------------------------------------------------------------------------------------

/** @brief Maps runway-relative metres (along, height) to a pixel box. */
struct ProfileCanvas final
{
    Box box;
    double alongMin{};
    double alongMax{};
    double heightMin{};
    double heightMax{};

    void Point(double along, double height, float& x, float& y) const noexcept
    {
        double const alongFraction = (along - alongMin) / (alongMax - alongMin);
        double const heightFraction = (height - heightMin) / (heightMax - heightMin);
        x = box.x + static_cast<float>(alongFraction) * box.width;
        y = box.Bottom() - static_cast<float>(heightFraction) * box.height;
    }
};

/** @brief A segment between two runway-relative points, clipped to the profile. */
void ProfileSegment(Draw2d const& draw, ProfileCanvas const& canvas, double along0, double height0, double along1,
    double height1, float thickness, PpokitColor color)
{
    float x0 = 0.F;
    float y0 = 0.F;
    float x1 = 0.F;
    float y1 = 0.F;
    canvas.Point(along0, height0, x0, y0);
    canvas.Point(along1, height1, x1, y1);
    if (ClipSegment(canvas.box, x0, y0, x1, y1))
    {
        draw.Line(x0, y0, x1, y1, thickness, color);
    }
}

/** @brief A labelled tick at `along`, inside the plot. */
void ProfileTick(Draw2d const& draw, ProfileCanvas const& canvas, double along)
{
    float x = 0.F;
    float y = 0.F;
    canvas.Point(along, canvas.heightMin, x, y);
    if (x < canvas.box.x || x > canvas.box.Right())
    {
        return;
    }
    draw.Line(x, canvas.box.y, x, canvas.box.Bottom(), 1.F, Rgba(73, 101, 123, 100));
    char label[32];
    std::snprintf(label, sizeof(label), "%.0f m", along);
    draw.Text(label, x + 3.F, canvas.box.Bottom() - 16.F, 13.F, Rgba(167, 188, 199));
}

/** @brief A labelled point, when it lies in the plot. */
void ProfileMarker(Draw2d const& draw, ProfileCanvas const& canvas, double along, double height, PpokitColor color,
    char const* label)
{
    float x = 0.F;
    float y = 0.F;
    canvas.Point(along, height, x, y);
    if (!canvas.box.Contains(x, y))
    {
        return;
    }
    draw.Circle(x, y, 4.F, true, color);
    draw.Text(label, x + 5.F, y - 15.F, 13.F, color);
}

/** @brief The runway-relative vertical approach profile in `area`. */
void DrawApproachProfile(Draw2d const& draw, Box const& area, Scene const& scene)
{
    constexpr double MinimumViewStartMeters = -2500.0;
    constexpr double VelocityProjectionSeconds = 4;
    constexpr PpokitColor GlidePlaneColor = Rgba(92, 208, 166);
    constexpr PpokitColor ThresholdColor = Rgba(245, 192, 85);
    draw.Fill(area, OpaquePanelColor);
    float const left = area.x + 10.F;
    draw.Text("Approach profile | runway-relative", left, area.y + 4.F, 15.F, HeaderColor);
    flight::Guidance const& guidance = scene.guidance;
    bool const valid =
        flight::IsFinite(scene.aircraft) && std::isfinite(guidance.along) && std::isfinite(guidance.height);
    if (!valid)
    {
        draw.Text("Approach profile unavailable for this flight.", left, area.y + 24.F, 14.F, DimTextColor);
        return;
    }
    // Only the runway and the glide slope are needed by the glide-plane helpers.
    flight::Config reference{};
    reference.runway = scene.runway;
    reference.mission.glideDegrees = scene.glideDegrees;
    double const aimAlong = flight::AimPointAlongMeters(reference);
    double const along = guidance.along;
    // Horizontal extent: from 2.5 km out (or further, around the aircraft) to past the aim point.
    double const startAlong = std::min(MinimumViewStartMeters, along - std::max(100.0, std::abs(along) * 0.1));
    double const endAlong = std::max(aimAlong + 400.0, along + 100.0);
    double const aircraftPathHeight = flight::ApproachPathHeight(reference, along);
    double const startHeight = flight::ApproachPathHeight(reference, startAlong);
    char text[192];
    std::snprintf(text, sizeof(text), "Height %.1f m | path error %+.1f m | sink %.2f / target %.2f m/s",
        guidance.height, guidance.height - aircraftPathHeight, -guidance.relativeVelocity.up, TargetSink(scene));
    draw.Text(text, left, area.y + 22.F, 14.F, TextColor);
    draw.Text("Green: glide plane to the aim point | Blue: aircraft and 4 s velocity | Amber: altitude error", left,
        area.y + 39.F, 13.F, DimTextColor);
    Box const plot{area.x + 10.F, area.y + 58.F, area.width - 20.F, area.height - 64.F};
    if (plot.width < 80.F || plot.height < 35.F)
    {
        return;
    }
    draw.Fill(plot, Rgba(12, 24, 35));
    // Vertical extent: the deck, the aircraft and the glide plane at the left edge, with 12 % padding.
    double const maxHeight = std::max({10.0, guidance.height, startHeight});
    double const minHeight = std::min(0.0, guidance.height);
    double const padding = std::max(5.0, (maxHeight - minHeight) * 0.12);
    ProfileCanvas const canvas{plot, startAlong, endAlong, minHeight - padding, maxHeight + padding};
    ProfileTick(draw, canvas, 0);
    ProfileTick(draw, canvas, aimAlong);
    ProfileSegment(draw, canvas, 0, 0, std::min(endAlong, scene.runway.lengthMeters), 0, 4.F, Rgba(180, 194, 194));
    // The glide plane is straight: it crosses the threshold at its crossing height and meets the aim point.
    ProfileSegment(draw, canvas, startAlong, startHeight, aimAlong, 0, 2.F, GlidePlaneColor);
    ProfileMarker(draw, canvas, 0, scene.runway.thresholdHeightMeters, ThresholdColor, "Threshold");
    ProfileMarker(draw, canvas, aimAlong, 0, GlidePlaneColor, "Aim point");
    ProfileSegment(draw, canvas, along, aircraftPathHeight, along, guidance.height, 1.5F, Rgba(255, 181, 93, 220));
    double const projectedAlong = along + guidance.relativeVelocity.north * VelocityProjectionSeconds;
    double const projectedHeight = guidance.height + guidance.relativeVelocity.up * VelocityProjectionSeconds;
    ProfileSegment(
        draw, canvas, along, guidance.height, projectedAlong, projectedHeight, 1.5F, Rgba(119, 190, 255, 210));
    float x = 0.F;
    float y = 0.F;
    canvas.Point(along, guidance.height, x, y);
    if (plot.Contains(x, y))
    {
        draw.Triangle(x + 7.F, y, x - 6.F, y - 4.F, x - 6.F, y + 4.F, Rgba(133, 208, 255));
    }
}

// ---------------------------------------------------------------------------------------
// Labels of the scene.
// ---------------------------------------------------------------------------------------

/** @brief Where the window draws: its size, the 3D scene's area, the panel and the profile. */
struct Layout final
{
    float width{};
    float height{};
    Box scene;
    Box panel;
    Box profile;
    std::array<Box, CameraModes> cameraButtons{};
    std::array<Box, PanelTabs> tabs{};
    Box panelBody;
};

/** @brief The layout of a window of `width` x `height` pixels. */
Layout LayoutOf(float width, float height) noexcept
{
    constexpr float PanelWidth = 470.F;
    constexpr float ProfileHeight = 220.F;
    constexpr float StatusLine = 26.F;
    constexpr float TabHeight = 26.F;
    Layout layout;
    layout.width = width;
    layout.height = height;
    float const panelWidth = std::min(PanelWidth, width * 0.45F);
    float const bottom = std::max(0.F, height - StatusLine);
    layout.panel = Box{width - panelWidth, 0.F, panelWidth, bottom};
    float const profileHeight = std::min(ProfileHeight, bottom * 0.4F);
    layout.profile = Box{0.F, bottom - profileHeight, width - panelWidth, profileHeight};
    layout.scene = Box{0.F, 0.F, width - panelWidth, bottom - profileHeight};
    float x = 10.F;
    for (std::size_t mode = 0; mode < CameraModes; ++mode)
    {
        float const buttonWidth = mode == FlightplanOverview ? 170.F : 125.F;
        layout.cameraButtons[mode] = Box{x, 10.F, buttonWidth, 24.F};
        x += buttonWidth + 6.F;
    }
    float const tabWidth = (panelWidth - 20.F) / PanelTabs;
    for (std::size_t tab = 0; tab < PanelTabs; ++tab)
    {
        layout.tabs[tab] = Box{layout.panel.x + 10.F + tab * tabWidth, 8.F, tabWidth - 4.F, TabHeight};
    }
    layout.panelBody = Box{layout.panel.x + 12.F, 8.F + TabHeight + 10.F, panelWidth - 24.F, bottom - TabHeight - 26.F};
    return layout;
}

/** @brief The physical threshold, labelled where it projects into the scene. */
void DrawThresholdLabel(
    Draw2d const& draw, Layout const& layout, ViewCamera const& camera, flight::Runway const& runway)
{
    constexpr float FontSize = 16.F;
    float x = 0.F;
    float y = 0.F;
    if (!Project(camera, flight::ThresholdPoint(runway), layout.width, layout.height, layout.scene, x, y))
    {
        return;
    }
    draw.Circle(x, y, 8.F, false, ThresholdLabelColor);
    draw.Line(x - 12.F, y, x - 5.F, y, 1.F, ThresholdLabelColor);
    draw.Line(x + 5.F, y, x + 12.F, y, 1.F, ThresholdLabelColor);
    char const label[] = "THRESHOLD";
    float const labelWidth = draw.Width(label, FontSize);
    float const textX =
        std::clamp(x + 12.F, layout.scene.x, std::max(layout.scene.x, layout.scene.Right() - labelWidth));
    float const textY =
        std::clamp(y - 20.F, layout.scene.y, std::max(layout.scene.y, layout.scene.Bottom() - FontSize));
    draw.Fill(Box{textX, textY, labelWidth, FontSize}, Rgba(14, 27, 44, 230));
    draw.Text(label, textX, textY, FontSize, ThresholdLabelColor);
}

/** @brief A compact instrument tag joined to the projected aircraft. */
void DrawAircraftTag(Draw2d const& draw, Layout const& layout, ViewCamera const& camera, Scene const& scene)
{
    constexpr float Width = 190.F;
    constexpr float Height = 146.F;
    constexpr float Margin = 8.F;
    constexpr float FontSize = 16.F;
    constexpr float RowHeight = 21.F;
    constexpr float TagOffset = 28.F;
    Box const& area = layout.scene;
    double const slope = flight::DisplaySlopeDegrees(scene.guidance);
    if (area.width < Width + 2 * Margin || area.height < Height + 2 * Margin || !flight::IsFinite(scene.aircraft) ||
        !std::isfinite(scene.guidance.height) || !std::isfinite(slope) || !std::isfinite(scene.runway.elevationMeters))
    {
        return;
    }
    float x = 0.F;
    float y = 0.F;
    if (!Project(camera, scene.aircraft.position, layout.width, layout.height, area, x, y))
    {
        return;
    }
    // Right of the aircraft when it fits, else left; the connector meets the nearest edge.
    bool const fitsRight = x + TagOffset + Width + Margin <= area.Right();
    float const preferredX = fitsRight ? x + TagOffset : x - TagOffset - Width;
    float const panelX = std::clamp(preferredX, area.x + Margin, area.Right() - Width - Margin);
    float const panelY = std::clamp(y - Height / 2, area.y + Margin, area.Bottom() - Height - Margin);
    float const edgeX = panelX > x ? panelX : panelX + Width;
    float const edgeY = std::clamp(y, panelY + 12.F, panelY + Height - 12.F);
    PpokitColor const accent = scene.mission.outcome == flight::Outcome::Crash ? Rgba(255, 112, 105) : HeaderColor;
    draw.Line(x, y, edgeX, edgeY, 1.F, accent);
    draw.Circle(x, y, 5.F, false, accent);
    draw.Fill(Box{panelX, panelY, Width, Height}, PanelColor);
    draw.Fill(Box{panelX, panelY + 10.F, 3.F, 18.F}, accent);
    draw.Text("AIRCRAFT", panelX + 12.F, panelY + 10.F, 15.F, accent);
    draw.Line(panelX + 12.F, panelY + 31.F, panelX + Width - 12.F, panelY + 31.F, 1.F, SeparatorColor);
    double heading = std::fmod(Degrees(scene.aircraft.heading), 360.0);
    if (heading < 0)
    {
        heading += 360.0;
    }
    int const roundedHeading = static_cast<int>(std::floor(heading + 0.5)) % 360;
    char values[5][40];
    // The position is in the runway's local tangent frame: the altitude is approximate away from its origin.
    std::snprintf(values[0], sizeof(values[0]), "~%.0f m", scene.runway.elevationMeters + scene.aircraft.position.up);
    std::snprintf(values[1], sizeof(values[1]), "%.0f m", scene.guidance.height);
    std::snprintf(values[2], sizeof(values[2]), "%.1f m/s", scene.aircraft.airspeed);
    std::snprintf(values[3], sizeof(values[3]), "%03d deg", roundedHeading);
    std::snprintf(values[4], sizeof(values[4]), "%+.1f deg", slope);
    constexpr std::array<char const*, 5> labels{"ALT", "H / RWY", "SPEED", "HEADING", "SLOPE"};
    for (std::size_t row = 0; row < labels.size(); ++row)
    {
        float const rowY = panelY + 39.F + static_cast<float>(row) * RowHeight;
        draw.Text(labels[row], panelX + 12.F, rowY, FontSize, DimTextColor);
        draw.Text(values[row], panelX + Width - 12.F - draw.Width(values[row], FontSize), rowY, FontSize, TextColor);
    }
}

/** @brief Deck speed, swell and present heave, at the bottom left of the scene. */
void DrawDeckBanner(Draw2d const& draw, Layout const& layout, flight::DeckState const& deck)
{
    constexpr float FontSize = 16.F;
    constexpr double MetresPerSecondToKnots = 3600.0 / 1852.0;
    if (!deck.motion.Moving())
    {
        return;
    }
    char text[160];
    std::snprintf(text, sizeof(text), "DECK %.1f m/s (%.0f kt)   SWELL +/-%.1f m, %.0f s   HEAVE %+.2f m",
        deck.motion.speedMps, deck.motion.speedMps * MetresPerSecondToKnots, deck.motion.swellAmplitudeMeters,
        deck.motion.swellPeriodSeconds, deck.heaveMeters);
    float const width = draw.Width(text, FontSize);
    float const y = layout.scene.Bottom() - FontSize - 12.F;
    draw.Fill(Box{10.F, y - 4.F, width + 12.F, FontSize + 8.F}, Rgba(14, 27, 44, 220));
    draw.Text(text, 16.F, y, FontSize, WakeColor);
}

/** @brief A clickable label, highlighted when selected. */
void DrawButton(Draw2d const& draw, Box const& box, char const* label, bool selected, PpokitColor accent)
{
    draw.Fill(box, selected ? WithAlpha(accent, 200) : PanelColor);
    draw.Outline(box, selected ? accent : SeparatorColor);
    constexpr float FontSize = 15.F;
    float const width = draw.Width(label, FontSize);
    draw.Text(label, box.x + (box.width - width) / 2.F, box.y + (box.height - FontSize) / 2.F, FontSize,
        selected ? BackgroundColor : TextColor);
}

// ---------------------------------------------------------------------------------------
// The view.
// ---------------------------------------------------------------------------------------

class FlightView final
{
public:
    static char const* Name() { return "flight"; }

    explicit FlightView(std::string const&) {}

    void Receive(ppokit::view::Frame const& frame)
    {
        if (frame.Find("mission") == nullptr)
        {
            return;
        }
        scene_ = Decode(frame);
        received_ = true;
        if (scene_.crash.event != shownCrashEvent_)
        {
            // A new crash freezes the scene on its pose for a short animation.
            shownCrashEvent_ = scene_.crash.event;
            crashShownAt_ = std::chrono::steady_clock::now();
            crashAnimating_ = scene_.crash.event != 0;
        }
    }

    void Input(PpokitViewInput const& input)
    {
        Layout const layout = LayoutOf(width_, height_);
        bool const down = (input.buttons & 1U) != 0U;
        bool const pressed = down && !leftDown_;
        leftDown_ = down;
        if (pressed)
        {
            Click(layout, input.mouseX, input.mouseY);
        }
        if (!down)
        {
            orbiting_ = false;
        }
        if (orbiting_ && camera_ == AircraftChase)
        {
            orbitYaw_ += input.dragX * 0.01;
            orbitPitch_ = std::clamp(orbitPitch_ - input.dragY * 0.01, -1.2, 1.2);
        }
        if ((input.buttons & 2U) != 0U)
        {
            // The right button brings the chase camera back behind the aircraft.
            orbitYaw_ = 0.0;
            orbitPitch_ = 0.0;
        }
        if (input.wheel != 0.F)
        {
            Scroll(layout, input.mouseX, input.mouseY, input.wheel);
        }
    }

    void Render(ppokit::view::Canvas const& canvas)
    {
        width_ = canvas.Width();
        height_ = canvas.Height();
        PpokitDraw const& raw = canvas.Raw();
        Draw2d const draw(raw);
        canvas.Clear(BackgroundColor);
        if (!received_)
        {
            draw.Text("waiting for the FlightControl3D Visualiseur (flight_watch@2)", 20.F, 20.F, 22.F, TextColor);
            return;
        }
        Layout const layout = LayoutOf(width_, height_);
        double const crashAge = std::chrono::duration<double>(std::chrono::steady_clock::now() - crashShownAt_).count();
        bool const showCrash = crashAnimating_ && crashAge >= 0.0 && crashAge < CrashAnimationSeconds;
        Scene const frame = showCrash ? CrashFrame() : scene_;
        ViewCamera const camera = CameraFor(frame);
        DrawScene(raw, frame, camera, showCrash ? crashAge / CrashAnimationSeconds : -1.0);
        DrawThresholdLabel(draw, layout, camera, frame.runway);
        DrawDeckBanner(draw, layout, frame.deck);
        if (showCrash)
        {
            draw.Text("CRASH", 16.F, 64.F, 26.F, Rgba(255, 140, 54));
        }
        else
        {
            DrawAircraftTag(draw, layout, camera, frame);
        }
        DrawCameraButtons(draw, layout);
        DrawApproachProfile(draw, layout.profile, scene_);
        DrawPanel(draw, layout);
    }

private:
    static constexpr double CrashAnimationSeconds = 1.5;
    static constexpr double MinimumChaseMeters = 25.0;
    static constexpr double MaximumChaseMeters = 300.0;

    /** @brief The scene at the crash: its pose, red, without guidance. */
    Scene CrashFrame() const
    {
        Scene frame = scene_;
        frame.aircraft = scene_.crash.aircraft;
        frame.runway = scene_.crash.runway;
        frame.mission.outcome = flight::Outcome::Crash;
        frame.guidance.targetValid = false;
        return frame;
    }

    ViewCamera CameraFor(Scene const& frame) const noexcept
    {
        switch (camera_)
        {
        case RunwayEnd:
            return RunwayCamera(frame);
        case FlightplanOverview:
            return OverviewCamera(frame);
        default:
            return ChaseCamera(frame, chaseDistance_, orbitYaw_, orbitPitch_);
        }
    }

    void DrawScene(PpokitDraw const& raw, Scene const& frame, ViewCamera const& camera, double explosion) const
    {
        Draw3d const draw(raw);
        raw.begin3d(raw.context, camera.position.data(), camera.target.data(), camera.up.data(), camera.fovy);
        DrawSea(draw, frame);
        DrawSwell(draw, frame.runway, frame.deck, frame.seaHeight);
        DrawWake(draw, frame.runway, frame.deck, frame.seaHeight);
        DrawLandingAreaSides(draw, frame.runway, frame.seaHeight);
        DrawLandingArea(draw, frame.runway);
        DrawRunway(draw, frame.runway);
        DrawGuidance(draw, frame);
        if (flight::IsFinite(frame.aircraft))
        {
            DrawAircraft(draw, frame, camera_);
        }
        if (explosion >= 0.0)
        {
            DrawCrashExplosion(draw, frame.aircraft.position, explosion);
        }
        raw.end3d(raw.context);
    }

    void DrawCameraButtons(Draw2d const& draw, Layout const& layout) const
    {
        for (std::size_t mode = 0; mode < CameraModes; ++mode)
        {
            DrawButton(draw, layout.cameraButtons[mode], CameraNames[mode], camera_ == static_cast<CameraMode>(mode),
                HeaderColor);
        }
        if (camera_ == AircraftChase)
        {
            char text[96];
            std::snprintf(
                text, sizeof(text), "Distance %.0f m (wheel) | drag: turn | right click: behind", chaseDistance_);
            Box const& first = layout.cameraButtons[0];
            draw.Text(text, first.x, first.Bottom() + 6.F, 14.F, DimTextColor);
        }
    }

    void DrawPanel(Draw2d const& draw, Layout const& layout)
    {
        draw.Fill(layout.panel, OpaquePanelColor);
        for (std::size_t tab = 0; tab < PanelTabs; ++tab)
        {
            bool const crashTab = tab == CrashTab && scene_.crash.has;
            PpokitColor const accent = crashTab ? CrashTextColor : HeaderColor;
            DrawButton(draw, layout.tabs[tab], TabNames[tab], tab_ == static_cast<PanelTab>(tab), accent);
        }
        Column column(draw, layout.panelBody, scroll_);
        switch (tab_)
        {
        case ControlsTab:
            column.Title("Controls");
            DrawControls(column, scene_);
            column.Space();
            DrawRewards(column, scene_);
            break;
        case DiagnosticsTab:
            column.Title("Convergence diagnostics");
            DrawDiagnostics(column, scene_);
            break;
        case CrashTab:
            DrawCrashDetails(column, scene_);
            break;
        default:
            DrawStatus(column, scene_);
            column.Space();
            DrawTargets(column, scene_);
            column.Space();
            DrawRunwayPanel(column, scene_);
            column.Space();
            DrawLandingDetails(column, scene_);
            break;
        }
        // Scrolling stops once the last row is in view.
        rows_ = column.Rows();
    }

    void Click(Layout const& layout, float x, float y)
    {
        for (std::size_t mode = 0; mode < CameraModes; ++mode)
        {
            if (layout.cameraButtons[mode].Contains(x, y))
            {
                camera_ = static_cast<CameraMode>(mode);
                return;
            }
        }
        for (std::size_t tab = 0; tab < PanelTabs; ++tab)
        {
            if (layout.tabs[tab].Contains(x, y))
            {
                tab_ = static_cast<PanelTab>(tab);
                scroll_ = 0;
                return;
            }
        }
        orbiting_ = layout.scene.Contains(x, y);
    }

    void Scroll(Layout const& layout, float x, float y, float wheel)
    {
        if (layout.panel.Contains(x, y))
        {
            auto const visible = static_cast<std::size_t>(layout.panelBody.height / Column::LineHeight);
            std::size_t const last = rows_ > visible ? rows_ - visible : 0;
            int const step = wheel > 0.F ? -3 : 3;
            long const next = static_cast<long>(scroll_) + step;
            scroll_ = static_cast<std::size_t>(std::clamp<long>(next, 0, static_cast<long>(last)));
            return;
        }
        if (layout.scene.Contains(x, y) && camera_ == AircraftChase)
        {
            chaseDistance_ = std::clamp(chaseDistance_ * (1.0 - 0.1 * wheel), MinimumChaseMeters, MaximumChaseMeters);
        }
    }

    Scene scene_{};
    bool received_ = false;
    float width_ = 1280.F;
    float height_ = 720.F;
    CameraMode camera_ = AircraftChase;
    double chaseDistance_ = 80.0;
    double orbitYaw_ = 0.0;
    double orbitPitch_ = 0.0;
    bool leftDown_ = false;
    bool orbiting_ = false;
    PanelTab tab_ = FlightTab;
    std::size_t scroll_ = 0;
    std::size_t rows_ = 0;
    std::uint64_t shownCrashEvent_ = 0;
    std::chrono::steady_clock::time_point crashShownAt_{};
    bool crashAnimating_ = false;
};

} // namespace

PPOKIT_COMPONENTS(FlightWatch)
PPOKIT_VIEWS(FlightView)
