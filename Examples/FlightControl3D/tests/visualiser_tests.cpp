/**
 * @file Examples/FlightControl3D/tests/visualiser_tests.cpp
 * @brief The FlightControl3D Visualiseur end to end, and what only training moves.
 * @details
 * - ppokit-run plays T6's policy in a copy of the project whose flight_watch sends to this
 *   test's port; the frames arrive in UDP and FlightView draws the last one on a recording
 *   canvas: the 3D scene (runway surfaces), the profile, and each tab of the panel, chosen
 *   by clicks as a user would.
 * - FlightView on known frames: the values the desktop instruments showed (targets, slope,
 *   episode clock, deck motion, sink rates, profile, reward bars, crash reasons), written
 *   as the panels write them.
 * - The environment DLL moves the curriculum only while it learns: an airborne episode
 *   played (set_learning 0) leaves the promotion window as it is, a trained one counts.
 */
#include "flight_world.hpp"

#include "ppokit/environment/environment_library.hpp"
#include "ppokit/view/frames.hpp"
#include "ppokit/view/view_library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

namespace fs = std::filesystem;
namespace world = ppo::examples::flight::world;
using Json = nlohmann::json;
/** @brief network.json keeps its order: the order of the blocks is part of the networks' identity. */
using OrderedJson = nlohmann::ordered_json;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

Json ReadJson(fs::path const& path)
{
    std::ifstream stream(path);
    return Json::parse(stream);
}

template<typename Document>
void WriteJson(fs::path const& path, Document const& json)
{
    std::ofstream(path) << json.dump(2) << '\n';
}

std::string ReadText(fs::path const& path)
{
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

/** @brief What the view drew: texts, and how many 3D triangles and 2D triangles. */
struct Recording final
{
    std::vector<std::string> texts;
    std::size_t triangles3d = 0;
    std::size_t triangles2d = 0;
    std::size_t scenes = 0;
    std::size_t modelLoads = 0;
    std::size_t modelsDrawn = 0;
    std::size_t blueAircraftTriangles = 0;
    std::size_t redAircraftTriangles = 0;
    std::vector<std::array<float, 3>> blueAircraftVertices;
};

Recording& Of(void* context)
{
    return *static_cast<Recording*>(context);
}

void RecordClear(void*, PpokitColor) {}
void RecordLine(void*, float, float, float, float, float, PpokitColor) {}
void RecordCircle(void*, float, float, float, int, PpokitColor) {}
void RecordRectangle(void*, float, float, float, float, int, PpokitColor) {}
void RecordText(void* context, char const* text, float, float, float, PpokitColor)
{
    Of(context).texts.emplace_back(text);
}
void RecordBegin3d(void* context, float const*, float const*, float const*, float)
{
    ++Of(context).scenes;
}
void RecordEnd3d(void*) {}
void RecordLine3d(void*, float const*, float const*, PpokitColor) {}
void RecordSphere(void*, float const*, float, PpokitColor) {}
void RecordBox(void*, float const*, float const*, PpokitColor) {}
void RecordGrid(void*, int, float) {}
std::int32_t RecordLoadModel(void* context, char const*)
{
    ++Of(context).modelLoads;
    return 0;
}
void RecordDrawModel(void* context, std::int32_t, float const*, PpokitColor) { ++Of(context).modelsDrawn; }
void RecordTriangle2d(void* context, float, float, float, float, float, float, PpokitColor)
{
    ++Of(context).triangles2d;
}
void RecordTriangle3d(void* context, float const* first, float const* second, float const* third, PpokitColor color)
{
    Recording& recording = Of(context);
    ++recording.triangles3d;
    if (color == 0x66bfffffU)
    {
        ++recording.blueAircraftTriangles;
        for (float const* vertex : {first, second, third})
            recording.blueAircraftVertices.push_back({vertex[0], vertex[1], vertex[2]});
    }
    if (color == 0xe62937ffU) ++recording.redAircraftTriangles;
}
/** @brief A fixed-width font: 8 pixels a character at size 16. */
float RecordMeasure(void*, char const* text, float size)
{
    return static_cast<float>(std::string(text).size()) * size * 0.5F;
}

PpokitDraw RecordingDraw(Recording& recording)
{
    PpokitDraw draw{};
    draw.context = &recording;
    draw.width = 1280.F;
    draw.height = 720.F;
    draw.clear = &RecordClear;
    draw.line2d = &RecordLine;
    draw.circle2d = &RecordCircle;
    draw.rectangle2d = &RecordRectangle;
    draw.text2d = &RecordText;
    draw.begin3d = &RecordBegin3d;
    draw.end3d = &RecordEnd3d;
    draw.line3d = &RecordLine3d;
    draw.sphere3d = &RecordSphere;
    draw.box3d = &RecordBox;
    draw.grid3d = &RecordGrid;
    draw.loadModel = &RecordLoadModel;
    draw.drawModel = &RecordDrawModel;
    draw.triangle2d = &RecordTriangle2d;
    draw.triangle3d = &RecordTriangle3d;
    draw.measureText = &RecordMeasure;
    return draw;
}

/** @brief Whether a text starts with `prefix`. */
bool Drew(Recording const& recording, std::string const& prefix)
{
    return std::any_of(recording.texts.begin(), recording.texts.end(),
        [&prefix](std::string const& text) { return text.rfind(prefix, 0) == 0; });
}

/** @brief A copy of T6 whose Visualiseur sends to `port`, at its highest rate. */
fs::path CopyProject(fs::path const& folder, int port)
{
    fs::path const original(PPO_FLIGHT_T6_PROJECT);
    fs::create_directories(folder / "checkpoints");
    fs::create_directories(folder / "references");
    fs::create_directories(folder / "eval");
    for (char const* file : {"scenario.json", "task.json", "training.json", "eval/validation.json",
             "models.json", "references/t6_best_870.ppokit", "checkpoints/policy.ppokit", "checkpoints/ppokit_curriculum_level.txt"})
    {
        fs::copy_file(original / file, folder / file, fs::copy_options::overwrite_existing);
    }
    std::ifstream networkFile(original / "network.json");
    OrderedJson network = OrderedJson::parse(networkFile);
    network["observers"]["nodes"]["watch"]["config"] = {{"port", port}, {"rate", 240}};
    WriteJson(folder / "network.json", network);
    Json project = ReadJson(original / "project.json");
    project["libraries"] = {{"environment", FLIGHT_ENVIRONMENT_LIBRARY}, {"network", FLIGHT_NETWORK_LIBRARY}};
    WriteJson(folder / "project.json", project);
    return folder;
}

/** @brief Runs ppokit-run with `arguments` (its output discarded); returns its exit code. */
int RunPpokit(std::string const& arguments)
{
#if defined(_WIN32)
    std::string const command = "\"\"" + std::string(PPOKIT_RUN) + "\" " + arguments + " > NUL 2>&1\"";
#else
    std::string const command = "\"" + std::string(PPOKIT_RUN) + "\" " + arguments + " > /dev/null 2>&1";
#endif
    int const status = std::system(command.c_str());
#if defined(_WIN32)
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

/** @brief A left click at (x, y): pressed in one frame, released in the next. */
void Click(ppokit::view::ViewerSession& session, float x, float y)
{
    PpokitViewInput input{};
    input.mouseX = x;
    input.mouseY = y;
    input.buttons = 1U;
    session.Input(input);
    input.buttons = 0U;
    session.Input(input);
}

/** @brief Renders the session's view on a new recording. */
Recording Rendered(ppokit::view::ViewerSession& session)
{
    Recording recording;
    session.Render(RecordingDraw(recording));
    return recording;
}

/** @brief Plays T6 with its Visualiseur and checks what the view draws of the frames. */
void CheckPlayedEpisode(fs::path const& root)
{
    auto const library = ppokit::view::ViewLibrary::Open(FLIGHT_VISUALISER_LIBRARY);
    ppokit::view::ViewerSession session(library, "", 0);
    Check(std::string(session.ViewName()) == "flight", "the FlightControl3D view is the DLL's view");
    Recording const waiting = Rendered(session);
    Check(!waiting.texts.empty() && waiting.texts[0].find("waiting") == 0, "the view says it waits");

    fs::path const project = CopyProject(root / "T6", session.Port());
    std::string const level = ReadText(project / "checkpoints/ppokit_curriculum_level.txt");
    std::atomic<int> code{-1};
    std::thread run([&] {
        code = RunPpokit("play --episodes 1 --realtime --step-seconds 0.002 --project \"" + project.string() + "\"");
    });
    // Receive while the episode plays, so the socket's buffer never fills.
    while (code.load() < 0)
    {
        session.Poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    run.join();
    session.Poll();
    Check(code.load() == 0, "ppokit-run plays T6 with its Visualiseur");
    Check(session.Received() > 100, "the frames of the episode arrive in UDP");
    Check(ReadText(project / "checkpoints/ppokit_curriculum_level.txt") == level,
        "playing leaves the curriculum level as it was");

    // The flight tab (the first): status, guidance, runway, landing; the scene and the profile.
    Recording const flight = Rendered(session);
    // The runway alone is 36 quads: surface, 2 edges, 24 centreline dashes, threshold bar, 8 stripes.
    constexpr std::size_t RunwayTriangles = 2 * 36;
    Check(flight.scenes == 1 && flight.triangles3d >= RunwayTriangles + 2,
        "the view draws the 3D scene: the sea and the marked runway (" + std::to_string(flight.scenes) + " scene(s), "
            + std::to_string(flight.triangles3d) + " triangles)");
    Check(Drew(flight, "Aircraft dynamics: T-6 Texan II"), "the status names the T-6");
    Check(Drew(flight, "Inference | level"), "the status says ppokit-run plays (the curriculum stays)");
    Check(Drew(flight, "Guidance: ") && Drew(flight, "Aim point: ") && Drew(flight, "Episode time "),
        "the status shows the guidance, the aim point and the episode clock");
    Check(Drew(flight, "CAS (m/s)") && Drew(flight, "Slope (deg)") && Drew(flight, "Track azimuth (deg)"),
        "the targets table shows the speed, the slope and the track");
    Check(Drew(flight, "Runway dimensions: ") && Drew(flight, "Runway threshold above sea: "),
        "the runway panel shows the runway and the sea");
    Check(Drew(flight, "Sink rate target / actual: "), "the status shows the sink rate and its target");
    Check(Drew(flight, "Episodes ") && Drew(flight, "Crashes "), "the status counts the episodes and their ends");
    Check(Drew(flight, "Approach profile | runway-relative"), "the view draws the approach profile");
    Check(Drew(flight, "Aircraft chase") && Drew(flight, "Runway / flightplan") && Drew(flight, "Runway"),
        "the three cameras can be chosen");

    // Each tab, chosen by a click on its label (the panel is on the right of a 1280 x 720 window).
    float const panelX = 1280.F - 470.F;
    float const tabWidth = (470.F - 20.F) / 4.F;
    Click(session, panelX + 10.F + 1.5F * tabWidth, 20.F);
    Recording const controls = Rendered(session);
    Check(Drew(controls, "Network mean") && Drew(controls, "elevator") && Drew(controls, "Episode rewards"),
        "the controls tab shows the played network mean, the applied controls and the reward bars");
    Click(session, panelX + 10.F + 2.5F * tabWidth, 20.F);
    Recording const diagnostics = Rendered(session);
    Check(Drew(diagnostics, "Window: the last") && Drew(diagnostics, "Observation"),
        "the diagnostics tab shows the window of the episode and the observations");
    Click(session, panelX + 10.F + 3.5F * tabWidth, 20.F);
    Recording const crash = Rendered(session);
    Check(Drew(crash, "Last crash reasons"), "the crash tab lists the last crash reasons");
    // The overview camera draws the aircraft's marker in the same scene.
    Click(session, 10.F + 125.F + 6.F + 85.F, 22.F);
    Recording const overview = Rendered(session);
    Check(
        overview.scenes == 1 && overview.triangles3d == flight.triangles3d, "the overview camera draws the same scene");
}

/** @brief A world row, an action and rewards, sent to a view as flight_watch@2 sends them. */
struct KnownFrame final
{
    std::vector<float> row = std::vector<float>(world::Size, 0.F);
    std::vector<float> action = std::vector<float>(6, 0.F);
    std::vector<float> rewards = std::vector<float>(7, 0.F);

    float& At(std::size_t group, std::size_t index) { return row[group + index]; }
};

/** @brief The view of the DLL, fed one frame at a time and drawn on a recording. */
class DirectView final
{
public:
    explicit DirectView(std::shared_ptr<ppokit::view::ViewLibrary const> library)
        : library_(std::move(library)), api_(library_->Find(""))
    {
        char error[256] = {};
        view_ = api_.create("{}", error, sizeof(error));
        if (view_ == nullptr)
        {
            throw std::runtime_error(std::string("the flight view cannot be created: ") + error);
        }
    }
    ~DirectView() { api_.destroy(view_); }
    DirectView(DirectView const&) = delete;
    DirectView& operator=(DirectView const&) = delete;

    void Send(KnownFrame const& frame)
    {
        static std::vector<ppokit::view::FramePort> const ports = {{"aircraft", world::AircraftSize},
            {"airfield", world::AirfieldSize}, {"deck", world::DeckSize}, {"guidance", world::GuidanceSize},
            {"mission", world::MissionSize}, {"landing", world::LandingSize}, {"crash", world::CrashSize},
            {"episode_rewards", world::EpisodeRewardsSize}, {"diagnostics", world::DiagnosticsSize},
            {"action", 6}, {"rewards", 7}};
        ppokit::view::FrameWriter writer(ports, 1);
        std::size_t const offsets[] = {world::AircraftAt, world::AirfieldAt, world::DeckAt, world::GuidanceAt,
            world::MissionAt, world::LandingAt, world::CrashAt, world::EpisodeRewardsAt, world::DiagnosticsAt};
        std::vector<float const*> values;
        for (std::size_t const offset : offsets)
        {
            values.push_back(frame.row.data() + offset);
        }
        values.push_back(frame.action.data());
        values.push_back(frame.rewards.data());
        std::size_t const size = writer.Write(++sequence_, 1, values.data());
        api_.receive(view_, writer.Data(), size);
    }

    void Click(float x, float y)
    {
        PpokitViewInput input{};
        input.mouseX = x;
        input.mouseY = y;
        input.buttons = 1U;
        api_.input(view_, &input);
        input.buttons = 0U;
        api_.input(view_, &input);
    }

    Recording Render()
    {
        Recording recording;
        PpokitDraw const draw = RecordingDraw(recording);
        api_.render(view_, &draw);
        return recording;
    }

private:
    std::shared_ptr<ppokit::view::ViewLibrary const> library_;
    PpokitViewApi const& api_;
    void* view_ = nullptr;
    std::uint64_t sequence_ = 0;
};

/** @brief The centre of the panel's tab `tab` in a 1280 x 720 window. */
float TabX(int tab)
{
    float const panelX = 1280.F - 470.F;
    float const tabWidth = (470.F - 20.F) / 4.F;
    return panelX + 10.F + (static_cast<float>(tab) + 0.5F) * tabWidth;
}

/** @brief The instruments of the desktop, on frames with known values. */
void CheckKnownFrames()
{
    constexpr double DegreesToRadians = 3.14159265358979323846 / 180.0;
    DirectView view(ppokit::view::ViewLibrary::Open(FLIGHT_VISUALISER_LIBRARY));
    KnownFrame frame;
    frame.At(world::AircraftAt, world::AircraftAirspeed) = 61.5F;
    frame.At(world::AircraftAt, world::AircraftPitch) = static_cast<float>(8 * DegreesToRadians);
    frame.At(world::AirfieldAt, world::RunwayLength) = 3600.F;
    frame.At(world::AirfieldAt, world::RunwayWidth) = 120.F;
    frame.At(world::DeckAt, world::ThresholdHeight) = 15.F;
    frame.At(world::GuidanceAt, world::TargetSpeed) = 56.F;
    frame.At(world::GuidanceAt, world::TargetSlope) = static_cast<float>(-3 * DegreesToRadians);
    frame.At(world::GuidanceAt, world::RelativeAlong) = 100.F;
    frame.At(world::GuidanceAt, world::RelativeUp) = static_cast<float>(-100 * std::tan(2 * DegreesToRadians));
    frame.At(world::GuidanceAt, world::AimAlong) = 286.F;
    frame.At(world::GuidanceAt, world::GlideDegrees) = 3.F;
    frame.At(world::GuidanceAt, world::Along) = -300.F;
    frame.At(world::GuidanceAt, world::Height) = 40.F;
    frame.At(world::MissionAt, world::ElapsedSeconds) = 12.5F;
    frame.At(world::MissionAt, world::MaximumSeconds) = 900.F;
    frame.At(world::MissionAt, world::LastOutcome) = 2.F; // Crash
    frame.At(world::MissionAt, world::LastSeconds) = 7.5F;
    frame.At(world::MissionAt, world::AircraftType) = static_cast<float>(world::Pc7);
    view.Send(frame);
    Recording flight = view.Render();
    bool const leftWing = std::any_of(flight.blueAircraftVertices.begin(), flight.blueAircraftVertices.end(),
        [](auto const& vertex) { return std::abs(vertex[0] + 0.06F) < 1e-5F; });
    bool const rightWing = std::any_of(flight.blueAircraftVertices.begin(), flight.blueAircraftVertices.end(),
        [](auto const& vertex) { return std::abs(vertex[0] - 0.06F) < 1e-5F; });
    Check(flight.modelLoads == 0 && flight.modelsDrawn == 0 && flight.blueAircraftTriangles == 36
            && leftWing && rightWing,
        "the master silhouette has its closed tapered fuselage and twelve metre wingspan without a fighter model");
    Check(Drew(flight, "Aircraft dynamics: Pilatus PC-7"), "the status names the aircraft of the scenario");
    Check(Drew(flight, "56.0") && Drew(flight, "61.5") && Drew(flight, "+5.5"),
        "the targets compare the target and actual CAS");
    Check(Drew(flight, "-3.00") && Drew(flight, "-2.00"), "the targets compare the flight-path slopes, not the pitch");
    Check(Drew(flight, "Guidance: Approach") && Drew(flight, "Episode time 12.5 / 900 simulated s")
            && Drew(flight, "Last episode: Crash after 7.5 simulated s"),
        "the status shows the guidance and the simulated durations");
    Check(Drew(flight, "Deck: static") && !Drew(flight, "Swell"), "a static deck is shown without motion details");
    Check(Drew(flight, "Aim point: 286 m past the threshold"), "the status shows the aim point");

    // A moving deck, a commanded descent and a sink rate.
    frame.At(world::DeckAt, world::DeckSpeed) = 6.F;
    frame.At(world::DeckAt, world::SwellAmplitude) = 0.6F;
    frame.At(world::DeckAt, world::SwellPeriod) = 9.F;
    frame.At(world::DeckAt, world::Heave) = 0.3F;
    frame.At(world::DeckAt, world::DeckVerticalSpeed) = 0.25F;
    frame.At(world::GuidanceAt, world::DirectionUp) = -0.05F;
    frame.At(world::GuidanceAt, world::RelativeUp) = -4.F;
    view.Send(frame);
    Recording const moving = view.Render();
    Check(Drew(moving, "Deck speed: 6.0 m/s along the runway") && Drew(moving, "Deck vertical speed: +0.25 m/s")
            && Drew(moving, "Swell: +/-0.60 m, period 9.0 s, wavelength 126 m") && Drew(moving, "heave +0.30 m"),
        "a moving deck shows its speed, its swell and the present heave");
    Check(Drew(moving, "Sink rate target / actual: 2.80 / 4.00 m/s"), "the status compares the sink rates");
    Check(Drew(moving, "Approach profile | runway-relative") && Drew(moving, "Height 40.0 m | path error "),
        "the profile shows the height and its error to the glide plane");
    Check(Drew(moving, "DECK 6.0 m/s (12 kt)   SWELL +/-0.6 m, 9 s   HEAVE +0.30 m"), "the scene shows the deck banner");

    frame.At(world::MissionAt, world::Outcome) = 2.F;
    view.Send(frame);
    Check(view.Render().redAircraftTriangles == 36, "the original silhouette turns red for a crash");
    frame.At(world::MissionAt, world::Outcome) = 0.F;

    // The episode's reward bars, against the active Stop and Crash weights.
    frame.At(world::EpisodeRewardsAt, world::EpisodeSums) = -25.F;
    frame.At(world::EpisodeRewardsAt, world::ActiveWeights + 5) = 100.F;
    frame.At(world::EpisodeRewardsAt, world::ActiveWeights + 6) = -100.F;
    view.Send(frame);
    view.Click(TabX(1), 20.F);
    Recording const controls = view.Render();
    Check(Drew(controls, "Path  -25.00"), "the reward bars show the signed episode sum");

    // The crash panel: each failed measurement and its limit.
    std::size_t const speed = world::CrashConditions + 8 * world::CrashConditionValues;
    std::size_t const pitchLow = world::CrashConditions + 10 * world::CrashConditionValues;
    view.Click(TabX(3), 20.F);
    Check(Drew(view.Render(), "No crash recorded"), "the crash panel has an explicit empty state");
    frame.At(world::CrashAt, world::HasCrash) = 1.F;
    frame.At(world::CrashAt, speed) = 1.F;
    frame.At(world::CrashAt, speed + 1) = 65.F;
    frame.At(world::CrashAt, speed + 2) = 60.F;
    frame.At(world::CrashAt, pitchLow) = 1.F;
    frame.At(world::CrashAt, pitchLow + 1) = -7.F;
    frame.At(world::CrashAt, pitchLow + 2) = -5.F;
    view.Send(frame);
    Recording const crash = view.Render();
    Check(Drew(crash, "Speed (deck-relative)") && Drew(crash, "65.0 / max 60.0 m/s") && Drew(crash, "Pitch")
            && Drew(crash, "-7.0 / min -5.0 deg"),
        "the crash panel shows each failed measurement and its minimum or maximum");
}

/** @brief Values of the world row's mission group after a step or reset. */
struct MissionRow final
{
    double windowEpisodes = 0.0;
    bool startedAirborne = false;
    double mode = 0.0;
};

MissionRow MissionOf(std::vector<float> const& row)
{
    MissionRow mission;
    mission.windowEpisodes = row[world::MissionAt + world::WindowEpisodes];
    mission.startedAirborne = row[world::MissionAt + world::StartedAirborne] > 0.5F;
    mission.mode = row[world::MissionAt + world::Mode];
    return mission;
}

/**
 * @brief Plays episodes with neutral actions until an airborne one ends.
 * @return The mission values written by its last step.
 */
MissionRow PlayAirborneEpisode(ppokit::environment::Environment& environment)
{
    PpokitEnvironmentInfo const& info = environment.Info();
    std::vector<float> observation(info.observationSize);
    std::vector<std::uint8_t> available(1);
    std::vector<float> row(info.worldSize);
    std::vector<float> action(info.actionSize, 0.F);
    std::vector<double> rewards(1);
    std::vector<double> channels(info.rewardChannels);
    constexpr int MaximumEpisodes = 8;
    constexpr int MaximumSteps = 20000;
    for (int episode = 0; episode < MaximumEpisodes; ++episode)
    {
        environment.Reset(observation.data(), available.data(), row.data());
        bool const airborne = MissionOf(row).startedAirborne;
        for (int step = 0; step < MaximumSteps; ++step)
        {
            ppokit::runtime::StepEnd const end = environment.Step(action.data(), observation.data(), available.data(),
                row.data(), rewards.data(), channels.data());
            if (end != ppokit::runtime::StepEnd::Running)
            {
                break;
            }
        }
        if (airborne)
        {
            return MissionOf(row);
        }
    }
    throw std::runtime_error("no airborne episode in the training stream");
}

/** @brief Curriculum level of CheckLearningMode: airborne starts, with a promotion window. */
constexpr int LearningModeLevel = 8;

/** @brief Only learning moves the curriculum's promotion window. */
void CheckLearningMode(fs::path const& root)
{
    fs::path const project = CopyProject(root / "T6_learning", 47999);
    // A stage with airborne starts and a promotion window (the final stage has none), whatever
    // level the repository's checkpoint has reached.
    std::ofstream(project / "checkpoints" / "ppokit_curriculum_level.txt") << LearningModeLevel << '\n';
    auto const library = ppokit::environment::EnvironmentLibrary::Open(FLIGHT_ENVIRONMENT_LIBRARY);
    Check(library->Info().worldSize == world::Size, "the environment writes the world row of flight_world.hpp");
    ppokit::environment::Environment environment(library, project, 11);
    environment.SetLearning(false);
    MissionRow const played = PlayAirborneEpisode(environment);
    Check(played.mode == world::PlayMode && played.windowEpisodes == 0.0,
        "an airborne episode played leaves the promotion window empty");
    environment.SetLearning(true);
    MissionRow const trained = PlayAirborneEpisode(environment);
    Check(trained.mode == world::TrainMode && trained.windowEpisodes == 1.0,
        "an airborne episode trained counts in the promotion window");
}

/** @brief The training state of the environment, parsed. */
Json StateOf(ppokit::environment::Environment const& environment)
{
    return Json::parse(environment.TrainingState());
}

/**
 * @brief The first observation of the next training episode (the world row also shows the
 *        episodes played before, which differ between a run and its resumption).
 */
std::vector<float> NextStart(ppokit::environment::Environment& environment)
{
    PpokitEnvironmentInfo const& info = environment.Info();
    std::vector<float> observation(info.observationSize);
    std::vector<float> row(info.worldSize);
    std::vector<std::uint8_t> available(1);
    environment.Reset(observation.data(), available.data(), row.data());
    return observation;
}

/** @brief Whether restoring `state` fails with a message containing `expected`. */
bool RestoreFails(ppokit::environment::Environment& environment, std::string const& state, std::string const& expected)
{
    try
    {
        environment.RestoreTrainingState(state);
    }
    catch (std::exception const& error)
    {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

/**
 * @brief The training state follows the run (audit P0-03, P1-02): a fresh run starts at
 *        level 0 whatever an earlier run left, and a resumed run continues its curriculum
 *        window and its stream of starts exactly.
 */
void CheckTrainingState(fs::path const& root)
{
    fs::path const project = CopyProject(root / "T6_state", 47998);
    // An earlier run left an advanced curriculum (the level file of earlier versions).
    std::ofstream(project / "checkpoints" / "ppokit_curriculum_level.txt") << LearningModeLevel << '\n';
    auto const library = ppokit::environment::EnvironmentLibrary::Open(FLIGHT_ENVIRONMENT_LIBRARY);

    ppokit::environment::Environment fresh(library, project, 11);
    Check(StateOf(fresh).at("level") == LearningModeLevel, "a checkpoint without a state keeps the level file's level");
    fresh.RestoreTrainingState({});
    Json const restarted = StateOf(fresh);
    Check(restarted.at("level") == 0 && restarted.at("window").at("episodes") == 0,
        "a fresh run starts at level 0 with an empty window, whatever an earlier run reached");

    ppokit::environment::Environment trained(library, project, 11);
    trained.SetLearning(true);
    PlayAirborneEpisode(trained);
    std::string const saved = trained.TrainingState();
    Json const state = Json::parse(saved);
    Check(state.at("level") == LearningModeLevel && state.at("window").at("episodes") == 1,
        "the state holds the level and the window of the run");
    std::vector<float> const next = NextStart(trained);

    ppokit::environment::Environment resumed(library, project, 99);
    resumed.SetLearning(true);
    resumed.RestoreTrainingState(saved);
    Check(resumed.TrainingState() == saved, "a restored state reads back the same");
    Check(NextStart(resumed) == next, "a resumed run draws the same next start as the run it continues");

    Json beyond = state;
    beyond["level"] = 99;
    Check(RestoreFails(resumed, beyond.dump(), "curriculum level 99"), "a level beyond the curriculum is refused");
    Json impossible = state;
    impossible["window"]["stops"] = 2;
    Check(RestoreFails(resumed, impossible.dump(), "more successes than episodes"),
        "a window with more stops than episodes is refused");
    Check(RestoreFails(resumed, "{}", "not a flight.training_state/1"), "a state of another schema is refused");
    Check(RestoreFails(resumed, "level 3", "training state"), "a state that is not JSON is refused");
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path() / "flight_visualiser";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    try
    {
        CheckKnownFrames();
        CheckPlayedEpisode(root);
        CheckLearningMode(root);
        CheckTrainingState(root);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " FlightControl3D Visualiseur check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "FlightControl3D Visualiseur tests passed\n";
    return EXIT_SUCCESS;
}
