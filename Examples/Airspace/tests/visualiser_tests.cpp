/**
 * @file Examples/Airspace/tests/visualiser_tests.cpp
 * @brief The Airspace Visualiseur end to end: ppokit-run plays RLSNA and RLMission, the world
 *        row reaches airspace_watch, its frames arrive in UDP and AirspaceView draws the scene.
 * @details Copies of the projects (and of the models they read) whose observer sends to this
 * test's port; the production policy plays one episode while the test receives the frames,
 * then renders the last one on a recording canvas: the work area, the aircraft of the scene
 * (triangles), the controlled aircraft's rings and the line of data.
 */
#include "ppokit/view/view_library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

namespace fs = std::filesystem;
// Ordered: the node order of network.json is part of the graph a policy was trained for.
using Json = nlohmann::ordered_json;

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

void WriteJson(fs::path const& path, Json const& json)
{
    std::ofstream(path) << json.dump(2) << '\n';
}

/** @brief What the view drew: triangles, circles and texts. */
struct Recording final
{
    std::size_t triangles = 0;
    std::size_t circles = 0;
    std::vector<std::string> texts;
};

void RecordClear(void*, PpokitColor) {}
void RecordLine(void*, float, float, float, float, float, PpokitColor) {}
void RecordCircle(void* context, float, float, float, int, PpokitColor)
{
    ++static_cast<Recording*>(context)->circles;
}
void RecordRectangle(void*, float, float, float, float, int, PpokitColor) {}
void RecordText(void* context, char const* text, float, float, float, PpokitColor)
{
    static_cast<Recording*>(context)->texts.emplace_back(text);
}
void RecordTriangle(void* context, float, float, float, float, float, float, PpokitColor)
{
    ++static_cast<Recording*>(context)->triangles;
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
    draw.triangle2d = &RecordTriangle;
    return draw;
}

/** @brief The environment and network DLLs of a project, by family. */
struct Libraries final
{
    char const* environment;
    char const* network;
};

/**
 * @brief A copy of the family's project, its models and the settings it reads (Airspace's and
 *        Shared's, at the same relative places), whose Visualiseur sends to `port`.
 * @return The project folder.
 */
fs::path CopyProject(fs::path const& root, std::string const& family, Libraries const& libraries, int port)
{
    fs::path const airspace(PPO_AIRSPACE_FOLDER);
    fs::path const copy = root / "Examples" / "Airspace";
    fs::path const folder = copy / "projects" / family;
    fs::create_directories(folder);
    fs::create_directories(copy / "models");
    fs::create_directories(root / "Examples" / "Shared");
    fs::copy(airspace / "projects" / family, folder, fs::copy_options::recursive);
    fs::copy(airspace / "models" / family, copy / "models" / family, fs::copy_options::recursive);
    fs::copy(airspace / "settings", copy / "settings", fs::copy_options::recursive);
    fs::copy(PPO_SHARED_SETTINGS, root / "Examples" / "Shared" / "settings", fs::copy_options::recursive);
    Json network = ReadJson(folder / "network.json");
    network["observers"]["nodes"]["watch"]["config"] = {{"port", port}, {"rate", 240}};
    WriteJson(folder / "network.json", network);
    Json project = ReadJson(folder / "project.json");
    project["libraries"] = {{"environment", libraries.environment}, {"network", libraries.network}};
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

/** @brief Whether the recording holds a text that starts with `prefix`. */
bool DrewText(Recording const& drawn, std::string const& prefix)
{
    return std::any_of(drawn.texts.begin(), drawn.texts.end(),
        [&prefix](std::string const& text) { return text.compare(0, prefix.size(), prefix) == 0; });
}

/**
 * @brief Plays episodes of a family's project with a policy and checks the view of the last frame.
 * @param episodes Episodes to play (the Playground's random episodes can be short).
 * @param aircraft Aircraft of the scene (the view draws a glyph and its shadow for each).
 */
void CheckFamily(std::shared_ptr<ppokit::view::ViewLibrary const> const& library, fs::path const& root,
    std::string const& family, Libraries const& libraries, std::string const& policy, int episodes,
    std::size_t aircraft)
{
    ppokit::view::ViewerSession session(library, "", 0);
    Check(std::string(session.ViewName()) == "airspace", "the Airspace view is the DLL's view");
    Recording waiting;
    session.Render(RecordingDraw(waiting));
    Check(!waiting.texts.empty() && waiting.texts[0].find("waiting") == 0, "the view says it waits");

    fs::path const project = CopyProject(root / family, family, libraries, session.Port());
    std::atomic<int> code{-1};
    std::thread run([&] {
        // Paced: the watch sends at most `rate` frames a second of wall time.
        code = RunPpokit("play --episodes " + std::to_string(episodes) + " --realtime --step-seconds 0.002 --policy \"" + policy
            + "\" --project \"" + project.string() + "\"");
    });
    // Receive while the episode plays, so the socket's buffer never fills.
    while (code.load() < 0)
    {
        session.Poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    run.join();
    session.Poll();
    Check(code.load() == 0, "ppokit-run plays " + family + " with its Visualiseur");
    Check(session.Received() > 10, family + ": the frames of the episodes arrive in UDP");

    Recording drawn;
    session.Render(RecordingDraw(drawn));
    Check(DrewText(drawn, "Work Area"), family + ": the view draws the work area");
    Check(DrewText(drawn, "aircraft alive: "), family + ": the view counts the aircraft");
    Check(DrewText(drawn, "controlled: Pk "), family + ": the view shows the controlled aircraft's data");
    Check(drawn.triangles >= 2 * aircraft, family + ": the view draws the scene's aircraft (shadow and glyph)");
    Check(drawn.circles >= 2, family + ": the view rings the controlled aircraft");
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path() / "airspace_visualiser";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    try
    {
        auto const library = ppokit::view::ViewLibrary::Open(AIRSPACE_VISUALISER_LIBRARY);
        // RLSNA plays the evaluator's first case: the interceptor and its target.
        CheckFamily(library, root, "RLSNA", {RLSNA_ENVIRONMENT_LIBRARY, RLSNA_NETWORK_LIBRARY},
            (root / "RLSNA/Examples/Airspace/models/RLSNA/prod/policy.ppokit").string(), 1, 2);
        // RLMission plays Playground episodes with the eval policy: its aircraft and a red one.
        CheckFamily(library, root, "RLMission", {RLMISSION_ENVIRONMENT_LIBRARY, RLMISSION_NETWORK_LIBRARY},
            (root / "RLMission/Examples/Airspace/models/RLMission/eval/policy.ppokit").string(), 3, 2);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " Airspace Visualiseur check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Airspace Visualiseur tests passed\n";
    return EXIT_SUCCESS;
}
