/**
 * @file Examples/BlueDefense/tests/visualiser_tests.cpp
 * @brief The BlueDefense Visualiseur end to end: play and training send world frames to BlueView.
 * @details A copy of the BluePatrol project whose observer sends to this test's port; the
 * production actor plays one episode while the test receives the frames, then renders the
 * last one on a recording canvas; a separate fresh graph-backend training also sends frames.
 */
#include "ppokit/view/view_library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {

namespace fs = std::filesystem;
// Preserve the network node order when changing observer settings in a project copy.
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

/** @brief What the view drew: discs, rectangle outlines (width, height) and texts. */
struct Recording final
{
    std::vector<std::pair<float, float>> circles;
    std::vector<std::pair<float, float>> outlines;
    std::vector<std::string> texts;
};

void RecordClear(void*, PpokitColor) {}
void RecordLine(void*, float, float, float, float, float, PpokitColor) {}
void RecordCircle(void* context, float x, float y, float, int, PpokitColor)
{
    static_cast<Recording*>(context)->circles.emplace_back(x, y);
}
void RecordRectangle(void* context, float, float, float width, float height, int filled, PpokitColor)
{
    if (filled == 0)
    {
        static_cast<Recording*>(context)->outlines.emplace_back(width, height);
    }
}
void RecordText(void* context, char const* text, float, float, float, PpokitColor)
{
    static_cast<Recording*>(context)->texts.emplace_back(text);
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
    return draw;
}

/** @brief A copy of BluePatrol whose Visualiseur sends to `port`, at its highest rate. */
fs::path CopyProject(fs::path const& folder, int port)
{
    fs::path const original(PPO_BLUEDEFENSE_PROJECT);
    fs::create_directories(folder / "checkpoints" / "shared" / "prod");
    Json scenario = ReadJson(original / "scenario.json");
    // The scenario names its agent profiles relative to its folder.
    scenario["agentProfiles"] = (fs::path(PPO_SHARED_SETTINGS) / "AgentProfiles.json").generic_string();
    WriteJson(folder / "scenario.json", scenario);
    for (char const* file : {"blue_patrol_training.json", "task.json", "training.json", "checkpoints/shared/prod/policy.ppokit"})
    {
        fs::copy_file(original / file, folder / file, fs::copy_options::overwrite_existing);
    }
    Json network = ReadJson(original / "network.json");
    network["observers"]["nodes"]["watch"]["config"] = {{"port", port}, {"rate", 240}};
    WriteJson(folder / "network.json", network);
    Json project = ReadJson(original / "project.json");
    project["libraries"] = {{"environment", BLUE_ENVIRONMENT_LIBRARY}, {"network", BLUE_GRAPH_NETWORK_LIBRARY}};
    fs::path const curriculum = (original / project.at("externalCurriculum").get<std::string>()).lexically_normal();
    fs::copy_file(curriculum, folder / "curriculum.json", fs::copy_options::overwrite_existing);
    project["externalCurriculum"] = "curriculum.json";
    WriteJson(folder / "project.json", project);
    return folder;
}

/** @brief Runs ppokit-run with `arguments` (its output discarded); returns its exit code. */
int RunPpokit(std::string const& arguments)
{
    char const* const overrideProgram = std::getenv("PPOKIT_RUN_TEST_EXECUTABLE");
    std::string const program = overrideProgram != nullptr && overrideProgram[0] != '\0'
        ? overrideProgram : PPOKIT_RUN;
#if defined(_WIN32)
    std::string const command = "\"\"" + program + "\" " + arguments + "\"";
#else
    std::string const command = "\"" + program + "\" " + arguments;
#endif
    int const status = std::system(command.c_str());
#if defined(_WIN32)
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

/** @brief Check that a fresh BluePatrol graph training publishes physical world frames. */
void TrainingFrames(fs::path const& folder, std::shared_ptr<ppokit::view::ViewLibrary const> const& library)
{
    ppokit::view::ViewerSession session(library, "", 0);
    fs::path const project = CopyProject(folder, session.Port());
    auto settings = ReadJson(fs::path(PPO_BLUEDEFENSE_PROJECT) / "training.json");
    settings["ppo"]["rollout_steps"] = 256;
    settings["ppo"]["minibatch"] = 256;
    settings["ppo"]["epochs"] = 1;
    WriteJson(project / "training.json", settings);

    std::atomic<int> code{-1};
    std::thread run([&] { code = RunPpokit("train --fresh --updates 1 --project \"" + project.string() + "\""); });
    while (code.load() < 0)
    {
        session.Poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    run.join();
    session.Poll();
    Recording drawn;
    session.Render(RecordingDraw(drawn));
    Check(code.load() == 0, "fresh graph training completes");
    Check(session.Received() > 0 && !drawn.circles.empty(),
        "BluePatrol training sends world frames that the view renders");
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path() / "blue_visualiser";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    try
    {
        auto const library = ppokit::view::ViewLibrary::Open(BLUE_VISUALISER_LIBRARY);
        ppokit::view::ViewerSession session(library, "", 0);
        Check(std::string(session.ViewName()) == "blue", "the BlueDefense view is the DLL's view");
        Recording waiting;
        session.Render(RecordingDraw(waiting));
        Check(!waiting.texts.empty() && waiting.texts[0].find("waiting") == 0, "the view says it waits");

        fs::path const project = CopyProject(root / "BluePatrol", session.Port());
        std::atomic<int> code{-1};
        std::thread run([&] {
            // Paced (5 ms a step): the watch sends at most `rate` frames a second of wall time,
            // and an episode played as fast as possible lasts a fraction of a second.
            code = RunPpokit("play --episodes 1 --realtime --step-seconds 0.005"
                " --policy checkpoints/shared/prod/policy.ppokit --project \""
                + project.string() + "\"");
        });
        // Receive while the episode plays, so the socket's buffer never fills.
        while (code.load() < 0)
        {
            session.Poll();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        run.join();
        session.Poll();
        Check(code.load() == 0, "ppokit-run plays BluePatrol with its Visualiseur");
        Check(session.Received() > 100, "the frames of the episode arrive in UDP");

        Json const scenario = ReadJson(project / "scenario.json");
        std::size_t const blues = scenario.at("blueDefense").at("blueTeam").at("activeCount").get<std::size_t>();
        std::size_t const reds = scenario.at("blueDefense").at("redTeam").at("activeCount").get<std::size_t>();
        Recording drawn;
        session.Render(RecordingDraw(drawn));
        std::string counts;
        for (std::string const& text : drawn.texts)
        {
            if (text.find("blues ") == 0)
            {
                counts = text;
            }
        }
        Check(counts.find("/" + std::to_string(blues) + "   reds ") != std::string::npos
                && counts.find("/" + std::to_string(reds) + "   objective") != std::string::npos,
            "the view counts the scenario's blues and reds: " + counts);
        Check(drawn.circles.size() >= 1 + blues, "the view draws the objective and the blues");
        auto const drew = [&drawn](std::string const& text) {
            return std::find(drawn.texts.begin(), drawn.texts.end(), text) != drawn.texts.end();
        };
        Check(drew("DEFENDED AREA"), "the view draws the defended area");
        Check(drew("L") && (blues < 2 || drew("W1")), "the view draws the role badges (L, W1...)");
        // The enabled restricted zones of the scenario, at least (a friendly one may be spawned).
        std::size_t zones = 0;
        for (Json const& item : scenario.at("blueDefense").at("restrictedZones").at("items"))
        {
            zones += item.value("enabled", true) ? 1U : 0U;
        }
        std::size_t const at = counts.find("zones ");
        Check(at != std::string::npos && std::stoul(counts.substr(at + 6)) >= zones,
            "the view counts the world's restricted zones: " + counts);
        // Until moved, the view fits the outer failure rectangle (95 % of the limiting side).
        bool fitted = false;
        for (auto const& [width, height] : drawn.outlines)
        {
            fitted = fitted || std::abs(width - 0.95F * 1280.F) < 1.F || std::abs(height - 0.95F * 720.F) < 1.F;
        }
        Check(fitted, "the view fits the outer failure rectangle");
        TrainingFrames(root / "BluePatrolTraining", library);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " BlueDefense Visualiseur check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "BlueDefense Visualiseur tests passed\n";
    return EXIT_SUCCESS;
}
