/**
 * @file PPOKit/tests/view_tests.cpp
 * @brief A Visualiseur end to end, without a window: its observer sends frames in UDP, the
 *        viewer session receives them and its view draws them through a recording PpokitDraw.
 *        Also the frame format itself, and damaged frames.
 */
#include "ppokit/component/library.hpp"
#include "ppokit/view/frames.hpp"
#include "ppokit/view/view_library.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ppokit;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief What a view drew in one frame. */
struct Recording final
{
    std::vector<std::pair<float, float>> circles;
    std::vector<std::string> texts;
};

void RecordClear(void*, PpokitColor) {}
void RecordLine(void*, float, float, float, float, float, PpokitColor) {}
void RecordCircle(void* context, float x, float y, float, int, PpokitColor)
{
    static_cast<Recording*>(context)->circles.emplace_back(x, y);
}
void RecordRectangle(void*, float, float, float, float, int, PpokitColor) {}
void RecordText(void* context, char const* text, float, float, float, PpokitColor)
{
    static_cast<Recording*>(context)->texts.emplace_back(text);
}

PpokitDraw RecordingDraw(Recording& recording)
{
    PpokitDraw draw{};
    draw.context = &recording;
    draw.width = 640.F;
    draw.height = 480.F;
    draw.clear = &RecordClear;
    draw.line2d = &RecordLine;
    draw.circle2d = &RecordCircle;
    draw.rectangle2d = &RecordRectangle;
    draw.text2d = &RecordText;
    return draw;
}

void Frames()
{
    view::FrameWriter writer({{"pose", 3}, {"rewards", 2}}, 4);
    float const pose[6] = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F};
    float const rewards[4] = {0.5F, -1.F, 0.25F, 2.F};
    float const* const values[2] = {pose, rewards};
    std::size_t const size = writer.Write(9, 2, values);
    std::optional<view::Frame> const frame = view::ReadFrame(writer.Data(), size);
    Check(frame && frame->sequence == 9 && frame->agents == 2 && frame->ports.size() == 2, "a frame reads back");
    Check(frame && frame->Value("pose", 1, 2) == 6.F && frame->Value("rewards", 0, 1) == -1.F,
        "values are found by port, agent and index");
    Check(frame && frame->Value("wind", 0, 0, -7.F) == -7.F, "an absent port gives the fallback");
    Check(!view::ReadFrame(writer.Data(), size - 1), "a truncated frame is refused");
    std::vector<unsigned char> damaged(writer.Data(), writer.Data() + size);
    damaged[0] = 'X';
    Check(!view::ReadFrame(damaged.data(), damaged.size()), "a datagram that is not a frame is refused");
    Check(writer.Write(1, 5, values) == 0, "more agents than planned are not written");
}

void EndToEnd()
{
    auto const library = view::ViewLibrary::Open(PPOKIT_TEST_VIEW);
    view::ViewerSession session(library, "", 0);
    Check(std::string(session.ViewName()) == "dots" && session.SecondsSinceLast() < 0.0,
        "the first view is shown, waiting for data");
    Recording waiting;
    session.Render(RecordingDraw(waiting));
    Check(waiting.texts.size() == 1 && waiting.texts[0] == "waiting", "the view says it waits");

    // The observer half of the same DLL, as ppokit-run calls it after a step of two agents.
    auto const components = component::Library::Open(PPOKIT_TEST_VIEW, {});
    component::ComponentRef const dots = components->Find("dots", 1);
    Check(dots.manifest->capabilities.observer, "the Visualiseur's observer is an observer");
    std::string const config = component::ValidateConfig(*dots.manifest,
        "{\"port\": " + std::to_string(session.Port()) + ", \"rate\": 100000}");
    component::Instance observer(dots, config, "{\"N\":2}", 1);
    float points[8] = {0.1F, 0.2F, 0.3F, 0.4F, 1.F, 2.F, 3.F, 4.F};
    std::uint32_t const shape[2] = {2, 2};
    PpokitBuffer input{points, PPOKIT_DTYPE_F32, 2, shape};
    observer.Forward(2, &input, nullptr, nullptr);

    for (int attempt = 0; attempt < 200 && session.Received() == 0; ++attempt)
    {
        session.Poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    Check(session.Received() == 1 && session.SecondsSinceLast() >= 0.0, "the frame arrives in UDP");
    Recording drawn;
    session.Render(RecordingDraw(drawn));
    Check(drawn.circles.size() == 4 && drawn.circles[3].first == 300.F && drawn.circles[3].second == 400.F,
        "the view draws every point of both agents");
    Check(!drawn.texts.empty() && drawn.texts.back() == "frame 0", "the view shows the frame number");
    PpokitViewInput zoom{};
    zoom.wheel = 1.F;
    session.Input(zoom);
    Recording zoomed;
    session.Render(RecordingDraw(zoomed));
    Check(zoomed.circles.size() == 4 && zoomed.circles[0].first == 20.F, "input reaches the view");

    std::string message;
    try
    {
        view::ViewerSession unknown(library, "radar", 0);
    }
    catch (std::exception const& error)
    {
        message = error.what();
    }
    Check(message.find("has no view 'radar' (views: dots)") != std::string::npos, "an unknown view is named");
}

} // namespace

int main()
{
    try
    {
        Frames();
        EndToEnd();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " view check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit view tests passed\n";
    return EXIT_SUCCESS;
}
