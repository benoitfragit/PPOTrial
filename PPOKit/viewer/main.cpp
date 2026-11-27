/**
 * @file PPOKit/viewer/main.cpp
 * @brief ppokit-viewer: a window on the world of a run, drawn by an example's Visualiseur view.
 * @details It listens for the frames a Visualiseur observer sends in UDP and hands them to
 * its view, which draws through the viewer. It attaches and detaches at any time: start it
 * before, during or after a run. Nothing else: no rewards, no training controls (that is
 * the studio's job).
 *
 * @verbatim
 * ppokit-viewer --view LIBRARY [--view-name NAME] [--port 47000] [--config JSON]
 * @endverbatim
 */
#include "raylib_draw.hpp"

#include "ppokit/view/view_library.hpp"

#include <raylib.h>

#include "ppokit/ipc/control.hpp"
#include <memory>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

namespace {

struct Options final
{
    std::string library;
    std::string controlEndpoint;
    std::string viewName;
    std::uint16_t port = 47000;
    std::string config = "{}";
};

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index + 1 < argc; index += 2)
    {
        std::string const option = argv[index];
        std::string const value = argv[index + 1];
        if (option == "--view")
        {
            options.library = value;
        }
        else if (option == "--control") options.controlEndpoint = value;
        else if (option == "--view-name")
        {
            options.viewName = value;
        }
        else if (option == "--port")
        {
            options.port = static_cast<std::uint16_t>(std::stoul(value));
        }
        else if (option == "--config")
        {
            options.config = value;
        }
        else
        {
            throw std::invalid_argument("unknown option '" + option + "'");
        }
    }
    if (options.library.empty())
    {
        throw std::invalid_argument("--view LIBRARY is required (the example's Visualiseur DLL)");
    }
    return options;
}

/** @brief The mouse since the previous frame, for the view's camera. */
PpokitViewInput ReadInput() noexcept
{
    PpokitViewInput input{};
    Vector2 const mouse = GetMousePosition();
    input.mouseX = mouse.x;
    input.mouseY = mouse.y;
    input.wheel = GetMouseWheelMove();
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
    {
        Vector2 const delta = GetMouseDelta();
        input.dragX = delta.x;
        input.dragY = delta.y;
        input.buttons |= 1U;
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
    {
        input.buttons |= 2U;
    }
    return input;
}

/** @brief One line at the bottom: the port, and whether frames arrive. */
void DrawStatus(ppokit::view::ViewerSession const& session)
{
    char text[160];
    double const age = session.SecondsSinceLast();
    if (age < 0.0)
    {
        std::snprintf(text, sizeof(text), "waiting for frames on UDP port %u", static_cast<unsigned>(session.Port()));
    }
    else if (age < 1.0)
    {
        std::snprintf(text, sizeof(text), "live on port %u  |  %llu frames", static_cast<unsigned>(session.Port()),
            static_cast<unsigned long long>(session.Received()));
    }
    else
    {
        std::snprintf(text, sizeof(text), "port %u  |  no frame for %.0f s (run paused or ended)",
            static_cast<unsigned>(session.Port()), age);
    }
    DrawText(text, 10, GetScreenHeight() - 24, 16, Color{200, 200, 200, 255});
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Options const options = ParseOptions(argc, argv);
        std::unique_ptr<ppokit::ipc::Worker> control;
        if (!options.controlEndpoint.empty()) control = std::make_unique<ppokit::ipc::Worker>(options.controlEndpoint, false);
        auto const library = ppokit::view::ViewLibrary::Open(options.library);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 720, "PPOKit viewer");
        SetTargetFPS(60);
        {
            ppokit::view::ViewerSession session(library, options.viewName, options.port, options.config);
            SetWindowTitle((std::string("PPOKit viewer - ") + session.ViewName()).c_str());
            ppokit::viewer::RaylibDrawing drawing;
            while (!WindowShouldClose() && (!control || control->Continue()))
            {
                session.Poll();
                session.Input(ReadInput());
                BeginDrawing();
                PpokitDraw const draw =
                    drawing.Table(static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight()));
                session.Render(draw);
                DrawStatus(session);
                EndDrawing();
            }
        }
        CloseWindow();
        if (control) control->Finish();
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ppokit-viewer: " << error.what() << '\n';
        return 1;
    }
}
