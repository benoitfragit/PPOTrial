/**
 * @file PPOKit/include/ppokit/view/sdk.hpp
 * @brief C++ SDK of a Visualiseur DLL: the observer that sends frames, and the views that
 *        draw them in ppokit-viewer.
 * @details The observer half is a component (ppokit/component/sdk.hpp) declared Observer();
 * a FrameSender sends what it is wired to, port by port, at most `rate` frames per second:
 *
 * @code
 * class FlightWatch final : public ppokit::sdk::Component
 * {
 * public:
 *     static ppokit::sdk::Manifest Describe()
 *     {
 *         ppokit::sdk::Manifest manifest("flight_watch", 1);
 *         manifest.Input("pose", {6}).Input("controls", {5}).Observer();
 *         manifest.Integer("port", 47000, 1024, 65535).Number("rate", 30, 1, 1000);
 *         return manifest;
 *     }
 *     FlightWatch(Config const& config, Shapes const&, std::uint64_t)
 *         : sender_(config, {{"pose", 6}, {"controls", 5}}) {}
 *     void Forward(ppokit::sdk::Call const& call) override { sender_.Send(call); }
 * private:
 *     ppokit::view::FrameSender sender_;
 * };
 * @endcode
 *
 * The view half is a class with a static Name() and the members below, exported with
 * PPOKIT_VIEWS(FlightView):
 *
 * @code
 * class FlightView final
 * {
 * public:
 *     static char const* Name() { return "flight"; }
 *     explicit FlightView(std::string const& configJson);
 *     void Receive(ppokit::view::Frame const& frame);        // the latest frame
 *     void Input(PpokitViewInput const& input);              // camera
 *     void Render(ppokit::view::Canvas const& canvas);       // draw
 * };
 * @endcode
 */
#pragma once

#include "ppokit/component/sdk.hpp"
#include "ppokit/live/datagram.hpp"
#include "ppokit/view/frames.hpp"
#include "ppokit/view/view_abi.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace ppokit::view {

/**
 * @brief Sends the inputs of an observer as frames, to "host" (default 127.0.0.1) on
 *        "port", at most "rate" frames per second. Nothing is allocated per step.
 */
class FrameSender final
{
public:
    /**
     * @param config The observer's configuration: "port" (int) and "rate" (frames per
     *        second); a missing setting takes its default (47000, 30).
     * @param ports The observer's input ports in manifest order, with their values per agent.
     * @param maximumAgents Largest number of agents of a step.
     */
    FrameSender(sdk::Config const& config, std::vector<FramePort> ports, std::uint32_t maximumAgents = 16)
        : writer_(ports, maximumAgents),
          sender_("127.0.0.1", static_cast<std::uint16_t>(Setting(config, "port", 47000.0))),
          interval_(1.0 / Setting(config, "rate", 30.0)),
          pointers_(ports.size(), nullptr)
    {
    }

    /** @brief Sends the call's inputs when the rate allows; returns whether a frame went out. */
    bool Send(sdk::Call const& call) noexcept
    {
        auto const now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_).count() < interval_)
        {
            return false;
        }
        last_ = now;
        for (std::size_t port = 0; port < pointers_.size(); ++port)
        {
            pointers_[port] = call.Input(port);
        }
        std::size_t const size = writer_.Write(sequence_++, call.Batch(), pointers_.data());
        return size > 0 && sender_.Send(writer_.Data(), size);
    }

private:
    /** @brief A setting of the configuration, or `fallback` when the manifest has none. */
    static double Setting(sdk::Config const& config, char const* name, double fallback)
    {
        try
        {
            return config.Number(name);
        }
        catch (std::exception const&)
        {
            return fallback;
        }
    }

    FrameWriter writer_;
    live::DatagramSender sender_;
    double interval_;
    std::vector<float const*> pointers_;
    std::uint64_t sequence_ = 0;
    std::chrono::steady_clock::time_point last_{};
};

/** @brief The viewer's drawing functions, with C++ conveniences. */
class Canvas final
{
public:
    explicit Canvas(PpokitDraw const& draw) noexcept : draw_(draw) {}

    float Width() const noexcept { return draw_.width; }
    float Height() const noexcept { return draw_.height; }
    PpokitDraw const& Raw() const noexcept { return draw_; }

    void Clear(PpokitColor color) const { draw_.clear(draw_.context, color); }
    void Line(float x0, float y0, float x1, float y1, float thickness, PpokitColor color) const
    {
        draw_.line2d(draw_.context, x0, y0, x1, y1, thickness, color);
    }
    void Circle(float x, float y, float radius, bool filled, PpokitColor color) const
    {
        draw_.circle2d(draw_.context, x, y, radius, filled ? 1 : 0, color);
    }
    void Rectangle(float x, float y, float width, float height, bool filled, PpokitColor color) const
    {
        draw_.rectangle2d(draw_.context, x, y, width, height, filled ? 1 : 0, color);
    }
    void Text(std::string const& text, float x, float y, float size, PpokitColor color) const
    {
        draw_.text2d(draw_.context, text.c_str(), x, y, size, color);
    }
    void Triangle(float x0, float y0, float x1, float y1, float x2, float y2, PpokitColor color) const
    {
        draw_.triangle2d(draw_.context, x0, y0, x1, y1, x2, y2, color);
    }
    /** @brief Width in pixels of `text` at `size`. */
    float TextWidth(std::string const& text, float size) const
    {
        return draw_.measureText(draw_.context, text.c_str(), size);
    }

private:
    PpokitDraw const& draw_;
};

namespace detail {

/** @brief What the C handle of a view points to. */
template<typename View>
struct ViewHandle final
{
    std::unique_ptr<View> view;
};

template<typename View>
void* CreateView(char const* config, char* error, std::size_t errorSize) noexcept
{
    try
    {
        auto handle = std::make_unique<ViewHandle<View>>();
        handle->view = std::make_unique<View>(std::string(config != nullptr ? config : "{}"));
        return handle.release();
    }
    catch (std::exception const& failure)
    {
        if (error != nullptr && errorSize > 0)
        {
            std::snprintf(error, errorSize, "%s", failure.what());
        }
    }
    return nullptr;
}

template<typename View>
void ReceiveFrame(void* handle, std::uint8_t const* data, std::size_t size) noexcept
{
    try
    {
        if (std::optional<Frame> frame = ReadFrame(data, size))
        {
            static_cast<ViewHandle<View>*>(handle)->view->Receive(*frame);
        }
    }
    catch (...)
    {
        // A frame that cannot be shown is dropped; the view keeps the previous one.
    }
}

template<typename View>
void InputView(void* handle, PpokitViewInput const* input) noexcept
{
    try
    {
        static_cast<ViewHandle<View>*>(handle)->view->Input(*input);
    }
    catch (...)
    {
    }
}

template<typename View>
void RenderView(void* handle, PpokitDraw const* draw) noexcept
{
    try
    {
        static_cast<ViewHandle<View>*>(handle)->view->Render(Canvas(*draw));
    }
    catch (...)
    {
    }
}

template<typename View>
void DestroyView(void* handle) noexcept
{
    delete static_cast<ViewHandle<View>*>(handle);
}

/** @brief The C table of view class T, built once. */
template<typename View>
PpokitViewApi const* ViewTable()
{
    static PpokitViewApi const table = {
        PPOKIT_VIEW_ABI, View::Name(), &CreateView<View>, &DestroyView<View>, &ReceiveFrame<View>, &InputView<View>,
        &RenderView<View>};
    return &table;
}

} // namespace detail
} // namespace ppokit::view

/** @brief Exports the views of this DLL (up to four): PPOKIT_VIEWS(FlightView) */
#define PPOKIT_VIEWS(...)                                                                                             \
    extern "C" PPOKIT_VIEW_EXPORT int ppokit_views(                                                                   \
        uint32_t requestedAbi, const PpokitViewApi* const** views, size_t* count)                                   \
    {                                                                                                                 \
        if (requestedAbi != PPOKIT_VIEW_ABI)                                                                          \
        {                                                                                                             \
            return 1;                                                                                                 \
        }                                                                                                             \
        static const PpokitViewApi* const table[] = {PPOKIT_VIEW_TABLES(__VA_ARGS__)};                               \
        *views = table;                                                                                               \
        *count = sizeof(table) / sizeof(table[0]);                                                                    \
        return 0;                                                                                                     \
    }

#define PPOKIT_VIEW_TABLE(T) ppokit::view::detail::ViewTable<T>()
#define PPOKIT_VIEW_COUNT(...) PPOKIT_SDK_EXPAND(PPOKIT_VIEW_COUNT_(__VA_ARGS__, 4, 3, 2, 1))
#define PPOKIT_VIEW_COUNT_(_1, _2, _3, _4, N, ...) N
#define PPOKIT_VIEW_TABLES(...)                                                                                       \
    PPOKIT_SDK_EXPAND(PPOKIT_SDK_JOIN(PPOKIT_VIEW_TABLES_, PPOKIT_VIEW_COUNT(__VA_ARGS__))(__VA_ARGS__))
#define PPOKIT_VIEW_TABLES_1(a) PPOKIT_VIEW_TABLE(a)
#define PPOKIT_VIEW_TABLES_2(a, ...) PPOKIT_VIEW_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_VIEW_TABLES_1(__VA_ARGS__))
#define PPOKIT_VIEW_TABLES_3(a, ...) PPOKIT_VIEW_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_VIEW_TABLES_2(__VA_ARGS__))
#define PPOKIT_VIEW_TABLES_4(a, ...) PPOKIT_VIEW_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_VIEW_TABLES_3(__VA_ARGS__))
