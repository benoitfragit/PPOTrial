/**
 * @file PPOKit/tests/components/test_view.cpp
 * @brief A Visualiseur DLL for the view tests: an observer that sends 2D points in UDP, and
 *        a view that draws them as circles, with the frame number.
 */
#include "ppokit/view/sdk.hpp"

#include <string>

namespace {

using ppokit::sdk::Call;
using ppokit::sdk::Config;
using ppokit::sdk::Manifest;
using ppokit::sdk::Shapes;

/** @brief Sends its "points" (N points of x, y per agent). */
class Dots final : public ppokit::sdk::Component
{
public:
    static Manifest Describe()
    {
        Manifest manifest("dots", 1);
        manifest.Input("points", {"N", 2}).Observer();
        manifest.Integer("port", 47000, 1, 65535).Number("rate", 30.0, 1.0, 100000.0);
        return manifest;
    }
    Dots(Config const& config, Shapes const& shapes, std::uint64_t)
        : sender_(config, {{"points", static_cast<std::uint32_t>(shapes["N"] * 2)}})
    {
    }
    void Forward(Call const& call) override { sender_.Send(call); }

private:
    ppokit::view::FrameSender sender_;
};

/** @brief Draws the points of the last frame received, 100 pixels per unit. */
class DotsView final
{
public:
    static char const* Name() { return "dots"; }
    explicit DotsView(std::string const& /*configJson*/) {}

    void Receive(ppokit::view::Frame const& frame) { frame_ = frame; }
    void Input(PpokitViewInput const& input) { zoom_ += input.wheel; }
    void Render(ppokit::view::Canvas const& canvas)
    {
        canvas.Clear(0x101010FFU);
        ppokit::view::Frame::Port const* const points = frame_.Find("points");
        if (points == nullptr)
        {
            canvas.Text("waiting", 10.F, 10.F, 20.F, 0xFFFFFFFFU);
            return;
        }
        for (std::size_t index = 0; index + 1 < points->values.size(); index += 2)
        {
            canvas.Circle(points->values[index] * 100.F * zoom_, points->values[index + 1] * 100.F * zoom_, 3.F, true,
                0x00FF00FFU);
        }
        canvas.Text("frame " + std::to_string(frame_.sequence), 10.F, 10.F, 20.F, 0xFFFFFFFFU);
    }

private:
    ppokit::view::Frame frame_;
    float zoom_ = 1.F;
};

} // namespace

PPOKIT_COMPONENTS(Dots)
PPOKIT_VIEWS(DotsView)
