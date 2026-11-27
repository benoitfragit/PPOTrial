/**
 * @file PPOKit/viewer/raylib_draw.hpp
 * @brief The drawing table (PpokitDraw) of ppokit-viewer, implemented with raylib.
 */
#pragma once

#include "ppokit/view/view_abi.h"

#include <map>
#include <memory>
#include <string>

namespace ppokit::viewer {

/** @brief Models loaded for the views, by id; unloaded with the window. */
class RaylibDrawing final
{
public:
    RaylibDrawing();
    ~RaylibDrawing();
    RaylibDrawing(RaylibDrawing const&) = delete;
    RaylibDrawing& operator=(RaylibDrawing const&) = delete;

    /** @brief The table for one frame of the current window size. */
    PpokitDraw Table(float width, float height) noexcept;

    /** @brief Loads `path` once; returns its id, or -1. */
    int LoadModelOnce(char const* path);
    /** @brief Draws model `id` with a column-major transform. */
    void DrawModelWith(int id, float const* transform, PpokitColor tint);

private:
    struct LoadedModel;
    std::map<std::string, int> ids_;
    std::map<int, std::unique_ptr<LoadedModel>> models_;
    int next_ = 0;
};

} // namespace ppokit::viewer
