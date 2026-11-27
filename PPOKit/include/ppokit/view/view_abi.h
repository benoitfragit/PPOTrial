/**
 * @file PPOKit/include/ppokit/view/view_abi.h
 * @brief Stable C API of a view: what ppokit-viewer loads to show an example's world.
 * @details A Visualiseur DLL holds two halves of one example's visualization:
 * - an observer component (ppokit/component/abi.h, capability "observer") that ppokit-run
 *   calls after every step and that sends what it is wired to in UDP frames
 *   (ppokit/view/frames.hpp);
 * - one or more views, listed by `ppokit_views`, that ppokit-viewer creates, feeds with the
 *   frames received and asks to draw.
 *
 * A view draws through the PpokitDraw table the viewer gives it: it needs no graphics
 * library of its own and never touches the viewer's window. The viewer may attach to a
 * run at any time and leave it: a view shows the last frame received.
 */
#ifndef PPOKIT_VIEW_ABI_H
#define PPOKIT_VIEW_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief ABI version; the viewer refuses a DLL of another one. */
#define PPOKIT_VIEW_ABI 2u

#if defined(_WIN32)
#define PPOKIT_VIEW_EXPORT __declspec(dllexport)
#else
#define PPOKIT_VIEW_EXPORT __attribute__((visibility("default")))
#endif

/** @brief A color, 0xRRGGBBAA. */
typedef uint32_t PpokitColor;

/**
 * @brief Drawing functions the viewer provides for one frame. Coordinates are in pixels
 *        (2D, origin top-left) or in world units inside begin3d / end3d.
 */
typedef struct PpokitDraw
{
    void* context;
    float width;
    float height;
    void (*clear)(void* context, PpokitColor color);
    void (*line2d)(void* context, float x0, float y0, float x1, float y1, float thickness, PpokitColor color);
    void (*circle2d)(void* context, float x, float y, float radius, int filled, PpokitColor color);
    void (*rectangle2d)(void* context, float x, float y, float width, float height, int filled, PpokitColor color);
    void (*text2d)(void* context, const char* text, float x, float y, float size, PpokitColor color);
    /** @brief Starts 3D drawing from a camera at `position` looking at `target`, `up` up, `fovy` degrees. */
    void (*begin3d)(void* context, const float* position, const float* target, const float* up, float fovy);
    void (*end3d)(void* context);
    void (*line3d)(void* context, const float* from, const float* to, PpokitColor color);
    void (*sphere3d)(void* context, const float* center, float radius, PpokitColor color);
    void (*box3d)(void* context, const float* center, const float* size, PpokitColor color);
    void (*grid3d)(void* context, int slices, float spacing);
    /** @brief Loads a model file (glTF, OBJ...) once; returns its id, or -1. */
    int32_t (*loadModel)(void* context, const char* path);
    /** @brief Draws a model with a column-major 4x4 transform. */
    void (*drawModel)(void* context, int32_t model, const float* transform, PpokitColor tint);
    /** @brief A filled triangle in pixels, whatever the order of its corners. */
    void (*triangle2d)(void* context, float x0, float y0, float x1, float y1, float x2, float y2, PpokitColor color);
    /** @brief A filled triangle inside begin3d / end3d, seen from both sides. */
    void (*triangle3d)(void* context, const float* first, const float* second, const float* third, PpokitColor color);
    /** @brief Width in pixels of `text` drawn by text2d at `size`. */
    float (*measureText)(void* context, const char* text, float size);
} PpokitDraw;

/** @brief Mouse and keyboard of the viewer's window since the previous frame. */
typedef struct PpokitViewInput
{
    float mouseX;
    float mouseY;
    /** @brief Mouse movement while the left button is held. */
    float dragX;
    float dragY;
    float wheel;
    /** @brief Bit 0: left button down, bit 1: right button down. */
    uint32_t buttons;
} PpokitViewInput;

/** @brief One view of a DLL. */
typedef struct PpokitViewApi
{
    uint32_t abi;
    /** @brief Name given to ppokit-viewer (--view-name); the first view is the default. */
    const char* name;
    /** @brief Creates the view. @param configJson Options, e.g. {"models": "Examples/Shared/3D"}. */
    void* (*create)(const char* configJson, char* error, size_t errorSize);
    void (*destroy)(void* view);
    /** @brief A datagram received from the run (a frame of ppokit/view/frames.hpp). */
    void (*receive)(void* view, const uint8_t* data, size_t size);
    void (*input)(void* view, const PpokitViewInput* input);
    void (*render)(void* view, const PpokitDraw* draw);
} PpokitViewApi;

/** @brief Signature of the function listing a DLL's views. */
typedef int (*PpokitViewsFunction)(uint32_t requestedAbi, const PpokitViewApi* const** views, size_t* count);

/** @brief Name of that function. */
#define PPOKIT_VIEWS_ENTRY "ppokit_views"

#ifdef __cplusplus
}
#endif

#endif /* PPOKIT_VIEW_ABI_H */
