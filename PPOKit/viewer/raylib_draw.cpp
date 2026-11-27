/**
 * @file PPOKit/viewer/raylib_draw.cpp
 * @brief PpokitDraw with raylib: what a view asks, drawn in the viewer's window.
 */
#include "raylib_draw.hpp"

#include <raylib.h>

#include <cstdint>

namespace ppokit::viewer {

struct RaylibDrawing::LoadedModel final
{
    Model model;
};

namespace {

Color ToColor(PpokitColor color) noexcept
{
    return Color{static_cast<unsigned char>((color >> 24U) & 0xFFU), static_cast<unsigned char>((color >> 16U) & 0xFFU),
        static_cast<unsigned char>((color >> 8U) & 0xFFU), static_cast<unsigned char>(color & 0xFFU)};
}

Vector3 ToVector(float const* values) noexcept
{
    return Vector3{values[0], values[1], values[2]};
}

/** @brief A column-major 4x4 array as a raylib matrix (raylib names its fields by column-major index). */
Matrix ToMatrix(float const* a) noexcept
{
    Matrix matrix{};
    matrix.m0 = a[0];
    matrix.m1 = a[1];
    matrix.m2 = a[2];
    matrix.m3 = a[3];
    matrix.m4 = a[4];
    matrix.m5 = a[5];
    matrix.m6 = a[6];
    matrix.m7 = a[7];
    matrix.m8 = a[8];
    matrix.m9 = a[9];
    matrix.m10 = a[10];
    matrix.m11 = a[11];
    matrix.m12 = a[12];
    matrix.m13 = a[13];
    matrix.m14 = a[14];
    matrix.m15 = a[15];
    return matrix;
}

RaylibDrawing* Self(void* context) noexcept
{
    return static_cast<RaylibDrawing*>(context);
}

void Clear(void*, PpokitColor color)
{
    ClearBackground(ToColor(color));
}

void Line2d(void*, float x0, float y0, float x1, float y1, float thickness, PpokitColor color)
{
    DrawLineEx(Vector2{x0, y0}, Vector2{x1, y1}, thickness, ToColor(color));
}

void Circle2d(void*, float x, float y, float radius, int filled, PpokitColor color)
{
    if (filled != 0)
    {
        DrawCircleV(Vector2{x, y}, radius, ToColor(color));
    }
    else
    {
        DrawCircleLines(static_cast<int>(x), static_cast<int>(y), radius, ToColor(color));
    }
}

void Rectangle2d(void*, float x, float y, float width, float height, int filled, PpokitColor color)
{
    Rectangle const rectangle{x, y, width, height};
    if (filled != 0)
    {
        DrawRectangleRec(rectangle, ToColor(color));
    }
    else
    {
        DrawRectangleLinesEx(rectangle, 1.F, ToColor(color));
    }
}

void Text2d(void*, char const* text, float x, float y, float size, PpokitColor color)
{
    DrawText(text, static_cast<int>(x), static_cast<int>(y), static_cast<int>(size), ToColor(color));
}

void Begin3d(void*, float const* position, float const* target, float const* up, float fovy)
{
    Camera3D camera{};
    camera.position = ToVector(position);
    camera.target = ToVector(target);
    camera.up = ToVector(up);
    camera.fovy = fovy;
    camera.projection = CAMERA_PERSPECTIVE;
    BeginMode3D(camera);
}

void End3d(void*)
{
    EndMode3D();
}

void Line3d(void*, float const* from, float const* to, PpokitColor color)
{
    DrawLine3D(ToVector(from), ToVector(to), ToColor(color));
}

void Sphere3d(void*, float const* center, float radius, PpokitColor color)
{
    DrawSphere(ToVector(center), radius, ToColor(color));
}

void Box3d(void*, float const* center, float const* size, PpokitColor color)
{
    DrawCubeV(ToVector(center), ToVector(size), ToColor(color));
}

void Grid3d(void*, int slices, float spacing)
{
    DrawGrid(slices, spacing);
}

void Triangle2d(void*, float x0, float y0, float x1, float y1, float x2, float y2, PpokitColor color)
{
    // raylib fills counter-clockwise triangles only: both orders, one of them is drawn.
    Vector2 const first{x0, y0};
    Vector2 const second{x1, y1};
    Vector2 const third{x2, y2};
    DrawTriangle(first, second, third, ToColor(color));
    DrawTriangle(first, third, second, ToColor(color));
}

void Triangle3d(void*, float const* first, float const* second, float const* third, PpokitColor color)
{
    // Back faces are culled: both windings make it visible from above and below.
    DrawTriangle3D(ToVector(first), ToVector(second), ToVector(third), ToColor(color));
    DrawTriangle3D(ToVector(third), ToVector(second), ToVector(first), ToColor(color));
}

float MeasureTextFunction(void*, char const* text, float size)
{
    return static_cast<float>(MeasureText(text, static_cast<int>(size)));
}

std::int32_t LoadModelFunction(void* context, char const* path)
{
    return Self(context)->LoadModelOnce(path);
}

void DrawModelFunction(void* context, std::int32_t model, float const* transform, PpokitColor tint)
{
    Self(context)->DrawModelWith(model, transform, tint);
}

} // namespace

RaylibDrawing::RaylibDrawing() = default;

RaylibDrawing::~RaylibDrawing()
{
    for (auto& [id, loaded] : models_)
    {
        UnloadModel(loaded->model);
    }
}

PpokitDraw RaylibDrawing::Table(float width, float height) noexcept
{
    PpokitDraw draw{};
    draw.context = this;
    draw.width = width;
    draw.height = height;
    draw.clear = &Clear;
    draw.line2d = &Line2d;
    draw.circle2d = &Circle2d;
    draw.rectangle2d = &Rectangle2d;
    draw.text2d = &Text2d;
    draw.begin3d = &Begin3d;
    draw.end3d = &End3d;
    draw.line3d = &Line3d;
    draw.sphere3d = &Sphere3d;
    draw.box3d = &Box3d;
    draw.grid3d = &Grid3d;
    draw.loadModel = &LoadModelFunction;
    draw.drawModel = &DrawModelFunction;
    draw.triangle2d = &Triangle2d;
    draw.triangle3d = &Triangle3d;
    draw.measureText = &MeasureTextFunction;
    return draw;
}

int RaylibDrawing::LoadModelOnce(char const* path)
{
    auto const known = ids_.find(path);
    if (known != ids_.end())
    {
        return known->second;
    }
    if (!FileExists(path))
    {
        ids_[path] = -1;
        return -1;
    }
    int const id = next_++;
    models_[id] = std::make_unique<LoadedModel>(LoadedModel{LoadModel(path)});
    ids_[path] = id;
    return id;
}

void RaylibDrawing::DrawModelWith(int id, float const* transform, PpokitColor tint)
{
    auto const found = models_.find(id);
    if (found == models_.end())
    {
        return;
    }
    found->second->model.transform = ToMatrix(transform);
    DrawModel(found->second->model, Vector3{0.F, 0.F, 0.F}, 1.F, ToColor(tint));
}

} // namespace ppokit::viewer
