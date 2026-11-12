/**
* @file Examples/Airspace/Renderer/src/environmentrenderer.cpp
* @brief Implementation for environmentrenderer.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "environmentrenderer.hpp"
#include "airspacecomponents.hpp"
#include "camera_follow.hpp"
#include "tactical_role.hpp"
#ifdef PPO_WITH_FLIGHT_CONTROL
#include "flight_view.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <rlImGui.h>
#include <rlgl.h>

#include "interface.hpp"
#include "serializer.hpp"

namespace ppo::renderer {

namespace {

constexpr float PI_F = 3.14159265359F;
constexpr float DEG_PER_RAD = 57.2957795131F;
constexpr float CAMERA_MOUSE_ORBIT_SPEED = 0.0075F;
constexpr float CAMERA_MOUSE_PAN_SPEED = 0.0100F;
constexpr float CAMERA_WHEEL_ZOOM_SPEED = 1.25F;
constexpr float CAMERA_FOLLOW_RESPONSE_PER_SECOND = 8.F;
constexpr float AIRCRAFT_MODEL_EXTENT = 2.45F;
constexpr float OBJECTIVE_MODEL_EXTENT = 5.80F;
constexpr float AIRCRAFT_MODEL_YAW_OFFSET_DEG = 180.F;
constexpr std::size_t MULTI_CAMERA_RENDERS_PER_FRAME = 3U;
constexpr int MULTI_CAMERA_MAX_TEXTURE_WIDTH = 960;
constexpr int MULTI_CAMERA_MAX_TEXTURE_HEIGHT = 540;

/**
* @brief Linearly interpolates between two 3D vectors.
* @param a Start vector.
* @param b End vector.
* @param amount Interpolation factor in the [0, 1] interval.
* @return Interpolated vector.
*/
Vector3
VectorLerpLocal(Vector3 const& a, Vector3 const& b, float amount)
{
    return Vector3{
        a.x + (b.x - a.x) * amount,
        a.y + (b.y - a.y) * amount,
        a.z + (b.z - a.z) * amount
    };
}

/**
* @brief Adds two 3D vectors component-wise.
* @param a Left operand.
* @param b Right operand.
* @return Sum vector.
*/
Vector3
VectorAdd(Vector3 const& a, Vector3 const& b)
{
    return Vector3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

/**
* @brief Subtracts two 3D vectors component-wise.
* @param a Left operand.
* @param b Right operand.
* @return Difference vector.
*/
Vector3
VectorSub(Vector3 const& a, Vector3 const& b)
{
    return Vector3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

/**
* @brief Multiplies one vector by one scalar.
* @param v Input vector.
* @param scalar Scalar multiplier.
* @return Scaled vector.
*/
Vector3
VectorScale(Vector3 const& v, float scalar)
{
    return Vector3{ v.x * scalar, v.y * scalar, v.z * scalar };
}

/**
* @brief Computes the Euclidean length of one vector.
* @param v Input vector.
* @return Vector norm.
*/
float
VectorLength(Vector3 const& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
* @brief Draws one thick 3D segment by extruding a cylinder between endpoints.
* @param start Segment start.
* @param end Segment end.
* @param thickness Segment thickness.
* @param color Segment color.
*/
void
DrawLineEx3D(Vector3 const& start, Vector3 const& end, float thickness, Color color)
{
    if (thickness <= 0.F)
        return;

    if (VectorLength(VectorSub(end, start)) <= 1.0e-5F)
        return;

    DrawCylinderEx(start, end, thickness * 0.5F, thickness * 0.5F, 8, color);
}

/**
* @brief Normalizes one vector and guards against near-zero lengths.
* @param v Input vector.
* @return Unit vector, or zero when the input length is negligible.
*/
Vector3
VectorNormalizeSafe(Vector3 const& v)
{
    float const length = VectorLength(v);
    if (length <= 1.0e-5F)
        return Vector3{ 0.F, 0.F, 0.F };

    return VectorScale(v, 1.F / length);
}

/**
* @brief Computes the cross product of two vectors.
* @param a Left operand.
* @param b Right operand.
* @return Cross-product vector.
*/
Vector3
VectorCross(Vector3 const& a, Vector3 const& b)
{
    return Vector3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

/**
 * @brief Computes the dot product of two vectors.
 * @param a Left operand.
 * @param b Right operand.
 * @return Scalar dot product.
 */
float VectorDot(Vector3 const& a, Vector3 const& b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
* @brief Intersects one ray with the horizontal ground plane.
* @param ray Input ray.
* @param hitPoint Output intersection point.
* @return True when the ray hits the ground in front of the origin.
*/
bool
TryIntersectGroundPlane(Ray const& ray, Vector3& hitPoint)
{
    if (std::fabs(ray.direction.y) <= 1.0e-5F)
        return false;

    float const distance = -ray.position.y / ray.direction.y;
    if (distance <= 0.F)
        return false;

    hitPoint = Vector3{
        ray.position.x + ray.direction.x * distance,
        0.F,
        ray.position.z + ray.direction.z * distance
    };
    return true;
}

/**
* @brief Converts one entity type to a human-readable label.
* @param type Entity type identifier.
* @return Static label string.
*/
char const*
EntityTypeLabel(ppo::model::eEntityType type)
{
    switch (type)
    {
    case ppo::model::eEntityType::Mobile:  return "Mobile";
    case ppo::model::eEntityType::RestrictedZone:       return "RestrictedZone";
    case ppo::model::eEntityType::Objective: return "Objective";
    case ppo::model::eEntityType::Missile:   return "Missile";
    default:                                 return "Unknown";
    }
}

/**
* @brief Converts one tactical side to a human-readable label.
* @param side Tactical side identifier.
* @return Static label string.
*/
char const*
SideLabel(ppo::model::eEntitySide side)
{
    switch (side)
    {
    case ppo::model::eEntitySide::Friend:  return "Friend";
    case ppo::model::eEntitySide::Hostile: return "Hostile";
    default:                               return "Unknown";
    }
}

/**
* @brief Returns the base tactical color used for one side.
* @param side Tactical side identifier.
* @param alpha Optional alpha channel.
* @return Side color.
*/
Color
ColorForSide(ppo::model::eEntitySide side, unsigned char alpha = 255)
{
    switch (side)
    {
    case ppo::model::eEntitySide::Hostile: return Color{ 235, 64, 52, alpha };
    case ppo::model::eEntitySide::Friend:  return Color{ 64, 116, 255, alpha };
    default:                               return Color{ 255, 255, 255, alpha };
    }
}

/**
* @brief Linearly blends two colors.
* @param a Start color.
* @param b End color.
* @param t Blend factor in the [0, 1] interval.
* @return Blended color.
*/
Color
LerpColor(Color a, Color b, float t) noexcept
{
    t = std::clamp(t, 0.F, 1.F);
    auto lerp = [t](unsigned char lhs, unsigned char rhs)
    {
        return static_cast<unsigned char>(
            static_cast<float>(lhs) + (static_cast<float>(rhs) - static_cast<float>(lhs)) * t);
    };

    return Color{
        lerp(a.r, b.r),
        lerp(a.g, b.g),
        lerp(a.b, b.b),
        lerp(a.a, b.a)
    };
}

/**
* @brief Returns a desaturated color used for destroyed entities.
* @param color Original tactical color.
* @return Wreck-tinted color.
*/
Color
DestroyedTint(Color color) noexcept
{
    return LerpColor(color, Color{ 62, 62, 62, color.a }, 0.72F);
}

/**
* @brief Returns the ImGui tactical color used for one side.
* @param side Tactical side identifier.
* @param alpha Optional alpha factor.
* @return Side color as one ImGui vector.
*/
ImVec4
ColorForSideUi(ppo::model::eEntitySide side, float alpha = 1.F)
{
    Color const color = ColorForSide(side, static_cast<unsigned char>(alpha * 255.F));
    return ImVec4{
        static_cast<float>(color.r) / 255.F,
        static_cast<float>(color.g) / 255.F,
        static_cast<float>(color.b) / 255.F,
        static_cast<float>(color.a) / 255.F
    };
}

/** @brief Déclare le type eSimulationControlGlyph, utilisé par Inspection et représentation graphique de l'environnement ECS. */
enum class eSimulationControlGlyph : unsigned char
{
    Play,
    Pause,
    Stop
};

bool
DrawSimulationControlButton(char const* id,
                            char const* label,
                            eSimulationControlGlyph glyph,
                            ImVec2 const& size,
                            ImVec4 const& baseColor)
{
    ImGui::PushID(id);
    ImGui::InvisibleButton("##control", size);

    bool const pressed = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    bool const hovered = ImGui::IsItemHovered();
    bool const held = ImGui::IsItemActive();

    ImVec2 const min = ImGui::GetItemRectMin();
    ImVec2 const max = ImGui::GetItemRectMax();
    ImVec2 const center{ 0.5F * (min.x + max.x), 0.5F * (min.y + max.y) - 4.F };
    ImDrawList* draw = ImGui::GetWindowDrawList();

    ImVec4 fill = baseColor;
    if (hovered)
    {
        fill.x = std::min(fill.x + 0.10F, 1.F);
        fill.y = std::min(fill.y + 0.10F, 1.F);
        fill.z = std::min(fill.z + 0.10F, 1.F);
    }
    if (held)
    {
        fill.x *= 0.85F;
        fill.y *= 0.85F;
        fill.z *= 0.85F;
    }

    draw->AddRectFilled(min, max, ImGui::ColorConvertFloat4ToU32(fill), 14.F);
    draw->AddRect(min,
                  max,
                  ImGui::ColorConvertFloat4ToU32(ImVec4(1.F, 1.F, 1.F, hovered ? 0.34F : 0.18F)),
                  14.F,
                  0,
                  hovered ? 2.2F : 1.4F);

    ImU32 const iconColor = ImGui::ColorConvertFloat4ToU32(ImVec4(0.97F, 0.985F, 1.F, 1.F));
    switch (glyph)
    {
    case eSimulationControlGlyph::Play:
    {
        ImVec2 const p0{ center.x - 10.F, center.y - 15.F };
        ImVec2 const p1{ center.x - 10.F, center.y + 15.F };
        ImVec2 const p2{ center.x + 18.F, center.y };
        draw->AddTriangleFilled(p0, p1, p2, iconColor);
        break;
    }
    case eSimulationControlGlyph::Pause:
    {
        ImVec2 const leftMin{ center.x - 15.F, center.y - 15.F };
        ImVec2 const leftMax{ center.x - 4.F, center.y + 15.F };
        ImVec2 const rightMin{ center.x + 4.F, center.y - 15.F };
        ImVec2 const rightMax{ center.x + 15.F, center.y + 15.F };
        draw->AddRectFilled(leftMin, leftMax, iconColor, 2.F);
        draw->AddRectFilled(rightMin, rightMax, iconColor, 2.F);
        break;
    }
    case eSimulationControlGlyph::Stop:
    {
        ImVec2 const squareMin{ center.x - 13.F, center.y - 13.F };
        ImVec2 const squareMax{ center.x + 13.F, center.y + 13.F };
        draw->AddRectFilled(squareMin, squareMax, iconColor, 3.F);
        break;
    }
    default:
        break;
    }

    ImVec2 const labelSize = ImGui::CalcTextSize(label);
    ImVec2 const labelPos{
        center.x - labelSize.x * 0.5F,
        max.y - labelSize.y - 8.F
    };
    draw->AddText(labelPos, ImGui::ColorConvertFloat4ToU32(ImVec4(1.F, 1.F, 1.F, 0.92F)), label);

    ImGui::PopID();
    return pressed;
}

/**
* @brief Converts one local north-east-down position to renderer coordinates.
* @param north North coordinate.
* @param east East coordinate.
* @param down Down coordinate.
* @param scale World scale factor.
* @return 3D world position used by the renderer.
*/
Vector3
ToWorld3D(float north, float east, float down, float scale)
{
    return Vector3{ east / scale, -down / scale, -north / scale };
}

/**
* @brief Builds the forward unit vector from one heading angle.
* @param rotation Heading angle in radians.
* @return Forward direction vector.
*/
Vector3
HeadingForward(float rotation)
{
    return Vector3{ std::sin(rotation), 0.F, -std::cos(rotation) };
}

/**
* @brief Returns the world scaling factor associated with one UI scale level.
* @param scaleLevel Scale selector used by the renderer.
* @return World scaling factor.
*/
float
WorldScaleForLevel(int scaleLevel) noexcept
{
    switch (std::clamp(scaleLevel, 0, 2))
    {
    case 0: return 1.F;
    case 1: return 1000.F;
    default: return 1852.F;
    }
}

/**
* @brief Returns the root directory used to load external 3D assets.
* @return Asset root path.
*/
std::filesystem::path
ModelsRootPath()
{
    return std::filesystem::path{ RESOURCE_PATH }.parent_path() / "3D";
}

/**
* @brief Returns the current steady-clock time in seconds.
* @return Current steady-clock time in seconds.
*/
double
GetSteadyTimeSeconds(void)
{
    using Clock = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<Seconds>(Clock::now().time_since_epoch()).count();
}

/**
* @brief Loads one external 3D model and normalizes its size for rendering.
* @param asset Asset slot to populate.
* @param path Model file path.
* @param targetExtent Target visual extent after normalization.
* @param yawOffsetDeg Static yaw offset applied at draw time.
*/
void
LoadModelAsset(ppo::renderer::EnvironmentRenderer::ModelAsset& asset,
               std::filesystem::path const& path,
               float targetExtent,
               float yawOffsetDeg)
{
    asset = {};
    asset.m_yawOffsetDeg = yawOffsetDeg;

    if (!std::filesystem::exists(path))
        return;

    Model model = LoadModel(path.string().c_str());
    if (model.meshCount <= 0)
    {
        UnloadModel(model);
        return;
    }

    BoundingBox const bounds = GetModelBoundingBox(model);
    Vector3 const extents{
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z
    };
    float const maxExtent = std::max({ std::fabs(extents.x), std::fabs(extents.y), std::fabs(extents.z), 1.0e-3F });

    asset.m_model = model;
    asset.m_bounds = bounds;
    asset.m_uniformScale = targetExtent / maxExtent;
    asset.m_groundOffset = -bounds.min.y * asset.m_uniformScale;
    asset.m_loaded = true;
}

/**
* @brief Builds the procedural fallback model used for objectives.
* @param asset Asset slot to populate.
* @param targetExtent Target visual extent after normalization.
*/
void
BuildGeneratedObjectiveModelAsset(ppo::renderer::EnvironmentRenderer::ModelAsset& asset,
                                  float targetExtent)
{
    asset = {};

    // Keep objective rendering self-contained and robust by generating the mesh in code.
    Mesh mesh = GenMeshCylinder(0.9F, 4.2F, 20);
    Model model = LoadModelFromMesh(mesh);
    if (model.meshCount <= 0)
    {
        UnloadModel(model);
        return;
    }

    BoundingBox const bounds = GetModelBoundingBox(model);
    Vector3 const extents{
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z
    };
    float const maxExtent = std::max({ std::fabs(extents.x), std::fabs(extents.y), std::fabs(extents.z), 1.0e-3F });

    asset.m_model = model;
    asset.m_bounds = bounds;
    asset.m_uniformScale = targetExtent / maxExtent;
    asset.m_groundOffset = -bounds.min.y * asset.m_uniformScale;
    asset.m_loaded = true;
}

/**
* @brief Releases one previously loaded model asset.
* @param asset Asset slot to clear.
*/
void
UnloadModelAsset(ppo::renderer::EnvironmentRenderer::ModelAsset& asset)
{
    if (asset.m_loaded)
        UnloadModel(asset.m_model);

    asset = {};
}

/**
* @brief Returns the archetype string stored in the entity snapshot.
* @param snapshot Snapshot registry.
* @param entity Target entity.
* @return Archetype name, or an empty string when unavailable.
*/
std::string
EntityArchetype(entt::registry const& snapshot, entt::entity entity)
{
    if (!snapshot.any_of<ppo::model::sEntityMetadata>(entity))
        return {};

    auto const& metadata = snapshot.get<ppo::model::sEntityMetadata>(entity);
    return metadata.m_archetype;
}

/**
* @brief Selects the aircraft model that should represent one entity.
* @param blueModel Friendly aircraft model asset.
* @param redModel Hostile aircraft model asset.
* @param snapshot Snapshot registry.
* @param entity Target entity.
* @param state Shared entity state.
* @return Pointer to the selected model asset, or null when unavailable.
*/
ppo::renderer::EnvironmentRenderer::ModelAsset const*
ResolveAircraftModelAsset(ppo::renderer::EnvironmentRenderer::ModelAsset const& blueModel,
                          ppo::renderer::EnvironmentRenderer::ModelAsset const& redModel,
                          entt::registry const& snapshot,
                          entt::entity entity,
                          ppo::model::sEntityState const& state)
{
    std::string const archetype = EntityArchetype(snapshot, entity);
    if (archetype == "trainingaircraft")
    {
        if (blueModel.m_loaded)
            return &blueModel;
    }
    else if (archetype == "aircraft")
    {
        if (redModel.m_loaded)
            return &redModel;
    }

    if (state.m_side == ppo::model::eEntitySide::Friend && blueModel.m_loaded)
        return &blueModel;

    if (state.m_side == ppo::model::eEntitySide::Hostile && redModel.m_loaded)
        return &redModel;

    return nullptr;
}

/**
* @brief Draws one aircraft model aligned with the entity heading.
* @param asset Model asset to draw.
* @param state Shared entity state.
* @param worldScale World scaling factor.
* @param tint Final tint color.
*/
void
DrawAircraftModel3D(ppo::renderer::EnvironmentRenderer::ModelAsset const& asset,
                    ppo::model::sEntityState const& state,
                    float worldScale,
                    Color tint)
{
    Vector3 center = ToWorld3D(state.m_x, state.m_y, 0.F, worldScale);
    center.y += asset.m_groundOffset;

    float const yawDeg = asset.m_yawOffsetDeg - state.m_rotation * DEG_PER_RAD;
    Vector3 const uniformScale{
        asset.m_uniformScale,
        asset.m_uniformScale,
        asset.m_uniformScale
    };

    DrawModelEx(asset.m_model, center, Vector3{ 0.F, 1.F, 0.F }, yawDeg, uniformScale, tint);
}

/**
* @brief Draws one static 3D model anchored to the entity position.
* @param asset Model asset to draw.
* @param state Shared entity state.
* @param worldScale World scaling factor.
* @param tint Final tint color.
*/
void
DrawStaticModel3D(ppo::renderer::EnvironmentRenderer::ModelAsset const& asset,
                  ppo::model::sEntityState const& state,
                  float worldScale,
                  Color tint)
{
    Vector3 center = ToWorld3D(state.m_x, state.m_y, 0.F, worldScale);
    center.y += asset.m_groundOffset;

    float const yawDeg = asset.m_yawOffsetDeg - state.m_rotation * DEG_PER_RAD;
    Vector3 const uniformScale{
        asset.m_uniformScale,
        asset.m_uniformScale,
        asset.m_uniformScale
    };

    DrawModelEx(asset.m_model, center, Vector3{ 0.F, 1.F, 0.F }, yawDeg, uniformScale, tint);
}

void
DrawAircraftGlyph3D(ppo::model::sEntityState const& state, float scale, Color color)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    Vector3 const forward = HeadingForward(state.m_rotation);
    Vector3 const right{ -forward.z, 0.F, forward.x };
    float const fuselageLength = 1.35F;
    float const wingSpan = 0.78F;
    float const tailSpan = 0.30F;
    float const bodyHeight = 0.14F;

    Vector3 const nose{
        center.x + forward.x * fuselageLength,
        center.y + bodyHeight * 0.95F,
        center.z + forward.z * fuselageLength
    };
    Vector3 const wingLeft{
        center.x - forward.x * 0.10F + right.x * wingSpan,
        center.y + bodyHeight * 0.28F,
        center.z - forward.z * 0.10F + right.z * wingSpan
    };
    Vector3 const wingRight{
        center.x - forward.x * 0.10F - right.x * wingSpan,
        center.y + bodyHeight * 0.28F,
        center.z - forward.z * 0.10F - right.z * wingSpan
    };
    Vector3 const tail{
        center.x - forward.x * 0.98F,
        center.y + bodyHeight * 0.18F,
        center.z - forward.z * 0.98F
    };
    Vector3 const tailLeft{
        tail.x + right.x * tailSpan,
        tail.y,
        tail.z + right.z * tailSpan
    };
    Vector3 const tailRight{
        tail.x - right.x * tailSpan,
        tail.y,
        tail.z - right.z * tailSpan
    };
    Vector3 const dorsal{
        center.x - forward.x * 0.34F,
        center.y + bodyHeight * 1.85F,
        center.z - forward.z * 0.34F
    };
    Vector3 const shadowOffset{ 0.07F, -0.05F, 0.08F };

    DrawTriangle3D(VectorAdd(nose, shadowOffset), VectorAdd(wingLeft, shadowOffset), VectorAdd(wingRight, shadowOffset), Fade(BLACK, 0.28F));
    DrawTriangle3D(VectorAdd(tail, shadowOffset), VectorAdd(wingLeft, shadowOffset), VectorAdd(wingRight, shadowOffset), Fade(BLACK, 0.20F));

    DrawTriangle3D(nose, wingLeft, wingRight, Fade(color, 0.96F));
    DrawTriangle3D(tail, wingRight, wingLeft, Fade(color, 0.80F));
    DrawTriangle3D(tail, tailLeft, tailRight, Fade(color, 0.64F));
    DrawTriangle3D(dorsal, wingRight, wingLeft, Fade(color, 0.72F));

    DrawLine3D(nose, tail, Fade(WHITE, 0.55F));
    DrawLine3D(wingLeft, wingRight, Fade(BLACK, 0.22F));
    DrawLine3D(tail, dorsal, Fade(color, 0.55F));
    DrawLine3D(tailLeft, tailRight, Fade(color, 0.48F));
}

/**
* @brief Selects the car visualization without changing the Mobile entity type.
*/
bool IsCarVisual(entt::registry const& snapshot, entt::entity id)
{
    if (!snapshot.any_of<ppo::model::sEntityMetadata>(id))
        return false;
    auto const& name = snapshot.get<ppo::model::sEntityMetadata>(id).m_archetype;
    return name == "defender_car" || name == "attacker_car";
}

/**
 * @brief Resolves the render-only episode role of one blue mobile.
 * @param snapshot Snapshot containing metadata and role overlay.
 * @param entity Candidate blue entity.
 * @return Leader, teammate, or none when no role is published.
*/
TacticalRole TacticalRoleForEntity(entt::registry const& snapshot,
                                   entt::entity const entity) noexcept
{
    if (!snapshot.ctx().contains<ppo::model::sBlueTeamRoleOverlay>()
        || !snapshot.any_of<ppo::model::sEntityMetadata>(entity))
    {
        return TacticalRole::None;
    }

    auto const& overlay =
        snapshot.ctx().get<ppo::model::sBlueTeamRoleOverlay>();
    if (!overlay.Valid())
        return TacticalRole::None;
    auto const& metadata =
        snapshot.get<ppo::model::sEntityMetadata>(entity);
    return ResolveTacticalRole(metadata.m_name, overlay.m_activeBlueCount,
                               overlay.m_leaderPhysicalIndex);
}

/**
 * @brief Returns the high-contrast accent associated with a tactical role.
 * @param role Render-only role marker.
 * @return Gold for leader, cyan for teammate, neutral gray otherwise.
 */
Color TacticalRoleColor(TacticalRole const role) noexcept
{
    if (role == TacticalRole::Leader)
        return Color{255, 196, 62, 255};
    if (role == TacticalRole::Teammate)
        return Color{65, 205, 255, 255};
    return Color{190, 205, 218, 255};
}

/**
 * @brief Draws a heading-aligned low-poly car without external assets.
 * @param state Current mobile state.
 * @param scale World-to-render scale.
 * @param body Team body color.
 * @param role Optional blue-team role used for roof accents.
 */
void DrawCarGlyph3D(ppo::model::sEntityState const& state, float scale,
                    Color body, TacticalRole const role)
{
    auto const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    Color const accent = TacticalRoleColor(role);
    rlPushMatrix();
    rlTranslatef(center.x, center.y, center.z);
    rlRotatef(-state.m_rotation * DEG_PER_RAD, 0.F, 1.F, 0.F);

    DrawCube(Vector3{0.08F, 0.025F, 0.08F}, 1.38F, 0.035F, 2.62F,
             Fade(BLACK, 0.38F));
    DrawCube(Vector3{0.F, 0.20F, 0.F}, 1.22F, 0.28F, 2.42F,
             ColorBrightness(body, -0.18F));
    DrawCube(Vector3{0.F, 0.34F, -0.47F}, 1.15F, 0.24F, 1.42F, body);
    DrawCube(Vector3{0.F, 0.35F, 0.77F}, 1.12F, 0.22F, 0.52F,
             ColorBrightness(body, -0.08F));
    DrawCubeWires(Vector3{0.F, 0.27F, 0.F}, 1.23F, 0.42F, 2.42F,
                  Fade(WHITE, 0.48F));

    DrawCube(Vector3{0.F, 0.57F, 0.18F}, 0.94F, 0.34F, 1.08F,
             Color{38, 58, 72, 255});
    DrawCube(Vector3{0.F, 0.755F, 0.18F}, 0.80F, 0.045F, 0.82F,
             accent);
    DrawCube(Vector3{0.F, 0.58F, -0.39F}, 0.84F, 0.22F, 0.055F,
             Color{144, 198, 224, 255});
    DrawCube(Vector3{0.F, 0.58F, 0.74F}, 0.84F, 0.20F, 0.055F,
             Color{79, 109, 126, 255});

    for (float side : {-0.63F, 0.63F})
        for (float axle : {-0.72F, 0.72F})
        {
            DrawCube(Vector3{side, 0.16F, axle}, 0.22F, 0.32F, 0.46F,
                     Color{17, 22, 29, 255});
            DrawCube(Vector3{side * 1.01F, 0.16F, axle}, 0.235F, 0.17F,
                     0.22F, Color{115, 127, 138, 255});
        }

    DrawCube(Vector3{0.F, 0.20F, -1.235F}, 0.52F, 0.11F, 0.045F,
             Color{20, 28, 34, 255});
    for (float side : {-0.39F, 0.39F})
    {
        DrawCube(Vector3{side, 0.35F, -1.235F}, 0.24F, 0.13F, 0.055F,
                 Color{225, 244, 255, 255});
        DrawCube(Vector3{side, 0.32F, 1.235F}, 0.23F, 0.12F, 0.055F,
                 Color{255, 62, 48, 255});
    }
    DrawCube(Vector3{0.F, 0.15F, 1.245F}, 0.76F, 0.08F, 0.04F,
             Color{30, 36, 42, 255});
    rlPopMatrix();
}

/**
 * @brief Draws screen-facing L/E badges above blue cars in one render target.
 * @param snapshot Snapshot containing entities and role overlay.
 * @param camera Camera used to project world positions.
 * @param width Render-target width in pixels.
 * @param height Render-target height in pixels.
 * @param worldScale World-to-render scale.
 */
void DrawTacticalRoleBadges(entt::registry const& snapshot,
                            Camera3D const& camera, int const width,
                            int const height, float const worldScale)
{
    if (!snapshot.ctx().contains<ppo::model::sBlueTeamRoleOverlay>()
        || width <= 0 || height <= 0)
    {
        return;
    }

    Vector3 const cameraForward =
        VectorNormalizeSafe(VectorSub(camera.target, camera.position));
    float const fontSize = std::clamp(
        0.052F * static_cast<float>(std::min(width, height)), 15.F, 24.F);
    float const radius = fontSize * 0.72F;
    Font const font = GetFontDefault();
    auto const entities = snapshot.view<ppo::model::sEntityState>();
    for (entt::entity const entity : entities)
    {
        auto const role = TacticalRoleForEntity(snapshot, entity);
        if (role == TacticalRole::None || !IsCarVisual(snapshot, entity))
            continue;

        auto const& state = entities.get<ppo::model::sEntityState>(entity);
        Vector3 world = ToWorld3D(state.m_x, state.m_y, 0.F, worldScale);
        world.y += 1.35F;
        if (VectorDot(cameraForward, VectorSub(world, camera.position)) <= 0.F)
            continue;

        Vector2 const screen = GetWorldToScreenEx(world, camera, width, height);
        if (screen.x < -radius || screen.x > static_cast<float>(width) + radius
            || screen.y < -radius || screen.y > static_cast<float>(height) + radius)
        {
            continue;
        }

        Color const accent = TacticalRoleColor(role);
        DrawCircleV(Vector2{screen.x + 2.F, screen.y + 2.F}, radius,
                    Fade(BLACK, 0.62F));
        DrawCircleV(screen, radius, Fade(accent, 0.94F));
        DrawCircleLines(static_cast<int>(screen.x), static_cast<int>(screen.y),
                        radius, Fade(WHITE, 0.92F));
        char const* const marker = TacticalRoleMarker(role);
        Vector2 const textSize = MeasureTextEx(font, marker, fontSize, 1.F);
        DrawTextEx(font, marker,
                   Vector2{screen.x - 0.5F * textSize.x,
                           screen.y - 0.54F * textSize.y},
                   fontSize, 1.F, Color{12, 20, 28, 255});
    }
}

/**
 * @brief Draws screen-facing labels for role-ordered patrol references.
 * @param overlay Exact references last supplied to generalized MAPPO.
 * @param camera Camera used to project reference positions.
 * @param width Render-target width in pixels.
 * @param height Render-target height in pixels.
 * @param worldScale World-to-render scale.
 */
void DrawPatrolReferenceLabels(
    ppo::model::sPatrolReferenceOverlay const& overlay,
    Camera3D const& camera, int const width, int const height,
    float const worldScale)
{
    if (!overlay.Valid() || width <= 0 || height <= 0)
        return;

    Vector3 const cameraForward =
        VectorNormalizeSafe(VectorSub(camera.target, camera.position));
    float const fontSize = std::clamp(
        0.045F * static_cast<float>(std::min(width, height)), 13.F, 20.F);
    float const radius = fontSize * 0.78F;
    Font const font = GetFontDefault();
    for (std::size_t roleIndex = 0U;
         roleIndex < overlay.m_activeReferenceCount; ++roleIndex)
    {
        auto const& reference = overlay.m_references[roleIndex];
        if (!reference.m_valid)
            continue;

        Vector3 world = ToWorld3D(reference.m_projectedNorth,
                                  reference.m_projectedEast,
                                  -0.48F * worldScale, worldScale);
        if (VectorDot(cameraForward, VectorSub(world, camera.position)) <= 0.F)
            continue;
        Vector2 const screen = GetWorldToScreenEx(world, camera, width, height);
        if (screen.x < -radius || screen.x > static_cast<float>(width) + radius
            || screen.y < -radius
            || screen.y > static_cast<float>(height) + radius)
        {
            continue;
        }

        Color const accent = TacticalRoleColor(
            roleIndex == 0U ? TacticalRole::Leader : TacticalRole::Teammate);
        char marker[4]{};
        if (roleIndex == 0U)
            std::snprintf(marker, sizeof(marker), "L");
        else
            std::snprintf(marker, sizeof(marker), "W%u",
                          static_cast<unsigned>(roleIndex));
        DrawCircleV(Vector2{screen.x + 2.F, screen.y + 2.F}, radius,
                    Fade(BLACK, 0.62F));
        DrawCircleV(screen, radius, Fade(accent, 0.92F));
        DrawCircleLines(static_cast<int>(screen.x), static_cast<int>(screen.y),
                        radius, Fade(WHITE, 0.88F));
        Vector2 const textSize = MeasureTextEx(font, marker, fontSize, 1.F);
        DrawTextEx(font, marker,
                   Vector2{screen.x - 0.5F * textSize.x,
                           screen.y - 0.54F * textSize.y},
                   fontSize, 1.F, Color{12, 20, 28, 255});
    }
}

/**
 * @brief Draws screen-facing labels for the ten frozen patrol waypoints.
 * @param overlay Generated waypoint sequence and active index.
 * @param camera Camera used to project waypoint positions.
 * @param width Render-target width in pixels.
 * @param height Render-target height in pixels.
 * @param worldScale World-to-render scale.
 */
void DrawPatrolWaypointLabels(
    ppo::model::sPatrolWaypointOverlay const& overlay,
    Camera3D const& camera, int const width, int const height,
    float const worldScale)
{
    if (!overlay.Valid() || width <= 0 || height <= 0)
        return;
    Vector3 const cameraForward =
        VectorNormalizeSafe(VectorSub(camera.target, camera.position));
    float const fontSize = std::clamp(
        0.032F * static_cast<float>(std::min(width, height)), 10.F, 15.F);
    Font const font = GetFontDefault();
    for (std::size_t index = 0U;
         index < ppo::model::sPatrolWaypointOverlay::WaypointCount; ++index)
    {
        auto const& waypoint = overlay.m_waypoints[index];
        Vector3 world = ToWorld3D(
            waypoint.m_north, waypoint.m_east,
            -0.40F * worldScale, worldScale);
        if (VectorDot(cameraForward, VectorSub(world, camera.position)) <= 0.F)
            continue;
        Vector2 const screen = GetWorldToScreenEx(world, camera, width, height);
        if (screen.x < -30.F || screen.x > static_cast<float>(width) + 30.F
            || screen.y < -20.F
            || screen.y > static_cast<float>(height) + 20.F)
        {
            continue;
        }
        bool const active = index == overlay.m_activeWaypointIndex;
        bool const validated = index < overlay.m_validatedWaypointCount;
        Color const color = active ? Color{255, 205, 72, 255}
            : validated ? Color{115, 135, 148, 210}
            : Color{90, 225, 170, 235};
        char marker[6]{};
        std::snprintf(marker, sizeof(marker), "WP%u",
            static_cast<unsigned>(index + 1U));
        Vector2 const textSize = MeasureTextEx(font, marker, fontSize, 1.F);
        DrawRectangleRounded(
            Rectangle{screen.x - 0.5F * textSize.x - 3.F,
                screen.y - 0.5F * textSize.y - 2.F,
                textSize.x + 6.F, textSize.y + 4.F},
            0.25F, 4, Fade(BLACK, 0.62F));
        DrawTextEx(font, marker,
            Vector2{screen.x - 0.5F * textSize.x,
                screen.y - 0.54F * textSize.y},
            fontSize, 1.F, color);
    }
}

/**
* @brief Draws a north/east rectangle on the tactical ground plane.
*/
void DrawDefendedArea3D(ppo::model::sDefendedArea const& area, float scale,
                        Color const color = Color{79, 145, 245, 225})
{
    float const n0 = area.m_north - area.m_halfNorth;
    float const n1 = area.m_north + area.m_halfNorth;
    float const e0 = area.m_east - area.m_halfEast;
    float const e1 = area.m_east + area.m_halfEast;
    auto a = ToWorld3D(n0, e0, -0.06F * scale, scale);
    auto b = ToWorld3D(n0, e1, -0.06F * scale, scale);
    auto c = ToWorld3D(n1, e1, -0.06F * scale, scale);
    auto d = ToWorld3D(n1, e0, -0.06F * scale, scale);
    DrawLineEx3D(a, b, 0.055F, color);
    DrawLineEx3D(b, c, 0.055F, color);
    DrawLineEx3D(c, d, 0.055F, color);
    DrawLineEx3D(d, a, 0.055F, color);
}

/**
 * @brief Draws the sampled leader view of the analytical patrol curve.
 * @param overlay Snapshot-only curve samples.
 * @param scale World-to-render scale.
 * @param reducedDetail True to use sparse, inexpensive line primitives.
 */
void DrawPatrolTrack3D(ppo::model::sPatrolTrackOverlay const& overlay,
                       float scale, bool const reducedDetail)
{
    std::size_t const count = std::min(
        overlay.m_sampleCount,
        ppo::model::sPatrolTrackOverlay::SampleCapacity);
    if (count < 2U)
        return;

    Color const color{70, 190, 255, 225};
    std::size_t const stride = reducedDetail ? 4U : 1U;
    for (std::size_t sample = stride; sample < count; sample += stride)
    {
        Vector3 const previous = ToWorld3D(
            overlay.m_north[sample - stride],
            overlay.m_east[sample - stride], -0.09F * scale, scale);
        Vector3 const current = ToWorld3D(
            overlay.m_north[sample],
            overlay.m_east[sample], -0.09F * scale, scale);
        if (reducedDetail)
            DrawLine3D(previous, current, color);
        else
            DrawLineEx3D(previous, current, 0.075F, color);
    }
    std::size_t const arrowStride = reducedDetail ? 32U : 16U;
    for (std::size_t sample = arrowStride;
         sample + 1U < count; sample += arrowStride)
    {
        Vector3 const tip = ToWorld3D(
            overlay.m_north[sample], overlay.m_east[sample],
            -0.11F * scale, scale);
        Vector3 const previous = ToWorld3D(
            overlay.m_north[sample - 1U], overlay.m_east[sample - 1U],
            -0.11F * scale, scale);
        Vector3 const forward = VectorNormalizeSafe(VectorSub(tip, previous));
        Vector3 const side{-forward.z, 0.F, forward.x};
        Vector3 const base = VectorSub(tip, VectorScale(forward, 0.8F));
        Vector3 const left = VectorAdd(base, VectorScale(side, 0.35F));
        Vector3 const right = VectorSub(base, VectorScale(side, 0.35F));
        if (reducedDetail)
        {
            DrawLine3D(tip, left, color);
            DrawLine3D(tip, right, color);
        }
        else
        {
            DrawLineEx3D(tip, left, 0.06F, color);
            DrawLineEx3D(tip, right, 0.06F, color);
        }
    }
}

/**
 * @brief Draws projected markers and observation-guidance arrows for MAPPO.
 * @param overlay Exact role-ordered references last supplied to the policy.
 * @param scale World-to-render scale.
 * @param reducedDetail True to use cheaper primitives in multi-camera views.
 */
void DrawPatrolReferences3D(
    ppo::model::sPatrolReferenceOverlay const& overlay,
    float const scale, bool const reducedDetail)
{
    if (!overlay.Valid())
        return;

    for (std::size_t roleIndex = 0U;
         roleIndex < overlay.m_activeReferenceCount; ++roleIndex)
    {
        auto const& reference = overlay.m_references[roleIndex];
        if (!reference.m_valid)
            continue;

        Color const accent = TacticalRoleColor(
            roleIndex == 0U ? TacticalRole::Leader : TacticalRole::Teammate);
        Vector3 const projected = ToWorld3D(
            reference.m_projectedNorth, reference.m_projectedEast,
            -0.08F * scale, scale);
        Vector3 const guidance = ToWorld3D(
            reference.m_guidanceNorth, reference.m_guidanceEast,
            -0.12F * scale, scale);
        DrawCylinderWires(projected, 0.32F, 0.32F, 0.07F,
                          reducedDetail ? 12 : 24, Fade(accent, 0.96F));
        if (reducedDetail)
            DrawLine3D(projected, guidance, Fade(accent, 0.48F));
        else
            DrawLineEx3D(projected, guidance, 0.035F,
                         Fade(accent, 0.48F));
        DrawSphere(guidance, reducedDetail ? 0.07F : 0.10F,
                   Fade(accent, 0.88F));

        Vector3 const forward = HeadingForward(reference.m_guidanceHeading);
        Vector3 const side{-forward.z, 0.F, forward.x};
        float const arrowLength = reducedDetail ? 0.72F : 1.05F;
        Vector3 const tip = VectorAdd(
            guidance, VectorScale(forward, arrowLength));
        Vector3 const arrowBase = VectorSub(
            tip, VectorScale(forward, 0.34F));
        Vector3 const left = VectorAdd(
            arrowBase, VectorScale(side, 0.20F));
        Vector3 const right = VectorSub(
            arrowBase, VectorScale(side, 0.20F));
        if (reducedDetail)
        {
            DrawLine3D(guidance, tip, accent);
            DrawLine3D(tip, left, accent);
            DrawLine3D(tip, right, accent);
        }
        else
        {
            DrawLineEx3D(guidance, tip, 0.065F, accent);
            DrawLineEx3D(tip, left, 0.055F, accent);
            DrawLineEx3D(tip, right, 0.055F, accent);
        }
    }
}

/**
 * @brief Draws the complete frozen waypoint sequence and its tangent headings.
 * @param overlay Generated 26-waypoint render view.
 * @param scale World-to-render scale.
 * @param reducedDetail True to use cheaper primitives in multi-camera views.
 */
void DrawPatrolWaypoints3D(
    ppo::model::sPatrolWaypointOverlay const& overlay,
    float const scale, bool const reducedDetail)
{
    if (!overlay.Valid())
        return;
    for (std::size_t index = 0U;
         index < ppo::model::sPatrolWaypointOverlay::WaypointCount; ++index)
    {
        auto const& waypoint = overlay.m_waypoints[index];
        bool const active = index == overlay.m_activeWaypointIndex;
        bool const validated = index < overlay.m_validatedWaypointCount;
        Color const color = active ? Color{255, 205, 72, 255}
            : validated ? Color{105, 125, 140, 190}
            : Color{90, 225, 170, 230};
        Vector3 const center = ToWorld3D(
            waypoint.m_north, waypoint.m_east, -0.16F * scale, scale);
        float const radius = active ? 0.38F : 0.25F;
        DrawCylinderWires(center, radius, radius, 0.08F,
            reducedDetail ? 10 : 20, color);
        Vector3 const forward = HeadingForward(waypoint.m_heading);
        Vector3 const tip = VectorAdd(
            center, VectorScale(forward, active ? 0.95F : 0.58F));
        if (reducedDetail)
            DrawLine3D(center, tip, color);
        else
            DrawLineEx3D(center, tip, active ? 0.065F : 0.04F, color);
    }
}

void
DrawSamGlyph3D(ppo::model::sEntityState const& state, float scale, Color color)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    Vector3 const forward = HeadingForward(state.m_rotation);
    Vector3 const right{ -forward.z, 0.F, forward.x };

    Color const baseColor = Fade(color, 0.92F);
    Color const accentColor{ 232, 236, 240, 220 };
    Color const launcherColor{ 90, 104, 118, 255 };
    Color const radarGlow{ 145, 235, 210, 165 };

    DrawCylinder(Vector3{ center.x, 0.15F, center.z }, 0.46F, 0.54F, 0.30F, 18, baseColor);
    DrawCylinderWires(Vector3{ center.x, 0.15F, center.z }, 0.49F, 0.49F, 0.32F, 18, Fade(BLACK, 0.35F));

    Vector3 const cabinCenter{
        center.x - forward.x * 0.03F,
        0.40F,
        center.z - forward.z * 0.03F
    };
    DrawCubeV(cabinCenter, Vector3{ 0.62F, 0.22F, 0.84F }, Fade(baseColor, 0.98F));
    DrawCubeWiresV(cabinCenter, Vector3{ 0.62F, 0.22F, 0.84F }, Fade(accentColor, 0.42F));

    Vector3 const mastBase{ center.x, 0.48F, center.z };
    Vector3 const mastTop{ center.x, 0.94F, center.z };
    DrawCylinderEx(mastBase, mastTop, 0.05F, 0.04F, 12, Fade(accentColor, 0.82F));

    Vector3 const radarCenter{
        center.x + right.x * 0.17F,
        1.02F,
        center.z + right.z * 0.17F
    };
    DrawSphere(radarCenter, 0.16F, radarGlow);
    DrawSphereWires(radarCenter, 0.19F, 10, 10, Fade(accentColor, 0.72F));

    auto drawLauncherTube = [&](float sideOffset)
    {
        Vector3 const start{
            center.x + right.x * sideOffset - forward.x * 0.18F,
            0.77F,
            center.z + right.z * sideOffset - forward.z * 0.18F
        };
        Vector3 const end{
            start.x + forward.x * 0.76F,
            1.02F,
            start.z + forward.z * 0.76F
        };

        DrawCylinderEx(start, end, 0.09F, 0.07F, 12, launcherColor);
        DrawCylinderEx(start, end, 0.055F, 0.045F, 10, Fade(accentColor, 0.84F));
        DrawSphere(end, 0.05F, Fade(accentColor, 0.95F));
    };

    drawLauncherTube(-0.16F);
    drawLauncherTube(0.16F);
}

void
DrawObjectiveGlyph3D(ppo::model::sEntityState const& state, float scale, Color color)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    float const pulse = 0.5F + 0.5F * std::sin(static_cast<float>(GetTime()) * 2.2F);

    DrawSphere(center, 0.28F, color);
    DrawSphereWires(center, 0.34F, 10, 10, Fade(WHITE, 0.45F));
    DrawCylinder(Vector3{ center.x, 0.34F, center.z }, 0.03F, 0.03F, 0.70F + 0.08F * pulse, 8, Fade(WHITE, 0.62F));

    Color const markerColor{ 255, 228, 130, static_cast<unsigned char>(190 + 50.F * pulse) };
    DrawCylinderWires(Vector3{ center.x, 0.06F, center.z },
                      0.58F + 0.06F * pulse,
                      0.58F + 0.06F * pulse,
                      0.08F,
                      30,
                      markerColor);
    DrawCylinderWires(Vector3{ center.x, 0.14F, center.z },
                      0.40F,
                      0.40F,
                      0.04F,
                      24,
                      Fade(markerColor, 0.85F));
}

void
DrawPrimaryTargetAura3D(ppo::model::sEntityState const& state, float scale)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    float const pulse = 0.5F + 0.5F * std::sin(static_cast<float>(GetTime()) * 2.8F);
    float const innerRadius = 0.50F + 0.10F * pulse;
    float const outerRadius = 0.86F + 0.16F * pulse;
    Color const glowColor{ 255, 205, 72, static_cast<unsigned char>(150 + 60 * pulse) };
    Color const coreColor{ 255, 241, 170, static_cast<unsigned char>(210 + 30 * pulse) };

    DrawCylinder(Vector3{ center.x, 0.12F, center.z },
                 outerRadius * 1.05F,
                 outerRadius * 1.30F,
                 0.26F + 0.08F * pulse,
                 36,
                 Fade(glowColor, 0.22F));
    DrawCylinder(Vector3{ center.x, 0.14F, center.z },
                 innerRadius * 1.15F,
                 outerRadius,
                 0.18F + 0.05F * pulse,
                 36,
                 Fade(coreColor, 0.28F));
    DrawCylinderWires(Vector3{ center.x, 0.14F, center.z }, outerRadius, outerRadius, 0.22F, 36, coreColor);
    DrawCylinderWires(Vector3{ center.x, 0.14F, center.z }, innerRadius, innerRadius, 0.16F, 36, Fade(coreColor, 0.75F));
    DrawSphereWires(Vector3{ center.x, center.y + 0.34F, center.z }, innerRadius, 14, 14, Fade(coreColor, 0.95F));
    DrawLine3D(Vector3{ center.x, center.y + 0.45F, center.z },
               Vector3{ center.x, center.y + 1.55F + 0.20F * pulse, center.z },
               Fade(coreColor, 0.90F));
}

void
DrawTrail3D(std::deque<ppo::renderer::EnvironmentRenderer::TrailPoint> const& trail,
            float duration,
            Color color,
            bool const reducedDetail)
{
    if (trail.size() < 2 || duration <= 0.F)
        return;

    double const now = GetTime();
    std::size_t const stride = reducedDetail ? 2U : 1U;
    for (std::size_t index = stride; index < trail.size(); index += stride)
    {
        auto const& previous = trail[index - stride];
        auto const& current = trail[index];
        float const age = static_cast<float>(now - current.m_timestamp);
        float const fade = std::clamp(1.F - age / duration, 0.F, 1.F);
        if (fade <= 0.01F)
            continue;

        Color const segmentColor{ color.r, color.g, color.b, static_cast<unsigned char>(180.F * fade) };
        DrawLine3D(previous.m_position, current.m_position, segmentColor);
        if (!reducedDetail)
        {
            DrawSphere(current.m_position, 0.04F + 0.015F * fade,
                       Fade(segmentColor, 0.35F));
        }
    }
}

void
DrawHemisphere(Vector3 const& center, float radius, Color color, int slices = 28, int rings = 12)
{
    for (int ring = 0; ring < rings; ++ring)
    {
        float const phi0 = (0.5F * PI_F * static_cast<float>(ring)) / static_cast<float>(rings);
        float const phi1 = (0.5F * PI_F * static_cast<float>(ring + 1)) / static_cast<float>(rings);

        for (int slice = 0; slice < slices; ++slice)
        {
            float const theta0 = (2.F * PI_F * static_cast<float>(slice)) / static_cast<float>(slices);
            float const theta1 = (2.F * PI_F * static_cast<float>(slice + 1)) / static_cast<float>(slices);

            auto point = [&](float phi, float theta)
            {
                return Vector3{
                    center.x + radius * std::cos(theta) * std::sin(phi),
                    center.y + radius * std::cos(phi),
                    center.z + radius * std::sin(theta) * std::sin(phi)
                };
            };

            Vector3 const a = point(phi0, theta0);
            Vector3 const b = point(phi0, theta1);
            Vector3 const c = point(phi1, theta1);
            Vector3 const d = point(phi1, theta0);

            DrawTriangle3D(a, b, c, color);
            DrawTriangle3D(a, c, d, color);
        }
    }
}

void
DrawCylinderSector(Vector3 const& center,
                   float radius,
                   float startAngle,
                   float endAngle,
                   float height,
                   Color color,
                   int slices = 40)
{
    float const step = (endAngle - startAngle) / static_cast<float>(std::max(slices, 1));
    float const y0 = center.y - height * 0.5F;
    float const y1 = center.y + height * 0.5F;

    for (int i = 0; i < slices; ++i)
    {
        float const a0 = startAngle + step * static_cast<float>(i);
        float const a1 = a0 + step;

        Vector3 const p00{ center.x + radius * std::sin(a0), y0, center.z - radius * std::cos(a0) };
        Vector3 const p01{ center.x + radius * std::sin(a1), y0, center.z - radius * std::cos(a1) };
        Vector3 const p10{ center.x + radius * std::sin(a0), y1, center.z - radius * std::cos(a0) };
        Vector3 const p11{ center.x + radius * std::sin(a1), y1, center.z - radius * std::cos(a1) };

        DrawTriangle3D(p00, p01, p11, color);
        DrawTriangle3D(p00, p11, p10, color);
    }
}

void
DrawSectorArc3D(Vector3 const& center,
                float radius,
                float startAngle,
                float endAngle,
                float height,
                Color color,
                int slices = 40,
                float thickness = 1.8F)
{
    float const step = (endAngle - startAngle) / static_cast<float>(std::max(slices, 1));
    float const y = center.y + height * 0.5F;

    for (int i = 1; i <= slices; ++i)
    {
        float const a0 = startAngle + step * static_cast<float>(i - 1);
        float const a1 = startAngle + step * static_cast<float>(i);
        Vector3 const p0{ center.x + radius * std::sin(a0), y, center.z - radius * std::cos(a0) };
        Vector3 const p1{ center.x + radius * std::sin(a1), y, center.z - radius * std::cos(a1) };
        DrawLineEx3D(p0, p1, thickness, color);
    }
}

void
DrawEngagementCone3D(Vector3 const& center,
                     float heading,
                     float halfAngle,
                     float radius,
                     float apexHeight,
                     Color fillColor,
                     Color edgeColor,
                     int const detailSlices)
{
    if (radius <= 0.05F || halfAngle <= 0.01F)
        return;

    Vector3 const forward = HeadingForward(heading);
    Vector3 const apex{
        center.x + forward.x * 0.28F,
        center.y + apexHeight,
        center.z + forward.z * 0.28F
    };
    Vector3 const origin{
        center.x + forward.x * 0.12F,
        center.y + 0.03F,
        center.z + forward.z * 0.12F
    };

    constexpr int maximumSlices = 40;
    int const slices = std::clamp(detailSlices, 8, maximumSlices);
    std::array<Vector3, static_cast<std::size_t>(maximumSlices) + 1U> arc{};

    for (int index = 0; index <= slices; ++index)
    {
        float const t = static_cast<float>(index) / static_cast<float>(slices);
        float const angle = heading - halfAngle + t * (2.F * halfAngle);
        arc[static_cast<std::size_t>(index)] = Vector3{
            center.x + radius * std::sin(angle),
            center.y + 0.02F,
            center.z - radius * std::cos(angle)
        };
    }

    for (std::size_t index = 1U;
         index <= static_cast<std::size_t>(slices); ++index)
    {
        DrawTriangle3D(apex, arc[index - 1], arc[index], fillColor);
        DrawTriangle3D(origin, arc[index - 1], arc[index], Fade(fillColor, 0.20F));
    }

    float const innerRadius = radius * 0.58F;
    Color const innerFill = Fade(edgeColor, 0.16F);
    for (int index = 1; index <= slices; ++index)
    {
        float const t0 = static_cast<float>(index - 1) / static_cast<float>(slices);
        float const t1 = static_cast<float>(index) / static_cast<float>(slices);
        float const a0 = heading - halfAngle + t0 * (2.F * halfAngle);
        float const a1 = heading - halfAngle + t1 * (2.F * halfAngle);
        Vector3 const p0{ center.x + innerRadius * std::sin(a0), center.y + 0.06F, center.z - innerRadius * std::cos(a0) };
        Vector3 const p1{ center.x + innerRadius * std::sin(a1), center.y + 0.06F, center.z - innerRadius * std::cos(a1) };
        DrawTriangle3D(apex, p0, p1, innerFill);
    }

    DrawLineEx3D(apex, arc[0], 0.055F, edgeColor);
    DrawLineEx3D(
        apex, arc[static_cast<std::size_t>(slices)], 0.055F, edgeColor);
    for (std::size_t index = 1U;
         index <= static_cast<std::size_t>(slices); ++index)
        DrawLineEx3D(arc[index - 1], arc[index], 0.045F, Fade(edgeColor, 0.92F));
}

void
DrawDestroyMarker3D(Vector3 const& center)
{
    DrawLine3D(Vector3{ center.x - 0.25F, center.y, center.z - 0.25F },
               Vector3{ center.x + 0.25F, center.y, center.z + 0.25F },
               WHITE);
    DrawLine3D(Vector3{ center.x - 0.25F, center.y, center.z + 0.25F },
               Vector3{ center.x + 0.25F, center.y, center.z - 0.25F },
               WHITE);
}

/**
* @brief Draws a wreck-style overlay on top of one destroyed entity.
* @param state Destroyed entity state.
* @param scale World scaling factor.
*/
void
DrawDestroyedWreck3D(ppo::model::sEntityState const& state, float scale)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);

    float baseRadius = 0.42F;
    float smokeHeight = 0.78F;
    switch (state.m_type)
    {
    case ppo::model::eEntityType::Objective:
        baseRadius = 0.95F;
        smokeHeight = 1.75F;
        break;
    case ppo::model::eEntityType::RestrictedZone:
        baseRadius = 0.58F;
        smokeHeight = 1.00F;
        break;
    case ppo::model::eEntityType::Mobile:
        baseRadius = 0.50F;
        smokeHeight = 0.92F;
        break;
    default:
        break;
    }

    float const pulse = 0.5F + 0.5F * std::sin(static_cast<float>(GetTime()) * 1.7F);
    Color const scorchColor{ 20, 20, 20, 210 };
    Color const emberColor{ 255, 102, 44, static_cast<unsigned char>(110.F + 60.F * pulse) };
    Color const smokeColor{ 54, 54, 58, static_cast<unsigned char>(120.F + 40.F * pulse) };

    DrawCylinder(Vector3{ center.x, 0.02F, center.z },
                 baseRadius * 1.15F,
                 baseRadius * 0.92F,
                 0.05F,
                 24,
                 Fade(scorchColor, 0.92F));
    DrawCylinderWires(Vector3{ center.x, 0.03F, center.z },
                      baseRadius * 1.24F,
                      baseRadius * 1.24F,
                      0.04F,
                      22,
                      Fade(emberColor, 0.42F));

    DrawSphere(Vector3{ center.x, center.y + 0.20F, center.z },
               baseRadius * 0.35F,
               Fade(emberColor, 0.42F));
    DrawSphere(Vector3{ center.x, center.y + smokeHeight, center.z },
               baseRadius * 0.60F,
               Fade(smokeColor, 0.46F));
    DrawSphere(Vector3{ center.x + 0.18F, center.y + smokeHeight * 0.68F, center.z - 0.12F },
               baseRadius * 0.42F,
               Fade(smokeColor, 0.38F));

    DrawLineEx3D(Vector3{ center.x - baseRadius * 0.70F, center.y + 0.06F, center.z - baseRadius * 0.55F },
                 Vector3{ center.x + baseRadius * 0.72F, center.y + 0.18F, center.z + baseRadius * 0.48F },
                 0.055F,
                 Fade(Color{ 34, 34, 34, 255 }, 0.92F));
    DrawLineEx3D(Vector3{ center.x - baseRadius * 0.64F, center.y + 0.10F, center.z + baseRadius * 0.52F },
                 Vector3{ center.x + baseRadius * 0.60F, center.y + 0.04F, center.z - baseRadius * 0.58F },
                 0.050F,
                 Fade(Color{ 44, 44, 44, 255 }, 0.88F));
}

/**
* @brief Draws a short-lived 3D explosion effect around a destroyed entity.
*/
void
DrawExplosion3D(ppo::renderer::EnvironmentRenderer::ExplosionEffect const& effect, float worldScale)
{
    double const now = GetSteadyTimeSeconds();
    float const age = static_cast<float>(now - effect.m_startTime);
    constexpr float kExplosionDuration = 2.0F;
    if (age < 0.F || age > kExplosionDuration)
        return;

    float const t = age / kExplosionDuration;
    float const easeOut = 1.F - (1.F - t) * (1.F - t);
    float const fade = std::clamp(1.F - t, 0.F, 1.F);
    float const shockRadius = effect.m_baseRadius + 3.35F * easeOut;
    float const fireballRadius = effect.m_baseRadius * (0.90F + 1.65F * easeOut);
    float const smokeRise = effect.m_heightScale * (0.95F + 1.35F * easeOut);
    float const sparkRise = effect.m_heightScale * 0.45F * (1.F - t);

    Vector3 const base = ToWorld3D(effect.m_x, effect.m_y, effect.m_z, worldScale);
    Vector3 const core{ base.x, base.y + 0.18F * effect.m_heightScale, base.z };
    Color const outerGlow{ 255, 112, 48, static_cast<unsigned char>(165.F * fade) };
    Color const fireColor{ 255, 176, 72, static_cast<unsigned char>(220.F * fade) };
    Color const smokeColor{ 90, 84, 84, static_cast<unsigned char>(125.F * fade) };

    DrawSphere(core, fireballRadius, fireColor);
    DrawSphere(Vector3{ core.x + 0.32F * effect.m_heightScale, core.y + 0.38F * effect.m_heightScale, core.z - 0.22F * effect.m_heightScale },
               fireballRadius * 0.78F,
               outerGlow);
    DrawSphere(Vector3{ core.x - 0.28F * effect.m_heightScale, core.y + 0.22F * effect.m_heightScale, core.z + 0.30F * effect.m_heightScale },
               fireballRadius * 0.64F,
               Fade(fireColor, 0.78F));
    DrawSphere(Vector3{ core.x, core.y + smokeRise, core.z },
               fireballRadius * (0.74F + 0.16F * effect.m_heightScale),
               smokeColor);

    DrawSphereWires(core, shockRadius, 18, 14, Color{ 255, 226, 188, static_cast<unsigned char>(210.F * fade) });
    DrawCylinderWires(Vector3{ core.x, core.y + 0.12F, core.z },
                      shockRadius * 0.82F,
                      shockRadius * 1.05F,
                      0.18F + 0.18F * easeOut * effect.m_heightScale,
                      28,
                      Fade(outerGlow, 0.92F));

    for (int i = 0; i < 10; ++i)
    {
        float const angle = (2.F * PI_F * static_cast<float>(i)) / 10.F + t * 1.1F;
        float const sparkLength = 0.70F + 1.25F * easeOut * effect.m_heightScale;
        Vector3 const start{
            core.x + 0.22F * std::sin(angle),
            core.y + 0.22F + 0.10F * static_cast<float>(i % 3),
            core.z - 0.22F * std::cos(angle)
        };
        Vector3 const end{
            start.x + sparkLength * std::sin(angle),
            start.y + sparkRise,
            start.z - sparkLength * std::cos(angle)
        };
        DrawLineEx3D(start, end, 0.038F * fade + 0.010F, Fade(Color{ 255, 214, 130, 255 }, 0.95F * fade));
    }
}

bool
MatchesFilter(std::string const& filter, ppo::model::sEntityState const& state, entt::entity entity)
{
    if (filter.empty())
        return true;

    std::string haystack = EntityTypeLabel(state.m_type);
    haystack += " ";
    haystack += SideLabel(state.m_side);
    haystack += " ";
    haystack += std::to_string(static_cast<entt::id_type>(entity));

    std::transform(haystack.begin(), haystack.end(), haystack.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return haystack.find(filter) != std::string::npos;
}

std::string
EntityDisplayName(ppo::model::sEntityState const& state, entt::entity entity)
{
    std::string label = EntityTypeLabel(state.m_type);
    label += " #";
    label += std::to_string(static_cast<entt::id_type>(entity));
    return label;
}

void
DrawPseudoLidarCone(Vector3 const& center,
                    float startAngle,
                    float endAngle,
                    float radius0,
                    float radius1,
                    Color color)
{
    float const coneHeight = 0.85F;
    Vector3 const apex{ center.x, center.y + coneHeight, center.z };
    Vector3 const a{ center.x + radius0 * std::sin(startAngle), center.y + 0.03F, center.z - radius0 * std::cos(startAngle) };
    Vector3 const b{ center.x + radius1 * std::sin(endAngle), center.y + 0.03F, center.z - radius1 * std::cos(endAngle) };
    Vector3 const aFoot{ a.x, center.y + 0.01F, a.z };
    Vector3 const bFoot{ b.x, center.y + 0.01F, b.z };

    DrawTriangle3D(apex, a, b, color);
    DrawTriangle3D(center, aFoot, bFoot, Fade(color, 0.22F));
    DrawLine3D(apex, a, Fade(WHITE, 0.18F));
    DrawLine3D(apex, b, Fade(WHITE, 0.18F));
}

void
DrawPseudoLidar3D(entt::registry const& snapshot,
                  entt::entity entity,
                  ppo::model::sEntityState const& state,
                  ppo::model::sSNAState const& snaState,
                  float worldScale)
{
    if (!snapshot.any_of<ppo::model::sPseudoLidarState>(entity))
        return;

    auto const& pseudoLidarState = snapshot.get<ppo::model::sPseudoLidarState>(entity);
    auto const& nearestCollisionRanges = pseudoLidarState.m_nearestCollisionRanges;
    auto const& nearestCollisionDistances = pseudoLidarState.m_nearestCollisionDistances;

    constexpr float noInterceptThreshold = 0.999F;
    constexpr float minDenominator = 1.0e-4F;

    float const maxAngleArea =
        pseudoLidarState.m_angleSpanRadians > 0.F ? pseudoLidarState.m_angleSpanRadians : PI_F;
    float const halfAngleArea = maxAngleArea * 0.5F;
    float const stepAngle = maxAngleArea / static_cast<float>(nearestCollisionRanges.size() - 1);

    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, worldScale);
    for (std::size_t index = 0; index + 1 < nearestCollisionRanges.size(); ++index)
    {
        float const r0 = nearestCollisionRanges[index];
        float const r1 = nearestCollisionRanges[index + 1];
        bool const hasIntercept0 = r0 < noInterceptThreshold;
        bool const hasIntercept1 = r1 < noInterceptThreshold;
        if (!hasIntercept0 && !hasIntercept1)
            continue;

        float const localStart = -halfAngleArea + static_cast<float>(index) * stepAngle;
        float const localEnd = localStart + stepAngle;
        float const worldStart = state.m_rotation + localStart;
        float const worldEnd = state.m_rotation + localEnd;

        auto const decodeInterceptDistance = [&](float ratio, float cachedDistance) -> float
        {
            if (cachedDistance > 0.F)
                return cachedDistance;

            if (ratio >= noInterceptThreshold)
                return 0.F;

            float const safeNoEscapeZone = std::max(snaState.m_noescapezone, 1.F);
            float const denominator = std::max(minDenominator, 1.F - ratio);
            return (ratio * safeNoEscapeZone) / denominator;
        };

        float const distance0 = decodeInterceptDistance(r0, nearestCollisionDistances[index]);
        float const distance1 = decodeInterceptDistance(r1, nearestCollisionDistances[index + 1]);
        float const radius0 = hasIntercept0 ? std::max(0.35F, distance0 / worldScale) : 0.F;
        float const radius1 = hasIntercept1 ? std::max(0.35F, distance1 / worldScale) : 0.F;

        float const minRatio = std::min(hasIntercept0 ? r0 : 1.F, hasIntercept1 ? r1 : 1.F);
        float const normalizedThreat = 1.F - std::clamp(minRatio, 0.F, 1.F);
        float const intensity = 0.35F + 0.65F * normalizedThreat;
        Color const beamColor{
            static_cast<unsigned char>(50 + 35 * intensity),
            static_cast<unsigned char>(180 + 50 * intensity),
            static_cast<unsigned char>(90 + 40 * intensity),
            static_cast<unsigned char>(50 + 70 * intensity)
        };

        DrawPseudoLidarCone(center, worldStart, worldEnd, radius0, radius1, beamColor);
    }
}

std::string
EntityDisplayName(entt::registry const& snapshot, entt::entity entity, ppo::model::sEntityState const& state)
{
    if (snapshot.any_of<ppo::model::sEntityMetadata>(entity))
    {
        auto const& metadata = snapshot.get<ppo::model::sEntityMetadata>(entity);
        if (!metadata.m_name.empty())
            return metadata.m_name + " (" + EntityTypeLabel(state.m_type) + ")";
    }

    return EntityDisplayName(state, entity);
}

/**
 * @brief Applies the shared frame-rate-independent follow contract to raylib.
 * @param camera Camera updated in place.
 * @param desiredPosition Desired world-space camera position.
 * @param desiredTarget Desired world-space look-at target.
 * @param forceSnap True after a target identity change.
 * @param snapDistance Maximum continuous target displacement.
 */
void UpdateCameraFollowPose(Camera3D& camera,
                            Vector3 const& desiredPosition,
                            Vector3 const& desiredTarget,
                            bool const forceSnap,
                            float const snapDistance) noexcept
{
    CameraFollowPose const current{
        {camera.position.x, camera.position.y, camera.position.z},
        {camera.target.x, camera.target.y, camera.target.z}};
    CameraFollowPose const desired{
        {desiredPosition.x, desiredPosition.y, desiredPosition.z},
        {desiredTarget.x, desiredTarget.y, desiredTarget.z}};
    CameraFollowPose const updated = AdvanceCameraFollow(
        current, desired, GetFrameTime(), CAMERA_FOLLOW_RESPONSE_PER_SECOND,
        forceSnap, snapDistance);
    camera.position = Vector3{updated.position.x, updated.position.y,
                              updated.position.z};
    camera.target = Vector3{updated.target.x, updated.target.y,
                            updated.target.z};
}

} // namespace

/**
* @brief Returns the singleton environment renderer instance.
* @param title Window title used on first construction.
* @return Shared renderer instance.
*/
EnvironmentRenderer&
/**
 * @brief Retourne une vue ou une valeur calculée depuis l'état courant de EnvironmentRenderer.
 */
EnvironmentRenderer::GetInstance(std::string const& title)
{
    static EnvironmentRenderer renderer{ title };
    return renderer;
}

/**
* @brief Builds the main renderer window, theme, and runtime resources.
* @param title Window title.
*/
EnvironmentRenderer::EnvironmentRenderer(std::string const& title) :
    m_title{ title }
{
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1600, 1000, m_title.c_str());
    SetTargetFPS(60);

    rlImGuiSetup(true);
    ImPlot::CreateContext();
    ApplyTheme();

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.13F;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    m_camera.position = Vector3{ 0.F, 12.F, 12.F };
    m_camera.target = Vector3{ 0.F, 0.F, 0.F };
    m_camera.up = Vector3{ 0.F, 1.F, 0.F };
    m_camera.fovy = 45.F;
    m_camera.projection = CAMERA_PERSPECTIVE;
    for (auto& view : m_tacticalCameraViews)
        view.m_camera = m_camera;
    ApplyScaleLevel(m_currentScaleLevel);

    auto& serializer = ppo::serialization::Serializer::GetInstance();
    serializer.Register(
        "Renderer.json",
        [this](ppo::serialization::json& ar)
        {
            if (!ar.contains("renderer") || !ar["renderer"].is_object())
                throw std::invalid_argument("Renderer.json requires a renderer object");

            auto const& section = ar["renderer"];

            ApplyScaleLevel(section.value("scaleLevel", m_currentScaleLevel));
            m_autoAttachCamera = section.value("autoAttachCamera", m_autoAttachCamera);
            m_multipleCameras = section.value(
                "multipleCameras", m_multipleCameras);
            m_showGrid = section.value("showGrid", m_showGrid);
            m_showDomains = section.value("showDomains", m_showDomains);
            m_showDestroyed = section.value("showDestroyed", m_showDestroyed);
            m_showMinimap = section.value("showMinimap", m_showMinimap);
            m_showEntities = section.value("showEntities", m_showEntities);
            m_showInspection = section.value("showInspection", m_showInspection);
            m_showSimulation = section.value("showSimulation", m_showSimulation);
            m_showPseudoLidar = section.value("showPseudoLidar", m_showPseudoLidar);
            m_showPatrolTrack = section.value("showPatrolTrack", m_showPatrolTrack);
            m_showPatrolReferences = section.value(
                "showPatrolReferences", m_showPatrolReferences);
            m_showPatrolWaypoints = section.value(
                "showPatrolWaypoints", m_showPatrolWaypoints);
            m_showHardBoundary = section.value("showHardBoundary", m_showHardBoundary);
            m_focusSelectionInMinimap = section.value("focusSelectionInMinimap", m_focusSelectionInMinimap);
            m_cameraDistance = std::clamp(section.value("cameraDistance", m_cameraDistance), 5.F, 60.F);
            m_cameraHeight = std::clamp(section.value("cameraHeight", m_cameraHeight), 2.F, 25.F);
            m_trailDuration = std::clamp(section.value("trailDuration", m_trailDuration), 2.F, 30.F);
            m_domainHeight = std::clamp(section.value("domainHeight", m_domainHeight), 0.3F, 5.F);
            m_mapZoom = std::clamp(section.value("minimapZoom", m_mapZoom), 0.3F, 4.F);

            if (section.contains("camera") && section["camera"].is_object())
            {
                auto const& camera = section["camera"];

                if (camera.contains("position") && camera["position"].is_array() && camera["position"].size() == 3)
                {
                    m_camera.position = Vector3{
                        camera["position"][0].get<float>(),
                        camera["position"][1].get<float>(),
                        camera["position"][2].get<float>()
                    };
                }

                if (camera.contains("target") && camera["target"].is_array() && camera["target"].size() == 3)
                {
                    m_camera.target = Vector3{
                        camera["target"][0].get<float>(),
                        camera["target"][1].get<float>(),
                        camera["target"][2].get<float>()
                    };
                }

                if (camera.contains("up") && camera["up"].is_array() && camera["up"].size() == 3)
                {
                    m_camera.up = Vector3{
                        camera["up"][0].get<float>(),
                        camera["up"][1].get<float>(),
                        camera["up"][2].get<float>()
                    };
                }

                m_camera.fovy = std::clamp(camera.value("fovy", m_camera.fovy), 20.F, 90.F);
            }
        },
        [this](ppo::serialization::json& ar)
        {
            auto& section = ar["renderer"];
            section["scaleLevel"] = m_currentScaleLevel;
            section["autoAttachCamera"] = m_autoAttachCamera;
            section["multipleCameras"] = m_multipleCameras;
            section["showGrid"] = m_showGrid;
            section["showDomains"] = m_showDomains;
            section["showDestroyed"] = m_showDestroyed;
            section["showMinimap"] = m_showMinimap;
            section["showEntities"] = m_showEntities;
            section["showInspection"] = m_showInspection;
            section["showSimulation"] = m_showSimulation;
            section["showPseudoLidar"] = m_showPseudoLidar;
            section["showPatrolTrack"] = m_showPatrolTrack;
            section["showPatrolReferences"] = m_showPatrolReferences;
            section["showPatrolWaypoints"] = m_showPatrolWaypoints;
            section["showHardBoundary"] = m_showHardBoundary;
            section["focusSelectionInMinimap"] = m_focusSelectionInMinimap;
            section["cameraDistance"] = m_cameraDistance;
            section["cameraHeight"] = m_cameraHeight;
            section["trailDuration"] = m_trailDuration;
            section["domainHeight"] = m_domainHeight;
            section["minimapZoom"] = m_mapZoom;
            section["camera"] = {
                { "position", { m_camera.position.x, m_camera.position.y, m_camera.position.z } },
                { "target",   { m_camera.target.x,   m_camera.target.y,   m_camera.target.z } },
                { "up",       { m_camera.up.x,       m_camera.up.y,       m_camera.up.z } },
                { "fovy",     m_camera.fovy }
            };
        });

    LoadModelAssets();
}

/**
* @brief Releases renderer resources and closes the window.
*/
EnvironmentRenderer::~EnvironmentRenderer(void)
{
    UnloadModelAssets();

    if (m_hasTacticalTexture)
        UnloadRenderTexture(m_tacticalTexture);
    ReleaseMultipleCameraTextures();

    ImPlot::DestroyContext();
    rlImGuiShutdown();

    if (IsWindowReady())
        CloseWindow();
}

/**
* @brief Applies the current world scale level to cached conversion factors.
* @param scaleLevel UI-selected scale level.
*/
void
EnvironmentRenderer::ApplyScaleLevel(int scaleLevel) noexcept
{
    m_currentScaleLevel = std::clamp(scaleLevel, 0, 2);
    m_worldScale = WorldScaleForLevel(m_currentScaleLevel);
}

/**
* @brief Loads every external or procedural model asset used by the renderer.
*/
void
EnvironmentRenderer::LoadModelAssets(void)
{
    std::filesystem::path const aircraftModelsRoot = ModelsRootPath() / "aircraft";
    LoadModelAsset(
        m_blueAircraftModel,
        aircraftModelsRoot / "blue_fighter_poly_pizza.glb",
        AIRCRAFT_MODEL_EXTENT,
        AIRCRAFT_MODEL_YAW_OFFSET_DEG);
    LoadModelAsset(
        m_redAircraftModel,
        aircraftModelsRoot / "red_jet_poly_pizza.glb",
        AIRCRAFT_MODEL_EXTENT,
        AIRCRAFT_MODEL_YAW_OFFSET_DEG);
    BuildGeneratedObjectiveModelAsset(m_objectiveModel, OBJECTIVE_MODEL_EXTENT);
}

/**
* @brief Releases every model asset owned by the renderer.
*/
void
EnvironmentRenderer::UnloadModelAssets(void)
{
    UnloadModelAsset(m_blueAircraftModel);
    UnloadModelAsset(m_redAircraftModel);
    UnloadModelAsset(m_objectiveModel);
}

/**
* @brief Applies the custom ImGui visual theme used by the application.
*/
void
EnvironmentRenderer::ApplyTheme(void)
{
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.F;
    style.ChildRounding = 10.F;
    style.FrameRounding = 7.F;
    style.PopupRounding = 8.F;
    style.GrabRounding = 6.F;
    style.ScrollbarRounding = 8.F;
    style.TabRounding = 8.F;
    style.WindowPadding = ImVec2(14.F, 12.F);
    style.FramePadding = ImVec2(10.F, 8.F);
    style.ItemSpacing = ImVec2(10.F, 8.F);
    style.ItemInnerSpacing = ImVec2(8.F, 6.F);
    style.WindowBorderSize = 1.F;
    style.FrameBorderSize = 0.F;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.035f, 0.045f, 0.060f, 1.0f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.055f, 0.070f, 0.090f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.055f, 0.070f, 0.090f, 0.98f);
    colors[ImGuiCol_Border] = ImVec4(0.18f, 0.24f, 0.30f, 0.65f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.08f, 0.11f, 0.15f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.16f, 0.22f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.22f, 0.28f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.11f, 0.17f, 0.22f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.18f, 0.28f, 0.36f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.22f, 0.34f, 0.44f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.10f, 0.18f, 0.24f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.14f, 0.25f, 0.33f, 1.0f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.18f, 0.31f, 0.40f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.07f, 0.10f, 0.14f, 1.0f);
    colors[ImGuiCol_TabActive] = ImVec4(0.16f, 0.24f, 0.31f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.21f, 0.32f, 0.41f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.05f, 0.08f, 0.11f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.11f, 0.15f, 1.0f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.42f, 0.67f, 1.0f, 1.0f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.56f, 0.96f, 1.0f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.48f, 0.73f, 1.0f, 1.0f);
    colors[ImGuiCol_Separator] = ImVec4(0.22f, 0.29f, 0.35f, 0.8f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.32f, 0.55f, 0.90f, 0.70f);
}

/**
* @brief Attaches one environment instance to the renderer.
* @param env Environment to observe and control.
*/
void
EnvironmentRenderer::SetEnvironment(ppo::model::Environment& env, View view) noexcept
{
    p_environment = &env;
    bool const flightView = view == View::FlightControl3D;
    if (m_flightControlView != flightView)
        m_layoutInitialized = false;
    m_flightControlView = flightView;
    SetExitKey(flightView ? KEY_NULL : KEY_ESCAPE);

    p_environment->ReadSnapshot([&](entt::registry const& snapshot)
    {
        if (snapshot.ctx().contains<ppo::model::sEnvironmentState>())
        {
            auto const& envState = snapshot.ctx().get<ppo::model::sEnvironmentState>();
            m_eastRange = envState.m_maxEast;
            m_northRange = envState.m_maxNorth;
            m_areawidth = envState.m_maxEast / m_worldScale;
            m_timestepMs = envState.m_dt * 1000.F;
        }
    });
}

/**
* @brief Returns the controller cache entry associated with one agent id.
* @param id Agent identifier.
* @return Shared controller state.
*/
std::shared_ptr<EnvironmentRenderer::AgentController>
EnvironmentRenderer::GetAgentController(unsigned long long id)
{
    if (id == 0)
        return nullptr;

    auto it = p_agents.find(id);
    if (it != p_agents.end())
        return it->second;

    auto agent = std::make_shared<AgentController>();
    agent->m_id = id;
    p_agents[id] = agent;
    return agent;
}

/**
* @brief Resizes or recreates the tactical render target when needed.
* @param width Requested texture width.
* @param height Requested texture height.
*/
void
EnvironmentRenderer::EnsureTacticalTexture(int width, int height)
{
    width = std::max(width, 32);
    height = std::max(height, 32);

    if (m_hasTacticalTexture
    &&  m_tacticalTexture.texture.width == width
    &&  m_tacticalTexture.texture.height == height)
        return;

    if (m_hasTacticalTexture)
        UnloadRenderTexture(m_tacticalTexture);

    m_tacticalTexture = LoadRenderTexture(width, height);
    m_hasTacticalTexture = true;
}

/**
* @brief Resizes or creates one render target used by a blue camera.
* @param view Blue camera view owning the target.
* @param width Requested texture width.
* @param height Requested texture height.
*/
void EnvironmentRenderer::EnsureTacticalTexture(
    TacticalCameraView& view, int width, int height)
{
    width = std::max(width, 32);
    height = std::max(height, 32);
    if (view.m_hasTexture
        && view.m_texture.texture.width == width
        && view.m_texture.texture.height == height)
    {
        return;
    }
    if (view.m_hasTexture)
        UnloadRenderTexture(view.m_texture);
    view.m_texture = LoadRenderTexture(width, height);
    view.m_hasTexture = true;
    BeginTextureMode(view.m_texture);
    ClearBackground(Color{10, 16, 22, 255});
    EndTextureMode();
}

/** @brief Releases all per-blue render textures without touching cameras. */
void EnvironmentRenderer::ReleaseMultipleCameraTextures(void) noexcept
{
    for (auto& view : m_tacticalCameraViews)
    {
        if (view.m_hasTexture)
            UnloadRenderTexture(view.m_texture);
        view.m_texture = {};
        view.m_hasTexture = false;
        view.m_blue = entt::null;
    }
    m_tacticalCameraRenderCursor = 0U;
}

/**
* @brief Builds the root dockspace used by the editor-like layout.
*/
void
EnvironmentRenderer::BuildDockspace(void)
{
    ImGuiViewport const* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.F);
    ImGui::Begin("##DockHost", nullptr, flags);
    ImGui::PopStyleVar(2);

    if (ImGui::BeginMenuBar())
    {
        ImGui::TextDisabled("PPOTrial  /  LEARNING WORKSPACE");
        ImGui::Separator();
        if (ImGui::BeginMenu("Panels"))
        {
            ImGui::MenuItem("Simulation", nullptr, &m_showSimulation);
            if (!m_flightControlView)
            {
                ImGui::MenuItem("Minimap", nullptr, &m_showMinimap);
                ImGui::MenuItem("Entities", nullptr, &m_showEntities);
                ImGui::MenuItem("Entity details", nullptr, &m_showInspection);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    ImGuiID const dockspaceId = ImGui::GetID("##MainDockspace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.F, 0.F), ImGuiDockNodeFlags_PassthruCentralNode);
    BuildDefaultLayout(dockspaceId);
    ImGui::End();
}

/**
* @brief Builds the default docking layout when no saved layout exists.
* @param dockspaceId Root dockspace identifier.
*/
void
EnvironmentRenderer::BuildDefaultLayout(ImGuiID dockspaceId)
{
    if (m_layoutInitialized)
        return;

    m_layoutInitialized = true;

    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);

    ImVec2 const size = ImGui::GetMainViewport()->Size;
    ImGui::DockBuilderSetNodeSize(dockspaceId, size);

    ImGuiID dockMain = dockspaceId;
    ImGuiID dockSidebar = ImGui::DockBuilderSplitNode(
        dockMain, ImGuiDir_Right, 0.31F, nullptr, &dockMain);
    ImGuiID dockControls = ImGui::DockBuilderSplitNode(
        dockSidebar, ImGuiDir_Down, m_flightControlView ? 0.30F : 0.22F, nullptr, &dockSidebar);

    ImGui::DockBuilderDockWindow("3D Tactical View", dockMain);
    if (m_flightControlView)
    {
        ImGuiID const dockInstruments = ImGui::DockBuilderSplitNode(
            dockSidebar, ImGuiDir_Up, 0.50F, nullptr, &dockSidebar);
        ImGui::DockBuilderDockWindow("FlightControl3D", dockInstruments);
    }
    ImGui::DockBuilderDockWindow("Learning", dockSidebar);
    ImGui::DockBuilderDockWindow("Simulation", dockControls);
    // Optional windows share a tabbed dock, never crowd the default view.
    ImGui::DockBuilderDockWindow("Minimap", dockControls);
    ImGui::DockBuilderDockWindow("Entities", dockControls);
    ImGui::DockBuilderDockWindow("Entity details", dockControls);

    ImGui::DockBuilderFinish(dockspaceId);
}

/**
* @brief Extracts mission telemetry displayed in the HUD and status bar.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::UpdateMissionTelemetry(entt::registry const& snapshot)
{
    UpdateTrackedEntity(snapshot);

    auto agentView = snapshot.view<ppo::model::sAgentStateSnaphot>();
    ppo::model::sAgentStateSnaphot const* trackedTrainingAgent = nullptr;

    for (auto entity : agentView)
    {
        auto const& [state] = agentView.get(entity);
        if (!state.m_common.m_training)
            continue;

        trackedTrainingAgent = &state;
        break;
    }

    if (trackedTrainingAgent == nullptr)
    {
        m_missionTelemetry.m_previousDone = false;
        m_missionTelemetry.m_agentId = 0;
        return;
    }

    m_missionTelemetry.m_agentId = trackedTrainingAgent->m_id;

    bool const done = trackedTrainingAgent->m_common.m_done;
    if (!m_missionTelemetry.m_previousDone && done)
    {
        bool success = false;

        if (m_trackedEntity != entt::null
        &&  snapshot.valid(m_trackedEntity)
        &&  snapshot.all_of<ppo::model::sEntityState, ppo::model::sSNAState>(m_trackedEntity))
        {
            auto const& ownState = snapshot.get<ppo::model::sEntityState>(m_trackedEntity);
            auto const& ownSnaState = snapshot.get<ppo::model::sSNAState>(m_trackedEntity);

            if (ownSnaState.m_target != entt::null
            &&  snapshot.valid(ownSnaState.m_target)
            &&  snapshot.any_of<ppo::model::sEntityState>(ownSnaState.m_target))
            {
                auto const& targetState = snapshot.get<ppo::model::sEntityState>(ownSnaState.m_target);

                success = targetState.m_destroyed && targetState.m_destroyedById == ownState.m_entityId;
                if (!success)
                {
                    float const dx = targetState.m_x - ownState.m_x;
                    float const dy = targetState.m_y - ownState.m_y;
                    float const range = std::sqrt(dx * dx + dy * dy);
                    success = range <= ownSnaState.m_escapezone;
                }
            }
        }

        ++m_missionTelemetry.m_totalEpisodes;
        m_missionTelemetry.m_successEpisodes += success ? 1U : 0U;
        m_missionTelemetry.m_lastEpisodeSuccess = success;
    }

    m_missionTelemetry.m_previousDone = done;
}

/**
* @brief Updates trail history for visible entities.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::UpdateTrails(entt::registry const& snapshot)
{
    double const now = GetTime();
    double const duration = std::max(1.0, static_cast<double>(m_trailDuration));

    auto entityView = snapshot.view<ppo::model::sEntityState>();
    for (auto entity : entityView)
    {
        auto const& [state] = entityView.get(entity);
        if (state.m_entityId == entt::null || state.m_destroyed)
            continue;

        auto& trail = m_entityTrails[static_cast<unsigned long long>(state.m_entityId)];
        Vector3 const worldPosition = ToWorld3D(state.m_x, state.m_y, 0.F, m_worldScale);
        Vector3 const liftedPosition{ worldPosition.x, worldPosition.y + 0.05F, worldPosition.z };

        bool shouldAppend = trail.empty();
        if (!shouldAppend)
        {
            auto const& last = trail.back();
            float const dx = liftedPosition.x - last.m_position.x;
            float const dy = liftedPosition.y - last.m_position.y;
            float const dz = liftedPosition.z - last.m_position.z;
            float const distanceSq = dx * dx + dy * dy + dz * dz;
            shouldAppend = distanceSq > 0.015F || (now - last.m_timestamp) > 0.18;
        }

        if (shouldAppend)
            trail.push_back(TrailPoint{ liftedPosition, now });

        while (!trail.empty() && (now - trail.front().m_timestamp) > duration)
            trail.pop_front();
    }

    for (auto it = m_entityTrails.begin(); it != m_entityTrails.end();)
    {
        while (!it->second.empty() && (now - it->second.front().m_timestamp) > duration)
            it->second.pop_front();

        if (it->second.empty())
            it = m_entityTrails.erase(it);
        else
            ++it;
    }
}

/**
* @brief Updates active explosion effects from entity destruction events.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::UpdateExplosionEffects(entt::registry const& snapshot)
{
    double const now = GetSteadyTimeSeconds();
    constexpr double kExplosionDuration = 2.0;

    for (auto it = m_explosionEffects.begin(); it != m_explosionEffects.end();)
    {
        if ((now - it->second.m_startTime) > kExplosionDuration)
            it = m_explosionEffects.erase(it);
        else
            ++it;
    }

    if (!snapshot.ctx().contains<ppo::model::sExplosionEventBuffer>())
        return;

    auto const& buffer = snapshot.ctx().get<ppo::model::sExplosionEventBuffer>();
    for (auto const& event : buffer.m_events)
    {
        m_explosionEffects[event.m_eventId] = ExplosionEffect{
            event.m_eventId,
            event.m_x,
            event.m_y,
            event.m_z,
            event.m_startTime,
            event.m_baseRadius,
            event.m_heightScale,
            event.m_type
        };
    }
}

/**
* @brief Chooses which entity is currently tracked and inspected.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::UpdateTrackedEntity(entt::registry const& snapshot)
{
    if (m_trackedEntity != entt::null && snapshot.valid(m_trackedEntity)
        && snapshot.any_of<ppo::model::sEntityState>(m_trackedEntity))
    {
        auto const& tracked =
            snapshot.get<ppo::model::sEntityState>(m_trackedEntity);
        if (!tracked.m_destroyed)
            return;
    }

    m_trackedEntity = entt::null;
    entt::entity fallbackMobile = entt::null;
    auto entityView = snapshot.view<ppo::model::sEntityState>();
    for (auto entity : entityView)
    {
        auto const& [state] = entityView.get(entity);
        if (state.m_type != ppo::model::eEntityType::Mobile || state.m_destroyed)
            continue;

        if (fallbackMobile == entt::null)
            fallbackMobile = entity;
        if (state.m_side == ppo::model::eEntitySide::Friend)
        {
            m_trackedEntity = entity;
            break;
        }
    }
    if (m_trackedEntity == entt::null)
        m_trackedEntity = fallbackMobile;

    if (m_selectedEntity == entt::null)
        m_selectedEntity = m_trackedEntity;
}

/**
* @brief Repositions the camera when auto-attach is enabled.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::UpdateAttachedCamera(entt::registry const& snapshot)
{
    if (!m_autoAttachCamera)
        return;

    UpdateTrackedEntity(snapshot);

    if (m_trackedEntity == entt::null || !snapshot.any_of<ppo::model::sEntityState>(m_trackedEntity))
    {
        m_cameraTrackedEntity = entt::null;
        return;
    }

    auto const& state = snapshot.get<ppo::model::sEntityState>(m_trackedEntity);
    Vector3 const focus = ToWorld3D(state.m_x, state.m_y, 0.F, m_worldScale);
    Vector3 const forward = HeadingForward(state.m_rotation);

    Vector3 const desiredTarget = Vector3{
        focus.x + forward.x * 1.4F,
        focus.y + 0.35F,
        focus.z + forward.z * 1.4F
    };

    Vector3 const desiredPosition = Vector3{
        focus.x - forward.x * m_cameraDistance,
        focus.y + m_cameraHeight,
        focus.z - forward.z * m_cameraDistance
    };

    bool const forceSnap = m_cameraTrackedEntity != m_trackedEntity;
    m_cameraTrackedEntity = m_trackedEntity;
    UpdateCameraFollowPose(
        m_camera, desiredPosition, desiredTarget, forceSnap,
        std::max(10.F, 0.8F * m_cameraDistance));
}

/**
* @brief Updates one blue-follow camera from the renderer's shared settings.
* @param snapshot Snapshot containing the blue state.
* @param blue Blue mobile followed by the camera.
* @param camera Camera state to update.
*/
void EnvironmentRenderer::UpdateBlueCamera(
    entt::registry const& snapshot, entt::entity const blue,
    Camera3D& camera, bool const forceSnap) const
{
    if (blue == entt::null || !snapshot.valid(blue)
        || !snapshot.any_of<ppo::model::sEntityState>(blue))
    {
        return;
    }
    auto const& state = snapshot.get<ppo::model::sEntityState>(blue);
    Vector3 const focus = ToWorld3D(
        state.m_x, state.m_y, 0.F, m_worldScale);
    Vector3 const forward = HeadingForward(state.m_rotation);
    Vector3 const desiredTarget{
        focus.x + forward.x * 1.4F,
        focus.y + 0.35F,
        focus.z + forward.z * 1.4F
    };
    Vector3 const desiredPosition{
        focus.x - forward.x * m_cameraDistance,
        focus.y + m_cameraHeight,
        focus.z - forward.z * m_cameraDistance
    };
    UpdateCameraFollowPose(
        camera, desiredPosition, desiredTarget, forceSnap,
        std::max(10.F, 0.8F * m_cameraDistance));
    camera.up = m_camera.up;
    camera.fovy = m_camera.fovy;
    camera.projection = m_camera.projection;
}

/**
* @brief Collects friendly mobiles in stable metadata-name order.
* @param snapshot Snapshot registry used for rendering.
* @return At most nine blue camera targets.
*/
EnvironmentRenderer::BlueCameraTargets
EnvironmentRenderer::CollectBlueCameraTargets(
    entt::registry const& snapshot) const
{
    BlueCameraTargets result{};
    result.m_entities.fill(entt::null);
    auto const entities = snapshot.view<ppo::model::sEntityState>();
    for (entt::entity const entity : entities)
    {
        auto const& state = entities.get<ppo::model::sEntityState>(entity);
        if (state.m_type != ppo::model::eEntityType::Mobile
            || state.m_side != ppo::model::eEntitySide::Friend
            || result.m_count >= result.m_entities.size())
        {
            continue;
        }
        result.m_entities[result.m_count++] = entity;
    }
    std::sort(
        result.m_entities.begin(),
        result.m_entities.begin()
            + static_cast<std::ptrdiff_t>(result.m_count),
        [&snapshot](entt::entity const left, entt::entity const right)
        {
            auto const* leftMetadata =
                snapshot.try_get<ppo::model::sEntityMetadata>(left);
            auto const* rightMetadata =
                snapshot.try_get<ppo::model::sEntityMetadata>(right);
            if (leftMetadata != nullptr && rightMetadata != nullptr)
                return leftMetadata->m_name < rightMetadata->m_name;
            if (leftMetadata != nullptr || rightMetadata != nullptr)
                return leftMetadata != nullptr;
            return entt::to_integral(left) < entt::to_integral(right);
        });
    return result;
}

/**
* @brief Updates free-camera orbit, pan, and zoom interactions.
* @param snapshot Snapshot registry used for rendering.
* @param hovered True when the tactical viewport is hovered.
* @param viewportOrigin Top-left viewport position.
* @param viewportSize Viewport size in pixels.
*/
void
EnvironmentRenderer::UpdateFreeCamera(entt::registry const& snapshot, bool hovered, Vector2 viewportOrigin, Vector2 viewportSize)
{
    if (m_autoAttachCamera || !hovered)
        return;

    ImGuiIO const& io = ImGui::GetIO();
    Vector2 const mouseDelta = GetMouseDelta();
    Vector3 const worldUp{ 0.F, 1.F, 0.F };
    Vector3 offset = VectorSub(m_camera.position, m_camera.target);
    float const radius = std::max(0.5F, VectorLength(offset));

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
    {
        float const yaw = -mouseDelta.x * CAMERA_MOUSE_ORBIT_SPEED;
        float const pitch = -mouseDelta.y * CAMERA_MOUSE_ORBIT_SPEED;

        float angle = std::atan2(offset.x, offset.z);
        float elevation = std::asin(std::clamp(offset.y / radius, -0.98F, 0.98F));

        angle += yaw;
        elevation = std::clamp(elevation + pitch, -1.35F, 1.35F);

        float const horizontalRadius = std::cos(elevation) * radius;
        offset = Vector3{
            std::sin(angle) * horizontalRadius,
            std::sin(elevation) * radius,
            std::cos(angle) * horizontalRadius
        };

        m_camera.position = VectorAdd(m_camera.target, offset);
    }

    Vector3 forward = VectorNormalizeSafe(VectorSub(m_camera.target, m_camera.position));
    if (VectorLength(forward) <= 1.0e-5F)
        forward = Vector3{ 0.F, 0.F, -1.F };

    Vector3 right = VectorNormalizeSafe(VectorCross(forward, worldUp));
    if (VectorLength(right) <= 1.0e-5F)
        right = Vector3{ 1.F, 0.F, 0.F };

    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE))
    {
        Vector3 const pan =
            VectorAdd(VectorScale(right, -mouseDelta.x * CAMERA_MOUSE_PAN_SPEED * radius),
                      VectorScale(worldUp, mouseDelta.y * CAMERA_MOUSE_PAN_SPEED * radius));
        m_camera.position = VectorAdd(m_camera.position, pan);
        m_camera.target = VectorAdd(m_camera.target, pan);
    }

    float const wheel = GetMouseWheelMove();
    if (std::fabs(wheel) > 0.F)
    {
        Vector2 const mouse = GetMousePosition();
        Vector2 const localMouse{ mouse.x - viewportOrigin.x, mouse.y - viewportOrigin.y };
        Vector3 groundHit{};
        bool const hasGroundHit =
            viewportSize.x > 1.F
         && viewportSize.y > 1.F
         && TryIntersectGroundPlane(GetScreenToWorldRayEx(localMouse,
                                                         m_camera,
                                                         static_cast<int>(viewportSize.x),
                                                         static_cast<int>(viewportSize.y)),
                                    groundHit);

        if (hasGroundHit)
            m_camera.target = VectorLerpLocal(m_camera.target, groundHit, 0.28F);

        Vector3 const zoomDirection =
            hasGroundHit ? VectorNormalizeSafe(VectorSub(groundHit, m_camera.position)) : forward;
        Vector3 const zoomStep =
            VectorScale(zoomDirection, wheel * std::max(1.F, radius * 0.15F) * CAMERA_WHEEL_ZOOM_SPEED);
        Vector3 const candidate = VectorAdd(m_camera.position, zoomStep);
        if (VectorLength(VectorSub(m_camera.target, candidate)) >= 0.75F)
            m_camera.position = candidate;
    }

    if (!io.WantTextInput)
    {
        float const frameMoveSpeed = m_cameraMoveSpeed * GetFrameTime() * (IsKeyDown(KEY_LEFT_SHIFT) ? 2.5F : 1.F);
        Vector3 planarForward = VectorNormalizeSafe(Vector3{ forward.x, 0.F, forward.z });
        if (VectorLength(planarForward) <= 1.0e-5F)
            planarForward = Vector3{ 0.F, 0.F, -1.F };

        Vector3 const planarRight = VectorNormalizeSafe(VectorCross(planarForward, worldUp));
        Vector3 translation{ 0.F, 0.F, 0.F };

        if (IsKeyDown(KEY_W))
            translation = VectorAdd(translation, VectorScale(planarForward, frameMoveSpeed));
        if (IsKeyDown(KEY_S))
            translation = VectorAdd(translation, VectorScale(planarForward, -frameMoveSpeed));
        if (IsKeyDown(KEY_A))
            translation = VectorAdd(translation, VectorScale(planarRight, -frameMoveSpeed));
        if (IsKeyDown(KEY_D))
            translation = VectorAdd(translation, VectorScale(planarRight, frameMoveSpeed));
        if (IsKeyDown(KEY_E))
            translation = VectorAdd(translation, VectorScale(worldUp, frameMoveSpeed));
        if (IsKeyDown(KEY_Q))
            translation = VectorAdd(translation, VectorScale(worldUp, -frameMoveSpeed));

        if (VectorLength(translation) > 0.F)
        {
            m_camera.position = VectorAdd(m_camera.position, translation);
            m_camera.target = VectorAdd(m_camera.target, translation);
        }

        if (IsKeyPressed(KEY_F)
        &&  m_trackedEntity != entt::null
        &&  snapshot.valid(m_trackedEntity)
        &&  snapshot.any_of<ppo::model::sEntityState>(m_trackedEntity))
        {
            auto const& trackedState = snapshot.get<ppo::model::sEntityState>(m_trackedEntity);
            Vector3 const focus = ToWorld3D(trackedState.m_x, trackedState.m_y, 0.F, m_worldScale);
            Vector3 const delta = VectorSub(focus, m_camera.target);
            m_camera.target = VectorAdd(m_camera.target, delta);
            m_camera.position = VectorAdd(m_camera.position, delta);
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            Vector2 const mouse = GetMousePosition();
            Vector2 const localMouse{ mouse.x - viewportOrigin.x, mouse.y - viewportOrigin.y };
            Vector3 groundHit{};
            if (viewportSize.x > 1.F
            &&  viewportSize.y > 1.F
            &&  TryIntersectGroundPlane(GetScreenToWorldRayEx(localMouse,
                                                              m_camera,
                                                              static_cast<int>(viewportSize.x),
                                                              static_cast<int>(viewportSize.y)),
                                        groundHit))
            {
                m_camera.target = VectorLerpLocal(m_camera.target, groundHit, 0.65F);
            }
        }

        if (IsKeyPressed(KEY_R))
        {
            m_camera.position = Vector3{ 0.F, 12.F, 12.F };
            m_camera.target = Vector3{ 0.F, 0.F, 0.F };
            m_cameraDistance = 12.F;
            m_cameraHeight = 4.5F;
        }
    }
}

/**
* @brief Draws the play, pause, and stop transport controls.
*/
void
EnvironmentRenderer::DrawSimulationControls(entt::registry const& snapshot)
{
    if (p_environment == nullptr)
        return;

    ImVec2 const controlButtonSize{ 92.F, 47.F };
    if (DrawSimulationControlButton("play", "PLAY", eSimulationControlGlyph::Play, controlButtonSize, ImVec4(0.17F, 0.58F, 0.33F, 1.F)))
    {
        for (auto const& [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = true;

#ifdef PPO_WITH_FLIGHT_CONTROL
        if (m_flightControlView)
        {
            m_flightPlayRequested = true;
            m_flightRestartBlock = snapshot.ctx().contains<ppo::examples::flight::FlightSnapshot>()
                && snapshot.ctx().get<ppo::examples::flight::FlightSnapshot>().finished;
        }
        else
#endif
            p_environment->Play();
    }

    ImGui::SameLine();
    if (DrawSimulationControlButton("pause", "PAUSE", eSimulationControlGlyph::Pause, controlButtonSize, ImVec4(0.76F, 0.50F, 0.12F, 1.F)))
    {
        for (auto const& [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = false;

        p_environment->Pause();
    }

    ImGui::SameLine();
    if (DrawSimulationControlButton("stop", "STOP", eSimulationControlGlyph::Stop, controlButtonSize, ImVec4(0.72F, 0.20F, 0.20F, 1.F)))
    {
        m_stopRequested = true;

        for (auto const& [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = false;
    }

#ifdef PPO_WITH_FLIGHT_CONTROL
    if (m_flightControlView)
    {
        ppo::examples::flight::DrawFlightDisplayControls();
        return;
    }
#endif
    if (!ImGui::CollapsingHeader("World and display settings"))
        return;

    ImGui::SeparatorText("World");
    bool refreshRanges = false;
    if (ImGui::SliderFloat("Area width (km)", &m_areawidth, 1.F, 500.F))
        refreshRanges = true;

    if (ImGui::SliderFloat("Time step (ms)", &m_timestepMs, 1.F, 200.F))
        p_environment->SetTimeStep(m_timestepMs / 1000.F);

    char const* const scaleLevels[]{ "m", "km", "NM" };
    if (ImGui::Combo("Scale level", &m_currentScaleLevel, scaleLevels, IM_ARRAYSIZE(scaleLevels)))
    {
        ApplyScaleLevel(m_currentScaleLevel);
        refreshRanges = true;
    }

    ImGui::SeparatorText("Visuals");
    ImGui::Checkbox("Multiple blue cameras", &m_multipleCameras);
    if (ImGui::Checkbox("Follow tracked mobile", &m_autoAttachCamera)
        && m_autoAttachCamera)
    {
        m_cameraTrackedEntity = entt::null;
    }
    ImGui::Checkbox("Show grid", &m_showGrid);
    ImGui::Checkbox("Show domains", &m_showDomains);
    ImGui::Checkbox("Show pseudo lidar", &m_showPseudoLidar);
    if (snapshot.ctx().contains<ppo::model::sPatrolTrackOverlay>())
        ImGui::Checkbox("Show patrol track", &m_showPatrolTrack);
    if (snapshot.ctx().contains<ppo::model::sPatrolReferenceOverlay>())
        ImGui::Checkbox("Show patrol references", &m_showPatrolReferences);
    if (snapshot.ctx().contains<ppo::model::sPatrolWaypointOverlay>())
        ImGui::Checkbox("Show patrol waypoints", &m_showPatrolWaypoints);
    if (snapshot.ctx().contains<ppo::model::sHardBoundaryOverlay>())
        ImGui::Checkbox("Show outer failure boundary", &m_showHardBoundary);
    ImGui::Checkbox("Show destroyed markers", &m_showDestroyed);
    ImGui::Checkbox("Center minimap on selection", &m_focusSelectionInMinimap);

    ImGui::SliderFloat("Camera distance", &m_cameraDistance, 5.F, 60.F);
    ImGui::SliderFloat("Camera height", &m_cameraHeight, 2.F, 25.F);
    ImGui::SliderFloat("Camera field of view", &m_camera.fovy,
                       20.F, 90.F, "%.0f deg");
    ImGui::SliderFloat("Trail duration", &m_trailDuration, 2.F, 30.F, "%.1f s");
    ImGui::SliderFloat("Mobile domain height", &m_domainHeight, 0.3F, 5.F);
    ImGui::SliderFloat("Minimap zoom", &m_mapZoom, 0.3F, 4.F);

    if (refreshRanges)
    {
        float const ratio = static_cast<float>(GetScreenHeight()) / static_cast<float>(std::max(GetScreenWidth(), 1));
        float const east = m_areawidth * m_worldScale;
        float const north = east * ratio;
        m_eastRange = east;
        m_northRange = north;
        p_environment->SetMaxRange(north, east);
    }
}

/**
* @brief Draws the one-line simulation status bar.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::DrawStatusBar(entt::registry const& snapshot)
{
    auto entityView = snapshot.view<ppo::model::sEntityState>();
    std::size_t total = 0;
    std::size_t alive = 0;
    std::size_t friendlies = 0;
    std::size_t hostiles = 0;

    for (auto entity : entityView)
    {
        auto const& [state] = entityView.get(entity);
        ++total;
        alive += state.m_destroyed ? 0U : 1U;
        friendlies += (state.m_side == ppo::model::eEntitySide::Friend) ? 1U : 0U;
        hostiles += (state.m_side == ppo::model::eEntitySide::Hostile) ? 1U : 0U;
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05F, 0.08F, 0.10F, 1.F));
    if (ImGui::BeginChild("##statusbar", ImVec2(0.F, 74.F), true))
    {
        ImGui::Columns(4, "##statuscols", false);
        ImGui::Text("Entities");
        ImGui::Text("%zu total", total);
        ImGui::Text("%zu alive", alive);
        ImGui::NextColumn();
        ImGui::TextColored(ColorForSideUi(ppo::model::eEntitySide::Friend), "Friendly");
        ImGui::Text("%zu units", friendlies);
        ImGui::Text("Tracked: %s", m_trackedEntity == entt::null ? "none" : "active");
        ImGui::NextColumn();
        ImGui::TextColored(ColorForSideUi(ppo::model::eEntitySide::Hostile), "Hostile");
        ImGui::Text("%zu units", hostiles);
        ImGui::Text("Scale: %.0f", m_worldScale);
        ImGui::NextColumn();
        ImGui::Text("Selection");
        if (m_selectedEntity != entt::null && snapshot.valid(m_selectedEntity) && snapshot.any_of<ppo::model::sEntityState>(m_selectedEntity))
        {
            auto const& state = snapshot.get<ppo::model::sEntityState>(m_selectedEntity);
            ImGui::Text("%s", EntityDisplayName(snapshot, m_selectedEntity, state).c_str());
            ImGui::Text("%s", SideLabel(state.m_side));
        }
        else
        {
            ImGui::TextUnformatted("No selection");
            ImGui::TextUnformatted(" ");
        }
        ImGui::Columns(1);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

/**
* @brief Draws the main training controls for one agent.
* @param agent Controller cache entry.
* @param state Agent snapshot to display.
*/
void
EnvironmentRenderer::DrawTrainingControls(std::shared_ptr<AgentController> const& agent,
                                          ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr || p_environment == nullptr)
        return;

    agent->m_rollout = static_cast<int>(state.m_common.m_maxStep);
    if (m_flightControlView)
    {
        ImGui::Text("Rollout buffer: %d aircraft steps", agent->m_rollout);
        return;
    }
    if (state.m_name == "Blue generalized shared MAPPO")
    {
        ImGui::Text("Rollout buffer: %d team steps", agent->m_rollout);
        ImGui::TextDisabled(
            "Set in Training/BluePatrol.json; restart to change.");
        return;
    }

    if (ImGui::SliderInt(
            "Rollout Buffer", &agent->m_rollout, 1, 10000))
    {
        ppo::model::sAgentRealTimeCommand command{};
        command.m_id = agent->m_id;
        command.m_rolloutBuffer = agent->m_rollout;
        command.m_type =
            ppo::model::sAgentRealTimeCommand::eTypeCommand::RolloutBuffer;
        p_environment->AddAgentCommand(command);
    }
}

/**
* @brief Draws reward-weight controls for one agent.
* @param agent Controller cache entry.
* @param state Agent snapshot to display.
*/
void
EnvironmentRenderer::DrawRewardControls(std::shared_ptr<AgentController> const& agent,
                                        ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr || p_environment == nullptr)
        return;

    agent->m_clamp = state.m_common.m_rewardClamp;
    agent->m_scalings = state.m_scalings;

    if (m_flightControlView)
    {
        for (std::size_t i = 0; i < state.m_scalings.size(); ++i)
        {
            char const* const name = i < state.m_rewardnames.size()
                ? state.m_rewardnames[i].c_str() : "Reward";
            ImGui::Text("%s: weight %+.3f", name, state.m_scalings[i]);
            if (i < state.m_rewards.size())
                ImGui::TextDisabled("  current %+.4f", state.m_rewards[i]);
        }
        return;
    }

    if (ImGui::SliderFloat("Reward clamp", &agent->m_clamp, 0.5F, 100.F))
    {
        ppo::model::sAgentRealTimeCommand command{};
        command.m_id = agent->m_id;
        command.m_rewardClamp = agent->m_clamp;
        command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardClamp;
        p_environment->AddAgentCommand(command);
    }

    for (std::size_t i = 0; i < state.m_scalings.size(); ++i)
    {
        ImGui::PushID(static_cast<int>(i));
        std::string const rewardLabel =
            (i < state.m_rewardnames.size() && !state.m_rewardnames[i].empty())
                ? state.m_rewardnames[i]
                : ("Reward " + std::to_string(i));

        if (i >= agent->m_scalings.size())
        {
            ImGui::PopID();
            break;
        }

        float const maxWeight = state.m_name == "Blue generalized shared MAPPO"
            ? 10.F : 1.F;
        if (ImGui::SliderFloat(
                rewardLabel.c_str(), &agent->m_scalings[i], 0.F, maxWeight))
        {
            ppo::model::sAgentRealTimeCommand command{};
            command.m_id = agent->m_id;
            command.m_rewardScaling = agent->m_scalings[i];
            command.m_rewardScalingType = static_cast<unsigned char>(i);
            command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardScaling;
            p_environment->AddAgentCommand(command);
        }
        if (i < state.m_rewards.size())
        {
            ImGui::SameLine();
            ImGui::TextDisabled("current %+.4f", state.m_rewards[i]);
        }
        ImGui::PopID();
    }
}

/**
* @brief Draws one scrolling plot for a single agent metric.
* @param plot Plot cache entry.
*/
void
EnvironmentRenderer::DrawOneAgentPlot(AgentController::Plot& plot)
{
    if (ImPlot::BeginPlot(plot.m_name.c_str(), ImGui::GetContentRegionAvail()))
    {
        ImPlot::SetupAxes("t", "value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::PlotLine(plot.m_name.c_str(), plot.data(), static_cast<int>(plot.size()));

        if (ImGui::BeginPopupContextItem())
        {
            if (ImGui::MenuItem("Pin plot"))
                p_pinnedPlot = (p_pinnedPlot == &plot) ? nullptr : &plot;

            ImGui::EndPopup();
        }

        ImPlot::EndPlot();
    }
}

/**
* @brief Draws one plot grid for one agent.
* @param agent Controller cache entry.
* @param state Agent snapshot used to label the plots.
*/
void
EnvironmentRenderer::DrawAgentPlot(std::shared_ptr<AgentController> const& agent,
                                   ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr)
        return;

    agent->m_plots[0].m_name = state.m_name + "-Loss";
    agent->m_plots[1].m_name = state.m_name + "-ActorLoss";
    agent->m_plots[2].m_name = state.m_name + "-ValueLoss";
    agent->m_plots[3].m_name = state.m_name + "-EntropyLoss";

    if (ImGui::BeginTable("##agentplots", 2, ImGuiTableFlags_SizingStretchSame))
    {
        for (int index = 0; index < 4; ++index)
        {
            if (index % 2 == 0)
                ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::PushID(index);
            ImGui::BeginChild("##plotcell", ImVec2(0.F, 220.F), false);
            DrawOneAgentPlot(agent->m_plots[index]);
            ImGui::EndChild();
            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

/**
* @brief Samples real training and evaluation events independently of visible tabs.
*/
void
EnvironmentRenderer::UpdateLearningTelemetry(entt::registry const& snapshot)
{
    auto const view = snapshot.view<ppo::model::sAgentStateSnaphot>();
    for (auto entity : view)
    {
        auto const& [state] = view.get(entity);
        auto agent = GetAgentController(state.m_id);
        if (agent == nullptr)
            continue;

        unsigned int const version = state.m_common.m_version;
        if (!agent->m_seenVersion)
        {
            agent->m_lastPlottedVersion = version;
            agent->m_seenVersion = true;
        }
        else if (version > agent->m_lastPlottedVersion)
        {
            // A version change means a PPO update occurred. A skipped
            // version is never recreated from duplicated render samples.
            agent->m_plots[0].push(state.m_common.m_loss);
            agent->m_plots[1].push(state.m_common.m_actorloss);
            agent->m_plots[2].push(state.m_common.m_valueloss);
            agent->m_plots[3].push(state.m_common.m_entropyloss);
            agent->m_lastPlottedVersion = version;
        }
        else if (version < agent->m_lastPlottedVersion)
        {
            agent->m_lastPlottedVersion = version;
            for (auto& plot : agent->m_plots)
                plot.clear();
            agent->m_evaluationScores.Clear();
            agent->m_evaluationSuccess.Clear();
        }

        auto const& common = state.m_common;
        if (common.m_hasLatestEvaluation
            && agent->m_evaluationScores.Record(common.m_completedEvaluations,
                                                common.m_latestEvaluationNotation))
        {
            agent->m_evaluationSuccess.Record(common.m_completedEvaluations,
                                              100.0 * common.m_latestEvaluationSuccessRate);
        }
    }
}

/**
* @brief Draws evidence of task performance without changing the learned policy.
*/
void
EnvironmentRenderer::DrawLearningProgress(std::shared_ptr<AgentController> const& agent,
                                          ppo::model::sAgentStateSnaphot const& state)
{
    auto const& common = state.m_common;
    if (common.m_evaluationinprogress)
    {
        ImGui::TextUnformatted("Evaluating model");
        ImGui::ProgressBar(std::clamp(common.m_evaluationprogress, 0.F, 1.F), ImVec2(-1.F, 0.F));
        if (m_flightControlView)
            ImGui::Text("Episodes completed: %u", common.m_evaluationTestsFinished);
        else
        {
            ImGui::Text("Tests completed: %u  |  Step-limit timeouts: %u",
                        common.m_evaluationTestsFinished, common.m_evaluationTimeoutTests);
            ImGui::Text("Simulation steps: %llu  |  %.0f steps/s",
                        common.m_evaluationStepsExecuted, common.m_evaluationStepsPerSecond);
            ImGui::Text("Elapsed: %.1f s", common.m_evaluationElapsedSeconds);
            ImGui::TextDisabled("Timeout = configured step limit reached, not necessarily a failed task.");
        }
    }

    auto const& scores = agent->m_evaluationScores.Samples();
    if (!common.m_hasLatestEvaluation || scores.empty())
    {
        ImGui::TextDisabled("Awaiting first complete evaluation.");
        ImGui::TextWrapped("The evaluation graph starts when a full test campaign completes.");
        return;
    }

    ImGui::SeparatorText("Model evaluation");
    ImGui::Text("LATEST  %.2f", common.m_latestEvaluationNotation);
    if (!m_flightControlView)
        ImGui::Text("BEST    %.2f", common.m_notation);
    ImGui::Text("SUCCESS %.1f%%", 100.F * common.m_latestEvaluationSuccessRate);
    if (!m_flightControlView)
    {
        ImGui::Text("SURVIVAL %.1f%%", 100.F * common.m_latestEvaluationSurvivalRate);
        ImGui::Text("TIMEOUT  %.1f%%", 100.F * common.m_latestEvaluationTimeoutRate);
    }
    ImGui::TextDisabled("Campaign #%llu  -  %u test cases",
        static_cast<unsigned long long>(common.m_completedEvaluations),
        common.m_latestEvaluationTestCount);

    if (scores.size() >= 2)
        ImGui::Text("Delta vs first visible evaluation: %+.2f score units",
            scores.back().score - scores.front().score);

    char const* trend = "Not enough campaigns";
    switch (agent->m_evaluationScores.RecentTrend())
    {
    case ppo::core::LearningHistory::Trend::Improving: trend = "Improving (indicative)"; break;
    case ppo::core::LearningHistory::Trend::Declining: trend = "Declining (indicative)"; break;
    case ppo::core::LearningHistory::Trend::Flat: trend = "Stable (indicative)"; break;
    default: break;
    }
    ImGui::Text("Recent trend: %s", trend);

    // The bounded history holds at most 256 entries. Stack storage avoids
    // allocating temporary plots on every frame.
    std::array<double, 256> xs{};
    std::array<double, 256> ys{};
    auto plot = [&](ppo::core::LearningHistory const& history,
                    char const* title, char const* series, char const* unit)
    {
        auto const& samples = history.Samples();
        auto const n = std::min(samples.size(), xs.size());
        for (std::size_t i = 0; i < n; ++i)
        {
            xs[i] = static_cast<double>(samples[i].campaign);
            ys[i] = samples[i].score;
        }
        if (ImPlot::BeginPlot(title, ImVec2(-1.F, 205.F)))
        {
            ImPlot::SetupAxes("Evaluation #", unit, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            if (n != 0)
                ImPlot::PlotLine(series, xs.data(), ys.data(), static_cast<int>(n));
            ImPlot::EndPlot();
        }
    };

    plot(agent->m_evaluationScores, "Evaluation score", "Score", "Score");
    plot(agent->m_evaluationSuccess, "Success rate", "Success", "Percent");

    ImGui::TextWrapped("A positive trend on a fixed evaluation suite is evidence of learning, not proof of generalization.");
    if (ImGui::Button("Export CSV"))
    {
        std::string safeName = state.m_name;
        for (char& character : safeName)
            if (!std::isalnum(static_cast<unsigned char>(character))
                && character != '_' && character != '-')
                character = '_';

        auto const stamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::filesystem::path const outputDir{ "learning_reports" };
        std::error_code error;
        std::filesystem::create_directories(outputDir, error);
        if (error)
            agent->m_exportStatus = "Failed to create learning_reports directory.";
        else
        {
            auto const csvPath = outputDir / (safeName + "_" + std::to_string(stamp) + ".csv");
            std::ofstream output{ csvPath };
            if (!output)
                agent->m_exportStatus = "Failed to open CSV file.";
            else
            {
                output << "campaign,score,success_percent\n";
                auto const& successes = agent->m_evaluationSuccess.Samples();
                for (std::size_t i = 0; i < scores.size(); ++i)
                {
                    output << scores[i].campaign << ',' << scores[i].score << ',';
                    if (i < successes.size() && successes[i].campaign == scores[i].campaign)
                        output << successes[i].score;
                    output << '\n';
                }
                output.close();
                agent->m_exportStatus = output ? ("CSV: " + csvPath.string()) : "Failed to write CSV.";
            }
        }
    }

    if (!agent->m_exportStatus.empty())
        ImGui::TextWrapped("%s", agent->m_exportStatus.c_str());
}

/**
* @brief Renders the full 3D tactical scene into the off-screen target.
* @param snapshot Snapshot registry used for rendering.
* @param target Render target receiving the scene.
* @param camera Camera used for this view.
* @param focusEntity Entity supplying target and pseudo-lidar overlays.
* @param reducedDetail True for the bounded multi-camera rendering path.
*/
void
EnvironmentRenderer::RenderTacticalScene(
    entt::registry const& snapshot, RenderTexture2D const& target,
    Camera3D const& camera, entt::entity const focusEntity,
    bool const reducedDetail)
{
#ifdef PPO_WITH_FLIGHT_CONTROL
    if (snapshot.ctx().contains<ppo::examples::flight::FlightSnapshot>())
    {
        ppo::examples::flight::DrawFlightScene(snapshot.ctx().get<ppo::examples::flight::FlightSnapshot>(),target);
        return;
    }
#endif
    BeginTextureMode(target);
    ClearBackground(Color{ 10, 16, 22, 255 });

    BeginMode3D(camera);
    if (m_showGrid)
    {
        DrawGrid(24, 5.F);
        DrawGrid(120, 1.F);
    }

    DrawPlane(Vector3{ 0.F, -0.02F, 0.F }, Vector2{ 320.F, 320.F }, Color{ 14, 21, 30, 255 });
    if (snapshot.ctx().contains<ppo::model::sDefendedArea>())
        DrawDefendedArea3D(snapshot.ctx().get<ppo::model::sDefendedArea>(), m_worldScale);
    if (m_showHardBoundary
        && snapshot.ctx().contains<ppo::model::sHardBoundaryOverlay>())
    {
        auto const& outside =
            snapshot.ctx().get<ppo::model::sHardBoundaryOverlay>();
        DrawDefendedArea3D(
            ppo::model::sDefendedArea{0.F, 0.F,
                                      outside.m_halfNorth, outside.m_halfEast},
            m_worldScale, Color{255, 166, 77, 230});
    }
    if (m_showPatrolTrack
        && snapshot.ctx().contains<ppo::model::sPatrolTrackOverlay>())
        DrawPatrolTrack3D(
            snapshot.ctx().get<ppo::model::sPatrolTrackOverlay>(),
            m_worldScale, reducedDetail);
    if (m_showPatrolReferences
        && snapshot.ctx().contains<ppo::model::sPatrolReferenceOverlay>())
    {
        DrawPatrolReferences3D(
            snapshot.ctx().get<ppo::model::sPatrolReferenceOverlay>(),
            m_worldScale, reducedDetail);
    }
    if (m_showPatrolWaypoints
        && snapshot.ctx().contains<ppo::model::sPatrolWaypointOverlay>())
    {
        DrawPatrolWaypoints3D(
            snapshot.ctx().get<ppo::model::sPatrolWaypointOverlay>(),
            m_worldScale, reducedDetail);
    }

    entt::entity primaryTarget = entt::null;
    entt::entity pseudoLidarEntity = entt::null;
    if (focusEntity != entt::null
    &&  snapshot.valid(focusEntity)
    &&  snapshot.any_of<ppo::model::sSNAState>(focusEntity))
    {
        pseudoLidarEntity = focusEntity;
        auto const& trackedSna =
            snapshot.get<ppo::model::sSNAState>(focusEntity);
        if (trackedSna.m_target != entt::null
        &&  snapshot.valid(trackedSna.m_target)
        &&  snapshot.any_of<ppo::model::sEntityState>(trackedSna.m_target))
        {
            primaryTarget = trackedSna.m_target;
        }
    }
    else if (m_selectedEntity != entt::null
         &&  snapshot.valid(m_selectedEntity)
         &&  snapshot.any_of<ppo::model::sSNAState>(m_selectedEntity))
    {
        pseudoLidarEntity = m_selectedEntity;
    }

    auto entityView = snapshot.view<ppo::model::sEntityState>();
    for (auto entity : entityView)
    {
        auto const& [state] = entityView.get(entity);
        auto trailIt = m_entityTrails.find(static_cast<unsigned long long>(state.m_entityId));
        if (trailIt != m_entityTrails.end())
            DrawTrail3D(trailIt->second, m_trailDuration,
                        ColorForSide(state.m_side, 180), reducedDetail);

        Color const baseColor = ColorForSide(state.m_side);
        bool const useDestroyedLook =
            state.m_destroyed
            && (state.m_type == ppo::model::eEntityType::Mobile
                || state.m_type == ppo::model::eEntityType::Objective);
        Color const renderColor = useDestroyedLook ? DestroyedTint(baseColor) : baseColor;
        Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, m_worldScale);
        if (state.m_type == ppo::model::eEntityType::Mobile)
        {
            if (IsCarVisual(snapshot, entity))
                DrawCarGlyph3D(state, m_worldScale, renderColor,
                               TacticalRoleForEntity(snapshot, entity));
            else if (auto const* aircraftModel = ResolveAircraftModelAsset(
                    m_blueAircraftModel,
                    m_redAircraftModel,
                    snapshot,
                    entity,
                    state))
            {
                DrawAircraftModel3D(*aircraftModel, state, m_worldScale, renderColor);
            }
            else
            {
                DrawAircraftGlyph3D(state, m_worldScale, renderColor);
            }
        }
        else if (state.m_type == ppo::model::eEntityType::Objective)
        {
            if (m_objectiveModel.m_loaded)
                DrawStaticModel3D(m_objectiveModel, state, m_worldScale, renderColor);
            else
                DrawObjectiveGlyph3D(state, m_worldScale, renderColor);
        }
        else if (state.m_type == ppo::model::eEntityType::RestrictedZone)
            DrawSamGlyph3D(state, m_worldScale, baseColor);

        if (!state.m_destroyed
        &&  m_showDomains
        &&  snapshot.any_of<ppo::model::sSNAState>(entity))
        {
            auto const& snaState = snapshot.get<ppo::model::sSNAState>(entity);
            if (state.m_type == ppo::model::eEntityType::RestrictedZone)
            {
                float const radius = snaState.m_noescapezone / m_worldScale;
                // Color the protected area blue in TwoVsTwo, while preserving
                // Airspace's historical red danger-area appearance.
                bool const isBlueArea = snapshot.ctx().contains<ppo::model::sDefendedArea>()
                    && state.m_side == ppo::model::eEntitySide::Friend;
                Color const fill = isBlueArea ? Color{ 75, 137, 255, 42 }
                                               : Color{ 248, 86, 76, 42 };
                Color const outline = isBlueArea ? Color{ 139, 188, 255, 165 }
                                                  : Color{ 255, 156, 120, 165 };
                DrawHemisphere(
                    center, radius, fill,
                    reducedDetail ? 14 : 28,
                    reducedDetail ? 6 : 12);
                DrawSphereWires(
                    center, radius,
                    reducedDetail ? 10 : 18,
                    reducedDetail ? 6 : 10, outline);
            }
            else
            {
                float const radius = snaState.m_noescapezone / m_worldScale;
                bool const isBlueCar = IsCarVisual(snapshot, entity)
                    && state.m_side == ppo::model::eEntitySide::Friend;
                Color const fill = isBlueCar ? Color{ 55, 120, 255, 74 }
                                              : Color{ 255, 72, 72, 74 };
                Color const outline = isBlueCar ? Color{ 146, 198, 255, 210 }
                                                 : Color{ 255, 188, 168, 210 };
                DrawEngagementCone3D(center,
                                     state.m_rotation,
                                     snaState.m_maxShootAngle,
                                     radius,
                                     m_domainHeight,
                                     fill,
                                     outline,
                                     reducedDetail ? 16 : 40);
            }

            if (m_showPseudoLidar
            &&  state.m_type == ppo::model::eEntityType::Mobile)
            {
                if (entity == pseudoLidarEntity)
                    DrawPseudoLidar3D(snapshot, entity, state, snaState, m_worldScale);
            }
        }

        if (state.m_destroyed && m_showDestroyed)
        {
            if (state.m_type == ppo::model::eEntityType::Mobile
            ||  state.m_type == ppo::model::eEntityType::Objective)
            {
                DrawDestroyedWreck3D(state, m_worldScale);
            }

            DrawDestroyMarker3D(center);
        }
      }

      if (primaryTarget != entt::null
      &&  snapshot.valid(primaryTarget)
      &&  snapshot.any_of<ppo::model::sEntityState>(primaryTarget))
      {
          auto const& targetState = snapshot.get<ppo::model::sEntityState>(primaryTarget);
          if (!targetState.m_destroyed)
              DrawPrimaryTargetAura3D(targetState, m_worldScale);
      }

      for (auto const& [_, effect] : m_explosionEffects)
          DrawExplosion3D(effect, m_worldScale);

      EndMode3D();
      if (m_showPatrolReferences
          && snapshot.ctx().contains<ppo::model::sPatrolReferenceOverlay>())
      {
          DrawPatrolReferenceLabels(
              snapshot.ctx().get<ppo::model::sPatrolReferenceOverlay>(),
              camera, target.texture.width, target.texture.height,
              m_worldScale);
      }
      if (m_showPatrolWaypoints
          && snapshot.ctx().contains<ppo::model::sPatrolWaypointOverlay>())
      {
          DrawPatrolWaypointLabels(
              snapshot.ctx().get<ppo::model::sPatrolWaypointOverlay>(),
              camera, target.texture.width, target.texture.height,
              m_worldScale);
      }
      DrawTacticalRoleBadges(snapshot, camera, target.texture.width,
                             target.texture.height, m_worldScale);
      EndTextureMode();
  }

/**
* @brief Draws one fitted child viewport per blue mobile.
* @param snapshot Snapshot registry used for rendering.
* @param targets Stable blue entity ordering for this frame.
* @param availableSize Total ImGui area available to the grid.
*/
void EnvironmentRenderer::DrawMultipleTacticalViews(
    entt::registry const& snapshot, BlueCameraTargets const& targets,
    ImVec2 const availableSize)
{
    constexpr int gap = 6;
    TacticalViewGrid const grid = MakeTacticalViewGrid(
        targets.m_count,
        static_cast<int>(availableSize.x),
        static_cast<int>(availableSize.y), gap);
    if (!grid.Valid())
        return;

    std::array<bool, kMaximumTacticalCameraViews> renderThisFrame{};
    std::size_t const renderCount = std::min(
        targets.m_count, MULTI_CAMERA_RENDERS_PER_FRAME);
    m_tacticalCameraRenderCursor %= targets.m_count;
    for (std::size_t offset = 0U; offset < renderCount; ++offset)
    {
        renderThisFrame[(m_tacticalCameraRenderCursor + offset)
                        % targets.m_count] = true;
    }

    for (std::size_t index = 0U; index < targets.m_count; ++index)
    {
        if (index % grid.columns != 0U)
            ImGui::SameLine(0.F, static_cast<float>(gap));

        ImGui::PushID(static_cast<int>(index));
        ImGui::BeginChild(
            "##blue-tactical-camera",
            ImVec2{static_cast<float>(grid.cellWidth),
                   static_cast<float>(grid.cellHeight)},
            true,
            ImGuiWindowFlags_NoScrollbar
                | ImGuiWindowFlags_NoScrollWithMouse);

        entt::entity const blue = targets.m_entities[index];
        TacticalRole const role = TacticalRoleForEntity(snapshot, blue);
        if (role != TacticalRole::None)
        {
            Color const roleColor = TacticalRoleColor(role);
            ImGui::TextColored(
                ImVec4{roleColor.r / 255.F, roleColor.g / 255.F,
                       roleColor.b / 255.F, 1.F},
                "[%s]", TacticalRoleMarker(role));
            ImGui::SameLine();
        }
        auto const* metadata =
            snapshot.try_get<ppo::model::sEntityMetadata>(blue);
        if (metadata != nullptr && !metadata->m_name.empty())
            ImGui::TextUnformatted(metadata->m_name.c_str());
        else
            ImGui::Text("Blue %u", static_cast<unsigned>(index + 1U));

        TacticalCameraView& view = m_tacticalCameraViews[index];
        bool const targetChanged = view.m_blue != blue;
        view.m_blue = blue;
        UpdateBlueCamera(snapshot, blue, view.m_camera, targetChanged);
        ImVec2 const imageSize = ImGui::GetContentRegionAvail();
        int const imageWidth = std::max(32, static_cast<int>(imageSize.x));
        int const imageHeight = std::max(32, static_cast<int>(imageSize.y));
        float const textureScale = std::min(
            {1.F,
             static_cast<float>(MULTI_CAMERA_MAX_TEXTURE_WIDTH)
                 / static_cast<float>(imageWidth),
             static_cast<float>(MULTI_CAMERA_MAX_TEXTURE_HEIGHT)
                 / static_cast<float>(imageHeight)});
        int const textureWidth = std::max(
            32, static_cast<int>(std::lround(
                    static_cast<float>(imageWidth) * textureScale)));
        int const textureHeight = std::max(
            32, static_cast<int>(std::lround(
                    static_cast<float>(imageHeight) * textureScale)));
        EnsureTacticalTexture(view, textureWidth, textureHeight);
        if (renderThisFrame[index])
        {
            RenderTacticalScene(
                snapshot, view.m_texture, view.m_camera, blue, true);
        }
        rlImGuiImageRect(
            &view.m_texture.texture, imageWidth, imageHeight,
            Rectangle{0.F, 0.F,
                      static_cast<float>(view.m_texture.texture.width),
                      -static_cast<float>(view.m_texture.texture.height)});
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
        {
            m_selectedEntity = blue;
            m_trackedEntity = blue;
        }
        ImGui::EndChild();
        ImGui::PopID();
    }

    for (std::size_t index = targets.m_count;
         index < m_tacticalCameraViews.size(); ++index)
    {
        auto& view = m_tacticalCameraViews[index];
        if (view.m_hasTexture)
            UnloadRenderTexture(view.m_texture);
        view.m_texture = {};
        view.m_hasTexture = false;
        view.m_blue = entt::null;
    }
    m_tacticalCameraRenderCursor =
        (m_tacticalCameraRenderCursor + renderCount) % targets.m_count;
}

/**
* @brief Draws the tactical 3D viewport window.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::DrawTacticalWindow(entt::registry const& snapshot)
{
    if (!ImGui::Begin("3D Tactical View"))
    {
        ImGui::End();
        return;
    }

#ifdef PPO_WITH_FLIGHT_CONTROL
    if (m_flightControlView)
    {
        if (!snapshot.ctx().contains<ppo::examples::flight::FlightSnapshot>())
        {
            ImGui::TextUnformatted("Preparing FlightControl3D preview...");
            ImGui::End();
            return;
        }
        auto const& flight = snapshot.ctx().get<ppo::examples::flight::FlightSnapshot>();
        ImGui::Text("Cessna 310 | FPS %d", GetFPS());
        ReleaseMultipleCameraTextures();
        ImVec2 const size = ImGui::GetContentRegionAvail();
        EnsureTacticalTexture(static_cast<int>(size.x), static_cast<int>(size.y));
        ppo::examples::flight::DrawFlightScene(flight, m_tacticalTexture);
        rlImGuiImageRenderTextureFit(&m_tacticalTexture, true);
        ImGui::End();
        return;
    }
#endif

    if (m_selectedEntity != entt::null && snapshot.valid(m_selectedEntity) && snapshot.any_of<ppo::model::sEntityState>(m_selectedEntity))
    {
        auto const& state = snapshot.get<ppo::model::sEntityState>(m_selectedEntity);
        ImGui::Text("%s", EntityDisplayName(state, m_selectedEntity).c_str());
        ImGui::SameLine();
        ImGui::TextColored(ColorForSideUi(state.m_side), "%s", SideLabel(state.m_side));
        TacticalRole const role = TacticalRoleForEntity(snapshot, m_selectedEntity);
        if (role != TacticalRole::None)
        {
            Color const roleColor = TacticalRoleColor(role);
            ImGui::SameLine();
            ImGui::TextColored(
                ImVec4{roleColor.r / 255.F, roleColor.g / 255.F,
                       roleColor.b / 255.F, 1.F},
                "[%s] %s", TacticalRoleMarker(role),
                role == TacticalRole::Leader ? "Leader" : "Teammate");
        }
        ImGui::SameLine();
        ImGui::Text("Heading %.1f deg", state.m_rotation * DEG_PER_RAD);
    }
    else
    {
        ImGui::TextUnformatted("No entity selected");
    }

    if (ImGui::Button("Track selection") && m_selectedEntity != entt::null)
        m_trackedEntity = m_selectedEntity;

    ImGui::SameLine();
    if (ImGui::Button("Reset camera"))
    {
        m_cameraDistance = 12.F;
        m_cameraHeight = 4.5F;
        m_cameraTrackedEntity = entt::null;
        for (auto& view : m_tacticalCameraViews)
        {
            view.m_camera = m_camera;
            view.m_blue = entt::null;
        }
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    float const successRate =
        (m_missionTelemetry.m_totalEpisodes == 0)
            ? 0.F
            : (100.F * static_cast<float>(m_missionTelemetry.m_successEpisodes) /
               static_cast<float>(m_missionTelemetry.m_totalEpisodes));

    ImGui::Text("FPS %d", GetFPS());
    ImGui::SameLine();
    ImGui::Text("Success %.1f%%", successRate);
    ImGui::SameLine();
    ImGui::Text("Episodes %u", m_missionTelemetry.m_totalEpisodes);
    ImGui::SameLine();
    ImGui::Text("%s", m_missionTelemetry.m_lastEpisodeSuccess ? "Last: Success" : "Last: Pending/Fail");
    Color const leaderColor = TacticalRoleColor(TacticalRole::Leader);
    Color const teammateColor = TacticalRoleColor(TacticalRole::Teammate);
    ImGui::TextColored(
        ImVec4{leaderColor.r / 255.F, leaderColor.g / 255.F,
               leaderColor.b / 255.F, 1.F},
        "L  Leader");
    ImGui::SameLine();
    ImGui::TextColored(
        ImVec4{teammateColor.r / 255.F, teammateColor.g / 255.F,
               teammateColor.b / 255.F, 1.F},
        "E  Teammate");
    BlueCameraTargets const cameraTargets =
        CollectBlueCameraTargets(snapshot);
    if (m_multipleCameras && cameraTargets.m_count != 0U)
    {
        if (m_hasTacticalTexture)
        {
            UnloadRenderTexture(m_tacticalTexture);
            m_tacticalTexture = {};
            m_hasTacticalTexture = false;
        }
        ImGui::TextDisabled(
            "All blue cameras share distance, height and field of view.");
        DrawMultipleTacticalViews(
            snapshot, cameraTargets, ImGui::GetContentRegionAvail());
    }
    else
    {
        ReleaseMultipleCameraTextures();
        ImVec2 const size = ImGui::GetContentRegionAvail();
        EnsureTacticalTexture(
            static_cast<int>(size.x), static_cast<int>(size.y));
        UpdateAttachedCamera(snapshot);
        RenderTacticalScene(
            snapshot, m_tacticalTexture, m_camera, m_trackedEntity, false);

        if (m_hasTacticalTexture)
        {
            rlImGuiImageRenderTextureFit(&m_tacticalTexture, true);
            bool const hovered = ImGui::IsItemHovered();
            ImVec2 const itemMin = ImGui::GetItemRectMin();
            ImVec2 const itemSize = ImGui::GetItemRectSize();
            UpdateFreeCamera(snapshot,
                             hovered,
                             Vector2{ itemMin.x, itemMin.y },
                             Vector2{ itemSize.x, itemSize.y });

            if (!m_autoAttachCamera)
                ImGui::TextDisabled("Free cam: RMB orbit | MMB pan | Wheel zoom on cursor | LMB focus point | WASD move | Q/E vertical | F focus | R reset");
        }
    }

    ImGui::End();
}

/**
* @brief Draws the top-down minimap window.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::DrawMinimapWindow(entt::registry const& snapshot)
{
    ImGui::Begin("Minimap");

    ImVec2 const cursor = ImGui::GetCursorScreenPos();
    ImVec2 const avail = ImGui::GetContentRegionAvail();
    float const size = std::max(32.F, std::min(avail.x, avail.y));
    ImVec2 const mapMin = cursor;
    ImVec2 const mapMax{ cursor.x + size, cursor.y + size };

    ImGui::InvisibleButton("##minimap_canvas", ImVec2(size, size));
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(mapMin, mapMax, IM_COL32(12, 18, 24, 255), 8.F);
    draw->AddRect(mapMin, mapMax, IM_COL32(190, 210, 230, 180), 8.F, 0, 1.5F);

    float centerNorth = 0.F;
    float centerEast = 0.F;

    float const centerX = (mapMin.x + mapMax.x) * 0.5F;
    float const centerY = (mapMin.y + mapMax.y) * 0.5F;
    float visibleEastRange = std::max(m_eastRange, 1.F) / std::max(m_mapZoom, 0.001F);
    float visibleNorthRange = std::max(m_northRange, 1.F) / std::max(m_mapZoom, 0.001F);
    if (m_showHardBoundary
        && snapshot.ctx().contains<ppo::model::sHardBoundaryOverlay>())
    {
        // Fit the entire terminal rectangle even when a previous zoom level
        // would clip it; no camera or historical Airspace scale is changed.
        auto const& outside =
            snapshot.ctx().get<ppo::model::sHardBoundaryOverlay>();
        visibleEastRange = std::max(
            visibleEastRange, 1.08F * outside.m_halfEast);
        visibleNorthRange = std::max(
            visibleNorthRange, 1.08F * outside.m_halfNorth);
    }
    float const scaleX = (size * 0.5F) / visibleEastRange;
    float const scaleY = (size * 0.5F) / visibleNorthRange;
    float const scale = std::min(scaleX, scaleY);

    auto toMinimap = [&](float north, float east)
    {
        return ImVec2{ centerX + (east - centerEast) * scale, centerY - (north - centerNorth) * scale };
    };

    // The legacy minimap ranges are half-extents. For TwoVsTwo the
    // authoritative world limits are full sizes, shared with the LiDAR.
    // Match the displayed work rectangle to the defended rectangle without
    // altering the historical Airspace minimap.
    float workHalfNorth = m_northRange;
    float workHalfEast = m_eastRange;
    if (snapshot.ctx().contains<ppo::model::sDefendedArea>()
        && snapshot.ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& world = snapshot.ctx().get<ppo::model::sEnvironmentState>();
        workHalfNorth = 0.5F * world.m_maxNorth;
        workHalfEast = 0.5F * world.m_maxEast;
    }
    ImVec2 const worldMin = toMinimap(-workHalfNorth, -workHalfEast);
    ImVec2 const worldMax = toMinimap(workHalfNorth, workHalfEast);
    ImVec2 const workMin{ worldMin.x, worldMax.y };
    ImVec2 const workMax{ worldMax.x, worldMin.y };

    draw->AddRectFilled(workMin, workMax, IM_COL32(20, 30, 40, 85), 6.F);
    draw->AddRect(workMin, workMax, IM_COL32(235, 210, 120, 220), 6.F, 0, 2.0F);
    if (m_showHardBoundary
        && snapshot.ctx().contains<ppo::model::sHardBoundaryOverlay>())
    {
        auto const& outside =
            snapshot.ctx().get<ppo::model::sHardBoundaryOverlay>();
        ImVec2 const farNorthEast = toMinimap(
            outside.m_halfNorth, outside.m_halfEast);
        ImVec2 const farSouthWest = toMinimap(
            -outside.m_halfNorth, -outside.m_halfEast);
        draw->AddRect(
            ImVec2{farSouthWest.x, farNorthEast.y},
            ImVec2{farNorthEast.x, farSouthWest.y},
            IM_COL32(255, 166, 77, 235), 0.F, 0, 2.F);
    }

    for (int ring = 1; ring <= 4; ++ring)
    {
        float const radius = (size * 0.5F * 0.2F) * static_cast<float>(ring);
        draw->AddCircle(ImVec2{ centerX, centerY }, radius, IM_COL32(80, 102, 120, 90), 96, 1.F);
    }

    draw->AddLine(ImVec2{ mapMin.x, centerY }, ImVec2{ mapMax.x, centerY }, IM_COL32(70, 94, 110, 110), 1.F);
    draw->AddLine(ImVec2{ centerX, mapMin.y }, ImVec2{ centerX, mapMax.y }, IM_COL32(70, 94, 110, 110), 1.F);
    draw->AddText(ImVec2{ workMin.x + 8.F, workMin.y + 6.F }, IM_COL32(235, 210, 120, 220), "Work Area");
    if (snapshot.ctx().contains<ppo::model::sDefendedArea>())
    {
        auto const& area = snapshot.ctx().get<ppo::model::sDefendedArea>();
        auto min = toMinimap(area.m_north + area.m_halfNorth,
                             area.m_east - area.m_halfEast);
        auto max = toMinimap(area.m_north - area.m_halfNorth,
                             area.m_east + area.m_halfEast);
        draw->AddRectFilled(min, max, IM_COL32(50, 100, 180, 24));
        draw->AddRect(min, max, IM_COL32(75, 160, 255, 235), 1.F, 0, 2.F);
        draw->AddText(ImVec2{min.x + 6.F, min.y + 6.F},
                      IM_COL32(145, 195, 255, 255), "DEFENDED AREA");
    }
    if (m_showPatrolTrack
        && snapshot.ctx().contains<ppo::model::sPatrolTrackOverlay>())
    {
        auto const& overlay =
            snapshot.ctx().get<ppo::model::sPatrolTrackOverlay>();
        std::size_t const count = std::min(
            overlay.m_sampleCount,
            ppo::model::sPatrolTrackOverlay::SampleCapacity);
        std::array<ImVec2,
                   ppo::model::sPatrolTrackOverlay::SampleCapacity> points{};
        ImU32 const color = IM_COL32(70, 190, 255, 235);
        if (count >= 2U)
        {
            for (std::size_t sample = 0U; sample < count; ++sample)
                points[sample] = toMinimap(
                    overlay.m_north[sample], overlay.m_east[sample]);
            draw->AddPolyline(points.data(), static_cast<int>(count),
                              color, 0, 2.5F);
            for (std::size_t sample = 16U;
                 sample + 1U < count; sample += 16U)
            {
                ImVec2 const tip = points[sample];
                ImVec2 const previous = points[sample - 1U];
                float const dx = tip.x - previous.x;
                float const dy = tip.y - previous.y;
                float const length = std::max(std::hypot(dx, dy), 0.001F);
                float const forwardX = dx / length;
                float const forwardY = dy / length;
                ImVec2 const base{tip.x - 8.F * forwardX,
                                  tip.y - 8.F * forwardY};
                ImVec2 const left{base.x - 3.5F * forwardY,
                                  base.y + 3.5F * forwardX};
                ImVec2 const right{base.x + 3.5F * forwardY,
                                   base.y - 3.5F * forwardX};
                draw->AddTriangleFilled(tip, left, right, color);
            }
        }
    }
    if (m_showPatrolWaypoints
        && snapshot.ctx().contains<ppo::model::sPatrolWaypointOverlay>())
    {
        auto const& overlay =
            snapshot.ctx().get<ppo::model::sPatrolWaypointOverlay>();
        if (overlay.Valid())
        {
            for (std::size_t index = 0U;
                 index < ppo::model::sPatrolWaypointOverlay::WaypointCount;
                 ++index)
            {
                auto const& waypoint = overlay.m_waypoints[index];
                bool const active = index == overlay.m_activeWaypointIndex;
                bool const validated = index < overlay.m_validatedWaypointCount;
                ImU32 const color = active ? IM_COL32(255, 205, 72, 255)
                    : validated ? IM_COL32(105, 125, 140, 205)
                    : IM_COL32(90, 225, 170, 235);
                ImVec2 const point = toMinimap(
                    waypoint.m_north, waypoint.m_east);
                draw->AddCircle(point, active ? 6.F : 4.F,
                    color, 16, active ? 2.5F : 1.6F);
                float const forwardX = std::sin(waypoint.m_heading);
                float const forwardY = -std::cos(waypoint.m_heading);
                draw->AddLine(point,
                    ImVec2{point.x + 8.F * forwardX,
                           point.y + 8.F * forwardY},
                    color, active ? 2.F : 1.2F);
                char marker[6]{};
                std::snprintf(marker, sizeof(marker), "WP%u",
                    static_cast<unsigned>(index + 1U));
                draw->AddText(
                    ImVec2{point.x + 6.F, point.y - 7.F}, color, marker);
            }
        }
    }
    if (m_showPatrolReferences
        && snapshot.ctx().contains<ppo::model::sPatrolReferenceOverlay>())
    {
        auto const& overlay =
            snapshot.ctx().get<ppo::model::sPatrolReferenceOverlay>();
        if (overlay.Valid())
        {
            for (std::size_t roleIndex = 0U;
                 roleIndex < overlay.m_activeReferenceCount; ++roleIndex)
            {
                auto const& reference = overlay.m_references[roleIndex];
                if (!reference.m_valid)
                    continue;

                Color const accent = TacticalRoleColor(
                    roleIndex == 0U
                        ? TacticalRole::Leader : TacticalRole::Teammate);
                ImU32 const color = IM_COL32(
                    accent.r, accent.g, accent.b, 245);
                ImVec2 const projected = toMinimap(
                    reference.m_projectedNorth, reference.m_projectedEast);
                ImVec2 const guidance = toMinimap(
                    reference.m_guidanceNorth, reference.m_guidanceEast);
                draw->AddLine(projected, guidance,
                              IM_COL32(accent.r, accent.g, accent.b, 115),
                              1.4F);
                draw->AddCircle(projected, 5.2F, color, 20, 2.F);
                draw->AddCircleFilled(guidance, 2.6F, color, 12);

                float const forwardX = std::sin(
                    reference.m_guidanceHeading);
                float const forwardY = -std::cos(
                    reference.m_guidanceHeading);
                ImVec2 const tip{guidance.x + 11.F * forwardX,
                                 guidance.y + 11.F * forwardY};
                ImVec2 const base{tip.x - 5.F * forwardX,
                                  tip.y - 5.F * forwardY};
                ImVec2 const left{base.x - 3.F * forwardY,
                                  base.y + 3.F * forwardX};
                ImVec2 const right{base.x + 3.F * forwardY,
                                   base.y - 3.F * forwardX};
                draw->AddLine(guidance, tip, color, 2.F);
                draw->AddTriangleFilled(tip, left, right, color);

                char marker[4]{};
                if (roleIndex == 0U)
                    std::snprintf(marker, sizeof(marker), "L");
                else
                    std::snprintf(marker, sizeof(marker), "W%u",
                                  static_cast<unsigned>(roleIndex));
                draw->AddText(
                    ImVec2{projected.x + 7.F, projected.y - 8.F},
                    color, marker);
            }
        }
    }

    draw->AddLine(ImVec2{ mapMin.x + 18.F, mapMax.y - 28.F }, ImVec2{ mapMin.x + 18.F, mapMax.y - 48.F }, IM_COL32_WHITE, 2.F);
    draw->AddText(ImVec2{ mapMin.x + 24.F, mapMax.y - 54.F }, IM_COL32_WHITE, "N");
    draw->AddLine(ImVec2{ mapMin.x + 18.F, mapMax.y - 28.F }, ImVec2{ mapMin.x + 38.F, mapMax.y - 28.F }, IM_COL32_WHITE, 2.F);
    draw->AddText(ImVec2{ mapMin.x + 42.F, mapMax.y - 36.F }, IM_COL32_WHITE, "E");

    entt::entity hoveredEntity = entt::null;
    float hoveredDistanceSq = std::numeric_limits<float>::max();
    entt::entity primaryTarget = entt::null;
    if (m_trackedEntity != entt::null
    &&  snapshot.valid(m_trackedEntity)
    &&  snapshot.any_of<ppo::model::sSNAState>(m_trackedEntity))
    {
        auto const& trackedSna = snapshot.get<ppo::model::sSNAState>(m_trackedEntity);
        if (trackedSna.m_target != entt::null
        &&  snapshot.valid(trackedSna.m_target)
        &&  snapshot.any_of<ppo::model::sEntityState>(trackedSna.m_target))
        {
            primaryTarget = trackedSna.m_target;
        }
    }

    auto entityView = snapshot.view<ppo::model::sEntityState>();
    for (auto entity : entityView)
    {
        auto const& [state] = entityView.get(entity);
        ImVec2 const center = toMinimap(state.m_x, state.m_y);
        Color const colorSide = ColorForSide(state.m_side);
        ImU32 const color = IM_COL32(colorSide.r, colorSide.g, colorSide.b, 255);
        bool const isSelected = (entity == m_selectedEntity);
        bool const isTracked = (entity == m_trackedEntity);

        if (center.x < mapMin.x - 12.F || center.x > mapMax.x + 12.F || center.y < mapMin.y - 12.F || center.y > mapMax.y + 12.F)
            continue;

        if (state.m_type == ppo::model::eEntityType::Mobile && IsCarVisual(snapshot, entity))
        {
            float const angle = state.m_rotation;
            ImVec2 const forward{std::sin(angle), -std::cos(angle)};
            ImVec2 const right{forward.y, -forward.x};
            auto corner = [&](float longitudinal, float lateral) {
                return ImVec2{center.x + forward.x * longitudinal + right.x * lateral,
                              center.y + forward.y * longitudinal + right.y * lateral};
            };
            ImVec2 const car[4] = {corner(11.F, -5.8F), corner(11.F, 5.8F),
                                   corner(-11.F, 5.8F), corner(-11.F, -5.8F)};
            draw->AddQuadFilled(car[0], car[1], car[2], car[3], color);
            draw->AddQuad(car[0], car[1], car[2], car[3], IM_COL32_WHITE, 1.4F);
            draw->AddLine(corner(2.F, -5.F), corner(2.F, 5.F),
                          IM_COL32(185, 220, 245, 230), 2.F);
        }
        else if (state.m_type == ppo::model::eEntityType::Mobile)
        {
            float const angle = state.m_rotation;
            float const len = 12.F;
            ImVec2 const nose{ center.x + std::sin(angle) * len, center.y - std::cos(angle) * len };
            ImVec2 const left{ center.x + std::sin(angle + 2.55F) * len * 0.75F, center.y - std::cos(angle + 2.55F) * len * 0.75F };
            ImVec2 const right{ center.x + std::sin(angle - 2.55F) * len * 0.75F, center.y - std::cos(angle - 2.55F) * len * 0.75F };
            draw->AddTriangleFilled(ImVec2{ nose.x + 1.F, nose.y + 1.F },
                                    ImVec2{ left.x + 1.F, left.y + 1.F },
                                    ImVec2{ right.x + 1.F, right.y + 1.F },
                                    IM_COL32(0, 0, 0, 120));
            draw->AddTriangleFilled(nose, left, right, color);
            draw->AddTriangle(nose, left, right, IM_COL32(255, 255, 255, 180), 1.4F);
        }
        else if (state.m_type == ppo::model::eEntityType::Objective)
        {
            draw->AddCircleFilled(center, 6.F, color);
            draw->AddCircle(center, 9.F, IM_COL32(255, 255, 255, 160), 24, 1.2F);
        }
        else if (state.m_type == ppo::model::eEntityType::RestrictedZone)
        {
            draw->AddCircleFilled(center, 5.F, color);
        }

        TacticalRole const role = TacticalRoleForEntity(snapshot, entity);
        if (role != TacticalRole::None)
        {
            Color const badgeColor = TacticalRoleColor(role);
            ImVec2 const badgeCenter{center.x, center.y - 17.F};
            draw->AddCircleFilled(
                ImVec2{badgeCenter.x + 1.5F, badgeCenter.y + 1.5F}, 8.F,
                IM_COL32(0, 0, 0, 150));
            draw->AddCircleFilled(
                badgeCenter, 8.F,
                IM_COL32(badgeColor.r, badgeColor.g, badgeColor.b, 245));
            char const* const marker = TacticalRoleMarker(role);
            ImVec2 const textSize = ImGui::CalcTextSize(marker);
            draw->AddText(
                ImVec2{badgeCenter.x - 0.5F * textSize.x,
                       badgeCenter.y - 0.5F * textSize.y},
                IM_COL32(12, 20, 28, 255), marker);
        }

        if (m_showDomains && snapshot.any_of<ppo::model::sSNAState>(entity))
        {
            auto const& snaState = snapshot.get<ppo::model::sSNAState>(entity);
            float const radiusPx = (snaState.m_noescapezone * scale);

            if (state.m_type == ppo::model::eEntityType::RestrictedZone)
            {
                draw->AddCircle(center, radiusPx, IM_COL32(colorSide.r, colorSide.g, colorSide.b, 110), 48, 1.2F);
            }
            else
            {
                int const segments = 28;
                std::array<ImVec2, 29> arc{};
                for (int i = 0; i <= segments; ++i)
                {
                    float const t = static_cast<float>(i) / static_cast<float>(segments);
                    float const a = state.m_rotation - snaState.m_maxShootAngle + t * (2.F * snaState.m_maxShootAngle);
                    arc[static_cast<std::size_t>(i)] = ImVec2{
                        center.x + std::sin(a) * radiusPx,
                        center.y - std::cos(a) * radiusPx};
                }
                std::array<ImVec2, 30> cone{};
                cone.front() = center;
                std::copy(arc.begin(), arc.end(), cone.begin() + 1);
                draw->AddConvexPolyFilled(cone.data(), static_cast<int>(cone.size()), IM_COL32(255, 72, 72, 34));
                draw->AddPolyline(arc.data(), static_cast<int>(arc.size()), IM_COL32(255, 196, 170, 170), 0, 1.5F);
                draw->AddLine(center, arc.front(), IM_COL32(255, 196, 170, 135), 1.1F);
                draw->AddLine(center, arc.back(), IM_COL32(255, 196, 170, 135), 1.1F);
            }
        }

        if (isTracked)
            draw->AddCircle(center, 10.F, IM_COL32(96, 180, 255, 180), 24, 1.4F);

        if (isSelected)
            draw->AddCircle(center, 12.F, IM_COL32(255, 255, 255, 220), 32, 1.8F);

        if (entity == primaryTarget && !state.m_destroyed)
        {
            float const pulse = 0.5F + 0.5F * std::sin(static_cast<float>(GetTime()) * 2.8F);
            float const haloRadius = 14.F + 3.F * pulse;
            draw->AddCircle(center, haloRadius, IM_COL32(255, 214, 92, 235), 40, 2.4F);
            draw->AddCircle(center, haloRadius + 5.F, IM_COL32(255, 244, 190, 120), 40, 1.2F);
            draw->AddLine(ImVec2{ center.x - 8.F, center.y }, ImVec2{ center.x + 8.F, center.y }, IM_COL32(255, 230, 140, 180), 1.4F);
            draw->AddLine(ImVec2{ center.x, center.y - 8.F }, ImVec2{ center.x, center.y + 8.F }, IM_COL32(255, 230, 140, 180), 1.4F);
        }

        auto trailIt = m_entityTrails.find(static_cast<unsigned long long>(state.m_entityId));
        if (trailIt != m_entityTrails.end() && trailIt->second.size() > 1)
        {
            double const now = GetTime();
            for (std::size_t index = 1; index < trailIt->second.size(); ++index)
            {
                auto const& previous = trailIt->second[index - 1];
                auto const& current = trailIt->second[index];
                float const fade = std::clamp(1.F - static_cast<float>(now - current.m_timestamp) / m_trailDuration, 0.F, 1.F);
                if (fade <= 0.01F)
                    continue;

                float const previousNorth = -previous.m_position.z * m_worldScale;
                float const previousEast = previous.m_position.x * m_worldScale;
                float const currentNorth = -current.m_position.z * m_worldScale;
                float const currentEast = current.m_position.x * m_worldScale;
                ImVec2 const p0 = toMinimap(previousNorth, previousEast);
                ImVec2 const p1 = toMinimap(currentNorth, currentEast);
                draw->AddLine(p0,
                              p1,
                              IM_COL32(colorSide.r, colorSide.g, colorSide.b, static_cast<int>(140.F * fade)),
                              1.6F);
            }
        }

        if (state.m_destroyed && m_showDestroyed)
        {
            draw->AddLine(ImVec2{ center.x - 6.F, center.y - 6.F }, ImVec2{ center.x + 6.F, center.y + 6.F }, IM_COL32_WHITE, 1.5F);
            draw->AddLine(ImVec2{ center.x - 6.F, center.y + 6.F }, ImVec2{ center.x + 6.F, center.y - 6.F }, IM_COL32_WHITE, 1.5F);
        }

        ImVec2 const mouse = ImGui::GetIO().MousePos;
        if (ImGui::IsWindowHovered()
        &&  mouse.x >= mapMin.x
        &&  mouse.x <= mapMax.x
        &&  mouse.y >= mapMin.y
        &&  mouse.y <= mapMax.y)
        {
            float const dx = mouse.x - center.x;
            float const dy = mouse.y - center.y;
            float const distSq = dx * dx + dy * dy;
            if (distSq < hoveredDistanceSq && distSq < 196.F)
            {
                hoveredDistanceSq = distSq;
                hoveredEntity = entity;
            }
        }
    }

    if (m_focusSelectionInMinimap
    &&  m_selectedEntity != entt::null
    &&  snapshot.valid(m_selectedEntity)
    &&  snapshot.any_of<ppo::model::sEntityState>(m_selectedEntity))
    {
        auto const& selected = snapshot.get<ppo::model::sEntityState>(m_selectedEntity);
        ImVec2 const focus = toMinimap(selected.m_x, selected.m_y);
        draw->AddCircle(focus, 16.F, IM_COL32(255, 255, 255, 110), 32, 1.2F);
        draw->AddLine(ImVec2{ focus.x - 18.F, focus.y }, ImVec2{ focus.x + 18.F, focus.y }, IM_COL32(255, 255, 255, 90), 1.0F);
        draw->AddLine(ImVec2{ focus.x, focus.y - 18.F }, ImVec2{ focus.x, focus.y + 18.F }, IM_COL32(255, 255, 255, 90), 1.0F);
    }

    if (hoveredEntity != entt::null && snapshot.any_of<ppo::model::sEntityState>(hoveredEntity))
    {
        auto const& state = snapshot.get<ppo::model::sEntityState>(hoveredEntity);
        ImGui::SetTooltip("%s\nNorth %.0f / East %.0f", EntityDisplayName(snapshot, hoveredEntity, state).c_str(), state.m_x, state.m_y);

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            m_selectedEntity = hoveredEntity;
    }

    ImGui::End();
}

/**
* @brief Draws the entity list and high-level tactical summary.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::DrawEntitiesWindow(entt::registry const& snapshot)
{
    ImGui::Begin("Entities");

    ImGui::InputTextWithHint("##entityfilter", "Filter by type / side / id", m_entityFilter, IM_ARRAYSIZE(m_entityFilter));
    ImGui::Separator();

    std::string filter = m_entityFilter;
    std::transform(filter.begin(), filter.end(), filter.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    auto entityView = snapshot.view<ppo::model::sEntityState>();
    if (ImGui::BeginTable("##entities", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY))
    {
        ImGui::TableSetupColumn("Entity");
        ImGui::TableSetupColumn("Side");
        ImGui::TableSetupColumn("Speed");
        ImGui::TableSetupColumn("State");
        ImGui::TableHeadersRow();

        for (auto entity : entityView)
        {
            auto const& [state] = entityView.get(entity);
            if (!MatchesFilter(filter, state, entity))
                continue;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            std::string const label = EntityDisplayName(snapshot, entity, state);
            if (ImGui::Selectable(label.c_str(), m_selectedEntity == entity, ImGuiSelectableFlags_SpanAllColumns))
                m_selectedEntity = entity;

            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                m_trackedEntity = entity;

            ImGui::TableNextColumn();
            ImGui::TextColored(ColorForSideUi(state.m_side), "%s", SideLabel(state.m_side));

            ImGui::TableNextColumn();
            ImGui::Text("%.1f", state.m_speed);

            ImGui::TableNextColumn();
            if (state.m_destroyed)
                ImGui::TextColored(ImVec4(1.F, 0.55F, 0.55F, 1.F), "Destroyed");
            else if (entity == m_trackedEntity)
                ImGui::TextColored(ImVec4(0.55F, 0.8F, 1.F, 1.F), "Tracked");
            else
                ImGui::TextUnformatted("Active");
        }

        ImGui::EndTable();
    }

    ImGui::End();
}

/**
* @brief Draws concise operational details for the selected entity.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::DrawInspectorWindow(entt::registry const& snapshot)
{
    ImGui::Begin("Entity details", &m_showInspection);

    if (m_selectedEntity == entt::null || !snapshot.valid(m_selectedEntity)
        || !snapshot.any_of<ppo::model::sEntityState>(m_selectedEntity))
    {
        ImGui::TextDisabled("Select an entity from the tactical view or list.");
        ImGui::End();
        return;
    }

    auto const& entityState = snapshot.get<ppo::model::sEntityState>(m_selectedEntity);
    ImGui::TextUnformatted(
        EntityDisplayName(snapshot, m_selectedEntity, entityState).c_str());
    ImGui::SameLine();
    ImGui::TextColored(ColorForSideUi(entityState.m_side), "%s", SideLabel(entityState.m_side));
    TacticalRole const role = TacticalRoleForEntity(snapshot, m_selectedEntity);
    if (role != TacticalRole::None)
    {
        Color const roleColor = TacticalRoleColor(role);
        ImGui::SameLine();
        ImGui::TextColored(
            ImVec4{roleColor.r / 255.F, roleColor.g / 255.F,
                   roleColor.b / 255.F, 1.F},
            "[%s] %s", TacticalRoleMarker(role),
            role == TacticalRole::Leader ? "Leader" : "Teammate");
    }
    ImGui::SameLine();
    if (ImGui::Button("Follow"))
        m_trackedEntity = m_selectedEntity;

    ImGui::SeparatorText("Overview");
    if (ImGui::BeginTable("##entity-overview", 2,
                          ImGuiTableFlags_RowBg
                              | ImGuiTableFlags_BordersInnerV))
    {
        auto const row = [](char const* const label, char const* const value) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", label);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(value);
        };
        char value[128]{};
        std::snprintf(value, sizeof(value), "N %.0f m   E %.0f m",
                      entityState.m_x, entityState.m_y);
        row("Position", value);
        std::snprintf(value, sizeof(value), "%.1f m/s", entityState.m_speed);
        row("Speed", value);
        std::snprintf(value, sizeof(value), "%.1f deg",
                      entityState.m_rotation * DEG_PER_RAD);
        row("Heading", value);
        row("State", entityState.m_destroyed ? "Destroyed" : "Active");

        if (snapshot.any_of<ppo::model::sEntityMetadata>(m_selectedEntity))
        {
            auto const& metadata = snapshot.get<ppo::model::sEntityMetadata>(m_selectedEntity);
            row("Scenario name", metadata.m_name.c_str());
            row("Platform", metadata.m_archetype.c_str());
        }
        ImGui::EndTable();
    }

    if (snapshot.any_of<ppo::model::sSNAState>(m_selectedEntity))
    {
        auto const& awareness =
            snapshot.get<ppo::model::sSNAState>(m_selectedEntity);
        ImGui::SeparatorText("Engagement");
        if (awareness.m_target != entt::null
            && snapshot.valid(awareness.m_target)
            && snapshot.any_of<ppo::model::sEntityState>(awareness.m_target))
        {
            auto const& target = snapshot.get<ppo::model::sEntityState>(awareness.m_target);
            ImGui::Text("Target: %s",
                        EntityDisplayName(snapshot, awareness.m_target, target).c_str());
        }
        else
            ImGui::TextDisabled("No active target");
        ImGui::Text("Closing speed %.1f m/s  |  PK %.2f",
                    awareness.m_closingspeed, awareness.m_pk);
        ImGui::Text("No-escape range %.0f m  |  Missiles %hu/%hu",
                    awareness.m_noescapezone, awareness.m_nbmissiles,
                    awareness.m_maxnbmissiles);
    }

    if (snapshot.any_of<ppo::model::sFlightModelState>(m_selectedEntity))
    {
        auto const& flight =
            snapshot.get<ppo::model::sFlightModelState>(m_selectedEntity);
        ImGui::SeparatorText("Motion");
        ImGui::Text("Acceleration %.2f  |  Turn acceleration %.2f",
                    flight.m_acc, flight.m_accpsi);
        ImGui::Text("Fuel %.1f / %.1f", flight.m_fuel, flight.m_maxfuel);
    }
    if (snapshot.any_of<ppo::model::sSNAControlState>(m_selectedEntity))
    {
        auto const& control =
            snapshot.get<ppo::model::sSNAControlState>(m_selectedEntity);
        ImGui::TextDisabled("Commands  acceleration %.2f  |  turn %.2f",
                            control.m_accCmd, control.m_rotSpeedCmd);
    }

    ImGui::End();
}

/**
* @brief Draws the agent monitoring and control window.
* @param snapshot Snapshot registry used for rendering.
*/
void
EnvironmentRenderer::DrawAgentsWindow(entt::registry const& snapshot)
{
    ImGui::Begin("Learning");

#ifdef PPO_WITH_FLIGHT_CONTROL
    if (m_flightControlView && snapshot.ctx().contains<ppo::examples::flight::FlightSnapshot>())
    {
        auto const& flight = snapshot.ctx().get<ppo::examples::flight::FlightSnapshot>();
        ImGui::Text("%s%s", flight.training ? "Hybrid PPO training" : "Independent evaluation",
            flight.failed ? " | Error" : flight.finished ? " | Complete" : "");
        if (flight.training)
        {
            ImGui::TextDisabled("Source: %s",flight.resumed ? "latest training checkpoint" : "new policy");
            ImGui::TextDisabled("Current block ends at update %llu",static_cast<unsigned long long>(flight.trainingUpdateLimit));
            if (flight.finished && !flight.failed)
                ImGui::TextUnformatted("Play continues training from this model.");
        }
        ImGui::Text("Episodes %llu | Touchdowns %llu | Stops %llu",
            static_cast<unsigned long long>(flight.episodes),static_cast<unsigned long long>(flight.touchdowns),
            static_cast<unsigned long long>(flight.stops));
        ImGui::Text("Crashes %llu | Timeouts %llu",static_cast<unsigned long long>(flight.crashes),
            static_cast<unsigned long long>(flight.timeouts));
        if (flight.hasBest)
            ImGui::Text("Validated BEST: %.1f%% successful stops",flight.bestScore);
        else
            ImGui::TextDisabled("No validated BEST yet");
    }
#endif

    auto const view = snapshot.view<ppo::model::sAgentStateSnaphot>();
    if (view.begin() == view.end())
        ImGui::TextDisabled("No training agent found.");

    for (auto entity : view)
    {
        auto const& [state] = view.get(entity);
        auto agent = GetAgentController(state.m_id);
        if (agent == nullptr)
            continue;

        ImGui::PushID(static_cast<int>(state.m_id));
        if (ImGui::CollapsingHeader(state.m_name.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextDisabled("Model version %u  |  Step %lld / %lld",
                                state.m_common.m_version,
                                state.m_common.m_step, state.m_common.m_maxStep);
            ImGui::Text("Instant reward: %.3f", state.m_common.m_reward);
            if (ImGui::BeginTabBar("##learning-tabs"))
            {
                if (ImGui::BeginTabItem("Progress"))
                {
                    DrawLearningProgress(agent, state);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(m_flightControlView ? "Rewards" : "Rewards & Settings"))
                {
                    if (!m_flightControlView)
                        ImGui::TextWrapped("Changes apply to the active training agent.");
                    DrawRewardControls(agent, state);
                    ImGui::SeparatorText("Rollout");
                    DrawTrainingControls(agent, state);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Diagnostics"))
                {
                    if (state.m_common.m_version == 0U)
                        ImGui::TextDisabled(
                            "Waiting for the first complete PPO rollout.");
                    else
                    {
                        ImGui::Text("Update %u  |  Total %.5f",
                            state.m_common.m_version, state.m_common.m_loss);
                        ImGui::Text("Actor %.5f  |  Value %.5f  |  Entropy %.5f",
                            state.m_common.m_actorloss,
                            state.m_common.m_valueloss,
                            state.m_common.m_entropyloss);
                    }
                    ImGui::TextDisabled(
                        "Losses sampled from the last PPO mini-batch.");
                    DrawAgentPlot(agent, state);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }
        ImGui::PopID();
    }

    ImGui::End();
}

/**
* @brief Executes one renderer frame.
* @return True while the application should keep running.
*/
bool
EnvironmentRenderer::Run(void)
{
    if (p_environment == nullptr || !IsWindowReady())
        return false;

    if (WindowShouldClose())
        return false;

    BeginDrawing();
    ClearBackground(Color{ 8, 12, 18, 255 });

    rlImGuiBegin();
    BuildDockspace();

    if (m_stopRequested)
    {
        p_environment->Stop();
        m_stopRequested = false;
    }
    if (m_flightPlayRequested)
    {
        // Stop/join must never run while ReadSnapshot holds the snapshot mutex.
        if (m_flightRestartBlock)
            p_environment->Stop();
        p_environment->Play();
        m_flightPlayRequested = false;
        m_flightRestartBlock = false;
    }

    p_environment->ReadSnapshot([&](entt::registry const& snapshot)
    {
        UpdateMissionTelemetry(snapshot);
        UpdateTrails(snapshot);
        UpdateExplosionEffects(snapshot);
        UpdateLearningTelemetry(snapshot);

        if (m_showSimulation)
        {
            if (ImGui::Begin("Simulation", &m_showSimulation))
            {
                DrawSimulationControls(snapshot);
                if (!m_flightControlView && ImGui::CollapsingHeader("Scene details"))
                    DrawStatusBar(snapshot);
                if (p_pinnedPlot != nullptr && ImGui::CollapsingHeader("Pinned diagnostics"))
                    DrawOneAgentPlot(*p_pinnedPlot);
            }
            ImGui::End();
        }

        DrawTacticalWindow(snapshot);
        if (!m_flightControlView && m_showMinimap)
            DrawMinimapWindow(snapshot);
        if (!m_flightControlView && m_showEntities)
            DrawEntitiesWindow(snapshot);
        if (!m_flightControlView && m_showInspection)
            DrawInspectorWindow(snapshot);
        DrawAgentsWindow(snapshot);
#ifdef PPO_WITH_FLIGHT_CONTROL
        if (snapshot.ctx().contains<ppo::examples::flight::FlightSnapshot>())
            ppo::examples::flight::DrawFlightInstruments(snapshot.ctx().get<ppo::examples::flight::FlightSnapshot>());
#endif
    });

    rlImGuiEnd();
    EndDrawing();

    return !WindowShouldClose();
}

} /* ppo::renderer */
