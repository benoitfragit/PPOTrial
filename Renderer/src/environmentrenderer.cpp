#include "environmentrenderer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <rlImGui.h>

#include "entt/meta/meta.hpp"
#include "entt/meta/resolve.hpp"

#include "interface.hpp"
#include "metaregistry.hpp"

namespace ppo::renderer {

namespace {

constexpr float PI_F = 3.14159265359F;
constexpr float DEG_PER_RAD = 57.2957795131F;
constexpr float CAMERA_MOUSE_ORBIT_SPEED = 0.0075F;
constexpr float CAMERA_MOUSE_PAN_SPEED = 0.0100F;
constexpr float CAMERA_WHEEL_ZOOM_SPEED = 1.25F;
constexpr float CAMERA_ATTACH_SMOOTHING = 0.12F;

Vector3
VectorLerpLocal(Vector3 const& a, Vector3 const& b, float amount)
{
    return Vector3{
        a.x + (b.x - a.x) * amount,
        a.y + (b.y - a.y) * amount,
        a.z + (b.z - a.z) * amount
    };
}

Vector3
VectorAdd(Vector3 const& a, Vector3 const& b)
{
    return Vector3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

Vector3
VectorSub(Vector3 const& a, Vector3 const& b)
{
    return Vector3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

Vector3
VectorScale(Vector3 const& v, float scalar)
{
    return Vector3{ v.x * scalar, v.y * scalar, v.z * scalar };
}

float
VectorLength(Vector3 const& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector3
VectorNormalizeSafe(Vector3 const& v)
{
    float const length = VectorLength(v);
    if (length <= 1.0e-5F)
        return Vector3{ 0.F, 0.F, 0.F };

    return VectorScale(v, 1.F / length);
}

Vector3
VectorCross(Vector3 const& a, Vector3 const& b)
{
    return Vector3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

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

char const*
EntityTypeLabel(ppo::model::eEntityType type)
{
    switch (type)
    {
    case ppo::model::eEntityType::Aircraft:  return "Aircraft";
    case ppo::model::eEntityType::SAM:       return "SAM";
    case ppo::model::eEntityType::Objective: return "Objective";
    case ppo::model::eEntityType::Missile:   return "Missile";
    default:                                 return "Unknown";
    }
}

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

Vector3
ToWorld3D(float north, float east, float down, float scale)
{
    return Vector3{ east / scale, -down / scale, -north / scale };
}

Vector3
HeadingForward(float rotation)
{
    return Vector3{ std::sin(rotation), 0.F, -std::cos(rotation) };
}

void
DrawAircraftGlyph3D(ppo::model::sEntityState const& state, float scale, Color color)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    Vector3 const forward = HeadingForward(state.m_rotation);
    Vector3 const right{ -forward.z, 0.F, forward.x };
    Vector3 const lift{ 0.F, 1.F, 0.F };

    float const length = 1.25F;
    float const width = 0.55F;
    float const bodyHeight = 0.24F;

    Vector3 const nose{
        center.x + forward.x * length,
        center.y + bodyHeight * 0.35F,
        center.z + forward.z * length
    };
    Vector3 const left{
        center.x - forward.x * length * 0.48F + right.x * width,
        center.y,
        center.z - forward.z * length * 0.48F + right.z * width
    };
    Vector3 const rightPt{
        center.x - forward.x * length * 0.48F - right.x * width,
        center.y,
        center.z - forward.z * length * 0.48F - right.z * width
    };
    Vector3 const topNose{ nose.x, nose.y + bodyHeight, nose.z };
    Vector3 const topLeft{ left.x, left.y + bodyHeight, left.z };
    Vector3 const topRight{ rightPt.x, rightPt.y + bodyHeight, rightPt.z };
    Vector3 const tail{
        center.x - forward.x * length * 0.72F,
        center.y + bodyHeight * 0.55F,
        center.z - forward.z * length * 0.72F
    };
    Vector3 const finTop{ tail.x + lift.x * 0.01F, tail.y + bodyHeight * 1.45F, tail.z + lift.z * 0.01F };

    DrawTriangle3D(nose, left, rightPt, Fade(BLACK, 0.40F));
    DrawTriangle3D(topNose, topRight, topLeft, Fade(BLACK, 0.25F));
    DrawTriangle3D(nose, left, rightPt, color);
    DrawTriangle3D(topNose, topRight, topLeft, Fade(color, 0.90F));
    DrawTriangle3D(nose, topNose, left, Fade(color, 0.82F));
    DrawTriangle3D(left, topNose, topLeft, Fade(color, 0.74F));
    DrawTriangle3D(nose, rightPt, topNose, Fade(color, 0.88F));
    DrawTriangle3D(rightPt, topRight, topNose, Fade(color, 0.76F));
    DrawTriangle3D(left, topLeft, rightPt, Fade(color, 0.70F));
    DrawTriangle3D(rightPt, topLeft, topRight, Fade(color, 0.66F));
    DrawTriangle3D(tail, finTop, topNose, Fade(color, 0.58F));

    DrawLine3D(center, nose, Fade(WHITE, 0.70F));
    DrawLine3D(center,
               Vector3{ center.x + forward.x * 1.80F, center.y + 0.08F, center.z + forward.z * 1.80F },
               Fade(WHITE, 0.82F));
    DrawLine3D(left, topLeft, Fade(BLACK, 0.45F));
    DrawLine3D(rightPt, topRight, Fade(BLACK, 0.45F));
    DrawSphere(Vector3{ center.x, center.y + 0.04F, center.z }, 0.12F, Fade(WHITE, 0.28F));
    DrawCylinder(Vector3{ center.x, -0.02F, center.z }, 0.18F, 0.28F, 0.02F, 16, Fade(BLACK, 0.22F));
}

void
DrawObjectiveGlyph3D(ppo::model::sEntityState const& state, float scale, Color color)
{
    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, scale);
    DrawSphere(center, 0.28F, color);
    DrawSphereWires(center, 0.34F, 10, 10, Fade(WHITE, 0.45F));
    DrawCylinder(Vector3{ center.x, 0.34F, center.z }, 0.03F, 0.03F, 0.70F, 8, Fade(WHITE, 0.55F));
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
            Color color)
{
    if (trail.size() < 2 || duration <= 0.F)
        return;

    double const now = GetTime();
    for (std::size_t index = 1; index < trail.size(); ++index)
    {
        auto const& previous = trail[index - 1];
        auto const& current = trail[index];
        float const age = static_cast<float>(now - current.m_timestamp);
        float const fade = std::clamp(1.F - age / duration, 0.F, 1.F);
        if (fade <= 0.01F)
            continue;

        Color const segmentColor{ color.r, color.g, color.b, static_cast<unsigned char>(180.F * fade) };
        DrawLine3D(previous.m_position, current.m_position, segmentColor);
        DrawSphere(current.m_position, 0.04F + 0.015F * fade, Fade(segmentColor, 0.35F));
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
DrawDestroyMarker3D(Vector3 const& center)
{
    DrawLine3D(Vector3{ center.x - 0.25F, center.y, center.z - 0.25F },
               Vector3{ center.x + 0.25F, center.y, center.z + 0.25F },
               WHITE);
    DrawLine3D(Vector3{ center.x - 0.25F, center.y, center.z + 0.25F },
               Vector3{ center.x + 0.25F, center.y, center.z - 0.25F },
               WHITE);
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
    constexpr std::size_t beamCount = 31;
    std::array<float, beamCount> nearestCollisionRanges{};
    nearestCollisionRanges.fill(1.F);
    constexpr float noInterceptThreshold = 0.999F;
    constexpr float minDenominator = 1.0e-4F;

    float const maxAngleArea = PI_F;
    float const halfAngleArea = maxAngleArea * 0.5F;
    float const stepAngle = maxAngleArea / static_cast<float>(beamCount - 1);

    auto sams = snapshot.view<ppo::model::sEntityState, ppo::model::sSNAState>();
    for (auto other : sams)
    {
        if (other == entity)
            continue;

        auto const& [otherState, otherSna] = sams.get(other);
        if (otherState.m_type != ppo::model::eEntityType::SAM
        ||  otherState.m_side == state.m_side
        ||  otherState.m_destroyed)
            continue;

        float angle = -halfAngleArea;
        for (std::size_t index = 0; index < beamCount; ++index)
        {
            float const angleRad = state.m_rotation + angle;
            float const mx = state.m_x - otherState.m_x;
            float const my = state.m_y - otherState.m_y;
            float const safetyRadius = otherSna.m_noescapezone;
            float const distanceToCenter = std::sqrt(mx * mx + my * my);
            float const b = mx * std::cos(angleRad) + my * std::sin(angleRad);
            float const c = mx * mx + my * my - safetyRadius * safetyRadius;
            float const delta = b * b - c;

            if (distanceToCenter <= safetyRadius)
            {
                nearestCollisionRanges[index] = 0.F;
                angle += stepAngle;
                continue;
            }

            if (delta >= 0.F)
            {
                float const sqrtDelta = std::sqrt(delta);
                float const d1 = -b - sqrtDelta;
                float const d2 = -b + sqrtDelta;

                if (d1 >= 0.F)
                    nearestCollisionRanges[index] = std::min(nearestCollisionRanges[index], d1 / (d1 + std::max(snaState.m_noescapezone, 1.F)));
            }

            angle += stepAngle;
        }
    }

    Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, worldScale);
    for (std::size_t index = 0; index + 1 < beamCount; ++index)
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

        auto const decodeInterceptDistance = [&](float ratio) -> float
        {
            if (ratio >= noInterceptThreshold)
                return 0.F;

            float const safeNoEscapeZone = std::max(snaState.m_noescapezone, 1.F);
            float const denominator = std::max(minDenominator, 1.F - ratio);
            return (ratio * safeNoEscapeZone) / denominator;
        };

        float const distance0 = decodeInterceptDistance(r0);
        float const distance1 = decodeInterceptDistance(r1);
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

std::string
FieldDisplayName(char const* rawName)
{
    if (rawName == nullptr)
        return "<unnamed>";

    std::string name{ rawName };
    if (name.rfind("m_", 0) == 0)
        name.erase(0, 2);

    return name;
}

char const*
MetaDisplayName(entt::meta_type const& type)
{
    if (auto const* info = static_cast<ppo::reflection::sMetaDescriptor*>(type.custom()); info != nullptr)
        return info->scenario_name;

    return type.name() != nullptr ? type.name() : "<unnamed>";
}

bool
TryGetArithmeticString(entt::meta_any const& any, std::string& out)
{
    if (auto const* value = any.try_cast<bool>())
    {
        out = *value ? "true" : "false";
        return true;
    }

    if (auto const* value = any.try_cast<unsigned long long>())
    {
        out = std::to_string(*value);
        return true;
    }

    if (auto const* value = any.try_cast<unsigned int>())
    {
        out = std::to_string(*value);
        return true;
    }

    if (auto const* value = any.try_cast<int>())
    {
        out = std::to_string(*value);
        return true;
    }

    if (auto const* value = any.try_cast<float>())
    {
        out = TextFormat("%.3f", *value);
        return true;
    }

    if (auto const* value = any.try_cast<double>())
    {
        out = TextFormat("%.3f", *value);
        return true;
    }

    return false;
}

std::string
TryResolveEntityName(entt::registry const* snapshot, entt::entity entity)
{
    if (snapshot == nullptr || entity == entt::null || !snapshot->valid(entity))
        return {};

    if (snapshot->any_of<ppo::model::sEntityState>(entity))
    {
        auto const& state = snapshot->get<ppo::model::sEntityState>(entity);
        return EntityDisplayName(*snapshot, entity, state);
    }

    return "Entity #" + std::to_string(static_cast<entt::id_type>(entity));
}

void
DrawMetaAnyRecursive(char const* label, entt::meta_any value, entt::registry const* snapshot = nullptr)
{
    if (!value)
    {
        ImGui::Text("%s: <null>", label);
        return;
    }

    auto const type = value.type();
    std::string stringValue{};

    if (TryGetArithmeticString(value, stringValue))
    {
        ImGui::Text("%s: %s", label, stringValue.c_str());
        return;
    }

    if (auto const* text = value.try_cast<std::string>())
    {
        ImGui::Text("%s: %s", label, text->c_str());
        return;
    }

    if (auto const* entityId = value.try_cast<ppo::model::EntityId>())
    {
        std::string const resolved = TryResolveEntityName(snapshot, *entityId);
        if (!resolved.empty())
            ImGui::Text("%s: %s", label, resolved.c_str());
        else
            ImGui::Text("%s: <entity:%u>", label, static_cast<unsigned int>(*entityId));
        return;
    }

    auto seq = value.as_sequence_container();
    if (seq)
    {
        if (ImGui::TreeNode(label, "%s [%zu]", label, seq.size()))
        {
            std::size_t index = 0;
            for (auto it = seq.begin(); it != seq.end(); ++it, ++index)
            {
                auto item = *it;
                std::string const itemLabel = "item[" + std::to_string(index) + "]";
                DrawMetaAnyRecursive(itemLabel.c_str(), item, snapshot);
            }
            ImGui::TreePop();
        }
        return;
    }

    auto map = value.as_associative_container();
    if (map)
    {
        if (ImGui::TreeNode(label, "%s {%zu}", label, map.size()))
        {
            std::size_t index = 0;
            for (auto it = map.begin(); it != map.end(); ++it, ++index)
            {
                auto pair = *it;
                std::string const keyLabel = "key[" + std::to_string(index) + "]";
                std::string const valueLabel = "value[" + std::to_string(index) + "]";
                DrawMetaAnyRecursive(keyLabel.c_str(), pair.first, snapshot);
                DrawMetaAnyRecursive(valueLabel.c_str(), pair.second, snapshot);
            }
            ImGui::TreePop();
        }
        return;
    }

    if (ImGui::TreeNode(label, "%s (%s)", label, MetaDisplayName(type)))
    {
        bool hasData = false;
        for (auto const& [_, data] : type.data())
        {
            hasData = true;
            auto field = data.get(value);
            std::string const fieldName = FieldDisplayName(data.name());
            DrawMetaAnyRecursive(fieldName.c_str(), field, snapshot);
        }

        if (!hasData)
            ImGui::TextUnformatted("<no reflected fields>");

        ImGui::TreePop();
    }
}

template<typename Component>
void
DrawComponentIfPresent(entt::registry const& snapshot, entt::entity entity)
{
    if (!snapshot.any_of<Component>(entity))
        return;

    auto const& component = snapshot.get<Component>(entity);
    auto type = entt::resolve<Component>();
    DrawMetaAnyRecursive(MetaDisplayName(type), entt::forward_as_meta(component), &snapshot);
}

} // namespace

EnvironmentRenderer&
EnvironmentRenderer::GetInstance(std::string const& title)
{
    static EnvironmentRenderer renderer{ title };
    return renderer;
}

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
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    m_camera.position = Vector3{ 0.F, 12.F, 12.F };
    m_camera.target = Vector3{ 0.F, 0.F, 0.F };
    m_camera.up = Vector3{ 0.F, 1.F, 0.F };
    m_camera.fovy = 45.F;
    m_camera.projection = CAMERA_PERSPECTIVE;
}

EnvironmentRenderer::~EnvironmentRenderer(void)
{
    if (m_hasTacticalTexture)
        UnloadRenderTexture(m_tacticalTexture);

    ImPlot::DestroyContext();
    rlImGuiShutdown();

    if (IsWindowReady())
        CloseWindow();
}

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

void
EnvironmentRenderer::SetEnvironment(ppo::model::Environment& env) noexcept
{
    p_environment = &env;

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
        ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.F);
    ImGui::Begin("##DockHost", nullptr, flags);
    ImGui::PopStyleVar(2);

    ImGuiID const dockspaceId = ImGui::GetID("##MainDockspace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.F, 0.F), ImGuiDockNodeFlags_PassthruCentralNode);
    BuildDefaultLayout(dockspaceId);
    ImGui::End();
}

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

    ImGuiID dockLeft = dockspaceId;
    ImGuiID dockRight = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Right, 0.27F, nullptr, &dockLeft);
    ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Down, 0.34F, nullptr, &dockLeft);
    ImGuiID dockTopLeft = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Left, 0.26F, nullptr, &dockLeft);

    ImGuiID dockRightBottom = ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.44F, nullptr, &dockRight);
    ImGuiID dockBottomRight = ImGui::DockBuilderSplitNode(dockBottom, ImGuiDir_Right, 0.45F, nullptr, &dockBottom);

    ImGui::DockBuilderDockWindow("Simulation", dockTopLeft);
    ImGui::DockBuilderDockWindow("Entities", dockTopLeft);
    ImGui::DockBuilderDockWindow("3D Tactical View", dockLeft);
    ImGui::DockBuilderDockWindow("Minimap", dockBottom);
    ImGui::DockBuilderDockWindow("Inspection", dockBottomRight);
    ImGui::DockBuilderDockWindow("Agents", dockRight);
    ImGui::DockBuilderDockWindow("Meta Catalog", dockRightBottom);

    ImGui::DockBuilderFinish(dockspaceId);
}

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

void
EnvironmentRenderer::UpdateTrackedEntity(entt::registry const& snapshot)
{
    if (m_trackedEntity != entt::null && snapshot.valid(m_trackedEntity) && snapshot.any_of<ppo::model::sEntityState>(m_trackedEntity))
        return;

    m_trackedEntity = entt::null;

    bool hasTrainingAgent = false;
    auto agentView = snapshot.view<ppo::model::sAgentStateSnaphot>();
    for (auto entity : agentView)
    {
        auto const& [state] = agentView.get(entity);
        if (state.m_name.find("Training") != std::string::npos)
        {
            hasTrainingAgent = true;
            break;
        }
    }

    auto entityView = snapshot.view<ppo::model::sEntityState>();
    for (auto entity : entityView)
    {
        auto const& [state] = entityView.get(entity);
        if (state.m_type != ppo::model::eEntityType::Aircraft || state.m_destroyed)
            continue;

        if (!hasTrainingAgent || state.m_side == ppo::model::eEntitySide::Friend)
        {
            m_trackedEntity = entity;
            break;
        }
    }

    if (m_selectedEntity == entt::null)
        m_selectedEntity = m_trackedEntity;
}

void
EnvironmentRenderer::UpdateAttachedCamera(entt::registry const& snapshot)
{
    if (!m_autoAttachCamera)
        return;

    UpdateTrackedEntity(snapshot);

    if (m_trackedEntity == entt::null || !snapshot.any_of<ppo::model::sEntityState>(m_trackedEntity))
        return;

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

    m_camera.target = VectorLerpLocal(m_camera.target, desiredTarget, CAMERA_ATTACH_SMOOTHING);
    m_camera.position = VectorLerpLocal(m_camera.position, desiredPosition, CAMERA_ATTACH_SMOOTHING);
}

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

void
EnvironmentRenderer::DrawSimulationControls(void)
{
    if (p_environment == nullptr)
        return;

    if (ImGui::Button("Play"))
    {
        for (auto const& [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = true;

        p_environment->Play();
    }

    ImGui::SameLine();
    if (ImGui::Button("Pause"))
    {
        for (auto const& [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = false;

        p_environment->Pause();
    }

    ImGui::SameLine();
    if (ImGui::Button("Stop"))
    {
        for (auto const& [_, agent] : p_agents)
            if (agent != nullptr)
                agent->m_started = false;

        p_environment->Stop();
    }

    ImGui::SeparatorText("World");

    bool refreshRanges = false;
    if (ImGui::SliderFloat("Area width (km)", &m_areawidth, 1.F, 500.F))
        refreshRanges = true;

    if (ImGui::SliderFloat("Time step (ms)", &m_timestepMs, 1.F, 200.F))
        p_environment->SetTimeStep(m_timestepMs / 1000.F);

    char const* const scaleLevels[]{ "m", "km", "NM" };
    if (ImGui::Combo("Scale level", &m_currentScaleLevel, scaleLevels, IM_ARRAYSIZE(scaleLevels)))
    {
        m_worldScale = (m_currentScaleLevel == 0) ? 1.F : ((m_currentScaleLevel == 1) ? 1000.F : 1852.F);
        refreshRanges = true;
    }

    ImGui::SeparatorText("Visuals");
    ImGui::Checkbox("Auto-attach camera", &m_autoAttachCamera);
    ImGui::Checkbox("Show grid", &m_showGrid);
    ImGui::Checkbox("Show domains", &m_showDomains);
    ImGui::Checkbox("Show pseudo lidar", &m_showPseudoLidar);
    ImGui::Checkbox("Show destroyed markers", &m_showDestroyed);
    ImGui::Checkbox("Show meta catalog", &m_showMetaCatalog);
    ImGui::Checkbox("Center minimap on selection", &m_focusSelectionInMinimap);

    ImGui::SliderFloat("Camera distance", &m_cameraDistance, 5.F, 60.F);
    ImGui::SliderFloat("Camera height", &m_cameraHeight, 2.F, 25.F);
    ImGui::SliderFloat("Trail duration", &m_trailDuration, 2.F, 30.F, "%.1f s");
    ImGui::SliderFloat("Aircraft domain height", &m_domainHeight, 0.3F, 5.F);
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

void
EnvironmentRenderer::DrawTrainingControls(std::shared_ptr<AgentController> const& agent,
                                          ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr || p_environment == nullptr)
        return;

    agent->m_rollout = static_cast<int>(state.m_common.m_maxStep);
    if (ImGui::SliderInt("Rollout Buffer", &agent->m_rollout, 1, 10000))
    {
        ppo::model::sAgentRealTimeCommand command{};
        command.m_id = agent->m_id;
        command.m_rolloutBuffer = agent->m_rollout;
        command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RolloutBuffer;
        p_environment->AddAgentCommand(command);
    }
}

void
EnvironmentRenderer::DrawRewardControls(std::shared_ptr<AgentController> const& agent,
                                        ppo::model::sAgentStateSnaphot const& state)
{
    if (agent == nullptr || p_environment == nullptr)
        return;

    agent->m_clamp = state.m_common.m_rewardClamp;
    agent->m_scalings = state.m_scalings;

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

        if (ImGui::SliderFloat(rewardLabel.c_str(), &agent->m_scalings[i], 0.F, 1.F))
        {
            ppo::model::sAgentRealTimeCommand command{};
            command.m_id = agent->m_id;
            command.m_rewardScaling = agent->m_scalings[i];
            command.m_rewardScalingType = static_cast<unsigned char>(i);
            command.m_type = ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardScaling;
            p_environment->AddAgentCommand(command);
        }
        ImGui::PopID();
    }
}

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
    agent->m_plots[4].m_name = state.m_name + "-MeanReward";

    if (ImGui::BeginTable("##agentplots", 2, ImGuiTableFlags_SizingStretchSame))
    {
        for (int index = 0; index < 5; ++index)
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

void
EnvironmentRenderer::RenderTacticalScene(entt::registry const& snapshot, int width, int height)
{
    EnsureTacticalTexture(width, height);
    UpdateAttachedCamera(snapshot);

    BeginTextureMode(m_tacticalTexture);
    ClearBackground(Color{ 10, 16, 22, 255 });

    BeginMode3D(m_camera);
    if (m_showGrid)
        DrawGrid(30, 2.F);

    DrawPlane(Vector3{ 0.F, -0.02F, 0.F }, Vector2{ 320.F, 320.F }, Color{ 18, 28, 38, 255 });

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
        auto trailIt = m_entityTrails.find(static_cast<unsigned long long>(state.m_entityId));
        if (trailIt != m_entityTrails.end())
            DrawTrail3D(trailIt->second, m_trailDuration, ColorForSide(state.m_side, 180));

        Color const baseColor = ColorForSide(state.m_side);
        Vector3 const center = ToWorld3D(state.m_x, state.m_y, 0.F, m_worldScale);
        bool const isSelected = (entity == m_selectedEntity);
        bool const isTracked = (entity == m_trackedEntity);

        if (state.m_type == ppo::model::eEntityType::Aircraft)
            DrawAircraftGlyph3D(state, m_worldScale, baseColor);
        else if (state.m_type == ppo::model::eEntityType::Objective)
            DrawObjectiveGlyph3D(state, m_worldScale, baseColor);
        else if (state.m_type == ppo::model::eEntityType::SAM)
            DrawHemisphere(center, 0.35F, Fade(baseColor, 0.65F));

        if (m_showDomains && snapshot.any_of<ppo::model::sSNAState>(entity))
        {
            auto const& snaState = snapshot.get<ppo::model::sSNAState>(entity);
            if (state.m_type == ppo::model::eEntityType::SAM)
            {
                float const radius = snaState.m_noescapezone / m_worldScale;
                DrawHemisphere(center, radius, Fade(baseColor, 0.16F));
            }
            else
            {
                float const radius = snaState.m_noescapezone / m_worldScale;
                DrawCylinderSector(center,
                                   radius,
                                   state.m_rotation - snaState.m_maxShootAngle,
                                   state.m_rotation + snaState.m_maxShootAngle,
                                   m_domainHeight,
                                   Fade(baseColor, 0.22F));
            }

            if (m_showPseudoLidar
            &&  state.m_type == ppo::model::eEntityType::Aircraft
            &&  (entity == m_trackedEntity || entity == m_selectedEntity))
            {
                DrawPseudoLidar3D(snapshot, entity, state, snaState, m_worldScale);
            }
        }

        if (isTracked)
            DrawCylinderWires(center, 0.42F, 0.42F, 0.8F, 16, Fade(SKYBLUE, 0.75F));

        if (isSelected)
        {
            DrawSphereWires(Vector3{ center.x, center.y + 0.22F, center.z }, 0.52F, 10, 10, Fade(WHITE, 0.85F));
            DrawLine3D(Vector3{ center.x, center.y + 0.35F, center.z },
                       Vector3{ center.x, center.y + 1.3F, center.z },
                       Fade(WHITE, 0.55F));
        }

          if (state.m_destroyed && m_showDestroyed)
              DrawDestroyMarker3D(center);
      }

      if (primaryTarget != entt::null
      &&  snapshot.valid(primaryTarget)
      &&  snapshot.any_of<ppo::model::sEntityState>(primaryTarget))
      {
          auto const& targetState = snapshot.get<ppo::model::sEntityState>(primaryTarget);
          if (!targetState.m_destroyed)
              DrawPrimaryTargetAura3D(targetState, m_worldScale);
      }

      EndMode3D();
      EndTextureMode();
  }

void
EnvironmentRenderer::DrawTacticalWindow(entt::registry const& snapshot)
{
    ImGui::Begin("3D Tactical View");

    if (m_selectedEntity != entt::null && snapshot.valid(m_selectedEntity) && snapshot.any_of<ppo::model::sEntityState>(m_selectedEntity))
    {
        auto const& state = snapshot.get<ppo::model::sEntityState>(m_selectedEntity);
        ImGui::Text("%s", EntityDisplayName(state, m_selectedEntity).c_str());
        ImGui::SameLine();
        ImGui::TextColored(ColorForSideUi(state.m_side), "%s", SideLabel(state.m_side));
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

    ImVec2 const size = ImGui::GetContentRegionAvail();
    RenderTacticalScene(snapshot, static_cast<int>(size.x), static_cast<int>(size.y));

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

    ImGui::End();
}

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
    float const visibleEastRange = std::max(m_eastRange, 1.F) / std::max(m_mapZoom, 0.001F);
    float const visibleNorthRange = std::max(m_northRange, 1.F) / std::max(m_mapZoom, 0.001F);
    float const scaleX = (size * 0.5F) / visibleEastRange;
    float const scaleY = (size * 0.5F) / visibleNorthRange;
    float const scale = std::min(scaleX, scaleY);

    auto toMinimap = [&](float north, float east)
    {
        return ImVec2{ centerX + (east - centerEast) * scale, centerY - (north - centerNorth) * scale };
    };

    ImVec2 const worldMin = toMinimap(-m_northRange, -m_eastRange);
    ImVec2 const worldMax = toMinimap(m_northRange, m_eastRange);
    ImVec2 const workMin{ worldMin.x, worldMax.y };
    ImVec2 const workMax{ worldMax.x, worldMin.y };

    draw->AddRectFilled(workMin, workMax, IM_COL32(20, 30, 40, 85), 6.F);
    draw->AddRect(workMin, workMax, IM_COL32(235, 210, 120, 220), 6.F, 0, 2.0F);

    for (int ring = 1; ring <= 4; ++ring)
    {
        float const radius = (size * 0.5F * 0.2F) * static_cast<float>(ring);
        draw->AddCircle(ImVec2{ centerX, centerY }, radius, IM_COL32(80, 102, 120, 90), 96, 1.F);
    }

    draw->AddLine(ImVec2{ mapMin.x, centerY }, ImVec2{ mapMax.x, centerY }, IM_COL32(70, 94, 110, 110), 1.F);
    draw->AddLine(ImVec2{ centerX, mapMin.y }, ImVec2{ centerX, mapMax.y }, IM_COL32(70, 94, 110, 110), 1.F);
    draw->AddText(ImVec2{ workMin.x + 8.F, workMin.y + 6.F }, IM_COL32(235, 210, 120, 220), "Work Area");

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

        if (state.m_type == ppo::model::eEntityType::Aircraft)
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
        else if (state.m_type == ppo::model::eEntityType::SAM)
        {
            draw->AddCircleFilled(center, 5.F, color);
        }

        if (m_showDomains && snapshot.any_of<ppo::model::sSNAState>(entity))
        {
            auto const& snaState = snapshot.get<ppo::model::sSNAState>(entity);
            float const radiusPx = (snaState.m_noescapezone * scale);

            if (state.m_type == ppo::model::eEntityType::SAM)
            {
                draw->AddCircle(center, radiusPx, IM_COL32(colorSide.r, colorSide.g, colorSide.b, 110), 48, 1.2F);
            }
            else
            {
                int const segments = 28;
                std::vector<ImVec2> arc{};
                arc.reserve(static_cast<std::size_t>(segments) + 1);
                for (int i = 0; i <= segments; ++i)
                {
                    float const t = static_cast<float>(i) / static_cast<float>(segments);
                    float const a = state.m_rotation - snaState.m_maxShootAngle + t * (2.F * snaState.m_maxShootAngle);
                    arc.push_back(ImVec2{ center.x + std::sin(a) * radiusPx, center.y - std::cos(a) * radiusPx });
                }
                draw->AddPolyline(arc.data(), static_cast<int>(arc.size()), IM_COL32(colorSide.r, colorSide.g, colorSide.b, 160), 0, 1.4F);
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

void
EnvironmentRenderer::DrawInspectorWindow(entt::registry const& snapshot)
{
    ImGui::Begin("Inspection");

    if (m_selectedEntity == entt::null || !snapshot.valid(m_selectedEntity))
    {
        ImGui::TextUnformatted("Select an entity to inspect it.");
        ImGui::End();
        return;
    }

    auto const& entityState = snapshot.get<ppo::model::sEntityState>(m_selectedEntity);
    ImGui::Text("%s", EntityDisplayName(snapshot, m_selectedEntity, entityState).c_str());
    ImGui::SameLine();
    ImGui::TextColored(ColorForSideUi(entityState.m_side), "%s", SideLabel(entityState.m_side));
    ImGui::SameLine();
    if (ImGui::Button("Track"))
        m_trackedEntity = m_selectedEntity;

    if (ImGui::BeginChild("##entitysummary", ImVec2(0.F, 110.F), true))
    {
        ImGui::Columns(3, "##summarycols", false);
        ImGui::Text("Position");
        ImGui::Text("N %.0f", entityState.m_x);
        ImGui::Text("E %.0f", entityState.m_y);
        ImGui::NextColumn();
        ImGui::Text("Kinematics");
        ImGui::Text("V %.1f", entityState.m_speed);
        ImGui::Text("Hdg %.1f deg", entityState.m_rotation * DEG_PER_RAD);
        ImGui::NextColumn();
        ImGui::Text("Lifecycle");
        ImGui::Text("%s", entityState.m_destroyed ? "Destroyed" : "Alive");
        ImGui::Text("Id %u", static_cast<unsigned int>(m_selectedEntity));
        if (snapshot.any_of<ppo::model::sEntityMetadata>(m_selectedEntity))
        {
            auto const& metadata = snapshot.get<ppo::model::sEntityMetadata>(m_selectedEntity);
            ImGui::Text("Name %s", metadata.m_name.c_str());
        }
        ImGui::Columns(1);
    }
    ImGui::EndChild();

    if (ImGui::BeginTable("##componenttable", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
    {
        ImGui::TableSetupColumn("Component", ImGuiTableColumnFlags_WidthFixed, 180.F);
        ImGui::TableSetupColumn("Reflected Data");
        ImGui::TableHeadersRow();

        auto drawRow = [&](char const* componentName, auto drawFn)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(componentName);
            ImGui::TableNextColumn();
            drawFn();
        };

        if (snapshot.any_of<ppo::model::sEntityMetadata>(m_selectedEntity))
        {
            drawRow("EntityMetadata", [&]()
            {
                DrawComponentIfPresent<ppo::model::sEntityMetadata>(snapshot, m_selectedEntity);
            });
        }

        if (snapshot.any_of<ppo::model::sEntityState>(m_selectedEntity))
        {
            drawRow("EntityState", [&]()
            {
                DrawComponentIfPresent<ppo::model::sEntityState>(snapshot, m_selectedEntity);
            });
        }

        if (snapshot.any_of<ppo::model::sSNAState>(m_selectedEntity))
        {
            drawRow("SNAState", [&]()
            {
                DrawComponentIfPresent<ppo::model::sSNAState>(snapshot, m_selectedEntity);
            });
        }

        if (snapshot.any_of<ppo::model::sFlightModelState>(m_selectedEntity))
        {
            drawRow("FlightModelState", [&]()
            {
                DrawComponentIfPresent<ppo::model::sFlightModelState>(snapshot, m_selectedEntity);
            });
        }

        if (snapshot.any_of<ppo::model::sSNAControlState>(m_selectedEntity))
        {
            drawRow("SNAControlState", [&]()
            {
                DrawComponentIfPresent<ppo::model::sSNAControlState>(snapshot, m_selectedEntity);
            });
        }

        ImGui::EndTable();
    }

    ImGui::End();
}

void
EnvironmentRenderer::DrawAgentsWindow(entt::registry const& snapshot)
{
    ImGui::Begin("Agents");

    auto agentView = snapshot.view<ppo::model::sAgentStateSnaphot>();
    static bool first = true;

    for (auto entity : agentView)
    {
        auto const& [state] = agentView.get(entity);
        auto agent = GetAgentController(state.m_id);
        if (agent == nullptr)
            continue;

        if (agent->m_started && !state.m_common.m_training && !first)
        {
            agent->m_plots[0].push(state.m_common.m_loss);
            agent->m_plots[1].push(state.m_common.m_actorloss);
            agent->m_plots[2].push(state.m_common.m_valueloss);
            agent->m_plots[3].push(state.m_common.m_entropyloss);

            if (state.m_common.m_step < agent->m_lastRewardStep)
            {
                agent->m_rewardMean = 0.F;
                agent->m_rewardSampleCount = 0;
                agent->m_plots[4].clear();
            }

            agent->m_lastRewardStep = state.m_common.m_step;
            ++agent->m_rewardSampleCount;
            float const sampleCount = static_cast<float>(agent->m_rewardSampleCount);
            agent->m_rewardMean += (state.m_common.m_reward - agent->m_rewardMean) / sampleCount;
            agent->m_plots[4].push(agent->m_rewardMean);
        }

        if (ImGui::CollapsingHeader(state.m_name.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Id %llu", state.m_id);
            ImGui::SameLine();
            ImGui::Text("Step %lld / %lld", state.m_common.m_step, state.m_common.m_maxStep);
            ImGui::SameLine();
            ImGui::Text("Reward %.3f", state.m_common.m_reward);

            if (ImGui::BeginTabBar((state.m_name + "##tabs").c_str()))
            {
                if (ImGui::BeginTabItem("Controls"))
                {
                    DrawTrainingControls(agent, state);
                    DrawRewardControls(agent, state);
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Snapshot"))
                {
                    DrawMetaAnyRecursive(state.m_name.c_str(), entt::forward_as_meta(state));
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Plots"))
                {
                    DrawAgentPlot(agent, state);
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
    }

    first = false;
    ImGui::End();
}

void
EnvironmentRenderer::DrawMetaCatalogWindow(void)
{
    if (!m_showMetaCatalog)
        return;

    ImGui::Begin("Meta Catalog");

    std::array<std::vector<std::string>, 5> buckets{};
    for (auto&& entry : entt::resolve())
    {
        auto const& type = entry.second;
        auto descriptor = type.custom();

        std::string prefix = "[Type]";
        std::size_t bucketIndex = 0;
        if (auto const* info = static_cast<ppo::reflection::sMetaDescriptor*>(descriptor); info != nullptr)
        {
            switch (info->category)
            {
            case ppo::reflection::eMetaCategory::Entity:    prefix = "[Entity]"; bucketIndex = 1; break;
            case ppo::reflection::eMetaCategory::Component: prefix = "[Component]"; bucketIndex = 2; break;
            case ppo::reflection::eMetaCategory::Agent:     prefix = "[Agent]"; bucketIndex = 3; break;
            case ppo::reflection::eMetaCategory::Enum:      prefix = "[Enum]"; bucketIndex = 4; break;
            default: break;
            }
        }

        std::string label = prefix + " ";
        label += (type.name() != nullptr) ? type.name() : "<unnamed>";
        buckets[bucketIndex].push_back(label);
    }

    char const* sectionNames[]{ "Other", "Entities", "Components", "Agents", "Enums" };
    for (std::size_t index = 0; index < buckets.size(); ++index)
    {
        if (buckets[index].empty())
            continue;

        std::sort(buckets[index].begin(), buckets[index].end());
        if (ImGui::CollapsingHeader(sectionNames[index], ImGuiTreeNodeFlags_DefaultOpen))
        {
            for (auto const& label : buckets[index])
                ImGui::BulletText("%s", label.c_str());
        }
    }

    ImGui::End();
}

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

    p_environment->ReadSnapshot([&](entt::registry const& snapshot)
    {
        UpdateMissionTelemetry(snapshot);
        UpdateTrails(snapshot);

        ImGui::Begin("Simulation");
        DrawStatusBar(snapshot);
        DrawSimulationControls();
        if (p_pinnedPlot != nullptr)
            DrawOneAgentPlot(*p_pinnedPlot);
        ImGui::End();

        DrawTacticalWindow(snapshot);
        DrawMinimapWindow(snapshot);
        DrawEntitiesWindow(snapshot);
        DrawInspectorWindow(snapshot);
        DrawAgentsWindow(snapshot);
        DrawMetaCatalogWindow();
    });

    rlImGuiEnd();
    EndDrawing();

    return !WindowShouldClose();
}

} /* ppo::renderer */
