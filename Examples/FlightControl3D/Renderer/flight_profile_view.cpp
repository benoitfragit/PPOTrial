#include "flight_view.hpp"
#include "flight_guidance.hpp"
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ppo::examples::flight {
namespace {
/** @brief Maps runway-relative metres to a bounded profile canvas. */
struct ProfileCanvas final
{
    ImVec2 minimum{},maximum{};
    double alongMin{},alongMax{},heightMin{},heightMax{};
    ImVec2 Point(double along,double height) const noexcept
    {
        double const x=(along-alongMin)/(alongMax-alongMin);
        double const y=(height-heightMin)/(heightMax-heightMin);
        return {minimum.x+static_cast<float>(x)*(maximum.x-minimum.x),
            maximum.y-static_cast<float>(y)*(maximum.y-minimum.y)};
    }
};

/** @brief Draw a labelled tick inside the profile's plotting area. */
void Tick(ImDrawList& draw,ProfileCanvas const& canvas,double along) noexcept
{
    auto const point=canvas.Point(along,canvas.heightMin);
    if(point.x<canvas.minimum.x || point.x>canvas.maximum.x) return;
    draw.AddLine({point.x,canvas.minimum.y},{point.x,canvas.maximum.y},IM_COL32(73,101,123,100));
    char label[32]{};
    std::snprintf(label,sizeof(label),"%.0f m",along);
    draw.AddText({point.x+3,canvas.maximum.y-16},IM_COL32(167,188,199,255),label);
}
} // namespace

void DrawFlightApproachProfile(FlightSnapshot const& state)
{
    ImGui::SeparatorText("Approach profile | runway-relative");
    if(!state.approachProfileValid)
    {
        ImGui::TextDisabled("Approach profile unavailable for this flight.");
        return;
    }
    Config reference{};
    reference.runway=state.runway;
    reference.mission.glideDegrees=state.glideDegrees;
    double const aimAlong=AimPointAlongMeters(reference);
    double const along=state.guidance.along;
    constexpr double MinimumViewStartMeters=-2500.0;
    double const startAlong=std::min(MinimumViewStartMeters,along-std::max(100.0,std::abs(along)*0.1));
    double const endAlong=std::max(aimAlong+400.0,along+100.0);
    double const aircraftPathHeight=ApproachPathHeight(reference,along);
    double const startHeight=ApproachPathHeight(reference,startAlong);
    double const targetSink=-state.guidance.targetSpeed
        *Dot(state.guidance.targetDirection,RunwayNormal(state.runway));
    double const sink=-state.guidance.relativeVelocity.up;
    double const error=state.guidance.height-aircraftPathHeight;
    ImGui::Text("Height %.1f m | path error %+.1f m | sink %.2f / target %.2f m/s",
        state.guidance.height,error,sink,targetSink);
    ImGui::TextDisabled("Green: glide plane to the aim point | Blue: aircraft and 4 s velocity | Amber: altitude error");

    ImVec2 const available=ImGui::GetContentRegionAvail();
    if(available.x<80.F || available.y<35.F) return;
    ImVec2 const origin=ImGui::GetCursorScreenPos();
    ImVec2 const size{available.x,std::min(available.y,200.F)};
    ImGui::InvisibleButton("##approach-profile",size);
    ImDrawList& draw=*ImGui::GetWindowDrawList();
    draw.AddRectFilled(origin,{origin.x+size.x,origin.y+size.y},IM_COL32(12,24,35,255),4.F);
    double const maxHeight=std::max({10.0,state.guidance.height,startHeight});
    double const minHeight=std::min(0.0,state.guidance.height);
    double const heightPadding=std::max(5.0,(maxHeight-minHeight)*0.12);
    ProfileCanvas const canvas{{origin.x+10,origin.y+8},{origin.x+size.x-10,origin.y+size.y-8},
        startAlong,endAlong,minHeight-heightPadding,maxHeight+heightPadding};
    draw.PushClipRect(canvas.minimum,canvas.maximum,true);
    Tick(draw,canvas,0);
    Tick(draw,canvas,aimAlong);
    ImVec2 const runwayStart=canvas.Point(0,0),runwayEnd=canvas.Point(std::min(endAlong,state.runway.lengthMeters),0);
    draw.AddLine(runwayStart,runwayEnd,IM_COL32(180,194,194,255),4.F);
    // The glide plane is straight: it crosses the threshold at its crossing height and meets the aim point.
    draw.AddLine(canvas.Point(startAlong,startHeight),canvas.Point(aimAlong,0),IM_COL32(92,208,166,255),2.F);
    auto const threshold=canvas.Point(0,state.runway.thresholdHeightMeters);
    draw.AddCircleFilled(threshold,4.F,IM_COL32(245,192,85,255));
    draw.AddText({threshold.x+5,threshold.y-15},IM_COL32(245,192,85,255),"Threshold");
    auto const aimPoint=canvas.Point(aimAlong,0);
    draw.AddCircleFilled(aimPoint,4.F,IM_COL32(92,208,166,255));
    draw.AddText({aimPoint.x+5,aimPoint.y-15},IM_COL32(92,208,166,255),"Aim point");
    auto const aircraft=canvas.Point(state.guidance.along,state.guidance.height);
    auto const pathPoint=canvas.Point(along,aircraftPathHeight);
    draw.AddLine(pathPoint,aircraft,IM_COL32(255,181,93,220),1.5F);
    auto const projected=canvas.Point(state.guidance.along+state.guidance.relativeVelocity.north*4,
        state.guidance.height+state.guidance.relativeVelocity.up*4);
    draw.AddLine(aircraft,projected,IM_COL32(119,190,255,210),1.5F);
    draw.AddTriangleFilled({aircraft.x+7,aircraft.y},{aircraft.x-6,aircraft.y-4},
        {aircraft.x-6,aircraft.y+4},IM_COL32(133,208,255,255));
    draw.PopClipRect();
}
} // namespace ppo::examples::flight
