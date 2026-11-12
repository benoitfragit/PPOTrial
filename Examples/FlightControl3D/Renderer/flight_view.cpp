#include "flight_view.hpp"
#include "flight_display.hpp"
#include <imgui.h>
#include <rlgl.h>
#include <algorithm>

namespace ppo::examples::flight {
namespace {
// Render-only choices; all physics, guidance and learning use the JSON settings.
int cameraMode=1;
float chaseDistance=180.F;
bool showLocalizer=true;
bool showGlide=true;
constexpr Color LocalizerColor{255,196,72,255};
constexpr Color GlideColor{105,237,164,255};
constexpr double RenderMeters=100.0;
Vector3 World(Vec3 p) noexcept
{ return {static_cast<float>(p.east/RenderMeters),static_cast<float>(p.up/RenderMeters),static_cast<float>(-p.north/RenderMeters)}; }
void Segment(Vec3 a,Vec3 b,Color color) { DrawLine3D(World(a),World(b),color); }
void Quad(Vec3 a,Vec3 b,Vec3 c,Vec3 d,Color color)
{
    DrawTriangle3D(World(a),World(b),World(c),color); DrawTriangle3D(World(a),World(c),World(d),color);
    DrawTriangle3D(World(c),World(b),World(a),color); DrawTriangle3D(World(d),World(c),World(a),color);
}
void DrawBeam(Vec3 origin,Vec3 end,Vec3 left,Vec3 right,Color color)
{
    Color fill=color; fill.a=30;
    DrawTriangle3D(World(origin),World(left),World(right),fill);
    DrawTriangle3D(World(origin),World(right),World(left),fill);
    Segment(origin,left,color); Segment(left,right,color); Segment(right,origin,color);
    Segment(origin,end,color);
}
Vec3 AircraftPoint(AircraftState const& a,double forward,double right,double up)
{
    Runway pose{}; pose.headingDegrees=a.heading/DegreesToRadians; pose.pitch=a.pitch; pose.roll=a.roll; pose.origin=a.position;
    return FromRunway(pose,forward,right,up);
}
char const* PhaseName(Phase p) noexcept
{ return p==Phase::Patrol ? "Hippodrome" : p==Phase::Approach ? "ILS / flare" : "Rollout"; }
char const* OutcomeName(Outcome p) noexcept
{ return p==Outcome::Flying ? "In flight" : p==Outcome::Stopped ? "Stopped on runway" : p==Outcome::Crash ? "Crash" : "Time limit"; }
char const* GuidanceModeName(GuidanceMode mode) noexcept
{
    switch(mode)
    {
    case GuidanceMode::Patrol: return "Hippodrome";
    case GuidanceMode::Localizer: return "LOC acquisition";
    case GuidanceMode::Glide: return "GLIDE tracking";
    case GuidanceMode::Flare: return "Flare";
    case GuidanceMode::Rollout: return "Runway rollout";
    }
    return "Unknown";
}
Color GuidanceColor(GuidanceMode mode) noexcept
{
    if(mode==GuidanceMode::Patrol) return {65,158,220,255};
    if(mode==GuidanceMode::Localizer) return LocalizerColor;
    if(mode==GuidanceMode::Glide) return GlideColor;
    return {220,230,240,255};
}
}
void DrawFlightScene(FlightSnapshot const& s,RenderTexture2D const& target)
{
    auto const ils=MakeIlsDisplayGeometry(s.runway,s.ilsRange,s.glideDegrees,
        s.localizerCaptureDegrees,s.glideCaptureDegrees);
    Camera3D camera{};
    camera.up={0,1,0}; camera.fovy=50.F; camera.projection=CAMERA_PERSPECTIVE;
    if (cameraMode==0)
    {
        auto behind=AircraftPoint(s.aircraft,-chaseDistance,0,chaseDistance*0.3);
        camera.position=World(behind); camera.target=World(AircraftPoint(s.aircraft,30,0,0));
    }
    else
    {
        Vec3 minimum=s.aircraft.position,maximum=s.aircraft.position;
        auto include=[&minimum,&maximum](Vec3 point)
        {
            minimum={std::min(minimum.north,point.north),std::min(minimum.east,point.east),std::min(minimum.up,point.up)};
            maximum={std::max(maximum.north,point.north),std::max(maximum.east,point.east),std::max(maximum.up,point.up)};
        };
        include(s.runway.origin); include(FromRunway(s.runway,s.runway.lengthMeters,0,0));
        for(std::size_t i=0;i<s.plan.count;++i) include(s.plan.points[i]);
        if(showLocalizer) { include(ils.localizerLeft); include(ils.localizerRight); }
        if(showGlide) { include(ils.glideLow); include(ils.glideHigh); }
        auto const centre=Scale(Add(minimum,maximum),0.5);
        double const range=std::max({maximum.north-minimum.north,maximum.east-minimum.east,s.runway.lengthMeters});
        camera.target=World(centre);
        camera.position=World(Add(centre,{range,range,range}));
    }
    BeginTextureMode(target);
    ClearBackground({14,27,44,255});
    BeginMode3D(camera);
    // The sea is a visual backdrop; the actual finite moving surface is in JSBSim.
    DrawPlane(World({s.aircraft.position.north,s.aircraft.position.east,0}),{1000.F,1000.F},{18,53,75,255});
    auto const& r=s.runway;
    auto const a=FromRunway(r,0,-r.widthMeters/2,0),b=FromRunway(r,r.lengthMeters,-r.widthMeters/2,0);
    auto const c=FromRunway(r,r.lengthMeters,r.widthMeters/2,0),d=FromRunway(r,0,r.widthMeters/2,0);
    Quad(a,b,c,d,{64,72,80,255});
    Segment(a,b,WHITE); Segment(b,c,WHITE); Segment(c,d,WHITE); Segment(d,a,WHITE);
    for(int j=0;j<24;++j)
    {
        double const along=r.lengthMeters*static_cast<double>(j)/24;
        Segment(FromRunway(r,along,0,0.1),FromRunway(r,along+r.lengthMeters/48,0,0.1),WHITE);
    }
    for(std::size_t j=0;j<s.plan.count;++j) Segment(s.plan.points[j],s.plan.points[(j+1)%s.plan.count],{65,158,220,255});
    Color const guidanceColor=GuidanceColor(s.guidance.mode);
    Segment(s.aircraft.position,s.guidance.target,guidanceColor);
    DrawSphere(World(s.guidance.target),0.035F,guidanceColor);
    Color const color=s.mission.outcome==Outcome::Crash ? RED : SKYBLUE;
    auto const nose=AircraftPoint(s.aircraft,4.5,0,0),tail=AircraftPoint(s.aircraft,-4.5,0,0);
    DrawCylinderEx(World(nose),World(tail),0.006F,0.004F,8,color);
    Quad(AircraftPoint(s.aircraft,1,0,0),AircraftPoint(s.aircraft,-1,-6,0),
        AircraftPoint(s.aircraft,-1.5,0,0),AircraftPoint(s.aircraft,-1,6,0),color);
    Quad(AircraftPoint(s.aircraft,-3,0,0),AircraftPoint(s.aircraft,-4,-2,0),
        AircraftPoint(s.aircraft,-4.5,0,0),AircraftPoint(s.aircraft,-4,2,0),color);
    Segment(tail,AircraftPoint(s.aircraft,-4,0,2),WHITE);
    if (s.aircraft.gearPosition>0.01)
    {
        for(double right : {-2.0,2.0})
        {
            auto const wheel=AircraftPoint(s.aircraft,-1,right,-1.7*s.aircraft.gearPosition);
            Segment(AircraftPoint(s.aircraft,-1,right,0),wheel,WHITE); DrawSphere(World(wheel),0.003F,DARKGRAY);
        }
    }
    if (cameraMode==1) DrawSphere(World(s.aircraft.position),0.8F,color);
    // Keep the references visible even when the tilted LOC plane crosses the
    // sea backdrop. These translucent overlays never write depth.
    rlDrawRenderBatchActive(); rlDisableDepthMask(); rlDisableDepthTest();
    if(showLocalizer) DrawBeam(ils.localizerOrigin,ils.localizerEnd,ils.localizerLeft,ils.localizerRight,LocalizerColor);
    if(showGlide) DrawBeam(ils.glideOrigin,ils.glideEnd,ils.glideLow,ils.glideHigh,GlideColor);
    rlDrawRenderBatchActive(); rlEnableDepthMask(); rlEnableDepthTest();
    EndMode3D(); EndTextureMode();
}
void DrawFlightDisplayControls()
{
    ImGui::Checkbox("LOC",&showLocalizer);
    ImGui::SameLine(); ImGui::Checkbox("GLIDE",&showGlide);
    ImGui::Combo("Camera",&cameraMode,"Aircraft chase\0Runway / flightplan\0");
    if(cameraMode==0) ImGui::SliderFloat("Distance (m)",&chaseDistance,40.F,600.F);
}
void DrawFlightInstruments(FlightSnapshot const& s)
{
    if (!ImGui::Begin("FlightControl3D")) { ImGui::End(); return; }
    if (s.failed) ImGui::TextColored({1,0.3F,0.3F,1},"Simulation stopped: see FLIGHT_ERROR in the log.");
    ImGui::Text("%s | %s%s",PhaseName(s.mission.phase),OutcomeName(s.mission.outcome),s.finished ? " | Complete" : "");
    ImGui::Text("Guidance: %s",GuidanceModeName(s.guidance.mode));
    ImGui::Text("Episode time %.1f / %.0f simulated s",s.elapsedSeconds,s.maximumEpisodeSeconds);
    if(s.lastEpisode.outcome!=Outcome::Flying)
        ImGui::Text("Last episode: %s after %.1f simulated s",OutcomeName(s.lastEpisode.outcome),s.lastEpisodeSeconds);
    if(ImGui::BeginTable("##flight-targets",3,ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
    {
        ImGui::TableSetupColumn("Measure"); ImGui::TableSetupColumn("Target"); ImGui::TableSetupColumn("Actual");
        ImGui::TableHeadersRow();
        ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted("CAS (m/s)");
        ImGui::TableNextColumn(); ImGui::Text("%.1f",s.guidance.targetSpeed);
        ImGui::TableNextColumn(); ImGui::Text("%.1f",s.aircraft.airspeed);
        ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted("Slope (deg)");
        ImGui::TableNextColumn(); ImGui::Text("%+.2f",s.guidance.targetSlope/DegreesToRadians);
        ImGui::TableNextColumn(); ImGui::Text("%+.2f",DisplaySlopeDegrees(s.guidance));
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Slope: + climb / - descent (deck-relative)");
    ImGui::Text("Runway %.2f km | Height %.0f m",s.guidance.distance/1000,s.guidance.height);
    ImGui::TextColored({LocalizerColor.r/255.F,LocalizerColor.g/255.F,LocalizerColor.b/255.F,1},
        "LOC   %+.2f deg",s.guidance.localizer/DegreesToRadians);
    ImGui::TextColored({GlideColor.r/255.F,GlideColor.g/255.F,GlideColor.b/255.F,1},
        "GLIDE %+.2f deg",s.guidance.glideError/DegreesToRadians);
    ImGui::TextDisabled("%s",s.guidance.ilsValid ? "ILS signal valid" : "Outside ILS sector");
    auto const& result=s.mission.touchedDown ? s.mission : s.lastEpisode;
    if(ImGui::CollapsingHeader("Aircraft / landing details"))
    {
        ImGui::Text("Gear %s | Extension %.0f%% | WOW %s",s.controls.gearDown ? "DOWN" : "UP",
            s.aircraft.gearPosition*100,s.aircraft.weightOnWheels ? "YES" : "NO");
        ImGui::Text("Roll %.1f | Pitch %.1f | Heading %.1f deg",s.aircraft.roll/DegreesToRadians,
            s.aircraft.pitch/DegreesToRadians,s.aircraft.heading/DegreesToRadians);
        ImGui::Text("Relative sink %.2f m/s",-s.guidance.relativeVelocity.up);
        if(result.touchedDown) ImGui::Text("Touchdown %.1f m | Ground roll %.1f m",result.touchdownFromThreshold,result.groundRoll);
        if(result.outcome==Outcome::Stopped) ImGui::Text("Stop %.1f m from threshold",result.stopFromThreshold);
    }
    ImGui::End();
}
} // namespace ppo::examples::flight
