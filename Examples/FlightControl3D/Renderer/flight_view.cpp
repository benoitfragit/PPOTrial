#include "flight_view.hpp"
#include "flight_display.hpp"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>

namespace ppo::examples::flight {
namespace {
// Render-only choices; all physics, guidance and learning use the JSON settings.
enum CameraMode { AircraftChase, FlightplanOverview, RunwayEnd };
int cameraMode=AircraftChase;
float chaseDistance=80.F;
std::uint64_t shownCrashEvent{};
double crashAnimationStart{};
constexpr double CrashAnimationSeconds=1.5;
constexpr double RenderMeters=100.0;
Vector3 World(Vec3 p) noexcept
{ return {static_cast<float>(p.east/RenderMeters),static_cast<float>(p.up/RenderMeters),static_cast<float>(-p.north/RenderMeters)}; }
void Segment(Vec3 a,Vec3 b,Color color) { DrawLine3D(World(a),World(b),color); }
void Quad(Vec3 a,Vec3 b,Vec3 c,Vec3 d,Color color)
{
    DrawTriangle3D(World(a),World(b),World(c),color); DrawTriangle3D(World(a),World(c),World(d),color);
    DrawTriangle3D(World(c),World(b),World(a),color); DrawTriangle3D(World(d),World(c),World(a),color);
}
/**
 * @brief Draw a marking in runway coordinates, in physical metres.
 * @param runway Current physical runway pose.
 * @param start Beginning along the runway, in metres.
 * @param end End along the runway, in metres.
 * @param left Left edge relative to the centreline, in metres.
 * @param right Right edge relative to the centreline, in metres.
 * @param color Display color of the marking.
 */
void DrawRunwayMarking(Runway const& runway,double start,double end,double left,double right,Color color)
{
    constexpr double markingHeight=0.1;
    Quad(FromRunway(runway,start,left,markingHeight),FromRunway(runway,end,left,markingHeight),
        FromRunway(runway,end,right,markingHeight),FromRunway(runway,start,right,markingHeight),color);
}
/**
 * @brief Draw the surrounding landable surface in the physical runway plane.
 * @param runway Runway pose, dimensions and surrounding margin.
 */
void DrawLandingArea(Runway const& runway)
{
    double const margin=runway.landingMarginMeters;
    if(margin<=0) return;
    double const halfWidth=runway.widthMeters/2;
    Color const ground{79,112,82,255};
    Quad(FromRunway(runway,-margin,-halfWidth-margin,0),FromRunway(runway,runway.lengthMeters+margin,-halfWidth-margin,0),
        FromRunway(runway,runway.lengthMeters+margin,-halfWidth,0),FromRunway(runway,-margin,-halfWidth,0),ground);
    Quad(FromRunway(runway,-margin,halfWidth,0),FromRunway(runway,runway.lengthMeters+margin,halfWidth,0),
        FromRunway(runway,runway.lengthMeters+margin,halfWidth+margin,0),FromRunway(runway,-margin,halfWidth+margin,0),ground);
    Quad(FromRunway(runway,-margin,-halfWidth,0),FromRunway(runway,0,-halfWidth,0),
        FromRunway(runway,0,halfWidth,0),FromRunway(runway,-margin,halfWidth,0),ground);
    Quad(FromRunway(runway,runway.lengthMeters,-halfWidth,0),FromRunway(runway,runway.lengthMeters+margin,-halfWidth,0),
        FromRunway(runway,runway.lengthMeters+margin,halfWidth,0),FromRunway(runway,runway.lengthMeters,halfWidth,0),ground);
}
/**
 * @brief Draw the edges of the elevated landing surface down to sea level.
 * @param runway Current physical runway pose and landing margin.
 * @param seaHeight Sea level in the local world frame, in metres.
 */
void DrawLandingAreaSides(Runway const& runway,double seaHeight)
{
    double const margin=runway.landingMarginMeters;
    double const halfWidth=runway.widthMeters/2+margin;
    std::array<Vec3,4> const corners{
        FromRunway(runway,-margin,-halfWidth,0),
        FromRunway(runway,runway.lengthMeters+margin,-halfWidth,0),
        FromRunway(runway,runway.lengthMeters+margin,halfWidth,0),
        FromRunway(runway,-margin,halfWidth,0)};
    for(std::size_t i=0;i<corners.size();++i)
    {
        auto const top=corners[i];
        auto const next=corners[(i+1)%corners.size()];
        Vec3 const bottom{top.north,top.east,seaHeight};
        Vec3 const nextBottom{next.north,next.east,seaHeight};
        Quad(top,next,nextBottom,bottom,{64,81,66,255});
        Segment(top,next,{150,169,139,255});
    }
}
/**
 * @brief Draw the physical runway and high-contrast markings at its exact dimensions.
 * @param runway Current physical runway pose and dimensions.
 */
void DrawRunway(Runway const& runway)
{
    double const halfWidth=runway.widthMeters/2;
    auto const a=FromRunway(runway,0,-halfWidth,0),b=FromRunway(runway,runway.lengthMeters,-halfWidth,0);
    auto const c=FromRunway(runway,runway.lengthMeters,halfWidth,0),d=FromRunway(runway,0,halfWidth,0);
    Quad(a,b,c,d,{38,43,50,255});
    constexpr Color edgeColor{255,210,48,255};
    double const edgeWidth=std::min(5.0,runway.widthMeters*0.04);
    DrawRunwayMarking(runway,0,runway.lengthMeters,-halfWidth,-halfWidth+edgeWidth,edgeColor);
    DrawRunwayMarking(runway,0,runway.lengthMeters,halfWidth-edgeWidth,halfWidth,edgeColor);
    double const thresholdDepth=std::min(12.0,runway.lengthMeters*0.02);
    double const dashWidth=std::min(6.0,runway.widthMeters*0.08);
    for(int j=0;j<24;++j)
    {
        double const along=runway.lengthMeters*static_cast<double>(j)/24;
        DrawRunwayMarking(runway,std::max(along,thresholdDepth),along+runway.lengthMeters/48,
            -dashWidth/2,dashWidth/2,WHITE);
    }
    // Threshold markings stay inside the physical surface and distinguish its entry end.
    DrawRunwayMarking(runway,0,thresholdDepth,-halfWidth+edgeWidth,halfWidth-edgeWidth,edgeColor);
    double const stripeEnd=std::min(55.0,runway.lengthMeters*0.08);
    double const stripeStart=thresholdDepth+(stripeEnd-thresholdDepth)*0.1;
    double const stripeHalfWidth=runway.widthMeters*0.0175;
    for(double side : {-1.0,1.0}) for(int stripe=0;stripe<4;++stripe)
    {
        double const centre=side*runway.widthMeters*(0.06+0.09*stripe);
        DrawRunwayMarking(runway,stripeStart,stripeEnd,centre-stripeHalfWidth,centre+stripeHalfWidth,WHITE);
    }
}
/**
 * @brief Label the physical threshold in screen space when it is visible.
 * @param runway Current physical runway pose.
 * @param camera Current 3D camera.
 * @param target Render target dimensions.
 */
void DrawThresholdLabel(Runway const& runway,Camera3D const& camera,RenderTexture2D const& target)
{
    auto const point=World(ThresholdPoint(runway));
    Vector3 const view{camera.target.x-camera.position.x,camera.target.y-camera.position.y,
        camera.target.z-camera.position.z};
    double const depth=(point.x-camera.position.x)*view.x+(point.y-camera.position.y)*view.y
        +(point.z-camera.position.z)*view.z;
    if(!std::isfinite(depth) || depth<=0 || target.texture.width<=0 || target.texture.height<=0) return;
    auto const screen=GetWorldToScreenEx(point,camera,target.texture.width,target.texture.height);
    if(!std::isfinite(screen.x) || !std::isfinite(screen.y)
        || screen.x<0 || screen.y<0 || screen.x>=target.texture.width || screen.y>=target.texture.height) return;
    int const x=static_cast<int>(screen.x),y=static_cast<int>(screen.y);
    DrawCircleLines(x,y,8,{255,220,80,255});
    DrawLine(x-12,y,x-5,y,{255,220,80,255});
    DrawLine(x+5,y,x+12,y,{255,220,80,255});
    constexpr char label[]="THRESHOLD";
    int const textX=std::clamp(x+12,0,std::max(0,target.texture.width-MeasureText(label,16)));
    int const textY=std::clamp(y-20,0,std::max(0,target.texture.height-16));
    DrawRectangle(textX,textY,MeasureText(label,16),16,{14,27,44,230});
    DrawText(label,textX,textY,16,{255,220,80,255});
}
/**
 * @brief Draws an expanding, fading impact fireball at the physical crash position.
 * @param position World position of the aircraft at the crash.
 * @param progress Animation fraction in [0,1].
 */
void DrawCrashExplosion(Vec3 position,double progress)
{
    double const t=std::clamp(progress,0.0,1.0);
    unsigned char const alpha=static_cast<unsigned char>(255.0*(1.0-t));
    auto const center=World(Add(position,{0,0,6.0+20.0*t}));
    DrawSphereEx(center,static_cast<float>((10.0+35.0*t)/RenderMeters),8,12,{238,75,18,alpha});
    DrawSphereEx(center,static_cast<float>((6.0+20.0*t)/RenderMeters),8,12,{255,196,55,alpha});
    for(int i=0;i<8;++i)
    {
        double const angle=2.0*Pi*static_cast<double>(i)/8.0;
        double const radius=12.0+38.0*t;
        auto const particle=Add(position,{radius*std::cos(angle),radius*std::sin(angle),10.0+25.0*t});
        DrawSphereEx(World(particle),static_cast<float>((5.0+10.0*t)/RenderMeters),6,8,{255,125,20,alpha});
    }
}
Vec3 AircraftPoint(AircraftState const& a,double forward,double right,double up)
{
    auto const axes=AircraftAxes(a.roll,a.pitch,a.heading);
    return Add(a.position,Add(Scale(axes.forward,forward),Add(Scale(axes.right,right),Scale(axes.up,up))));
}
char const* PhaseName(FlightSnapshot const& state) noexcept
{
    return state.mission.phase==Phase::Approach ? "Approach" : "Rollout";
}
char const* OutcomeName(Outcome p) noexcept
{ return p==Outcome::Flying ? "In flight" : p==Outcome::Stopped ? "Stopped on runway"
    : p==Outcome::Crash ? "Crash" : "Time limit"; }
char const* GuidanceModeName(GuidanceMode mode) noexcept
{
    switch(mode)
    {
    case GuidanceMode::Flare: return "Flare";
    case GuidanceMode::Rollout: return "Runway rollout";
    case GuidanceMode::Approach: return "Approach";
    }
    return "Unknown";
}
constexpr Color GuidanceColor{220,230,240,255};
/**
 * @brief Draw a compact instrument tag connected to the projected aircraft.
 * @param state Render snapshot supplying aircraft and runway-relative measurements.
 * @param camera Active camera used to project the aircraft.
 * @param target Camera render target defining the available pixel extent.
 */
void DrawAircraftTag(FlightSnapshot const& state,Camera3D const& camera,RenderTexture2D const& target)
{
    constexpr int width=190,height=146,margin=8,fontSize=16,rowHeight=21;
    if(target.texture.width<width+2*margin || target.texture.height<height+2*margin
        || !IsFinite(state.aircraft) || !std::isfinite(state.guidance.height)
        || !std::isfinite(DisplaySlopeDegrees(state.guidance)) || !std::isfinite(state.runway.elevationMeters)) return;
    auto const point=World(state.aircraft.position);
    Vector3 const view{camera.target.x-camera.position.x,camera.target.y-camera.position.y,
        camera.target.z-camera.position.z};
    double const depth=(point.x-camera.position.x)*view.x+(point.y-camera.position.y)*view.y
        +(point.z-camera.position.z)*view.z;
    if(!std::isfinite(depth) || depth<=0) return;
    auto const screen=GetWorldToScreenEx(point,camera,target.texture.width,target.texture.height);
    if(!std::isfinite(screen.x) || !std::isfinite(screen.y)
        || screen.x<0 || screen.y<0 || screen.x>=target.texture.width || screen.y>=target.texture.height) return;

    int const x=static_cast<int>(screen.x),y=static_cast<int>(screen.y);
    int const preferredX=x+28+width+margin<=target.texture.width ? x+28 : x-28-width;
    int const panelX=std::clamp(preferredX,margin,target.texture.width-width-margin);
    int const panelY=std::clamp(y-height/2,margin,target.texture.height-height-margin);
    int const edgeX=panelX>x ? panelX : panelX+width;
    int const edgeY=std::clamp(y,panelY+12,panelY+height-12);
    Color const accent=state.mission.outcome==Outcome::Crash ? Color{255,112,105,255} : Color{101,217,235,255};
    DrawLine(x,y,edgeX,edgeY,accent);
    DrawCircleLines(x,y,5,accent);
    Rectangle const panel{static_cast<float>(panelX),static_cast<float>(panelY),
        static_cast<float>(width),static_cast<float>(height)};
    DrawRectangleRounded(panel,0.08F,6,{9,20,31,225});
    DrawRectangle(panelX,panelY+10,3,18,accent);
    DrawText("AIRCRAFT",panelX+12,panelY+10,15,accent);
    DrawLine(panelX+12,panelY+31,panelX+width-12,panelY+31,{68,93,105,255});

    double heading=std::fmod(state.aircraft.heading/DegreesToRadians,360.0);
    if(heading<0) heading+=360.0;
    int const headingDegrees=static_cast<int>(std::floor(heading+0.5))%360;
    char values[5][40]{};
    // Position is expressed in the runway's local tangent frame, so this
    // altitude is approximate away from the geodetic reference point.
    std::snprintf(values[0],sizeof(values[0]),"~%.0f m",state.runway.elevationMeters+state.aircraft.position.up);
    std::snprintf(values[1],sizeof(values[1]),"%.0f m",state.guidance.height);
    std::snprintf(values[2],sizeof(values[2]),"%.1f m/s",state.aircraft.airspeed);
    std::snprintf(values[3],sizeof(values[3]),"%03d deg",headingDegrees);
    std::snprintf(values[4],sizeof(values[4]),"%+.1f deg",DisplaySlopeDegrees(state.guidance));
    constexpr char const* labels[]{"ALT", "H / RWY", "SPEED", "HEADING", "SLOPE"};
    for(int i=0;i<5;++i)
    {
        int const rowY=panelY+39+i*rowHeight;
        DrawText(labels[i],panelX+12,rowY,fontSize,{155,177,187,255});
        DrawText(values[i],panelX+width-12-MeasureText(values[i],fontSize),rowY,fontSize,{239,246,247,255});
    }
}
}
void DrawFlightScene(FlightSnapshot const& snapshot,RenderTexture2D const& target)
{
    double const now=GetTime();
    if(snapshot.crashEvent!=shownCrashEvent)
    {
        shownCrashEvent=snapshot.crashEvent;
        crashAnimationStart=snapshot.crashEvent ? now : 0;
    }
    double const crashAge=now-crashAnimationStart;
    bool const showCrash=snapshot.crashEvent!=0 && crashAge>=0 && crashAge<CrashAnimationSeconds;
    FlightSnapshot frame=snapshot;
    if(showCrash)
    {
        frame.aircraft=snapshot.crashAircraft;
        frame.runway=snapshot.crashRunway;
        frame.mission.outcome=Outcome::Crash;
        frame.guidance.targetValid=false;
    }
    auto const& s=frame;
    Camera3D camera{};
    camera.up={0,1,0}; camera.fovy=50.F; camera.projection=CAMERA_PERSPECTIVE;
    if (cameraMode==AircraftChase)
    {
        auto behind=AircraftPoint(s.aircraft,-chaseDistance,0,chaseDistance*0.3);
        camera.position=World(behind); camera.target=World(AircraftPoint(s.aircraft,30,0,0));
    }
    else if(cameraMode==RunwayEnd)
    {
        auto const position=RunwayCameraPosition(s.runway);
        auto const delta=Subtract(s.aircraft.position,position);
        double const distance=std::sqrt(Dot(delta,delta));
        camera.position=World(position);
        camera.target=World(distance>1.0 ? s.aircraft.position : ThresholdPoint(s.runway));
        camera.fovy=static_cast<float>(RunwayCameraFieldOfView(position,s.aircraft.position));
        auto const normal=RunwayNormal(s.runway);
        camera.up=World(distance>1.0 && std::abs(Dot(delta,normal))/distance>0.99
            ? RunwayForward(s.runway) : normal);
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
        if(s.runway.landingMarginMeters>0)
        {
            double const margin=s.runway.landingMarginMeters;
            double const halfWidth=s.runway.widthMeters/2+margin;
            for(double along : {-margin,s.runway.lengthMeters+margin})
                for(double cross : {-halfWidth,halfWidth})
                    include(FromRunway(s.runway,along,cross,0));
        }
        auto const centre=Scale(Add(minimum,maximum),0.5);
        double const range=std::max({maximum.north-minimum.north,maximum.east-minimum.east,s.runway.lengthMeters});
        camera.target=World(centre);
        camera.position=World(Add(centre,{range,range,range}));
    }
    BeginTextureMode(target);
    ClearBackground({14,27,44,255});
    BeginMode3D(camera);
    // The sea is a visual backdrop; the actual finite moving surface is in JSBSim.
    DrawPlane(World({s.aircraft.position.north,s.aircraft.position.east,s.seaHeightMeters}),{1000.F,1000.F},{18,53,75,255});
    DrawLandingAreaSides(s.runway,s.seaHeightMeters);
    DrawLandingArea(s.runway);
    DrawRunway(s.runway);
    if(s.guidance.targetValid)
    {
        Segment(s.aircraft.position,s.guidance.target,GuidanceColor);
        DrawSphere(World(s.guidance.target),0.035F,GuidanceColor);
        // The short vector displays the same 3D command used by the policy and rewards.
        Segment(s.aircraft.position,Add(s.aircraft.position,Scale(s.guidance.targetDirection,180.0)),GREEN);
    }
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
    if (cameraMode==FlightplanOverview) DrawSphere(World(s.aircraft.position),0.8F,color);
    if(showCrash) DrawCrashExplosion(s.aircraft.position,crashAge/CrashAnimationSeconds);
    EndMode3D();
    DrawThresholdLabel(s.runway,camera,target);
    if(!showCrash) DrawAircraftTag(s,camera,target);
    if(showCrash) DrawText("CRASH",16,16,26,{255,140,54,255});
    EndTextureMode();
}
void DrawFlightDisplayControls()
{
    ImGui::Combo("Camera",&cameraMode,"Aircraft chase\0Runway / flightplan\0Runway\0");
    if(cameraMode==AircraftChase) ImGui::SliderFloat("Distance (m)",&chaseDistance,25.F,300.F);
}
namespace {
/**
 * @brief Display the most recent crash conditions and their measured limits.
 * @param state Render snapshot, including crashes retained across resets.
 */
void DrawCrashDetails(FlightSnapshot const& state)
{
    ImGui::TextUnformatted("Last crash reasons");
    ImGui::Separator();
    if(!state.hasCrashDiagnostic)
    {
        ImGui::TextDisabled("No crash recorded");
        return;
    }
    constexpr std::array<char const*,CrashReasonCount> labels{
        "Invalid aircraft state", "Distance", "Altitude",
        "Outside landing area", "Wheel outside landing area", "Gear extension", "Structural contact",
        "Sink rate (deck-relative)", "Speed (deck-relative)", "Roll", "Pitch", "Pitch", "Heading", "No wheel contact"};
    for(std::size_t i=0;i<CrashReasonCount;++i)
    {
        auto const& condition=state.crashDiagnostic.conditions[i];
        if(!condition.failed) continue;
        auto const reason=static_cast<CrashReason>(i);
        ImGui::TextColored({1.F,0.45F,0.4F,1.F},"%s",labels[i]);
        switch(reason)
        {
        case CrashReason::Distance: case CrashReason::Altitude:
            ImGui::Text("%.1f / max %.1f m",condition.value,condition.limit);break;
        case CrashReason::Sink: case CrashReason::Speed:
            ImGui::Text("%.1f / max %.1f m/s",condition.value,condition.limit);break;
        case CrashReason::Gear:
            ImGui::Text("%.1f%% / min %.1f%%",100*condition.value,100*condition.limit);break;
        case CrashReason::PitchLow:
            ImGui::Text("%.1f / min %.1f deg",condition.value,condition.limit);break;
        case CrashReason::Roll: case CrashReason::PitchHigh: case CrashReason::Heading:
            ImGui::Text("%.1f / max %.1f deg",condition.value,condition.limit);break;
        default: break;
        }
    }
}
} // namespace
void DrawFlightInstruments(FlightSnapshot const& s)
{
    if (!ImGui::Begin("FlightControl3D")) { ImGui::End(); return; }
    if(!ImGui::BeginTable("##flight-layout",2,ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV))
    { ImGui::End();return; }
    ImGui::TableSetupColumn("Instruments",ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Crash reasons",ImGuiTableColumnFlags_WidthFixed,260.F);
    ImGui::TableNextRow();ImGui::TableNextColumn();
    if (s.failed) ImGui::TextColored({1,0.3F,0.3F,1},"Simulation stopped: see FLIGHT_ERROR in the log.");
    ImGui::Text("%s | %s%s",PhaseName(s),OutcomeName(s.mission.outcome),s.finished ? " | Complete" : "");
    ImGui::TextUnformatted("Aircraft dynamics: T-6 Texan II");
    if(s.training && s.curriculumLevels>0)
        ImGui::Text("Curriculum level %zu / %zu | window %zu / %zu stops",s.curriculumLevel,s.curriculumLevels-1,
            s.curriculumWindowStops,s.curriculumWindowEpisodes);
    if(s.gateEvaluated>0)
        ImGui::Text("Last independent validation: %zu / %zu stops at difficulty %.2f",s.gateSuccesses,s.gateEvaluated,
            s.lastValidation.difficulty);
    if(s.hasBest)
        ImGui::Text("BEST: %.0f%% stops at difficulty %.2f (update %llu)",s.bestScore,s.bestDifficulty,
            static_cast<unsigned long long>(s.bestUpdate));
    ImGui::Text("Guidance: %s",GuidanceModeName(s.guidance.mode));
    ImGui::Text("Aim point: %.0f m past the threshold",s.aimPointAlongMeters);
    double const targetSink=-s.guidance.targetSpeed*Dot(s.guidance.targetDirection,RunwayNormal(s.runway));
    ImGui::Text("Sink rate target / actual: %.2f / %.2f m/s",targetSink,-s.guidance.relativeVelocity.up);
    ImGui::Text("Episode time %.1f / %.0f simulated s",s.elapsedSeconds,s.maximumEpisodeSeconds);
    if(s.lastEpisode.outcome!=Outcome::Flying)
        ImGui::Text("Last episode: %s after %.1f simulated s",OutcomeName(s.lastEpisode.outcome),s.lastEpisodeSeconds);
    if(ImGui::BeginTable("##flight-targets",4,ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
    {
        ImGui::TableSetupColumn("Measure"); ImGui::TableSetupColumn("Target"); ImGui::TableSetupColumn("Actual");
        ImGui::TableSetupColumn("Error");
        ImGui::TableHeadersRow();
        ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted("CAS (m/s)");
        ImGui::TableNextColumn(); ImGui::Text("%.1f",s.guidance.targetSpeed);
        ImGui::TableNextColumn(); ImGui::Text("%.1f",s.aircraft.airspeed);
        ImGui::TableNextColumn(); ImGui::Text("%+.1f",s.aircraft.airspeed-s.guidance.targetSpeed);
        ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted("Slope (deg)");
        ImGui::TableNextColumn(); ImGui::Text("%+.2f",s.guidance.targetSlope/DegreesToRadians);
        ImGui::TableNextColumn(); ImGui::Text("%+.2f",DisplaySlopeDegrees(s.guidance));
        ImGui::TableNextColumn(); ImGui::Text("%+.2f",DisplaySlopeDegrees(s.guidance)-s.guidance.targetSlope/DegreesToRadians);
        double const track=std::atan2(s.guidance.velocityDirection.east,
            s.guidance.velocityDirection.north)/DegreesToRadians;
        double const directionDot=std::clamp(Dot(s.guidance.targetDirection,s.guidance.velocityDirection),-1.0,1.0);
        double const directionError=std::acos(directionDot)/DegreesToRadians;
        ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::TextUnformatted("Track azimuth (deg)");
        ImGui::TableNextColumn(); ImGui::Text("%+.1f",s.guidance.targetHeading/DegreesToRadians);
        ImGui::TableNextColumn(); ImGui::Text("%+.1f",track);
        ImGui::TableNextColumn(); ImGui::Text("%.1f 3D",directionError);
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Slope: + climb / - descent (deck-relative)");
    ImGui::Text("Runway dimensions: %.0f x %.0f m",s.runway.lengthMeters,s.runway.widthMeters);
    ImGui::Text("Runway threshold above sea: %.1f m",s.runway.origin.up-s.seaHeightMeters);
    if(s.runway.landingMarginMeters>0)
        ImGui::Text("Landing area margin: %.0f m",s.runway.landingMarginMeters);
    ImGui::Text("Runway %.2f km | Height %.0f m",s.guidance.distance/1000,s.guidance.height);
    ImGui::Text("Runway lateral: %+.0f m (%s)",s.guidance.cross,
        s.guidance.cross<0 ? "left" : s.guidance.cross>0 ? "right" : "center");
    if(ImGui::CollapsingHeader("Controls and rewards",ImGuiTreeNodeFlags_DefaultOpen))
    {
        if(!s.diagnostics.enabled)
            ImGui::TextDisabled("Network diagnostics disabled in the scenario configuration.");
        if(ImGui::BeginTable("##flight-controls",4,ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
        {
            ImGui::TableSetupColumn("Control");ImGui::TableSetupColumn("Network mean");
            ImGui::TableSetupColumn("Sampled raw");ImGui::TableSetupColumn("Applied");
            ImGui::TableHeadersRow();
            std::array<double,ContinuousActions> const applied{s.controls.elevator,s.controls.aileron,
                s.controls.rudder,s.controls.throttle,s.controls.brake};
            for(std::size_t i=0;i<ContinuousActions;++i)
            {
                ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::TextUnformatted(ActionNames[i]);
                ImGui::TableNextColumn();
                if(s.diagnostics.enabled) ImGui::Text("%+.3f",s.diagnostics.policy[i]); else ImGui::TextUnformatted("-");
                ImGui::TableNextColumn();
                if(s.diagnostics.enabled) ImGui::Text("%+.3f",s.diagnostics.rawAction[i]); else ImGui::TextUnformatted("-");
                ImGui::TableNextColumn();ImGui::Text("%+.3f",applied[i]);
            }
            ImGui::EndTable();
        }
        if(ImGui::BeginTable("##flight-rewards",3,ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
        {
            ImGui::TableSetupColumn("Reward");ImGui::TableSetupColumn("Last step");ImGui::TableSetupColumn("Last window mean");
            ImGui::TableHeadersRow();
            double const samples=static_cast<double>(std::max<std::uint64_t>(s.lastDiagnostics.samples,1));
            for(std::size_t i=0;i<RewardCount;++i)
            {
                ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::TextUnformatted(RewardNames[i]);
                ImGui::TableNextColumn();ImGui::Text("%+.5f",s.lastStep.rewards[i]);
                ImGui::TableNextColumn();
                if(s.diagnostics.enabled && s.lastDiagnostics.samples>0)
                    ImGui::Text("%+.5f",s.lastDiagnostics.rewardSums[i]/samples);
                else ImGui::TextUnformatted("-");
            }
            ImGui::EndTable();
        }
    }
    if(s.diagnostics.enabled && ImGui::CollapsingHeader("Convergence diagnostics"))
    {
        auto const saturatedObservations=std::count_if(s.lastDiagnostics.observationSaturations.begin(),
            s.lastDiagnostics.observationSaturations.end(),[](std::uint64_t count){return count>0;});
        auto const saturatedActions=std::accumulate(s.lastDiagnostics.appliedSaturations.begin(),
            s.lastDiagnostics.appliedSaturations.end(),std::uint64_t{});
        ImGui::Text("Window samples %llu | saturated inputs %zu/%zu | applied saturation hits %llu",
            static_cast<unsigned long long>(s.lastDiagnostics.samples),static_cast<std::size_t>(saturatedObservations),
            ObservationSize,static_cast<unsigned long long>(saturatedActions));
        ImGui::Text("Minibatch approximate KL max %.6f | actor steps %zu",
            s.learning.maximumApproximateKl,s.learning.actorSteps);
        ImGui::Text("Sampled minibatch KL max %.6f | skipped actor batches %zu",
            s.learning.maximumSampledKl,s.learning.actorKlSkips);
        ImGui::Text("Critic steps %zu",s.learning.criticSteps);
        if(s.lastDiagnostics.samples>0
            && ImGui::BeginTable("##action-diagnostics",5,ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH))
        {
            ImGui::TableSetupColumn("Action");ImGui::TableSetupColumn("Raw mean |x|");
            ImGui::TableSetupColumn("Applied mean |x|");ImGui::TableSetupColumn("Raw sat");
            ImGui::TableSetupColumn("Applied sat");ImGui::TableHeadersRow();
            double const samples=static_cast<double>(s.lastDiagnostics.samples);
            for(std::size_t i=0;i<ContinuousActions;++i)
            {
                ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::TextUnformatted(ActionNames[i]);
                ImGui::TableNextColumn();ImGui::Text("%.3f",s.lastDiagnostics.rawAbsoluteSums[i]/samples);
                ImGui::TableNextColumn();ImGui::Text("%.3f",s.lastDiagnostics.appliedAbsoluteSums[i]/samples);
                ImGui::TableNextColumn();ImGui::Text("%llu",
                    static_cast<unsigned long long>(s.lastDiagnostics.rawSaturations[i]));
                ImGui::TableNextColumn();ImGui::Text("%llu",
                    static_cast<unsigned long long>(s.lastDiagnostics.appliedSaturations[i]));
            }
            ImGui::EndTable();
        }
        if(ImGui::BeginTable("##observations",4,ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH
            | ImGuiTableFlags_ScrollY,ImVec2(0,220)))
        {
            ImGui::TableSetupColumn("Observation");ImGui::TableSetupColumn("Now");
            ImGui::TableSetupColumn("Mean |x|");ImGui::TableSetupColumn("Max |x| / sat");ImGui::TableHeadersRow();
            double const samples=static_cast<double>(std::max<std::uint64_t>(s.lastDiagnostics.samples,1));
            for(std::size_t i=0;i<ObservationSize;++i)
            {
                ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::TextUnformatted(ObservationNames[i]);
                ImGui::TableNextColumn();ImGui::Text("%+.3f",s.diagnostics.observation[i]);
                ImGui::TableNextColumn();ImGui::Text("%.3f",s.lastDiagnostics.observationAbsoluteSums[i]/samples);
                ImGui::TableNextColumn();ImGui::Text("%.3f / %llu",s.lastDiagnostics.observationMaximumAbsolute[i],
                    static_cast<unsigned long long>(s.lastDiagnostics.observationSaturations[i]));
            }
            ImGui::EndTable();
        }
    }
    auto const& result=s.mission.touchedDown ? s.mission : s.lastEpisode;
    if(ImGui::CollapsingHeader("Aircraft / landing details"))
    {
        ImGui::Text("Gear extension %.0f%% | WOW %s",
            s.aircraft.gearPosition*100,s.aircraft.weightOnWheels ? "YES" : "NO");
        ImGui::Text("Roll %.1f | Pitch %.1f | Heading %.1f deg",s.aircraft.roll/DegreesToRadians,
            s.aircraft.pitch/DegreesToRadians,s.aircraft.heading/DegreesToRadians);
        ImGui::Text("Relative sink %.2f m/s",-s.guidance.relativeVelocity.up);
        if(result.touchedDown)
        {
            ImGui::Text("Touchdown %.1f m | Ground roll %.1f m",result.touchdownFromThreshold,result.groundRoll);
            ImGui::Text("Sink %.2f m/s | Speed %.1f m/s",result.touchdownSinkMps,result.touchdownSpeedMps);
            ImGui::Text("Touchdown quality %.2f | Landing area factor %.2f",result.touchdownQuality,
                result.landingAreaFactor);
        }
        if(result.outcome==Outcome::Stopped) ImGui::Text("Stop %.1f m from threshold",result.stopFromThreshold);
    }
    ImGui::TableNextColumn();
    DrawCrashDetails(s);
    ImGui::EndTable();
    ImGui::End();
}
} // namespace ppo::examples::flight
