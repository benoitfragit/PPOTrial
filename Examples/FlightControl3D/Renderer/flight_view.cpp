#include "flight_view.hpp"
#include <imgui.h>
#include <algorithm>

namespace ppo::examples::flight {
namespace {
// Render-only choices; all physics, guidance and learning use the JSON settings.
int cameraMode=0;
float chaseDistance=180.F;
constexpr double RenderMeters=100.0;
Vector3 World(Vec3 p) noexcept
{ return {static_cast<float>(p.east/RenderMeters),static_cast<float>(p.up/RenderMeters),static_cast<float>(-p.north/RenderMeters)}; }
void Segment(Vec3 a,Vec3 b,Color color) { DrawLine3D(World(a),World(b),color); }
void Quad(Vec3 a,Vec3 b,Vec3 c,Vec3 d,Color color)
{
    DrawTriangle3D(World(a),World(b),World(c),color); DrawTriangle3D(World(a),World(c),World(d),color);
    DrawTriangle3D(World(c),World(b),World(a),color); DrawTriangle3D(World(d),World(c),World(a),color);
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
}
void DrawFlightScene(FlightSnapshot const& s,RenderTexture2D const& target)
{
    Camera3D camera{};
    camera.up={0,1,0}; camera.fovy=50.F; camera.projection=CAMERA_PERSPECTIVE;
    if (cameraMode==0)
    {
        auto behind=AircraftPoint(s.aircraft,-chaseDistance,0,chaseDistance*0.3);
        camera.position=World(behind); camera.target=World(AircraftPoint(s.aircraft,30,0,0));
    }
    else
    {
        auto const centre=FromRunway(s.runway,s.runway.lengthMeters/2,0,0);
        double const range=std::max(s.guidance.distance,s.runway.lengthMeters)*0.8;
        camera.target=World(Scale(Add(s.aircraft.position,centre),0.5));
        camera.position=World(Add(Scale(Add(s.aircraft.position,centre),0.5),{range,range,range}));
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
    auto const threshold=FromRunway(r,0,0,r.thresholdHeightMeters);
    auto const glide=FromRunway(r,-s.ilsRange,0,r.thresholdHeightMeters+s.ilsRange*std::tan(s.glideDegrees*DegreesToRadians));
    Segment(threshold,glide,{100,230,151,255});
    Segment(FromRunway(r,-s.ilsRange,0,0),FromRunway(r,0,0,0),{220,179,80,255});
    Segment(s.aircraft.position,s.guidance.target,{255,198,74,255});
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
    EndMode3D(); EndTextureMode();
}
void DrawFlightInstruments(FlightSnapshot const& s)
{
    if (!ImGui::Begin("FlightControl3D")) { ImGui::End(); return; }
    ImGui::Combo("Camera",&cameraMode,"Aircraft chase\0Runway / flightplan\0");
    ImGui::SliderFloat("Chase distance (m)",&chaseDistance,40.F,600.F);
    ImGui::Text("JSBSim 1.3.3 | Cessna 310 | %s",s.training ? "Hybrid PPO training" : "Independent-seed evaluation");
    if (s.failed) ImGui::TextColored({1,0.3F,0.3F,1},"Simulation stopped after an error; see FLIGHT_ERROR in the log.");
    ImGui::Text("%s | %s%s",PhaseName(s.mission.phase),OutcomeName(s.mission.outcome),s.finished ? " | Complete" : "");
    ImGui::Text("Runway distance %.0f m | Height above deck %.1f m",s.guidance.distance,s.guidance.height);
    ImGui::Text("CAS %.1f m/s (target %.1f) | Relative sink %.2f m/s",s.aircraft.airspeed,s.guidance.targetSpeed,-s.guidance.relativeVelocity.up);
    ImGui::Text("Roll %.1f deg | Pitch %.1f deg | Heading %.1f deg",s.aircraft.roll/DegreesToRadians,
        s.aircraft.pitch/DegreesToRadians,s.aircraft.heading/DegreesToRadians);
    ImGui::Text("LOC %+.2f deg | Glide %+.2f deg | %s",s.guidance.localizer/DegreesToRadians,
        s.guidance.glideError/DegreesToRadians,s.guidance.ilsValid ? "Signal valid" : "Outside ILS sector");
    ImGui::Text("Gear command %s | Physical extension %.0f%% | WOW %s",s.controls.gearDown ? "DOWN" : "UP",
        s.aircraft.gearPosition*100,s.aircraft.weightOnWheels ? "YES" : "NO");
    ImGui::Text("Deck heave %.2f m | Pitch %.2f deg | Roll %.2f deg",s.runway.origin.up,
        s.runway.pitch/DegreesToRadians,s.runway.roll/DegreesToRadians);
    ImGui::Separator();
    auto const& result=s.mission.touchedDown ? s.mission : s.lastEpisode;
    if (result.touchedDown) ImGui::Text("Touchdown: %.1f m from threshold | Ground roll: %.1f m",result.touchdownFromThreshold,result.groundRoll);
    if (result.outcome==Outcome::Stopped) ImGui::Text("Stop: %.1f m from threshold",result.stopFromThreshold);
    ImGui::Text("Episodes %llu | Touchdowns %llu | Stops %llu | Crashes %llu | Timeouts %llu",
        static_cast<unsigned long long>(s.episodes),static_cast<unsigned long long>(s.touchdowns),static_cast<unsigned long long>(s.stops),
        static_cast<unsigned long long>(s.crashes),static_cast<unsigned long long>(s.timeouts));
    ImGui::Text("Updates %llu | Actor loss %.4f | Value loss %.4f",static_cast<unsigned long long>(s.learning.updates),s.learning.policyLoss,s.learning.valueLoss);
    if(s.validating) ImGui::Text("Validating frozen candidate: %llu completed episodes",static_cast<unsigned long long>(s.validation.episodes));
    if(s.hasBest) ImGui::Text("Validated BEST: %.1f%% successful stops",s.bestScore);
    else ImGui::TextUnformatted("No validated BEST yet");
    if(s.validationCampaigns) ImGui::Text("Latest validation: %.1f%% successful stops",100*s.lastValidation.SuccessRate());
    ImGui::End();
}
} // namespace ppo::examples::flight
