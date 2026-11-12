#include "test_support.hpp"
#include "flight_view.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <string>

using namespace ppo::examples::flight;

namespace {
struct UiContext final
{
    UiContext()
    {
        ImGui::CreateContext();
        auto& io=ImGui::GetIO(); io.IniFilename=nullptr; io.DisplaySize={1600,1000}; io.DeltaTime=1.F/60.F;
        unsigned char* pixels=nullptr; int width=0,height=0;
        io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
    }
    ~UiContext() { ImGui::DestroyContext(); }
};

struct FrameText final { std::string controls{},instruments{}; };
struct ControlPositions final { ImVec2 localizer{},glide{}; };

FrameText Frame(FlightSnapshot const& state,ControlPositions& positions)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({10,10}); ImGui::SetNextWindowSize({480,260});
    ImGui::Begin("Display controls");
    auto const origin=ImGui::GetCursorScreenPos();
    float const half=ImGui::GetFrameHeight()/2;
    positions.localizer={origin.x+half,origin.y+half};
    positions.glide={origin.x+ImGui::GetFrameHeight()+ImGui::GetStyle().ItemInnerSpacing.x
        +ImGui::CalcTextSize("LOC").x+ImGui::GetStyle().ItemSpacing.x+half,origin.y+half};
    ImGui::LogToBuffer(); DrawFlightDisplayControls();
    FrameText result{}; result.controls=ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish(); ImGui::End();
    ImGui::SetNextWindowPos({500,10}); ImGui::SetNextWindowSize({480,400});
    ImGui::LogToBuffer(); DrawFlightInstruments(state);
    result.instruments=ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish(); ImGui::Render();
    return result;
}

FrameText Click(FlightSnapshot const& state,ControlPositions& positions,ImVec2 point)
{
    auto& io=ImGui::GetIO();
    io.AddMousePosEvent(point.x,point.y); Frame(state,positions);
    io.AddMouseButtonEvent(0,true); Frame(state,positions);
    io.AddMouseButtonEvent(0,false); return Frame(state,positions);
}
}

int main()
{
    try
    {
        UiContext context;
        FlightSnapshot state{};
        state.aircraft.airspeed=61.5; state.aircraft.pitch=8*DegreesToRadians;
        state.guidance.targetSpeed=56; state.guidance.targetSlope=-3*DegreesToRadians;
        state.guidance.mode=GuidanceMode::Localizer;
        state.guidance.relativeVelocity={100,0,-100*std::tan(2*DegreesToRadians)};
        state.elapsedSeconds=12.5;state.maximumEpisodeSeconds=900;
        state.lastEpisode.outcome=Outcome::Crash;state.lastEpisodeSeconds=7.5;
        ControlPositions positions{};
        auto text=Frame(state,positions);
        Check(text.controls.find("[x] LOC")!=std::string::npos && text.controls.find("[x] GLIDE")!=std::string::npos,
            "both ILS display toggles start enabled");
        Check(text.controls.find("Distance (m)")==std::string::npos,
            "overview hides the unused chase-camera distance control");
        Check(text.instruments.find("Target")!=std::string::npos && text.instruments.find("Actual")!=std::string::npos
            && text.instruments.find("56.0")!=std::string::npos && text.instruments.find("61.5")!=std::string::npos,
            "instruments compare target and actual CAS");
        Check(text.instruments.find("-3.00")!=std::string::npos && text.instruments.find("-2.00")!=std::string::npos,
            "instruments compare target and actual flight-path slope, not pitch");
        Check(text.instruments.find("LOC / GLIDE acquisition")!=std::string::npos
            && text.instruments.find("Signal not visible | LOC acquiring | GLIDE acquiring")!=std::string::npos
            && text.instruments.find("12.5 / 900")!=std::string::npos
            && text.instruments.find("Crash after 7.5")!=std::string::npos,
            "instruments identify guidance source and simulated episode durations");
        Check(text.instruments.find("Episodes")==std::string::npos && text.instruments.find("BEST")==std::string::npos
            && text.instruments.find("Actor")==std::string::npos,"training data is not duplicated in flight instruments");
        text=Click(state,positions,positions.localizer);
        Check(text.controls.find("[ ] LOC")!=std::string::npos && text.controls.find("[x] GLIDE")!=std::string::npos,
            "LOC toggle leaves GLIDE enabled");
        text=Click(state,positions,positions.glide);
        Check(text.controls.find("[ ] LOC")!=std::string::npos && text.controls.find("[ ] GLIDE")!=std::string::npos,
            "GLIDE can be disabled independently");
        text=Click(state,positions,positions.localizer);
        Check(text.controls.find("[x] LOC")!=std::string::npos && text.controls.find("[ ] GLIDE")!=std::string::npos,
            "LOC can be restored while GLIDE stays disabled");
    }
    catch(std::exception const& error) { std::cerr<<error.what()<<'\n'; return 1; }
    std::cout<<"Flight instruments and independent ILS toggles passed\n";
}
