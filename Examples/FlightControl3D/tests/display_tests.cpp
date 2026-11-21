/**
 * @file display_tests.cpp
 * @brief Headless ImGui checks of the instrument and profile panels (desktop builds only).
 *
 * Renders frames into ImGui log buffers and checks the displayed texts; no window is
 * opened.
 *
 * Read next: Renderer/flight_view.cpp, Renderer/flight_profile_view.cpp.
 */
#include "test_support.hpp"
#include "flight_view.hpp"
#include "flight_display.hpp"
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

FrameText Frame(FlightSnapshot const& state)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({10,10}); ImGui::SetNextWindowSize({480,260});
    ImGui::Begin("Display controls");
    ImGui::LogToBuffer(); DrawFlightDisplayControls();
    FrameText result{}; result.controls=ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish(); ImGui::End();
    ImGui::SetNextWindowPos({500,10}); ImGui::SetNextWindowSize({900,400});
    ImGui::LogToBuffer(); DrawFlightInstruments(state);
    result.instruments=ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish(); ImGui::Render();
    return result;
}

std::string ProfileFrame(FlightSnapshot const& state)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({10,10});ImGui::SetNextWindowSize({900,400});
    ImGui::Begin("Profile test");
    ImGui::LogToBuffer();
    DrawFlightApproachProfile(state);
    DrawFlightRewardBar(state,0);
    std::string const text=ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish();ImGui::End();ImGui::Render();
    return text;
}
}

int main()
{
    try
    {
        UiContext context;
        ImGui::NewFrame();ImGui::Begin("Reward ID test");
        ImGui::PushID("agent-1");ImGui::PushID(0);
        auto const firstId=ImGui::GetID("SinkRate");
        ImGui::PopID();ImGui::PushID(1);
        auto const secondId=ImGui::GetID("SinkRate");
        ImGui::PopID();ImGui::PopID();ImGui::PushID("agent-2");ImGui::PushID(0);
        auto const otherAgentId=ImGui::GetID("SinkRate");
        ImGui::PopID();ImGui::PopID();ImGui::End();ImGui::Render();
        Check(firstId!=secondId && firstId!=otherAgentId,
            "identical labels have independent IDs per reward and agent");
        Check(Near(RewardProgressFraction(-25,100,-100),0.25)
            && Near(RewardProgressFraction(25,50,-100),0.5)
            && Near(RewardProgressFraction(-200,100,-100),1)
            && Near(RewardProgressFraction(-25,50,0),0.5)
            && Near(RewardProgressFraction(0,0,0),0),
            "signed episode bars use active Stop/Crash references, saturate and handle zero weights");
        FlightSnapshot state{};
        state.aircraft.airspeed=61.5; state.aircraft.pitch=8*DegreesToRadians;
        state.guidance.targetSpeed=56; state.guidance.targetSlope=-3*DegreesToRadians;
        state.guidance.mode=GuidanceMode::Approach;
        state.guidance.relativeVelocity={100,0,-100*std::tan(2*DegreesToRadians)};
        state.elapsedSeconds=12.5;state.maximumEpisodeSeconds=900;
        state.lastEpisode.outcome=Outcome::Crash;state.lastEpisodeSeconds=7.5;
        auto text=Frame(state);
        Check(text.controls.find("LOC")==std::string::npos && text.controls.find("GLIDE")==std::string::npos,
            "obsolete beam controls are absent");
        Check(text.controls.find("Distance (m)")!=std::string::npos,
            "chase-camera distance remains adjustable for close flight tracking");
        Check(text.instruments.find("Target")!=std::string::npos && text.instruments.find("Actual")!=std::string::npos
            && text.instruments.find("56.0")!=std::string::npos && text.instruments.find("61.5")!=std::string::npos,
            "instruments compare target and actual CAS");
        Check(text.instruments.find("-3.00")!=std::string::npos && text.instruments.find("-2.00")!=std::string::npos,
            "instruments compare target and actual flight-path slope, not pitch");
        Check(text.instruments.find("Approach")!=std::string::npos
            && text.instruments.find("LOC")==std::string::npos
            && text.instruments.find("GLIDE")==std::string::npos
            && text.instruments.find("12.5 / 900")!=std::string::npos
            && text.instruments.find("Crash after 7.5")!=std::string::npos,
            "instruments identify guidance source and simulated episode durations");
        Check(text.instruments.find("Episodes")==std::string::npos && text.instruments.find("BEST")==std::string::npos
            && text.instruments.find("Actor")==std::string::npos,"training data is not duplicated in flight instruments");
        Check(text.instruments.find("No crash recorded")!=std::string::npos,
            "the crash panel has an explicit empty state");
        state.guidance.targetDirection={0,0,-0.05};
        state.guidance.relativeVelocity.up=-4;
        text=Frame(state);
        state.aimPointAlongMeters=286;text=Frame(state);
        Check(text.instruments.find("Aim point: 286 m past the threshold")!=std::string::npos
            && text.instruments.find("Sink rate target / actual: 2.80 / 4.00 m/s")!=std::string::npos,
            "diagnostics show the threshold and runway-relative speeds");
        state.approachProfileValid=true;state.glideDegrees=3;
        state.runway.thresholdHeightMeters=15;state.runway.lengthMeters=3600;
        state.guidance.along=-300;state.guidance.height=40;
        state.episodeRewardSums[0]=-25;
        state.activeRewardSettings.weights[static_cast<std::size_t>(Reward::Crash)]=-100;
        state.activeRewardSettings.weights[static_cast<std::size_t>(Reward::Stop)]=100;
        state.learning.hasRewardMeans=true;state.learning.meanRewards[0]=0.25;
        auto const profile=ProfileFrame(state);
        Check(profile.find("Approach profile")!=std::string::npos
            && profile.find("sink")!=std::string::npos
            && profile.find("Episode -25.00  /  PPO mean +0.2500")!=std::string::npos,
            "profile and reward bar show current geometry and signed episode/PPO values");
        state.hasCrashDiagnostic=true;
        state.crashDiagnostic.conditions[static_cast<std::size_t>(CrashReason::Speed)]={true,65.0,60.0};
        state.crashDiagnostic.conditions[static_cast<std::size_t>(CrashReason::PitchLow)]={true,-7.0,-5.0};
        text=Frame(state);
        Check(text.instruments.find("Last crash reasons")!=std::string::npos
            && text.instruments.find("Speed")!=std::string::npos
            && text.instruments.find("65.0 / max 60.0 m/s")!=std::string::npos
            && text.instruments.find("-7.0 / min -5.0 deg")!=std::string::npos,
            "the crash panel shows each failed measurement and its minimum or maximum");
    }
    catch(std::exception const& error) { std::cerr<<error.what()<<'\n'; return 1; }
    std::cout<<"Flight instruments passed\n";
}
