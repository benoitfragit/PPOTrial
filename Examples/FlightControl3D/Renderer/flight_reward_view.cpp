#include "flight_view.hpp"
#include "flight_display.hpp"
#include <imgui.h>
#include <cstdio>

namespace ppo::examples::flight {
void DrawFlightRewardBar(FlightSnapshot const& state,std::size_t channel)
{
    if(channel>=RewardCount) return;
    double const episode=state.episodeRewardSums[channel];
    double const cycle=state.learning.hasRewardMeans ? state.learning.meanRewards[channel] : 0;
    double const stop=state.activeRewardSettings.weights[static_cast<std::size_t>(Reward::Stop)];
    double const crash=state.activeRewardSettings.weights[static_cast<std::size_t>(Reward::Crash)];
    float const fraction=RewardProgressFraction(episode,stop,crash);
    char overlay[96]{};
    if(state.learning.hasRewardMeans)
        std::snprintf(overlay,sizeof(overlay),"Episode %+.2f  /  PPO mean %+.4f",episode,cycle);
    else
        std::snprintf(overlay,sizeof(overlay),"Episode %+.2f  /  PPO mean --",episode);
    ImGui::PushStyleColor(ImGuiCol_FrameBg,ImVec4(0.10F,0.15F,0.19F,1.F));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,episode<0
        ? ImVec4(0.72F,0.25F,0.24F,1.F) : ImVec4(0.16F,0.58F,0.43F,1.F));
    ImGui::ProgressBar(fraction,ImVec2(-1.F,0.F),overlay);
    ImGui::PopStyleColor(2);
}
} // namespace ppo::examples::flight
