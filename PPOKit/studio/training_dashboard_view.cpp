/**
 * @file PPOKit/studio/training_dashboard_view.cpp
 * @brief Drawing of the training dashboard (rewards, PPO statistics, settings).
 */
#include "training_dashboard_view.hpp"

#include "help_tooltip.hpp"
#include "studio_theme.hpp"

#include "ppokit/editor/ppo_help.hpp"
#include "ppokit/editor/run_health.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace ppokit::studio {
namespace {

/** @brief Frames between two checks of the journal on disk while following (~0.5 s at 60 Hz). */
constexpr int FollowInterval = 30;
/** @brief Frames between two attempts to attach to a run's live channel. */
constexpr int AttachInterval = 60;

/** @brief Number of events of a journal, to see whether a live one grew. */
std::size_t EventCount(runtime::RunLog const& log) noexcept
{
    return log.updates.size() + log.weightChanges.size() + log.probes.size();
}
constexpr float RewardPlotHeight = 320.F;
constexpr float MetricPlotHeight = 170.F;
constexpr ImVec4 PendingText{1.F, 0.78F, 0.3F, 1.F};
constexpr ImVec4 AppliedText{0.45F, 0.85F, 0.45F, 1.F};
constexpr ImVec4 RefusedText{1.F, 0.45F, 0.4F, 1.F};
/** @brief A run with no new update for this long (and three update intervals) looks stopped. */
constexpr double StoppedAfterSeconds = 120.0;
/** @brief Height of the row of health cards, in text lines. */
constexpr float HealthCardLines = 4.6F;

constexpr editor::RewardView RewardViews[] = {editor::RewardView::WeightedStepMean,
    editor::RewardView::EpisodeMean, editor::RewardView::Cumulative, editor::RewardView::StepMean};

/** @brief Journal metrics compared with the target KL of the run. */
bool IsKlMetric(std::string const& name)
{
    return name == "approx_kl" || name == "max_approx_kl" || name == "max_sampled_kl";
}

char const* TrendArrow(editor::Trend trend) noexcept
{
    switch (trend)
    {
    case editor::Trend::Up:
        return "rising";
    case editor::Trend::Down:
        return "falling";
    case editor::Trend::Flat:
        return "steady";
    case editor::Trend::Unknown:
        break;
    }
    return "";
}

char const* HealthWord(editor::Health health) noexcept
{
    switch (health)
    {
    case editor::Health::Good:
        return "looks healthy";
    case editor::Health::Watch:
        return "worth a look";
    case editor::Health::Problem:
        return "needs attention";
    case editor::Health::Unknown:
        break;
    }
    return "too early to say";
}

void PlotSeries(editor::Series const& series)
{
    if (series.values.empty())
    {
        return;
    }
    ImPlot::PlotLine(series.label.c_str(), series.updates.data(), series.values.data(),
        static_cast<int>(series.values.size()), ImPlotLineFlags_SkipNaN);
}

} // namespace

TrainingDashboardView::TrainingDashboardView(std::filesystem::path journal, bool legacyControl)
    : path_(std::move(journal)), weights_(path_), legacyControl_(legacyControl)
{
    Refresh();
}

void TrainingDashboardView::SetRewardView(editor::RewardView view)
{
    view_ = view;
    RebuildSeries();
}

bool TrainingDashboardView::Refresh()
{
    std::error_code error;
    std::uintmax_t const size = std::filesystem::file_size(path_, error);
    if (error)
    {
        error_ = "Waiting for " + path_.filename().string() + " (no journal yet)";
        return false;
    }
    std::filesystem::file_time_type const time = std::filesystem::last_write_time(path_, error);
    if (log_ && size == fileSize_ && time == fileTime_)
    {
        return false;
    }
    try
    {
        ShowLog(runtime::LoadRunLog(path_));
    }
    catch (schema::SchemaError const& failure)
    {
        // Keep the last good journal on screen.
        error_ = failure.what();
        return false;
    }
    fileSize_ = size;
    fileTime_ = time;
    return true;
}

void TrainingDashboardView::ShowLog(runtime::RunLog log)
{
    log_ = std::move(log);
    error_.clear();
    shownEvents_ = EventCount(*log_);
    if (visible_.size() != log_->header.channels.size())
    {
        visible_.assign(log_->header.channels.size(), true);
    }
    RebuildSeries();
    weights_.Sync(*log_);
}

void TrainingDashboardView::SetSupervisor(ipc::Supervisor* supervisor) noexcept
{
    supervisor_ = supervisor;
    live_.reset();
}

void TrainingDashboardView::PollLive()
{
    if (supervisor_)
    {
        if (auto answer = supervisor_->TakeAnswer()) weights_.Receive(*answer);
        return;
    }
    if (!legacyControl_) return;
    if (!Live())
    {
        live_.reset();
        if (++framesSinceAttach_ < AttachInterval)
        {
            return;
        }
        framesSinceAttach_ = 0;
        live_ = live::LiveClient::Attach(path_.parent_path());
        if (!live_)
        {
            return;
        }
    }
    live_->Poll();
    if (live_->HasLog() && (!log_ || EventCount(live_->Log()) != shownEvents_
                               || live_->Log().header.task != log_->header.task))
    {
        ShowLog(live_->Log());
    }
    while (std::optional<runtime::WeightAnswer> const answer = live_->TakeAnswer())
    {
        weights_.Receive(*answer);
    }
}

void TrainingDashboardView::RebuildSeries()
{
    if (!log_)
    {
        return;
    }
    rewardSeries_ = editor::RewardSeries(*log_, view_);
    returnSeries_ = editor::ReturnSeries(*log_);
    metricSeries_.clear();
    for (std::string const& name : editor::MetricNames(*log_))
    {
        metricSeries_.push_back(editor::MetricSeries(*log_, name));
    }
    health_ = editor::RunHealth(*log_);
    weightChanges_.clear();
    for (runtime::WeightChange const& change : log_->weightChanges)
    {
        weightChanges_.push_back(static_cast<double>(change.fromUpdate));
    }
}

void TrainingDashboardView::Draw()
{
    PollLive();
    if (!Live() && follow_ && ++framesSinceRefresh_ >= FollowInterval)
    {
        framesSinceRefresh_ = 0;
        Refresh();
        if (!supervisor_ && legacyControl_) weights_.Poll();
    }
    DrawHeader();
    if (!log_)
    {
        MutedWrapped("No training journal is available yet. Start a run with Train above if needed. "
                     "After its first update, results appear here automatically.");
        return;
    }
    DrawHealth();
    if (ImGui::BeginTabBar("dashboard"))
    {
        if (ImGui::BeginTabItem("Rewards", nullptr, TabFlags(Tab::Rewards)))
        {
            DrawRewards();
            ImGui::EndTabItem();
        }
        // The tab title says when edited weights are not sent yet.
        int const weightFlags = TabFlags(Tab::Weights)
            | (weights_.AnyEdited() ? ImGuiTabItemFlags_UnsavedDocument : ImGuiTabItemFlags_None);
        if (ImGui::BeginTabItem("Reward weights", nullptr, weightFlags))
        {
            DrawWeights();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("PPO", nullptr, TabFlags(Tab::Ppo)))
        {
            DrawMetrics();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Probes", nullptr, TabFlags(Tab::Probes)))
        {
            DrawProbes();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Settings", nullptr, TabFlags(Tab::Settings)))
        {
            DrawSettings();
            ImGui::EndTabItem();
        }
        pendingTab_.reset();
        ImGui::EndTabBar();
    }
}

int TrainingDashboardView::TabFlags(Tab tab) const noexcept
{
    return pendingTab_ == tab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
}

void TrainingDashboardView::DrawHeader()
{
    if (log_)
    {
        std::size_t const updates = log_->updates.size();
        double const elapsed = updates == 0 ? 0.0 : log_->updates.back().elapsedSeconds;
        ImGui::TextUnformatted(log_->header.task.c_str());
        ImGui::SameLine();
        ImGui::TextColored(MutedText, "update %zu  |  trained %s", updates, editor::FormatDuration(elapsed).c_str());
        ImGui::SameLine();
        if (Live())
        {
            DrawLiveState();
        }
        else
        {
            DrawRunState(updates, elapsed);
        }
    }
    else
    {
        ImGui::TextUnformatted(path_.filename().string().c_str());
    }
    ImGui::SameLine();
    ImGui::Checkbox("Follow", &follow_);
    ImGui::SetItemTooltip("Re-read the journal while the run writes it");
    ImGui::SameLine();
    if (ImGui::SmallButton("Reload"))
    {
        fileSize_ = 0;
        Refresh();
    }
    ImGui::SetItemTooltip("%s", path_.string().c_str());
    if (!error_.empty())
    {
        ImGui::TextColored(ImVec4(1.F, 0.6F, 0.3F, 1.F), "%s", error_.c_str());
    }
}

void TrainingDashboardView::DrawRunState(std::size_t updates, double elapsed)
{
    // The journal is written at each update: its age says whether the run still trains.
    double const age = std::chrono::duration<double>(std::filesystem::file_time_type::clock::now() - fileTime_).count();
    double const interval = updates == 0 ? 0.0 : elapsed / static_cast<double>(updates);
    bool const live = age < std::max(StoppedAfterSeconds, 3.0 * interval);
    if (live)
    {
        ImGui::TextColored(HealthColor(editor::Health::Good), "|  training (last update %s ago)",
            editor::FormatDuration(age).c_str());
    }
    else
    {
        ImGui::TextColored(MutedText, "|  stopped or paused (no update for %s)", editor::FormatDuration(age).c_str());
    }
}

void TrainingDashboardView::DrawLiveState()
{
    std::optional<live::RunStatus> const& status = live_->Status();
    if (!status)
    {
        ImGui::TextColored(MutedText, "|  live, waiting for the run's status");
        return;
    }
    bool const running = status->state == live::RunState::Running;
    ImGui::TextColored(running ? HealthColor(editor::Health::Good) : HealthColor(editor::Health::Watch),
        "|  live: %s, %llu steps, %llu episodes", live::RunStateName(status->state),
        static_cast<unsigned long long>(status->steps), static_cast<unsigned long long>(status->episodes));
    bool const ended = status->state == live::RunState::Stopping || status->state == live::RunState::Finished;
    ImGui::SameLine();
    ImGui::BeginDisabled(ended);
    if (running)
    {
        if (ImGui::SmallButton("Pause"))
        {
            live_->Pause();
        }
        ImGui::SetItemTooltip("Hold the run between two steps; nothing is lost");
    }
    else if (ImGui::SmallButton("Play"))
    {
        live_->Play();
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Stop"))
    {
        live_->Stop();
    }
    ImGui::SetItemTooltip("Save the training, then end the run");
    ImGui::EndDisabled();
    if (status->windowSteps == 0 || !log_ || log_->header.channels.size() != status->windowRewards.size())
    {
        return;
    }
    // The rewards of the window being collected, before the update that will summarize them.
    std::string window = "Collecting: " + std::to_string(status->windowSteps) + " steps, mean reward per step";
    for (std::size_t channel = 0; channel < status->windowRewards.size(); ++channel)
    {
        char value[64];
        std::snprintf(value, sizeof(value), "  %s %.4g", log_->header.channels[channel].label.c_str(),
            status->windowRewards[channel] / static_cast<double>(status->windowSteps));
        window += value;
    }
    ImGui::TextColored(MutedText, "%s", window.c_str());
}

void TrainingDashboardView::DrawHealth()
{
    if (health_.empty())
    {
        return;
    }
    float const height = ImGui::GetTextLineHeightWithSpacing() * HealthCardLines + 2.F * ImGui::GetStyle().WindowPadding.y;
    if (!ImGui::BeginTable("health", static_cast<int>(health_.size()), ImGuiTableFlags_SizingStretchSame))
    {
        return;
    }
    for (editor::HealthCard const& card : health_)
    {
        ImGui::TableNextColumn();
        ImGui::PushID(card.title.c_str());
        ImVec4 const color = HealthColor(card.health);
        ImGui::BeginChild("card", ImVec2(-1.F, height), ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
        // A colored bar on the left edge carries the verdict at a glance.
        ImVec2 const corner = ImGui::GetWindowPos();
        ImGui::GetWindowDrawList()->AddRectFilled(corner, ImVec2(corner.x + 4.F, corner.y + height),
            ImGui::ColorConvertFloat4ToU32(color), 6.F, ImDrawFlags_RoundCornersLeft);
        ImGui::TextColored(MutedText, "%s", card.title.c_str());
        if (!card.metric.empty())
        {
            HelpMarker(card.metric);
        }
        ImGui::TextColored(color, "%s", card.value.c_str());
        ImGui::SameLine();
        ImGui::TextColored(MutedText, "%s", TrendArrow(card.trend));
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextUnformatted(card.message.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
        ImGui::SetItemTooltip("%s: %s", card.title.c_str(), HealthWord(card.health));
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void TrainingDashboardView::DrawRewards()
{
    for (editor::RewardView const view : RewardViews)
    {
        editor::HelpEntry const* const help = editor::FindHelp(editor::RewardViewName(view));
        if (ImGui::RadioButton(help->title, view_ == view))
        {
            SetRewardView(view);
        }
        HelpTooltip(*help);
        ImGui::SameLine();
    }
    ImGui::NewLine();

    // Channel toggles, with their label and weight.
    for (std::size_t channel = 0; channel < log_->header.channels.size(); ++channel)
    {
        runtime::RunChannel const& entry = log_->header.channels[channel];
        bool visible = visible_[channel];
        ImGui::PushID(static_cast<int>(channel));
        if (ImGui::Checkbox(entry.label.c_str(), &visible))
        {
            visible_[channel] = visible;
        }
        ImGui::PopID();
        ImGui::SetItemTooltip("Channel %s\nweight now: %g (at start: %g)", entry.name.c_str(), weights_.Current(channel),
            entry.weight);
        ImGui::SameLine();
    }
    ImGui::NewLine();

    if (ImPlot::BeginPlot("##rewards", ImVec2(-1.F, RewardPlotHeight)))
    {
        ImPlot::SetupAxes("update", editor::FindHelp(editor::RewardViewName(view_))->title,
            ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
        for (std::size_t channel = 0; channel < rewardSeries_.size(); ++channel)
        {
            if (visible_[channel])
            {
                PlotSeries(rewardSeries_[channel]);
            }
        }
        if (view_ == editor::RewardView::EpisodeMean)
        {
            PlotSeries(returnSeries_);
        }
        if (!weightChanges_.empty())
        {
            ImPlot::PlotInfLines("weights changed", weightChanges_.data(), static_cast<int>(weightChanges_.size()));
        }
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("Episode return##return", ImVec2(-1.F, MetricPlotHeight)))
    {
        ImPlot::SetupAxes("update", "return", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        PlotSeries(returnSeries_);
        ImPlot::EndPlot();
    }
}

void TrainingDashboardView::DrawWeights()
{
    ImGui::PushTextWrapPos(0.F);
    if (supervisor_ || (legacyControl_ && log_->header.acceptsWeightRequests))
    {
        ImGui::TextUnformatted("Change the weights while the run trains. The run applies them at its next update, "
                               "saves them with its settings, and the change appears as a vertical line on the "
                               "reward plots.");
    }
    else
    {
        ImGui::TextDisabled("This run does not take weight changes while it trains (it was started without a "
                            "settings file it can update, or its trainer does not support it). The weights in "
                            "force are shown below; change them in the run's settings and restart it.");
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    DrawWeightEditor();
    DrawWeightHistory();
}

void TrainingDashboardView::SendWeights()
{
    if (supervisor_)
    {
        if (auto request = weights_.SendLive())
            if (!supervisor_->SetRewards(*request))
                weights_.Receive({request->id, false, "Another command is pending or the request is too large"});
        return;
    }
    if (!Live())
    {
        if (legacyControl_) weights_.Send();
        return;
    }
    if (std::optional<runtime::WeightRequest> const request = weights_.SendLive())
    {
        live_->SendWeights(*request);
    }
}

void TrainingDashboardView::DrawWeightEditor()
{
    bool const editable = supervisor_ ? supervisor_->Connected() : (legacyControl_ && log_->header.acceptsWeightRequests);
    if (!ImGui::BeginTable("weights", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        return;
    }
    ImGui::TableSetupColumn("Channel");
    ImGui::TableSetupColumn("In force");
    ImGui::TableSetupColumn(editable ? "New weight" : "");
    ImGui::TableHeadersRow();
    for (std::size_t channel = 0; channel < weights_.Channels(); ++channel)
    {
        runtime::RunChannel const& entry = log_->header.channels[channel];
        ImGui::PushID(static_cast<int>(channel));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(entry.label.c_str());
        ImGui::SetItemTooltip("Channel %s. Its weighted contribution is shown before any final reward clamp.", entry.name.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%g", weights_.Current(channel));
        ImGui::TableNextColumn();
        if (editable)
        {
            double draft = weights_.Draft(channel);
            // About 1 % of the weight per pixel, so that small and large weights both drag well.
            float const speed = static_cast<float>(std::max(std::abs(weights_.Current(channel)), 0.1) * 0.01);
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10.F);
            if (ImGui::DragScalar("##draft", ImGuiDataType_Double, &draft, speed, nullptr, nullptr, "%.4g"))
            {
                weights_.SetDraft(channel, draft);
            }
            ImGui::SetItemTooltip("Drag, or double-click to type. Sent only when you press Apply.");
            if (weights_.Edited(channel))
            {
                ImGui::SameLine();
                ImGui::TextColored(PendingText, "edited");
            }
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
    if (!editable)
    {
        return;
    }

    bool const pending = weights_.Status() == editor::LiveWeights::State::Pending;
    ImGui::BeginDisabled(!weights_.AnyEdited() || pending);
    if (ImGui::Button("Apply at next update"))
    {
        SendWeights();
    }
    ImGui::SetItemTooltip("Send the edited weights to the run; it answers at its next update");
    ImGui::SameLine();
    if (ImGui::Button("Discard"))
    {
        weights_.Discard();
    }
    ImGui::SetItemTooltip("Back to the weights in force");
    ImGui::EndDisabled();
    ImGui::SameLine();
    switch (weights_.Status())
    {
    case editor::LiveWeights::State::Idle:
        break;
    case editor::LiveWeights::State::Pending:
        ImGui::TextColored(PendingText, "%s (the run must be training)",
            weights_.Message().c_str());
        break;
    case editor::LiveWeights::State::Applied:
        ImGui::TextColored(AppliedText, "%s", weights_.Message().c_str());
        break;
    case editor::LiveWeights::State::Refused:
        ImGui::TextColored(RefusedText, "%s", weights_.Message().c_str());
        break;
    }
}

void TrainingDashboardView::DrawWeightHistory()
{
    if (log_->weightChanges.empty())
    {
        return;
    }
    ImGui::SeparatorText("Changes during this run");
    for (runtime::WeightChange const& change : log_->weightChanges)
    {
        std::string line = "From update " + std::to_string(change.fromUpdate) + ":";
        for (std::size_t channel = 0; channel < change.weights.size() && channel < change.previous.size(); ++channel)
        {
            if (change.weights[channel] != change.previous[channel])
            {
                std::ostringstream text;
                text << "  " << log_->header.channels[channel].label << " " << change.previous[channel] << " -> "
                     << change.weights[channel];
                line += text.str();
            }
        }
        ImGui::BulletText("%s", line.c_str());
    }
}

void TrainingDashboardView::DrawMetrics()
{
    if (metricSeries_.empty())
    {
        ImGui::TextDisabled("No update yet.");
        return;
    }
    if (!ImGui::BeginTable("metrics", 2))
    {
        return;
    }
    for (editor::Series const& series : metricSeries_)
    {
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(series.label.c_str());
        HelpMarker(series.name);
        std::string const id = "##" + series.name;
        if (ImPlot::BeginPlot(id.c_str(), ImVec2(-1.F, MetricPlotHeight), ImPlotFlags_NoLegend))
        {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            PlotSeries(series);
            DrawReference(series.name);
            ImPlot::EndPlot();
        }
    }
    ImGui::EndTable();
}

void TrainingDashboardView::DrawReference(std::string const& metric) const
{
    // Reference lines say where a curve should stay: under the target KL, above 0 for the critic.
    double reference = 0.0;
    bool draw = false;
    if (IsKlMetric(metric))
    {
        for (auto const& [name, value] : log_->header.settings)
        {
            if (name == "target_kl" && value > 0.0)
            {
                reference = value;
                draw = true;
            }
        }
    }
    else if (metric == "explained_variance")
    {
        draw = true;
    }
    if (draw)
    {
        ImPlot::SetNextLineStyle(HealthColor(editor::Health::Watch));
        ImPlot::PlotInfLines(IsKlMetric(metric) ? "target KL" : "zero", &reference, 1, ImPlotInfLinesFlags_Horizontal);
    }
}

void TrainingDashboardView::DrawProbes()
{
    std::vector<std::string> const networks = editor::ProbeNetworks(*log_);
    if (networks.empty())
    {
        DrawNoProbes();
        return;
    }
    probeNetwork_ = std::min(probeNetwork_, networks.size() - 1);
    for (std::size_t network = 0; network < networks.size(); ++network)
    {
        if (ImGui::RadioButton(networks[network].c_str(), probeNetwork_ == network))
        {
            probeNetwork_ = network;
        }
        ImGui::SameLine();
    }
    ImGui::TextDisabled("|");
    for (editor::ProbeStatistic const statistic : editor::ProbeStatistics)
    {
        editor::HelpEntry const* const help = editor::FindHelp(editor::ProbeStatisticName(statistic));
        ImGui::SameLine();
        if (ImGui::RadioButton(help->title, probeStatistic_ == statistic))
        {
            probeStatistic_ = statistic;
        }
        HelpTooltip(*help);
    }
    DrawProbeRemarks(networks[probeNetwork_]);
    DrawProbePlot(networks[probeNetwork_]);
}

void TrainingDashboardView::DrawNoProbes() const
{
    // Probes measure what every block outputs and how much gradient reaches it.
    double every = 0.0;
    for (auto const& [name, value] : log_->header.settings)
    {
        if (name == "probe_every")
        {
            every = value;
        }
    }
    ImGui::PushTextWrapPos(0.F);
    if (every > 0.0)
    {
        ImGui::TextColored(MutedText, "This run probes its networks every %g updates: the first probes appear at "
                           "update %g. They show what each block outputs (spread, dead or saturated units) and how "
                           "much gradient reaches it.", every, every);
    }
    else
    {
        ImGui::TextColored(MutedText, "No probes in this run. Probes show what each block of the actor and critic "
                           "outputs (spread, dead or saturated units) and how much gradient reaches it. In a PPOKit "
                           "example, switch them on in the training settings (Diagnostics) and restart the run; "
                           "FlightControl3D and Blue MAPPO probe every 10 updates.");
    }
    ImGui::PopTextWrapPos();
}

void TrainingDashboardView::DrawProbeRemarks(std::string const& network)
{
    std::vector<editor::ProbeRemark> const remarks = editor::CheckProbes(*log_, network);
    ImGui::PushTextWrapPos(0.F);
    if (remarks.empty())
    {
        ImGui::TextColored(AppliedText, "Every block of the %s looks healthy at the latest probed update.",
            network.c_str());
    }
    for (editor::ProbeRemark const& remark : remarks)
    {
        ImGui::TextColored(PendingText, "%s", remark.message.c_str());
    }
    ImGui::PopTextWrapPos();
}

void TrainingDashboardView::DrawProbePlot(std::string const& network)
{
    if (!ImPlot::BeginPlot("##probes", ImVec2(-1.F, RewardPlotHeight)))
    {
        return;
    }
    editor::HelpEntry const* const help = editor::FindHelp(editor::ProbeStatisticName(probeStatistic_));
    ImPlot::SetupAxes("update", help->title, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
    if (probeStatistic_ == editor::ProbeStatistic::GradientNorm)
    {
        // Gradient sizes span orders of magnitude: a vanishing block must stay visible.
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    }
    for (std::string const& block : editor::ProbeBlocks(*log_, network))
    {
        PlotSeries(editor::ProbeSeries(*log_, network, block, probeStatistic_));
    }
    ImPlot::EndPlot();
}

void TrainingDashboardView::DrawSettings()
{
    if (!ImGui::BeginTable("settings", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
    {
        return;
    }
    ImGui::TableSetupColumn("Parameter");
    ImGui::TableSetupColumn("Value");
    ImGui::TableHeadersRow();
    for (auto const& [name, value] : log_->header.settings)
    {
        ImGui::TableNextColumn();
        editor::HelpEntry const* const help = editor::FindHelp(name);
        ImGui::TextUnformatted(help != nullptr ? help->title : name.c_str());
        if (help != nullptr)
        {
            HelpTooltip(*help);
        }
        ImGui::TableNextColumn();
        ImGui::Text("%g", value);
    }
    ImGui::EndTable();
}

} // namespace ppokit::studio
