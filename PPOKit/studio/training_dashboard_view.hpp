/**
 * @file PPOKit/studio/training_dashboard_view.hpp
 * @brief ImGui + ImPlot dashboard of a training run, read from its journal (metrics.jsonl).
 * @details A header says whether the run is still training; health cards say, in plain
 * words, whether it is going well (editor/run_health.hpp). Five tabs: rewards per channel (per step weighted or raw, per episode,
 * cumulative) with the episode return and the weight changes; reward weights, which can
 * be changed while the run trains when the run reads weight requests; PPO statistics,
 * one small plot each; probes of the network blocks, with remarks in plain words; the
 * run's hyper-parameters. Every curve, statistic and parameter has a help
 * tooltip (ppokit/editor/ppo_help.hpp).
 *
 * A run that announces a live channel (ppokit/live/live_channel.hpp) is followed live: the
 * view attaches by itself, receives each event as the run writes it and the rewards of the
 * window being collected, and offers Pause, Play and Stop; weights are sent on the same
 * channel. Otherwise, with "Follow" on, the journal is re-read when it grows.
 *
 * Needs a current ImGui and ImPlot context, nothing else (the tests run it headless).
 */
#pragma once

#include "ppokit/editor/live_weights.hpp"
#include "ppokit/editor/probe_series.hpp"
#include "ppokit/editor/run_health.hpp"
#include "ppokit/editor/run_series.hpp"
#include "ppokit/live/live_channel.hpp"
#include "ppokit/ipc/control.hpp"
#include "ppokit/runtime/run_log.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

class TrainingDashboardView final
{
public:
    enum class Tab : std::uint8_t
    {
        Rewards,
        Weights,
        Ppo,
        Probes,
        Settings,
    };

    explicit TrainingDashboardView(std::filesystem::path journal, bool legacyControl = true);

    /** @brief Draws the dashboard in the current ImGui window. */
    void Draw();

    /**
     * @brief Re-reads the journal if it changed on disk.
     * @return True when a new version was loaded. A malformed journal keeps the last good
     *         one on screen and shows the error.
     */
    bool Refresh();

    /** @brief Loaded journal, or null before the run wrote its start event. */
    runtime::RunLog const* Log() const noexcept { return log_ ? &*log_ : nullptr; }
    std::string const& Error() const noexcept { return error_; }
    void SetFollow(bool follow) noexcept { follow_ = follow; }
    void SetRewardView(editor::RewardView view);
    /** @brief Opens a tab at the next Draw. */
    void SelectTab(Tab tab) noexcept { pendingTab_ = tab; }
    /** @brief The weights being edited for the running training. */
    editor::LiveWeights& Weights() noexcept { return weights_; }
    /** @brief True while attached to the run's live channel. */
    bool Live() const noexcept { return live_ && live_->Connected(); }
    /** @brief Attaches to the run if it announces a live channel, and reads what arrived. */
    void PollLive();
    /** @brief Route live edits to this owned process and read its durable journal from disk. */
    void SetSupervisor(ipc::Supervisor* supervisor) noexcept;


private:
    /** @brief SetSelected for the tab requested by SelectTab. */
    int TabFlags(Tab tab) const noexcept;
    void DrawHeader();
    /** @brief "training" or "stopped", from the age of the journal on disk. */
    void DrawRunState(std::size_t updates, double elapsed);
    /** @brief State, window rewards and Pause / Play / Stop of a run followed live. */
    void DrawLiveState();
    /** @brief Takes a new journal (file or live) on screen. */
    void ShowLog(runtime::RunLog log);
    /** @brief The health cards: is the run going well? */
    void DrawHealth();
    /** @brief Target KL or zero line on a PPO statistic plot. */
    void DrawReference(std::string const& metric) const;
    void DrawRewards();
    void DrawWeights();
    void DrawWeightEditor();
    /** @brief Sends the edited weights: on the live channel when attached, else as a request file. */
    void SendWeights();
    void DrawWeightHistory();
    void DrawMetrics();
    void DrawProbes();
    /** @brief Why there are no probes yet, and when they come. */
    void DrawNoProbes() const;
    void DrawProbeRemarks(std::string const& network);
    void DrawProbePlot(std::string const& network);
    void DrawSettings();
    void RebuildSeries();

    std::filesystem::path path_;
    std::optional<runtime::RunLog> log_;
    std::string error_;
    std::uintmax_t fileSize_ = 0;
    std::filesystem::file_time_type fileTime_{};
    bool follow_ = true;
    int framesSinceRefresh_ = 0;
    editor::RewardView view_ = editor::RewardView::WeightedStepMean;
    std::optional<Tab> pendingTab_;
    std::vector<bool> visible_;
    std::vector<editor::Series> rewardSeries_;
    editor::Series returnSeries_;
    std::vector<editor::Series> metricSeries_;
    std::vector<editor::HealthCard> health_;
    std::vector<double> weightChanges_;
    editor::LiveWeights weights_;
    bool legacyControl_;
    ipc::Supervisor* supervisor_ = nullptr;
    std::optional<live::LiveClient> live_;
    int framesSinceAttach_ = 0;
    /** @brief Events of the journal shown, to take a live journal only when it grew. */
    std::size_t shownEvents_ = 0;
    std::size_t probeNetwork_ = 0;
    editor::ProbeStatistic probeStatistic_ = editor::ProbeStatistic::Deviation;
};

} // namespace ppokit::studio
