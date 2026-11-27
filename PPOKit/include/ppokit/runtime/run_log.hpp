/**
 * @file PPOKit/include/ppokit/runtime/run_log.hpp
 * @brief Run journal (metrics.jsonl): what a training run reports, update after update.
 * @details One JSON object per line (`ppokit.run/1`), appended and flushed as soon as it
 * is known, so that a crash loses nothing and the studio can follow a run while it
 * trains. Three events:
 *
 * - `start`: task name, reward channels (name, label, weight), hyper-parameters, and
 *   whether the run reads weight requests while it trains;
 * - `update`: update number, elapsed time, steps and episodes, PPO statistics and the
 *   per-channel reward summary of the window (RewardLedger);
 * - `weights`: a change of reward weights and the update from which it applies;
 * - `probes`: what the probes measured on the blocks of a network during an update
 *   (ppokit/graph/probes.hpp), when they are on;
 * - `evaluation`: a campaign of an evaluation suite (eval/) run during training: its
 *   metrics, its gates, whether it passed and whether its policy became the BEST.
 *
 * The reader ignores an incomplete last line (the run is writing it) and reports any
 * other malformed line with its number.
 */
#pragma once

#include "ppokit/core/ppo_settings.hpp"
#include "ppokit/runtime/hosted_trainer.hpp"
#include "ppokit/runtime/probe_record.hpp"
#include "ppokit/runtime/reward_ledger.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ppokit::runtime {

/** @brief Named value of a PPO statistic or a hyper-parameter, in stable order. */
using NamedValues = std::vector<std::pair<std::string, double>>;

/** @brief A reward channel as the journal records it. */
struct RunChannel final
{
    std::string name;
    std::string label;
    double weight = 1.0;
};

/** @brief The `start` event. */
struct RunHeader final
{
    std::string task;
    std::vector<RunChannel> channels;
    NamedValues settings;
    /** @brief The run reads weight requests (ppokit/runtime/weight_request.hpp) at its update boundaries. */
    bool acceptsWeightRequests = false;
};

/** @brief One `update` event. */
struct UpdateRecord final
{
    std::uint64_t update = 0;
    double elapsedSeconds = 0.0;
    NamedValues ppo;
    RewardSummary rewards;
};

/** @brief One `weights` event: the new weights apply from update `fromUpdate`. */
struct WeightChange final
{
    std::uint64_t fromUpdate = 0;
    std::vector<double> previous;
    std::vector<double> weights;
};

/** @brief A gate of an `evaluation` event. */
struct EvaluationGate final
{
    std::string metric;
    /** @brief ">=" or "<=". */
    std::string comparison;
    double threshold = 0.0;
    double value = 0.0;
    bool passed = false;
};

/** @brief One `evaluation` event: a campaign of a suite on the policy of update `update`. */
struct EvaluationRecord final
{
    std::uint64_t update = 0;
    std::string suite;
    std::size_t episodes = 0;
    bool passed = false;
    /** @brief The policy became the suite's BEST. */
    bool promoted = false;
    /** @brief Passed, and the suite's goal holds (the training may end). */
    bool goalReached = false;
    /** @brief Built-in metrics then the aggregates of the episode metrics (stopped.rate...). */
    NamedValues metrics;
    std::vector<EvaluationGate> gates;
    /** @brief The suite's goal conditions; empty without a goal. */
    std::vector<EvaluationGate> goal;
};

/** @brief A whole journal, as read back. */
struct RunLog final
{
    RunHeader header;
    std::vector<UpdateRecord> updates;
    std::vector<WeightChange> weightChanges;
    std::vector<ProbeRecord> probes;
    std::vector<EvaluationRecord> evaluations;
};

/** @brief Stable names of the single-agent PPO statistics (policy_loss, approx_kl...). */
NamedValues PpoMetrics(core::UpdateStatistics const& statistics);
/** @brief Stable names of the hosted MAPPO statistics, the same names plus the team ones. */
NamedValues PpoMetrics(TeamUpdateStatistics const& statistics);
/** @brief The PPO hyper-parameters as journal settings (gamma, lambda, clip...). */
NamedValues PpoSettingsValues(core::PpoSettings const& settings);
NamedValues PpoSettingsValues(TeamPpoSettings const& settings);

/** @brief Receives every journal line as it is written, e.g. to stream it live to the studio. */
class RunLogSink
{
public:
    virtual ~RunLogSink() = default;
    /** @brief One complete event, without its newline. */
    virtual void JournalLine(std::string const& line) = 0;
};

/**
 * @brief Appends events to a journal file.
 * @details Each event opens the file, appends its line and closes it: the line is on disk
 * when the call returns, and the file is never held open between updates.
 */
class RunLogWriter final
{
public:
    /**
     * @brief Creates (or truncates) the journal and writes the `start` event.
     * @throws std::runtime_error when the file cannot be written.
     */
    RunLogWriter(std::filesystem::path path, RunHeader const& header);

    /**
     * @brief Continues the journal of a run resumed from update `resumedUpdates`.
     * @details When the file holds a journal of the same task and channels, its history up
     * to that update is kept (events after it described a future the checkpoint never
     * reached) and a `weights` event is added if the weights differ; otherwise a new
     * journal starts. When events are dropped, the kept history is rewritten atomically
     * before appending. The training time of the kept history is added to the elapsed time
     * of every new update, so the journal holds the total training time of the run.
     * @param ledger Optional newly started ledger. Its cumulative channels continue from
     * the exact saved update, or are unknown if that update's journal entry is unavailable.
     */
    static RunLogWriter Resume(std::filesystem::path path, RunHeader const& header, std::uint64_t resumedUpdates,
        RewardLedger* ledger = nullptr);

    /** @brief Appends an update; `record.elapsedSeconds` counts from the start of this session. */
    void WriteUpdate(UpdateRecord const& record);
    void WriteWeights(WeightChange const& change);
    void WriteProbes(ProbeRecord const& record);
    void WriteEvaluation(EvaluationRecord const& record);

    std::filesystem::path const& Path() const noexcept { return path_; }

    /** @brief Also hands every new line to `sink` (not owned; null to stop). */
    void SetSink(RunLogSink* sink) noexcept { sink_ = sink; }

private:
    /** @brief Appends to a journal already holding a start event. */
    RunLogWriter(std::filesystem::path path, std::vector<std::string> channelNames);
    void WriteLine(std::string const& line);

    std::filesystem::path path_;
    /** @brief Channel names of the header, keys of the reward summaries. */
    std::vector<std::string> channelNames_;
    /** @brief Training time of the history a resumed journal keeps, added to new updates. */
    double elapsedOffset_ = 0.0;
    RunLogSink* sink_ = nullptr;
};

/**
 * @brief Parses a journal.
 * @throws schema::SchemaError naming the line of a malformed event (except an incomplete
 *         last line, which is ignored).
 */
RunLog ParseRunLog(std::string_view text, std::string_view sourceName);

/**
 * @brief Adds one event line to a journal being received (live stream).
 * @details The first event must be `start`; a later `start` begins a new journal (the run
 * was restarted) and replaces `log`.
 * @throws std::exception on a malformed line; `log` keeps what it had.
 */
void AppendRunLogLine(RunLog& log, std::string_view line);

/** @brief Reads and parses a journal file. */
RunLog LoadRunLog(std::filesystem::path const& path);

} // namespace ppokit::runtime
