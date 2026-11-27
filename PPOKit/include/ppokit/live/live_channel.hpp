/**
 * @file PPOKit/include/ppokit/live/live_channel.hpp
 * @brief The live channel of a run: the studio watches and commands it while it trains.
 * @details A run opens a LiveServer on a free loopback port and announces it in
 * `checkpoints/live.json` (`{"schema": "ppokit.live/1", "port": 50123}`). The studio
 * finds that file, connects with a LiveClient and can attach or detach at any time; the run
 * never waits for it. One JSON object per line:
 *
 * - run to studio: the journal (ppokit/runtime/run_log.hpp) — all of it on connection, then
 *   each event as it is written —, `status` lines a few times per second (state, update,
 *   steps, episodes, per-channel reward sums of the current window) and `weights_answer`
 *   lines;
 * - studio to run: `{"command": "pause"}`, `"play"`, `"stop"` (save, then end) and
 *   `{"command": "weights", "id": 3, "weights": [[0, 2.0], [4, -5.0]]}` (channel index in
 *   journal order, new weight).
 *
 * The run calls Poll at its step boundaries: it costs a clock read, and talks to the
 * sockets at most every PollInterval.
 */
#pragma once

#include "ppokit/live/connection.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/runtime/weight_request.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::live {

/** @brief What a run is doing. */
enum class RunState : std::uint8_t
{
    Running,
    Paused,
    /** @brief Saving before it ends, after a stop. */
    Stopping,
    Finished,
};

/** @brief "running", "paused", "stopping" or "finished". */
char const* RunStateName(RunState state) noexcept;

/** @brief A `status` line: where the run is, and the rewards of the window being collected. */
struct RunStatus final
{
    RunState state = RunState::Running;
    std::uint64_t update = 0;
    std::uint64_t steps = 0;
    std::uint64_t episodes = 0;
    /** @brief Sum of each weighted reward channel over the current window, in journal order. */
    std::vector<double> windowRewards;
    std::uint64_t windowSteps = 0;
};

/** @brief A command of the studio. */
struct Command final
{
    enum class Kind : std::uint8_t
    {
        Pause,
        Play,
        Stop,
        Weights,
    };
    Kind kind = Kind::Pause;
    /** @brief Weights: the request, as a weights.request.json file would carry it. */
    runtime::WeightRequest weights;
};

/** @brief File that announces the live port of a run, in its checkpoints folder. */
std::filesystem::path LiveFilePath(std::filesystem::path const& checkpoints);

/** @brief Port announced in a checkpoints folder, if a run announces one. */
std::optional<std::uint16_t> AnnouncedPort(std::filesystem::path const& checkpoints);

/** @brief The run side: streams the journal and statuses, collects the studio's commands. */
class LiveServer final : public runtime::RunLogSink
{
public:
    /** @brief Fewest time between two talks to the sockets. */
    static constexpr std::chrono::milliseconds PollInterval{20};
    /** @brief Fewest time between two status lines. */
    static constexpr std::chrono::milliseconds StatusInterval{250};

    /**
     * @brief Listens on a free port and announces it in `checkpoints/live.json`.
     * @param journal Journal of the run, sent whole to each studio that connects.
     * @param channels Number of reward channels: a weights command naming another is refused.
     * @throws std::runtime_error when no port can be opened.
     */
    LiveServer(std::filesystem::path checkpoints, std::filesystem::path journal, std::size_t channels);
    /** @brief Removes the announcement. */
    ~LiveServer() override;
    LiveServer(LiveServer const&) = delete;
    LiveServer& operator=(LiveServer const&) = delete;

    std::uint16_t Port() const noexcept { return listener_.Port(); }
    /** @brief True when Poll would talk to the sockets now (PollInterval has passed). */
    bool Due() const noexcept { return std::chrono::steady_clock::now() - lastPoll_ >= PollInterval; }

    /**
     * @brief Accepts studios, reads their commands and sends the status when due.
     * @param status Current status; read only when a status line is due.
     */
    void Poll(RunStatus const& status);
    /** @brief Sends the status now (state changes), whatever the interval. */
    void SendStatus(RunStatus const& status);
    /** @brief Next command received, oldest first. */
    std::optional<Command> TakeCommand();
    /** @brief Tells the studios what became of a weights command. */
    void Answer(runtime::WeightAnswer const& answer);
    /** @brief Number of studios connected. */
    std::size_t Clients() const noexcept { return clients_.size(); }

    /** @brief Journal line just written: forwarded to every studio. */
    void JournalLine(std::string const& line) override;

private:
    void Broadcast(std::string const& line);
    void AcceptClients();
    void ReadCommands();

    std::filesystem::path checkpoints_;
    std::filesystem::path journal_;
    std::size_t channels_ = 0;
    Listener listener_;
    std::vector<Connection> clients_;
    std::deque<Command> commands_;
    std::chrono::steady_clock::time_point lastPoll_{};
    std::chrono::steady_clock::time_point lastStatus_{};
};

/** @brief The studio side: the run's journal and status as they arrive, and its commands. */
class LiveClient final
{
public:
    /** @brief Connects to the run announced in `checkpoints`; nothing when none listens. */
    static std::optional<LiveClient> Attach(std::filesystem::path const& checkpoints);

    /** @brief Reads what arrived. @return True when something changed. */
    bool Poll();
    bool Connected() const noexcept { return connection_.Open(); }

    /** @brief The journal received so far (empty until the start event arrives). */
    runtime::RunLog const& Log() const noexcept { return log_; }
    bool HasLog() const noexcept { return !log_.header.task.empty(); }
    std::optional<RunStatus> const& Status() const noexcept { return status_; }
    /** @brief Answers to weights commands, oldest first, removed as they are taken. */
    std::optional<runtime::WeightAnswer> TakeAnswer();
    /** @brief The last line that could not be read, for diagnostics. */
    std::string const& LastError() const noexcept { return error_; }

    void Pause();
    void Play();
    void Stop();
    void SendWeights(runtime::WeightRequest const& request);

private:
    explicit LiveClient(Connection connection) noexcept : connection_(std::move(connection)) {}
    void Read(std::string const& line);

    Connection connection_;
    runtime::RunLog log_;
    std::optional<RunStatus> status_;
    std::deque<runtime::WeightAnswer> answers_;
    std::string error_;
};

} // namespace ppokit::live
