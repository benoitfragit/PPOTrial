/** @file control.hpp
 * @brief Session-scoped UDP control for a supervised process. Only IPv4 loopback peers are accepted.
 */
#pragma once

#include "ppokit/runtime/weight_request.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ppokit::ipc
{

enum class Command : std::uint16_t
{
    Pause = 5,
    Resume,
    Stop,
    SetRewards
};
enum class State : std::uint8_t
{
    Starting,
    Running,
    Paused,
    Stopping,
    Saving,
    Stopped
};

/** @brief Human-readable last state acknowledged by the worker. */
char const *Name(State state) noexcept;

/** @brief The supervisor socket is bound before launch. Tick has a bounded receive budget. */
class Supervisor final
{
  public:
    Supervisor();
    ~Supervisor();
    Supervisor(Supervisor const &) = delete;
    Supervisor &operator=(Supervisor const &) = delete;

    /** @brief Private CLI endpoint. Keep this session capability out of console and journal output. */
    std::string Endpoint() const;
    void Tick();
    /** @brief Queue a command. Stop supersedes all pending commands and remains latched. */
    bool Request(Command command);
    bool SetRewards(runtime::WeightRequest const &request);
    std::optional<runtime::WeightAnswer> TakeAnswer();
    State LastState() const noexcept;
    bool Connected() const noexcept;
    /** @brief Last terminal payload from the worker, independent of process exit. */
    std::string const &LastError() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** @brief Worker side. Its communication thread handles retries and heartbeat during optimization or pause. */
class Worker final
{
  public:
    explicit Worker(std::string const &endpoint, bool canSetRewards);
    ~Worker();
    Worker(Worker const &) = delete;
    Worker &operator=(Worker const &) = delete;

    /** @brief At a simulation boundary, wait only on a requested pause. False once Stop is latched. */
    bool Continue();
    bool Stopping() const noexcept;
    void Saving() noexcept;
    /** @brief Only the learner thread takes and applies reward changes at a complete update. */
    std::optional<runtime::WeightRequest> TakeWeights();
    void Answer(runtime::WeightAnswer const &answer);
    /** @brief Signal successful completion after the last write. OS exit is observed separately. */
    void Finish();
    /** @brief Best-effort terminal diagnostic. The OS exit code remains authoritative. */
    void Fail(std::string const &reason);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ppokit::ipc
