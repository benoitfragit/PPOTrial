/**
 * @file PPOKit/include/ppokit/live/live_control.hpp
 * @brief The run side of live control: pause, play and stop between steps, weights at
 *        update boundaries, and the status of the window being collected.
 * @details Plugged into a RolloutCollector (runtime::StepWatcher). After each step it reads
 * the clock; every LiveServer::PollInterval it talks to the studios. While paused it
 * blocks, still answering the studios, until play or stop.
 */
#pragma once

#include "ppokit/live/live_channel.hpp"
#include "ppokit/runtime/reward_ledger.hpp"
#include "ppokit/runtime/step_watcher.hpp"

#include <cstdint>
#include <deque>
#include <optional>

namespace ppokit::live {

class LiveControl final : public runtime::StepWatcher
{
public:
    /**
     * @param server Live channel of the run (outlives this object).
     * @param ledger Reward ledger of the collection, read for the window status.
     * @param update Update the run starts from (a resumed run continues its count).
     */
    LiveControl(LiveServer& server, runtime::RewardLedger const& ledger, std::uint64_t update);

    /** @brief Counts the step; talks to the studios when due; blocks while paused. */
    bool AfterStep() override;
    /**
     * @brief An update has been made: its number and the episodes its window closed.
     *        Blocks here while paused.
     */
    void AtUpdate(std::uint64_t update, std::uint64_t episodes);
    /** @brief The studio asked to stop: save, then end. */
    bool StopRequested() const noexcept { return stop_; }
    /** @brief Next weights command, to apply at this update boundary. */
    std::optional<runtime::WeightRequest> TakeWeights();
    /** @brief Says the run has ended. */
    void Finish();

private:
    /** @brief Applies the commands received: pause, play, stop; keeps weights for later. */
    void Apply();
    void Refresh() noexcept;
    /** @brief Blocks while paused, still answering the studios. */
    void WaitWhilePaused();

    LiveServer& server_;
    runtime::RewardLedger const& ledger_;
    RunStatus status_;
    std::uint64_t episodesBefore_ = 0;
    bool paused_ = false;
    bool stop_ = false;
    std::deque<runtime::WeightRequest> weights_;
};

} // namespace ppokit::live
