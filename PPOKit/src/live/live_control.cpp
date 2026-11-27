/**
 * @file PPOKit/src/live/live_control.cpp
 * @brief Pause, play, stop and weights from the studio, between the steps of a run.
 */
#include "ppokit/live/live_control.hpp"

#include <algorithm>
#include <thread>

namespace ppokit::live {

LiveControl::LiveControl(LiveServer& server, runtime::RewardLedger const& ledger, std::uint64_t update)
    : server_(server), ledger_(ledger)
{
    status_.update = update;
    status_.windowRewards.assign(ledger.Channels(), 0.0);
}

void LiveControl::Refresh() noexcept
{
    // Same size every time: the copy reuses the status vector, it does not allocate.
    std::vector<double> const& window = ledger_.WindowWeighted();
    std::copy(window.begin(), window.end(), status_.windowRewards.begin());
    status_.windowSteps = ledger_.WindowSteps();
    status_.episodes = episodesBefore_ + ledger_.WindowEpisodes();
}

bool LiveControl::AfterStep()
{
    ++status_.steps;
    if (!server_.Due())
    {
        return !stop_;
    }
    Refresh();
    server_.Poll(status_);
    Apply();
    WaitWhilePaused();
    return !stop_;
}

void LiveControl::WaitWhilePaused()
{
    while (paused_ && !stop_)
    {
        // Paused: no step, but the studio still sees the run and can resume or stop it.
        std::this_thread::sleep_for(LiveServer::PollInterval);
        server_.Poll(status_);
        Apply();
    }
}

void LiveControl::Apply()
{
    RunState const before = status_.state;
    while (std::optional<Command> command = server_.TakeCommand())
    {
        switch (command->kind)
        {
        case Command::Kind::Pause:
            paused_ = true;
            break;
        case Command::Kind::Play:
            paused_ = false;
            break;
        case Command::Kind::Stop:
            stop_ = true;
            break;
        case Command::Kind::Weights:
            weights_.push_back(std::move(command->weights));
            break;
        }
    }
    if (stop_)
    {
        status_.state = RunState::Stopping;
    }
    else if (paused_)
    {
        status_.state = RunState::Paused;
    }
    else
    {
        status_.state = RunState::Running;
    }
    if (status_.state != before)
    {
        server_.SendStatus(status_);
    }
}

void LiveControl::AtUpdate(std::uint64_t update, std::uint64_t episodes)
{
    status_.update = update;
    episodesBefore_ += episodes;
    Refresh();
    // Weights, a pause or a stop sent during the update are seen at this boundary; a
    // pause holds the run here, before its next step.
    server_.Poll(status_);
    Apply();
    WaitWhilePaused();
}

std::optional<runtime::WeightRequest> LiveControl::TakeWeights()
{
    if (weights_.empty())
    {
        return std::nullopt;
    }
    runtime::WeightRequest request = std::move(weights_.front());
    weights_.pop_front();
    return request;
}

void LiveControl::Finish()
{
    Refresh();
    status_.state = RunState::Finished;
    server_.SendStatus(status_);
    server_.Poll(status_);
}

} // namespace ppokit::live
