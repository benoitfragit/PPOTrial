/**
 * @file PPOKit/src/editor/live_weights.cpp
 * @brief Draft reward weights of the studio and their round trip to a running training.
 */
#include "ppokit/editor/live_weights.hpp"

#include "ppokit/runtime/weight_request.hpp"

#include <exception>
#include <utility>

namespace ppokit::editor {

std::vector<double> WeightsInForce(runtime::RunLog const& log)
{
    std::vector<double> weights;
    for (runtime::RunChannel const& channel : log.header.channels)
    {
        weights.push_back(channel.weight);
    }
    for (runtime::WeightChange const& change : log.weightChanges)
    {
        if (change.weights.size() == weights.size())
        {
            weights = change.weights;
        }
    }
    return weights;
}

LiveWeights::LiveWeights(std::filesystem::path journal) : journal_(std::move(journal))
{
    // A previous studio session may have left an answer: never mistake it for ours.
    if (auto const answer = runtime::ReadWeightAnswer(journal_))
    {
        nextId_ = answer->id + 1;
    }
}

void LiveWeights::Sync(runtime::RunLog const& log)
{
    std::vector<double> const inForce = WeightsInForce(log);
    if (inForce.size() != current_.size())
    {
        // First journal, or another run: start from its weights.
        names_.clear();
        for (runtime::RunChannel const& channel : log.header.channels)
        {
            names_.push_back(channel.name);
        }
        current_ = inForce;
        draft_ = inForce;
        edited_.assign(inForce.size(), false);
        return;
    }
    current_ = inForce;
    for (std::size_t channel = 0; channel < current_.size(); ++channel)
    {
        // An edit already in force is no longer an edit.
        if (!edited_[channel] || draft_[channel] == current_[channel])
        {
            draft_[channel] = current_[channel];
            edited_[channel] = false;
        }
    }
}

void LiveWeights::Poll()
{
    if (state_ != State::Pending)
    {
        return;
    }
    if (auto const answer = runtime::ReadWeightAnswer(journal_))
    {
        Receive(*answer);
    }
}

void LiveWeights::Receive(runtime::WeightAnswer const& answer)
{
    if (state_ != State::Pending || answer.id != pendingId_)
    {
        return;
    }
    state_ = answer.applied ? State::Applied : State::Refused;
    message_ = answer.message;
}

std::optional<runtime::WeightRequest> LiveWeights::SendLive()
{
    runtime::WeightRequest request;
    request.id = nextId_;
    for (std::size_t channel = 0; channel < draft_.size(); ++channel)
    {
        if (edited_[channel])
        {
            request.weights.emplace_back(channel, draft_[channel]);
        }
    }
    if (request.weights.empty())
    {
        return std::nullopt;
    }
    pendingId_ = nextId_;
    ++nextId_;
    state_ = State::Pending;
    message_ = "Sent live: the run applies it at its next update.";
    return request;
}

bool LiveWeights::AnyEdited() const noexcept
{
    for (bool const edited : edited_)
    {
        if (edited)
        {
            return true;
        }
    }
    return false;
}

void LiveWeights::SetDraft(std::size_t channel, double weight)
{
    draft_.at(channel) = weight;
    edited_.at(channel) = weight != current_.at(channel);
}

void LiveWeights::Discard()
{
    draft_ = current_;
    edited_.assign(current_.size(), false);
}

bool LiveWeights::Send()
{
    std::vector<std::pair<std::string, double>> changes;
    for (std::size_t channel = 0; channel < draft_.size(); ++channel)
    {
        if (edited_[channel])
        {
            changes.emplace_back(names_[channel], draft_[channel]);
        }
    }
    if (changes.empty())
    {
        return true;
    }
    try
    {
        runtime::WriteWeightRequest(journal_, nextId_, changes);
    }
    catch (std::exception const& error)
    {
        state_ = State::Refused;
        message_ = std::string("Not sent: ") + error.what();
        return false;
    }
    pendingId_ = nextId_;
    ++nextId_;
    state_ = State::Pending;
    message_ = "Sent: the run applies it at its next update.";
    return true;
}

} // namespace ppokit::editor
