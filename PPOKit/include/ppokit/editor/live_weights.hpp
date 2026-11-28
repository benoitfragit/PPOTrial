/**
 * @file PPOKit/include/ppokit/editor/live_weights.hpp
 * @brief Reward weights edited in the studio and sent to a running training (no GUI).
 * @details The weights in force come from the journal (start weights, then each
 * `weights` event). The user edits a draft; Send writes a request beside the journal
 * (ppokit/runtime/weight_request.hpp) and the state stays Pending until the run answers
 * at its next update. A channel the user did not touch follows the weights in force.
 * When the studio is attached live to the run, SendLive builds the same request for the
 * live channel and Receive takes the run's answer from it.
 */
#pragma once

#include "ppokit/runtime/run_log.hpp"
#include "ppokit/runtime/weight_request.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::editor {

/** @brief Weights in force after the whole journal: the start weights, then each change. */
std::vector<double> WeightsInForce(runtime::RunLog const& log);

class LiveWeights final
{
public:
    enum class State : std::uint8_t
    {
        /** @brief Nothing sent yet. */
        Idle,
        /** @brief Sent; the run answers at its next update. */
        Pending,
        /** @brief The run applied the last request (shown until the next one). */
        Applied,
        /** @brief The run refused it; Message() says why. */
        Refused,
    };

    explicit LiveWeights(std::filesystem::path journal);

    /** @brief Follows the journal: new weights in force, channel names. */
    void Sync(runtime::RunLog const& log);
    /** @brief Reads the run's answer to a pending request. */
    void Poll();

    std::size_t Channels() const noexcept { return current_.size(); }
    double Current(std::size_t channel) const { return current_.at(channel); }
    double Draft(std::size_t channel) const { return draft_.at(channel); }
    bool Edited(std::size_t channel) const { return edited_.at(channel); }
    bool AnyEdited() const noexcept;

    void SetDraft(std::size_t channel, double weight);
    /** @brief Back to the weights in force. */
    void Discard();
    /**
     * @brief Sends the edited channels to the run.
     * @return False (with Message()) when the request cannot be written.
     */
    bool Send();
    /**
     * @brief The edited channels as a request for the live channel (channel indices), and
     *        Pending until Receive gets its answer. Nothing when no channel is edited.
     */
    std::optional<runtime::WeightRequest> SendLive();
    /** @brief The run's answer, received live. */
    void Receive(runtime::WeightAnswer const& answer);

    State Status() const noexcept { return state_; }
    /** @brief The run's answer, or why sending failed. */
    std::string const& Message() const noexcept { return message_; }

private:
    std::filesystem::path journal_;
    std::vector<std::string> names_;
    std::vector<double> current_;
    std::vector<double> draft_;
    std::vector<bool> edited_;
    std::uint64_t nextId_ = 1;
    std::uint64_t pendingId_ = 0;
    State state_ = State::Idle;
    std::string message_;
};

} // namespace ppokit::editor
