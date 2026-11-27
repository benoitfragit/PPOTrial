/**
 * @file PPOKit/include/ppokit/runtime/weight_request.hpp
 * @brief Changing reward weights while a run trains: a request file and its answer.
 * @details The studio and the run share nothing but the run folder. To change weights,
 * the studio writes `weights.request.json` beside the journal:
 *
 * @code{.json}
 * { "schema": "ppokit.weights/1", "id": 3, "weights": { "Path": 2.0, "Crash": -5.0 } }
 * @endcode
 *
 * Only the channels to change are listed. At its next update boundary the run takes the
 * request (the file is removed), applies it or refuses it, and writes
 * `weights.answer.json` with the same id, whether it was applied and why not. An applied
 * change is also a `weights` event of the journal, so the dashboard draws it.
 *
 * A run announces that it reads requests in its journal header
 * (RunHeader::acceptsWeightRequests); the studio offers the editor only then.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ppokit::runtime {

/** @brief One weight to change: channel index (journal order) and new weight. */
using ChannelWeight = std::pair<std::size_t, double>;

/** @brief A request the run has taken. */
struct WeightRequest final
{
    std::uint64_t id = 0;
    std::vector<ChannelWeight> weights;
};

/** @brief The run's answer to a request. */
struct WeightAnswer final
{
    std::uint64_t id = 0;
    bool applied = false;
    /** @brief Why it was refused, or a short confirmation. */
    std::string message;
};

/** @brief `weights.request.json` beside the journal `journal`. */
std::filesystem::path WeightRequestPath(std::filesystem::path const& journal);
/** @brief `weights.answer.json` beside the journal `journal`. */
std::filesystem::path WeightAnswerPath(std::filesystem::path const& journal);

/**
 * @brief Studio side: writes a request atomically (a run never reads half a request).
 * @param weights Channel name and new weight, for the channels to change.
 */
void WriteWeightRequest(std::filesystem::path const& journal, std::uint64_t id,
    std::vector<std::pair<std::string, double>> const& weights);

/** @brief Studio side: the run's latest answer, if any (a malformed answer reads as none). */
std::optional<WeightAnswer> ReadWeightAnswer(std::filesystem::path const& journal);

/** @brief Run side: takes the requests of one run folder and answers them. */
class WeightRequestInbox final
{
public:
    /** @param channelNames Channel names of the journal header, in order. */
    WeightRequestInbox(std::filesystem::path journal, std::vector<std::string> channelNames);

    /**
     * @brief Takes the pending request, if any, and removes its file.
     * @details A request that cannot be read (bad JSON, unknown channel, non-finite
     * weight) is answered as refused here and not returned. Cheap when there is none:
     * one existence check, so it can run at every update boundary.
     */
    std::optional<WeightRequest> Take();

    /** @brief Writes the answer to the request `id`. */
    void Answer(WeightAnswer const& answer) const;

private:
    std::filesystem::path journal_;
    std::vector<std::string> channelNames_;
};

} // namespace ppokit::runtime
