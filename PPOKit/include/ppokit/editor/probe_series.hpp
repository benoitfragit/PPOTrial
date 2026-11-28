/**
 * @file PPOKit/include/ppokit/editor/probe_series.hpp
 * @brief The probes of a run journal for the studio: curves per block, and what they mean.
 * @details Pure data, no ImGui. CheckProbes reads the latest probed update of a network
 * and says in plain words which block looks unhealthy: units that never fire, saturated
 * units, a constant output, no gradient reaching a block, or an exploding gradient.
 */
#pragma once

#include "ppokit/editor/run_series.hpp"
#include "ppokit/runtime/run_log.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ppokit::editor {

/** @brief What a probe curve shows (each has a help entry, see ProbeStatisticName). */
enum class ProbeStatistic : std::uint8_t
{
    Deviation,
    Mean,
    DeadShare,
    SaturatedShare,
    GradientNorm,
};

inline constexpr ProbeStatistic ProbeStatistics[] = {ProbeStatistic::Deviation, ProbeStatistic::Mean,
    ProbeStatistic::DeadShare, ProbeStatistic::SaturatedShare, ProbeStatistic::GradientNorm};

/** @brief Help name of a statistic, e.g. "probe_dead". */
char const* ProbeStatisticName(ProbeStatistic statistic) noexcept;

/** @brief Networks with probes in the journal (actor, critic), in first-seen order. */
std::vector<std::string> ProbeNetworks(runtime::RunLog const& log);

/** @brief Blocks of `network`, in graph order, from its first probed update. */
std::vector<std::string> ProbeBlocks(runtime::RunLog const& log, std::string const& network);

/** @brief One statistic of one block, update by update (NaN where it is not measured). */
Series ProbeSeries(runtime::RunLog const& log, std::string const& network, std::string const& block,
    ProbeStatistic statistic);

/** @brief A remark on one block. */
struct ProbeRemark final
{
    std::string block;
    std::string message;
};

/** @brief Remarks on the latest probed update of `network`, most serious first; empty when healthy. */
std::vector<ProbeRemark> CheckProbes(runtime::RunLog const& log, std::string const& network);

} // namespace ppokit::editor
