/**
 * @file PPOKit/src/editor/probe_series.cpp
 * @brief Probe curves and plain-language diagnostics of the blocks of a network.
 */
#include "ppokit/editor/probe_series.hpp"

#include "ppokit/editor/ppo_help.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ppokit::editor {
namespace {

constexpr double Missing = std::numeric_limits<double>::quiet_NaN();

/** @brief Above this share of zero outputs, most units of a ReLU block are dead. */
constexpr double DeadAlarm = 0.5;
/** @brief Above this share of saturated outputs, a tanh or sigmoid block barely passes gradient. */
constexpr double SaturationAlarm = 0.5;
/** @brief Below this output spread, the block outputs (almost) a constant. */
constexpr double ConstantSpread = 1e-6;
/** @brief Gradient norms per step outside [VanishingGradient, ExplodingGradient] are suspicious. */
constexpr double VanishingGradient = 1e-8;
constexpr double ExplodingGradient = 1e3;

double Value(runtime::BlockProbe const& probe, ProbeStatistic statistic) noexcept
{
    switch (statistic)
    {
    case ProbeStatistic::Deviation:
        return probe.deviation;
    case ProbeStatistic::Mean:
        return probe.mean;
    case ProbeStatistic::DeadShare:
        return probe.deadShare;
    case ProbeStatistic::SaturatedShare:
        return probe.saturatedShare;
    case ProbeStatistic::GradientNorm:
        return probe.hasGradient ? probe.gradientNorm : Missing;
    }
    return Missing;
}

std::string Percent(double share)
{
    std::ostringstream text;
    text << std::fixed << std::setprecision(0) << share * 100.0 << " %";
    return text.str();
}

std::string Shown(double value)
{
    std::ostringstream text;
    text << std::setprecision(3) << value;
    return text.str();
}

/** @brief The latest record of `network`, or null. */
runtime::ProbeRecord const* Latest(runtime::RunLog const& log, std::string const& network)
{
    for (auto record = log.probes.rbegin(); record != log.probes.rend(); ++record)
    {
        if (record->network == network)
        {
            return &*record;
        }
    }
    return nullptr;
}

/**
 * @brief Whether |x| >= 0.99 means a saturated unit: not for linear outputs (heads, values,
 *        linear layers), whose large values are normal.
 */
bool CanSaturate(std::string const& type)
{
    for (char const* linear : {"linear", "head", "value"})
    {
        if (type.find(linear) != std::string::npos)
        {
            return false;
        }
    }
    return true;
}

} // namespace

char const* ProbeStatisticName(ProbeStatistic statistic) noexcept
{
    switch (statistic)
    {
    case ProbeStatistic::Deviation:
        return "probe_std";
    case ProbeStatistic::Mean:
        return "probe_mean";
    case ProbeStatistic::DeadShare:
        return "probe_dead";
    case ProbeStatistic::SaturatedShare:
        return "probe_saturated";
    case ProbeStatistic::GradientNorm:
        return "probe_gradient_norm";
    }
    return "";
}

std::vector<std::string> ProbeNetworks(runtime::RunLog const& log)
{
    std::vector<std::string> networks;
    for (runtime::ProbeRecord const& record : log.probes)
    {
        if (std::find(networks.begin(), networks.end(), record.network) == networks.end())
        {
            networks.push_back(record.network);
        }
    }
    return networks;
}

std::vector<std::string> ProbeBlocks(runtime::RunLog const& log, std::string const& network)
{
    std::vector<std::string> blocks;
    for (runtime::ProbeRecord const& record : log.probes)
    {
        if (record.network != network)
        {
            continue;
        }
        for (runtime::BlockProbe const& probe : record.blocks)
        {
            blocks.push_back(probe.block);
        }
        break;
    }
    return blocks;
}

Series ProbeSeries(runtime::RunLog const& log, std::string const& network, std::string const& block,
    ProbeStatistic statistic)
{
    Series curve;
    curve.name = block;
    curve.label = block;
    for (runtime::ProbeRecord const& record : log.probes)
    {
        if (record.network != network)
        {
            continue;
        }
        auto const probe = std::find_if(record.blocks.begin(), record.blocks.end(),
            [&block](runtime::BlockProbe const& entry) { return entry.block == block; });
        curve.updates.push_back(static_cast<double>(record.update));
        curve.values.push_back(probe != record.blocks.end() ? Value(*probe, statistic) : Missing);
    }
    return curve;
}

std::vector<ProbeRemark> CheckProbes(runtime::RunLog const& log, std::string const& network)
{
    std::vector<ProbeRemark> remarks;
    runtime::ProbeRecord const* const latest = Latest(log, network);
    if (latest == nullptr)
    {
        return remarks;
    }
    for (runtime::BlockProbe const& probe : latest->blocks)
    {
        // Blocks without parameters (concat, select, pools) only pass their inputs on: their
        // zeros and constants come from the observation, not from a unit that stopped learning.
        if (probe.samples == 0 || !probe.hasGradient)
        {
            continue;
        }
        if (probe.hasGradient && probe.gradientNorm < VanishingGradient)
        {
            remarks.push_back({probe.block, "Almost no gradient reaches '" + probe.block
                + "': it no longer learns. Look for dead or saturated blocks before it, or a disconnected output."});
        }
        if (probe.hasGradient && probe.gradientNorm > ExplodingGradient)
        {
            remarks.push_back({probe.block, "The gradient of '" + probe.block + "' is very large ("
                + Shown(probe.gradientNorm) + " per step): lower the learning rate, or check the scale of the "
                "observations and rewards."});
        }
        if (probe.deadShare > DeadAlarm)
        {
            remarks.push_back({probe.block, Percent(probe.deadShare) + " of the outputs of '" + probe.block
                + "' are exactly zero: with ReLU, these units are dead and learn nothing. A lower learning rate, "
                "normalized inputs or a tanh activation can help."});
        }
        if (probe.saturatedShare > SaturationAlarm && CanSaturate(probe.type))
        {
            remarks.push_back({probe.block, Percent(probe.saturatedShare) + " of the outputs of '" + probe.block
                + "' are saturated (|x| >= 0.99): tanh or sigmoid units there pass almost no gradient. Normalize "
                "the inputs or lower the learning rate."});
        }
        if (probe.deviation < ConstantSpread)
        {
            remarks.push_back({probe.block, "'" + probe.block
                + "' outputs the same value for every sample: it carries no information to the next blocks."});
        }
    }
    return remarks;
}

} // namespace ppokit::editor
