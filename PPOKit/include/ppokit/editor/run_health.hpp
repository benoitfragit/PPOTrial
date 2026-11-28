/**
 * @file PPOKit/include/ppokit/editor/run_health.hpp
 * @brief "Is my run going well?": a few health cards read from a run journal.
 * @details Each card is one quantity of the latest updates (episode return, explained
 * variance, policy change, clipping, skipped actor steps), its trend over the last ten
 * updates and a verdict in plain words. The thresholds are rules of thumb that say where
 * to look, not proofs: only an evaluation on independent seeds says a policy learned.
 * No GUI: the studio draws the cards, the tests check the verdicts.
 */
#pragma once

#include "ppokit/runtime/run_log.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ppokit::editor {

/** @brief Verdict of a card. */
enum class Health : std::uint8_t
{
    Good,
    Watch,
    Problem,
    /** @brief Not enough updates yet to say. */
    Unknown,
};

/** @brief Direction of a quantity over the last ten updates. */
enum class Trend : std::uint8_t
{
    Up,
    Flat,
    Down,
    Unknown,
};

struct HealthCard final
{
    /** @brief Short title, e.g. "Episode return". */
    std::string title;
    /** @brief Latest value, formatted for display. */
    std::string value;
    Trend trend = Trend::Unknown;
    Health health = Health::Unknown;
    /** @brief What it means, in one sentence. */
    std::string message;
    /** @brief Journal metric behind the card (help entry), empty for the episode return. */
    std::string metric;
};

/** @brief Updates averaged for the latest value, and distance of the trend comparison. */
inline constexpr std::size_t HealthWindow = 5;
inline constexpr std::size_t TrendDistance = 10;

/** @brief The cards of a run, in display order; quantities the run does not journal have no card. */
std::vector<HealthCard> RunHealth(runtime::RunLog const& log);

/** @brief "45 s", "12 min", "3 h 05", "2 d 4 h". */
std::string FormatDuration(double seconds);

/** @brief The worst verdict of `cards` (Good for none). */
Health OverallHealth(std::vector<HealthCard> const& cards) noexcept;

} // namespace ppokit::editor
