/**
 * @file PPOKit/src/editor/run_health.cpp
 * @brief Health cards of a run journal: latest values, trends and plain-word verdicts.
 */
#include "ppokit/editor/run_health.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>

namespace ppokit::editor {
namespace {

/** @brief Relative change below which a trend is flat. */
constexpr double FlatShare = 0.05;
/** @brief Explained variance under which the critic is still far from the returns. */
constexpr double LearningCritic = 0.3;
/** @brief Clip fraction above which updates are mostly clipped. */
constexpr double HighClipFraction = 0.3;
/** @brief Approximate KL considered large when the run has no target. */
constexpr double DefaultKlLimit = 0.05;

std::optional<double> Value(runtime::NamedValues const& values, std::string const& name)
{
    for (auto const& [key, value] : values)
    {
        if (key == name)
        {
            return value;
        }
    }
    return std::nullopt;
}

/** @brief Values of one quantity per update, in update order (only the updates that have it). */
template<typename Read>
std::vector<double> Collect(runtime::RunLog const& log, Read read)
{
    std::vector<double> values;
    for (runtime::UpdateRecord const& update : log.updates)
    {
        if (std::optional<double> const value = read(update))
        {
            values.push_back(*value);
        }
    }
    return values;
}

/** @brief Mean of `count` values ending `fromEnd` values before the last one. */
std::optional<double> WindowMean(std::vector<double> const& values, std::size_t fromEnd, std::size_t count)
{
    if (values.size() < fromEnd + 1)
    {
        return std::nullopt;
    }
    std::size_t const end = values.size() - fromEnd;
    std::size_t const begin = end > count ? end - count : 0;
    double sum = 0.0;
    for (std::size_t index = begin; index < end; ++index)
    {
        sum += values[index];
    }
    return sum / static_cast<double>(end - begin);
}

Trend TrendOf(std::vector<double> const& values)
{
    std::optional<double> const recent = WindowMean(values, 0, HealthWindow);
    std::optional<double> const earlier = WindowMean(values, TrendDistance, HealthWindow);
    if (!recent || !earlier)
    {
        return Trend::Unknown;
    }
    double const scale = std::max({std::abs(*recent), std::abs(*earlier), 1e-9});
    double const change = (*recent - *earlier) / scale;
    if (change > FlatShare)
    {
        return Trend::Up;
    }
    if (change < -FlatShare)
    {
        return Trend::Down;
    }
    return Trend::Flat;
}

std::string Format(double value)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.3g", value);
    return text;
}

/** @brief A card for a quantity whose latest value is not finite: always a problem. */
bool RejectNonFinite(HealthCard& card, double value)
{
    if (std::isfinite(value))
    {
        return false;
    }
    card.value = "not a number";
    card.health = Health::Problem;
    card.message = "The run produced a non-finite value: training has diverged. Restart from the last good "
                   "checkpoint with a lower learning rate.";
    return true;
}

HealthCard ReturnCard(runtime::RunLog const& log)
{
    HealthCard card;
    card.title = "Episode return";
    std::vector<double> const returns = Collect(log, [](runtime::UpdateRecord const& update) {
        return update.rewards.episodes > 0 ? std::optional<double>(update.rewards.episodeReturnMean) : std::nullopt;
    });
    std::optional<double> const recent = WindowMean(returns, 0, HealthWindow);
    bool const recentEpisodes = std::any_of(
        log.updates.end() - static_cast<std::ptrdiff_t>(std::min(log.updates.size(), HealthWindow)),
        log.updates.end(), [](runtime::UpdateRecord const& update) { return update.rewards.episodes > 0; });
    if (!recent || !recentEpisodes)
    {
        card.value = "-";
        card.health = Health::Watch;
        card.message = "No episode ended in the last updates: episodes may be longer than a rollout, or the "
                       "task never ends them.";
        return card;
    }
    card.value = Format(*recent);
    if (RejectNonFinite(card, *recent))
    {
        return card;
    }
    card.trend = TrendOf(returns);
    switch (card.trend)
    {
    case Trend::Up:
        card.health = Health::Good;
        card.message = "Rising over the last 10 updates.";
        break;
    case Trend::Flat:
        card.health = Health::Good;
        card.message = "Stable over the last 10 updates: a plateau, or a curriculum stage being learned.";
        break;
    case Trend::Down:
        card.health = Health::Watch;
        card.message = "Falling over the last 10 updates: a harder curriculum stage, a weight change, or "
                       "updates that are too large (see the policy change).";
        break;
    case Trend::Unknown:
        card.health = Health::Unknown;
        card.message = "Needs about 15 updates to show a trend.";
        break;
    }
    return card;
}

std::vector<double> Metric(runtime::RunLog const& log, std::string const& name)
{
    return Collect(log, [&](runtime::UpdateRecord const& update) { return Value(update.ppo, name); });
}

std::optional<HealthCard> ExplainedVarianceCard(runtime::RunLog const& log)
{
    std::vector<double> const values = Metric(log, "explained_variance");
    std::optional<double> const recent = WindowMean(values, 0, HealthWindow);
    if (!recent)
    {
        return std::nullopt;
    }
    HealthCard card;
    card.title = "Critic accuracy";
    card.metric = "explained_variance";
    card.value = Format(*recent);
    if (RejectNonFinite(card, *recent))
    {
        return card;
    }
    card.trend = TrendOf(values);
    if (*recent < 0.0)
    {
        card.health = Health::Problem;
        card.message = "The critic predicts returns worse than a constant would: lower the critic learning "
                       "rate, or check the scale of the rewards.";
    }
    else if (*recent < LearningCritic)
    {
        card.health = Health::Watch;
        card.message = "The critic explains little of the returns yet: normal early on, a concern if it stays.";
    }
    else
    {
        card.health = Health::Good;
        card.message = "The critic follows the returns well.";
    }
    return card;
}

std::optional<HealthCard> PolicyChangeCard(runtime::RunLog const& log)
{
    // Owned learners journal the mean KL, Flight its largest minibatch KL.
    std::string name = "approx_kl";
    std::vector<double> values = Metric(log, name);
    if (values.empty())
    {
        name = "max_approx_kl";
        values = Metric(log, name);
    }
    std::optional<double> const recent = WindowMean(values, 0, HealthWindow);
    if (!recent)
    {
        return std::nullopt;
    }
    HealthCard card;
    card.title = "Policy change";
    card.metric = name;
    card.value = Format(*recent);
    if (RejectNonFinite(card, *recent))
    {
        return card;
    }
    card.trend = TrendOf(values);
    double const target = Value(log.header.settings, "target_kl").value_or(0.0);
    double const limit = target > 0.0 ? target : DefaultKlLimit;
    std::string const reference = target > 0.0 ? "the target KL (" + Format(target) + ")" : "0.05";
    if (*recent > 2.0 * limit)
    {
        card.health = Health::Problem;
        card.message = "Updates move the policy far more than " + reference + ": lower the actor learning rate "
                       "or the epochs.";
    }
    else if (*recent > limit)
    {
        card.health = Health::Watch;
        card.message = "Updates move the policy a little more than " + reference + ".";
    }
    else
    {
        card.health = Health::Good;
        card.message = "Updates move the policy by a healthy amount (below " + reference + ").";
    }
    return card;
}

std::optional<HealthCard> ClipCard(runtime::RunLog const& log)
{
    std::vector<double> const values = Metric(log, "clip_fraction");
    std::optional<double> const recent = WindowMean(values, 0, HealthWindow);
    if (!recent)
    {
        return std::nullopt;
    }
    HealthCard card;
    card.title = "Clipped samples";
    card.metric = "clip_fraction";
    char text[32];
    std::snprintf(text, sizeof text, "%.0f %%", 100.0 * *recent);
    card.value = text;
    if (RejectNonFinite(card, *recent))
    {
        return card;
    }
    card.trend = TrendOf(values);
    if (*recent > HighClipFraction)
    {
        card.health = Health::Watch;
        card.message = "Many samples hit the clip range: the policy outruns its data. Lower the actor learning "
                       "rate or the epochs.";
    }
    else
    {
        card.health = Health::Good;
        card.message = "Few samples hit the clip range.";
    }
    return card;
}

std::optional<HealthCard> SkipCard(runtime::RunLog const& log)
{
    std::vector<double> const values = Metric(log, "actor_kl_skips");
    if (values.empty())
    {
        return std::nullopt;
    }
    HealthCard card;
    card.title = "Skipped actor steps";
    card.metric = "actor_kl_skips";
    double const last = values.back();
    card.value = Format(last);
    card.trend = TrendOf(values);
    if (last > 0.0)
    {
        card.health = Health::Watch;
        card.message = "The KL guard skipped actor steps in the last update: the policy tried to move too far.";
    }
    else
    {
        card.health = Health::Good;
        card.message = "No actor step skipped by the KL guard in the last update.";
    }
    return card;
}

} // namespace

std::vector<HealthCard> RunHealth(runtime::RunLog const& log)
{
    std::vector<HealthCard> cards;
    if (log.updates.empty())
    {
        return cards;
    }
    cards.push_back(ReturnCard(log));
    for (auto const& card : {ExplainedVarianceCard(log), PolicyChangeCard(log), ClipCard(log), SkipCard(log)})
    {
        if (card)
        {
            cards.push_back(*card);
        }
    }
    return cards;
}

std::string FormatDuration(double seconds)
{
    char text[32];
    if (!std::isfinite(seconds) || seconds < 0.0)
    {
        return "-";
    }
    auto const total = static_cast<long long>(seconds + 0.5);
    if (total < 60)
    {
        std::snprintf(text, sizeof text, "%lld s", total);
    }
    else if (total < 3600)
    {
        std::snprintf(text, sizeof text, "%lld min", total / 60);
    }
    else if (total < 86400)
    {
        std::snprintf(text, sizeof text, "%lld h %02lld", total / 3600, (total % 3600) / 60);
    }
    else
    {
        std::snprintf(text, sizeof text, "%lld d %lld h", total / 86400, (total % 86400) / 3600);
    }
    return text;
}

Health OverallHealth(std::vector<HealthCard> const& cards) noexcept
{
    Health overall = Health::Good;
    for (HealthCard const& card : cards)
    {
        if (card.health == Health::Problem)
        {
            return Health::Problem;
        }
        if (card.health == Health::Watch)
        {
            overall = Health::Watch;
        }
    }
    return overall;
}

} // namespace ppokit::editor
