#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ppo::core {

// Observational history only: neither the PPO update nor evaluation policy
// depends on this class. One sample represents one COMPLETED evaluation.
class LearningHistory
{
public:
    struct Sample
    {
        std::uint64_t campaign{ 0 };
        double score{ 0.0 };
    };

    enum class Trend
    {
        NotEnoughData,
        Improving,
        Flat,
        Declining
    };

    explicit LearningHistory(std::size_t capacity = 256)
        : m_capacity{ std::max<std::size_t>(capacity, 6U) }
    {
        m_samples.reserve(m_capacity);
    }

    // Ignore duplicate, out-of-order and non-finite evaluator snapshots.
    bool Record(std::uint64_t campaign, double score)
    {
        if (campaign == 0 || !std::isfinite(score)
            || (!m_samples.empty() && campaign <= m_lastCampaign))
            return false;

        m_lastCampaign = campaign;
        if (m_samples.size() == m_capacity)
            m_samples.erase(m_samples.begin());
        m_samples.push_back({ campaign, score });
        return true;
    }

    void Clear() noexcept
    {
        m_samples.clear();
        m_lastCampaign = 0;
    }

    std::vector<Sample> const& Samples() const noexcept { return m_samples; }
    std::uint64_t LastCampaign() const noexcept { return m_lastCampaign; }

    // Two non-overlapping groups of three campaigns. This is a heuristic,
    // NOT a statistical significance test or evidence of generalization.
    Trend RecentTrend() const noexcept
    {
        if (m_samples.size() < 6U)
            return Trend::NotEnoughData;

        auto const n = m_samples.size();
        double early = 0.0;
        double late = 0.0;
        for (std::size_t i = 0; i < 3U; ++i)
        {
            early += m_samples[n - 6U + i].score / 3.0;
            late += m_samples[n - 3U + i].score / 3.0;
        }
        double const tolerance = 0.05 * std::max({ 1.0, std::fabs(early), std::fabs(late) });
        if (late - early > tolerance)
            return Trend::Improving;
        if (early - late > tolerance)
            return Trend::Declining;
        return Trend::Flat;
    }

    double RecentMean() const noexcept
    {
        if (m_samples.empty())
            return std::numeric_limits<double>::quiet_NaN();

        auto const count = std::min<std::size_t>(3U, m_samples.size());
        double sum = 0.0;
        for (std::size_t i = m_samples.size() - count; i < m_samples.size(); ++i)
            sum += m_samples[i].score;
        return sum / static_cast<double>(count);
    }

private:
    std::size_t m_capacity;
    std::uint64_t m_lastCampaign{ 0 };
    std::vector<Sample> m_samples{};
};

} // namespace ppo::core
