/**
 * @file Framework/RL/Core/include/ppo/core/learning_monitor.hpp
 * @brief Interface ou implémentation du composant learning monitor.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
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
/** @brief Déclare le type LearningHistory, utilisé par Interface ou implémentation du composant learning monitor. */
class LearningHistory
{
public:
/** @brief Déclare le type Sample, utilisé par Interface ou implémentation du composant learning monitor. */
    struct Sample
    {
        std::uint64_t campaign{ 0 };
        double score{ 0.0 };
    };

/** @brief Déclare le type Trend, utilisé par Interface ou implémentation du composant learning monitor. */
    enum class Trend
    {
        NotEnoughData,
        Improving,
        Flat,
        Declining
    };

/**
 * @brief Réalise l'opération LearningHistory; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
    explicit LearningHistory(std::size_t capacity = 256)
        : m_capacity{ std::max<std::size_t>(capacity, 6U) }
    {
        m_samples.reserve(m_capacity);
    }

    // Ignore duplicate, out-of-order and non-finite evaluator snapshots.
/**
 * @brief Réalise l'opération Record; voir les préconditions et effets dans son corps C++.
 */
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

/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 * @details Applique des bornes numériques.
 */
    void Clear() noexcept
    {
        m_samples.clear();
        m_lastCampaign = 0;
    }

    std::vector<Sample> const& Samples() const noexcept { return m_samples; }
/**
 * @brief Retourne une vue ou une valeur calculée depuis l'état courant.
 * @details Applique des bornes numériques.
 */
    std::uint64_t LastCampaign() const noexcept { return m_lastCampaign; }

    // Two non-overlapping groups of three campaigns. This is a heuristic,
    // NOT a statistical significance test or evidence of generalization.
/**
 * @brief Réalise l'opération RecentTrend; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
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

/**
 * @brief Réalise l'opération RecentMean; voir les préconditions et effets dans son corps C++.
 * @details Applique des bornes numériques.
 */
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
