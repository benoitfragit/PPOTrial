/**
 * @file Framework/RL/Core/tests/learning_monitor_tests.cpp
 * @brief Tests et assertions déterministes pour learning monitor tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "ppo/core/learning_monitor.hpp"

#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void Check(bool condition)
{
    if (!condition)
        throw std::runtime_error("learning monitor regression test failed");
}
} // namespace

int main()
{
    using History = ppo::core::LearningHistory;
    History history{ 6 };
    Check(history.RecentTrend() == History::Trend::NotEnoughData);
    Check(std::isnan(history.RecentMean()));

    Check(history.Record(1, 10.0));
    Check(!history.Record(1, 500.0));
    Check(!history.Record(0, 12.0));
    Check(!history.Record(2, std::numeric_limits<double>::quiet_NaN()));
    Check(!history.Record(2, std::numeric_limits<double>::infinity()));
    Check(history.Record(2, 11.0));
    Check(!history.Record(1, 13.0));
    Check(history.Record(3, 10.0));
    Check(history.Record(4, 21.0));
    Check(history.Record(5, 22.0));
    Check(history.Record(6, 23.0));
    Check(history.Samples().size() == 6U);
    Check(history.RecentTrend() == History::Trend::Improving);
    Check(std::fabs(history.RecentMean() - 22.0) < 1.e-8);

    Check(history.Record(7, 9.0));
    Check(history.Samples().size() == 6U);
    Check(history.Samples().front().campaign == 2U);
    Check(history.Record(8, 8.0));
    Check(history.Record(9, 7.0));
    Check(history.RecentTrend() == History::Trend::Declining);

    history.Clear();
    Check(history.Samples().empty());
    Check(history.LastCampaign() == 0U);
    for (std::uint64_t i = 1; i <= 6; ++i)
        Check(history.Record(i, 10.0));
    Check(history.RecentTrend() == History::Trend::Flat);
    return 0;
}
