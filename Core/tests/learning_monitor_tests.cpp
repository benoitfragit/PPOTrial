#include "ppo/core/learning_monitor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main()
{
    using History = ppo::core::LearningHistory;
    History history{ 6 };
    assert(history.RecentTrend() == History::Trend::NotEnoughData);
    assert(std::isnan(history.RecentMean()));

    assert(history.Record(1, 10.0));
    assert(!history.Record(1, 500.0));
    assert(!history.Record(0, 12.0));
    assert(!history.Record(2, std::numeric_limits<double>::quiet_NaN()));
    assert(!history.Record(2, std::numeric_limits<double>::infinity()));
    assert(history.Record(2, 11.0));
    assert(!history.Record(1, 13.0));
    assert(history.Record(3, 10.0));
    assert(history.Record(4, 21.0));
    assert(history.Record(5, 22.0));
    assert(history.Record(6, 23.0));
    assert(history.Samples().size() == 6U);
    assert(history.RecentTrend() == History::Trend::Improving);
    assert(std::fabs(history.RecentMean() - 22.0) < 1.e-8);

    assert(history.Record(7, 9.0));
    assert(history.Samples().size() == 6U);
    assert(history.Samples().front().campaign == 2U);
    assert(history.Record(8, 8.0));
    assert(history.Record(9, 7.0));
    assert(history.RecentTrend() == History::Trend::Declining);

    history.Clear();
    assert(history.Samples().empty());
    assert(history.LastCampaign() == 0U);
    for (std::uint64_t i = 1; i <= 6; ++i)
        assert(history.Record(i, 10.0));
    assert(history.RecentTrend() == History::Trend::Flat);
    return 0;
}
