/**
 * @file PPOKit/include/ppokit/runtime/promotion_window.hpp
 * @brief The promotion rule of every curriculum: a window of episodes, then a success rate.
 * @details Shared by the stage curriculum of train.json (ppokit/runtime/curriculum.hpp)
 * and by examples with their own notion of difficulty (FlightControl3D's levels). Each
 * decision starts a fresh window, whether it promotes or not, so a promotion is judged
 * only on episodes of the level being left.
 */
#pragma once

#include <cstddef>
#include <stdexcept>

namespace ppokit::runtime {

class PromotionWindow final
{
public:
    /**
     * @brief Counts one episode of the current level.
     * @param window Episodes per decision.
     * @param promotionRate Success rate over a full window that promotes.
     * @return True when this episode completes a window that promotes.
     */
    bool Record(bool success, std::size_t window, double promotionRate) noexcept
    {
        ++episodes_;
        if (success)
        {
            ++successes_;
        }
        if (episodes_ < window)
        {
            return false;
        }
        double const rate = static_cast<double>(successes_) / static_cast<double>(episodes_);
        Clear();
        return rate >= promotionRate;
    }

    /** @brief Forgets the current window (a level change from outside, a resume). */
    void Clear() noexcept
    {
        episodes_ = 0;
        successes_ = 0;
    }

    /**
     * @brief Resumes a window saved by a training run (Episodes(), Successes()).
     * @throws std::invalid_argument when there are more successes than episodes.
     */
    void Restore(std::size_t episodes, std::size_t successes)
    {
        if (successes > episodes)
        {
            throw std::invalid_argument("a promotion window cannot hold more successes than episodes");
        }
        episodes_ = episodes;
        successes_ = successes;
    }

    std::size_t Episodes() const noexcept { return episodes_; }
    std::size_t Successes() const noexcept { return successes_; }

private:
    std::size_t episodes_ = 0;
    std::size_t successes_ = 0;
};

} // namespace ppokit::runtime
