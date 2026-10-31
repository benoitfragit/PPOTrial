/**
 * @file tactical_view_layout_tests.cpp
 * @brief Deterministic regressions for P=1..9 tactical camera layouts.
 */
#include "tactical_view_layout.hpp"

#include <cstddef>
#include <stdexcept>

namespace {

/**
 * @brief Requires one exact grid shape.
 * @param count Requested blue-camera count.
 * @param columns Expected columns.
 * @param rows Expected rows.
 */
void RequireShape(std::size_t const count, std::size_t const columns,
                  std::size_t const rows)
{
    auto const grid = ppo::renderer::MakeTacticalViewGrid(
        count, 1200, 800);
    if (!grid.Valid() || grid.viewCount != count
        || grid.columns != columns || grid.rows != rows
        || grid.cellWidth < 32 || grid.cellHeight < 32)
    {
        throw std::runtime_error("Invalid tactical camera grid shape");
    }
}

} // namespace

/** @brief Verifies balanced layouts and invalid camera counts. */
int main()
{
    RequireShape(1U, 1U, 1U);
    RequireShape(2U, 2U, 1U);
    RequireShape(3U, 2U, 2U);
    RequireShape(4U, 2U, 2U);
    RequireShape(5U, 3U, 2U);
    RequireShape(6U, 3U, 2U);
    RequireShape(7U, 3U, 3U);
    RequireShape(8U, 3U, 3U);
    RequireShape(9U, 3U, 3U);
    if (ppo::renderer::MakeTacticalViewGrid(0U, 1200, 800).Valid()
        || ppo::renderer::MakeTacticalViewGrid(10U, 1200, 800).Valid())
    {
        throw std::runtime_error("Invalid tactical camera count accepted");
    }
    return 0;
}
