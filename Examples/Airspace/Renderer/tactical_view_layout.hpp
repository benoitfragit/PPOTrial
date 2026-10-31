/**
 * @file tactical_view_layout.hpp
 * @brief Fixed-capacity layout contract for blue tactical camera grids.
 */
#pragma once

#include <algorithm>
#include <cstddef>

namespace ppo::renderer {

/** @brief Maximum number of blue cameras supported by the P/N runtime. */
inline constexpr std::size_t kMaximumTacticalCameraViews = 9U;

/** @brief Integer grid geometry used to render P tactical views. */
struct TacticalViewGrid final
{
    std::size_t viewCount{};
    std::size_t columns{};
    std::size_t rows{};
    int cellWidth{};
    int cellHeight{};

    /** @brief Returns true when the grid contains drawable cells. */
    constexpr bool Valid() const noexcept
    {
        return viewCount != 0U && columns != 0U && rows != 0U
            && cellWidth >= 32 && cellHeight >= 32;
    }
};

/**
 * @brief Fits one cell per blue into a balanced grid without allocations.
 * @param viewCount Number of active blue views in [1,9].
 * @param availableWidth Total drawable width in pixels.
 * @param availableHeight Total drawable height in pixels.
 * @param gap Inter-cell spacing in pixels.
 * @return Valid grid, or an empty value for invalid inputs.
 */
constexpr TacticalViewGrid MakeTacticalViewGrid(
    std::size_t const viewCount,
    int const availableWidth,
    int const availableHeight,
    int const gap = 6) noexcept
{
    if (viewCount == 0U || viewCount > kMaximumTacticalCameraViews
        || availableWidth < 32 || availableHeight < 32 || gap < 0)
    {
        return {};
    }

    std::size_t const columns = viewCount == 1U ? 1U
        : viewCount <= 4U ? 2U : 3U;
    std::size_t const rows = (viewCount + columns - 1U) / columns;
    int const horizontalGaps = gap * static_cast<int>(columns - 1U);
    int const verticalGaps = gap * static_cast<int>(rows - 1U);
    int const cellWidth = std::max(
        32, (availableWidth - horizontalGaps)
            / static_cast<int>(columns));
    int const cellHeight = std::max(
        32, (availableHeight - verticalGaps)
            / static_cast<int>(rows));
    return {viewCount, columns, rows, cellWidth, cellHeight};
}

} // namespace ppo::renderer
