#pragma once

#include "flight_config.hpp"
#include <filesystem>
#include <optional>

namespace ppo::examples::flight {

/**
 * @brief Adaptive start-difficulty curriculum for training episodes.
 *
 * The level grows from a short aligned final (level 0) to the full start
 * envelope (last level). Each level is promoted once the stop rate over a
 * window of airborne training episodes reaches the configured threshold. The
 * validation campaigns evaluate the same rolling and airborne mix up to the
 * current difficulty, see EvaluationStart.
 */
class FlightCurriculum final
{
public:
    explicit FlightCurriculum(CurriculumSettings const& settings) noexcept;
    /** @brief Current level, from 0 to Levels() - 1. */
    std::size_t Level() const noexcept { return level_; }
    std::size_t Levels() const noexcept { return settings_.levels; }
    /** @brief Maximum start difficulty in [0, 1] for the current level. */
    double Difficulty() const noexcept;
    /** @brief Stops and airborne episodes recorded in the current window. */
    std::size_t WindowStops() const noexcept { return stops_; }
    std::size_t WindowEpisodes() const noexcept { return episodes_; }
    /**
     * @brief Records the outcome of an airborne training episode.
     * @param stopped True when the episode ended with a confirmed stop.
     * @return True when this episode completes a window that promotes the level.
     */
    bool Record(bool stopped) noexcept;
    /** @brief Restores a saved level, clamped to the configured levels, with an empty window. */
    void Restore(std::size_t level) noexcept;
private:
    CurriculumSettings settings_;
    std::size_t level_{};
    std::size_t episodes_{}, stops_{};
};

/**
 * @brief Reads a curriculum level saved beside the latest checkpoint.
 * @param path Level file.
 * @return The saved level, or nothing when the file does not exist.
 */
std::optional<std::size_t> LoadCurriculumLevel(std::filesystem::path const& path);

/** @brief Atomically saves the curriculum level beside the latest checkpoint. */
void SaveCurriculumLevel(std::filesystem::path const& path, std::size_t level);

} // namespace ppo::examples::flight
