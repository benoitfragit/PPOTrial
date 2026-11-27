/**
 * @file flight_curriculum.hpp
 * @brief Adaptive start-difficulty curriculum of the training episodes.
 *
 * Role in the chain: session -> **curriculum** -> environment difficulty. The level grows
 * from a short aligned final (level 0, difficulty 0) to the full start envelope (last
 * level, difficulty 1); difficulty = level / (levels - 1). Each level is promoted once
 * the stop rate over a window of airborne training episodes reaches
 * CurriculumSettings::promotionSuccessRate; the window then restarts empty.
 *
 * The level is saved beside the latest checkpoint (latest/curriculum_level.txt) so that a
 * resumed run continues at the same difficulty. Validation campaigns evaluate the same
 * rolling and airborne mix up to the current difficulty, see EvaluationStart.
 *
 * Not here: the difficulty draw of each episode (FlightEnvironment::Reset) and the start
 * sampling (Core/flight_spawn.cpp).
 *
 * Read next: flight_curriculum.cpp, Runtime/flight_session.cpp (AdvanceCurriculum).
 */
#pragma once

#include "flight_config.hpp"

#include "ppokit/runtime/promotion_window.hpp"

#include <filesystem>
#include <optional>

namespace ppo::examples::flight {

/**
 * @brief Adaptive start-difficulty curriculum for training episodes.
 *
 * The level grows from a short aligned final (level 0) to the full start
 * envelope (last approach level), then through the configured stages; the last
 * stage is the full task (see ResolveCurriculum). Each level is promoted once the
 * stop rate over a window of airborne training episodes reaches the configured
 * threshold. The
 * validation campaigns evaluate the same rolling and airborne mix up to the
 * current difficulty, see EvaluationStart.
 */
class FlightCurriculum final
{
public:
    explicit FlightCurriculum(CurriculumSettings const& settings) noexcept;
    /** @brief Current level, from 0 to Levels() - 1. */
    std::size_t Level() const noexcept
    {
        return level_;
    }
    /** @brief Number of levels: the approach levels, then one level per stage. */
    std::size_t Levels() const noexcept
    {
        return CurriculumLevels(settings_);
    }
    /** @brief Maximum start difficulty in [0, 1] for the current level. */
    double Difficulty() const noexcept;
    /** @brief Stops recorded in the current window. */
    std::size_t WindowStops() const noexcept
    {
        return window_.Successes();
    }
    /** @brief Airborne episodes recorded in the current window. */
    std::size_t WindowEpisodes() const noexcept
    {
        return window_.Episodes();
    }
    /**
     * @brief Records the outcome of an airborne training episode.
     * @param stopped True when the episode ended with a confirmed stop.
     * @return True when this episode completes a window that promotes the level.
     */
    bool Record(bool stopped) noexcept;
    /** @brief Restores a saved level, clamped to the configured levels, with an empty window. */
    void Restore(std::size_t level) noexcept;
    /**
     * @brief Resumes a training run where it was saved: its level and its current window.
     * @throws std::invalid_argument for a level beyond the configured ones or a window with
     *         more stops than episodes.
     */
    void Resume(std::size_t level, std::size_t windowEpisodes, std::size_t windowStops);

private:
    CurriculumSettings settings_;
    std::size_t level_{};
    /** @brief Episodes of the current level and their stops (PPOKit's promotion rule). */
    ppokit::runtime::PromotionWindow window_;
};

/**
 * @brief Reads a curriculum level saved beside the latest checkpoint.
 * @param path Level file.
 * @return The saved level, or nothing when the file does not exist.
 * @throws std::runtime_error for a file that does not hold exactly one integer.
 */
std::optional<std::size_t> LoadCurriculumLevel(std::filesystem::path const& path);

/** @brief Atomically saves the curriculum level beside the latest checkpoint. */
void SaveCurriculumLevel(std::filesystem::path const& path, std::size_t level);

} // namespace ppo::examples::flight
