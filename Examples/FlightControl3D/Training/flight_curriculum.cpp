/**
 * @file flight_curriculum.cpp
 * @brief Window-based promotion of the curriculum level and its persistence.
 *
 * Read next: flight_curriculum.hpp.
 */
#include "flight_curriculum.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ppo::examples::flight {

FlightCurriculum::FlightCurriculum(CurriculumSettings const& settings) noexcept : settings_(settings)
{
}

double FlightCurriculum::Difficulty() const noexcept
{
    if (Levels() <= 1)
    {
        return 1.0;
    }
    return static_cast<double>(level_) / static_cast<double>(Levels() - 1);
}

bool FlightCurriculum::Record(bool stopped) noexcept
{
    // The full task has no further level.
    if (level_ + 1 >= Levels())
    {
        return false;
    }
    bool const promoted = window_.Record(stopped, settings_.windowEpisodes, settings_.promotionSuccessRate);
    if (promoted)
    {
        ++level_;
    }
    return promoted;
}

void FlightCurriculum::Restore(std::size_t level) noexcept
{
    level_ = std::min(level, Levels() - 1);
    window_.Clear();
}

void FlightCurriculum::Resume(std::size_t level, std::size_t windowEpisodes, std::size_t windowStops)
{
    if (level >= Levels())
    {
        throw std::invalid_argument("FlightControl3D: curriculum level " + std::to_string(level) + " of "
            + std::to_string(Levels()) + " levels");
    }
    window_.Restore(windowEpisodes, windowStops);
    level_ = level;
}

std::optional<std::size_t> LoadCurriculumLevel(std::filesystem::path const& path)
{
    if (!std::filesystem::exists(path))
    {
        return std::nullopt;
    }
    std::ifstream input(path);
    std::size_t level{};
    std::string trailing;
    if (!(input >> level) || (input >> trailing))
    {
        throw std::runtime_error("FlightControl3D: invalid saved curriculum level");
    }
    return level;
}

void SaveCurriculumLevel(std::filesystem::path const& path, std::size_t level)
{
    // Written beside, then swapped in: a crash never leaves a half-written level, and no backup stays.
    ppokit::artifacts::WriteFileAtomically(
        path, [level](std::ostream& output) { output << level << '\n'; }, ppokit::artifacts::AtomicWriteOptions{},
        "FlightControl3D: curriculum level");
}
} // namespace ppo::examples::flight
