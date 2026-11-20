#include "flight_curriculum.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ppo::examples::flight {

FlightCurriculum::FlightCurriculum(CurriculumSettings const& settings) noexcept : settings_(settings) {}

double FlightCurriculum::Difficulty() const noexcept
{
    if (settings_.levels <= 1) return 1.0;
    return static_cast<double>(level_) / static_cast<double>(settings_.levels - 1);
}

bool FlightCurriculum::Record(bool stopped) noexcept
{
    if (level_ + 1 >= settings_.levels) return false; // The full task has no further level.
    ++episodes_;
    if (stopped) ++stops_;
    if (episodes_ < settings_.windowEpisodes) return false;
    double const rate = static_cast<double>(stops_) / static_cast<double>(episodes_);
    bool const promoted = rate >= settings_.promotionSuccessRate;
    if (promoted) ++level_;
    // Each decision starts a fresh window, so a promotion is judged only on
    // episodes drawn from the level being left.
    episodes_ = 0;
    stops_ = 0;
    return promoted;
}

void FlightCurriculum::Restore(std::size_t level) noexcept
{
    level_ = std::min(level, settings_.levels - 1);
    episodes_ = 0;
    stops_ = 0;
}

std::optional<std::size_t> LoadCurriculumLevel(std::filesystem::path const& path)
{
    if (!std::filesystem::exists(path)) return std::nullopt;
    std::ifstream input(path);
    std::size_t level{};
    std::string trailing;
    if (!(input >> level) || (input >> trailing))
        throw std::runtime_error("FlightControl3D: invalid saved curriculum level");
    return level;
}

void SaveCurriculumLevel(std::filesystem::path const& path, std::size_t level)
{
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        output << level << '\n';
        output.flush();
        if (!output) throw std::runtime_error("FlightControl3D: curriculum level write failed");
    }
    std::filesystem::rename(temporary, path);
}
} // namespace ppo::examples::flight
