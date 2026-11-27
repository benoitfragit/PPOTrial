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
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace ppo::examples::flight {
namespace {

/** @brief Flight's historical levels expressed as a common episode-window plan. */
ppokit::runtime::OrchestrationPlan FlightPlan(CurriculumSettings const& settings)
{
    ppokit::runtime::OrchestrationPlan plan;
    plan.id = "flight-start-difficulty";
    for (std::size_t level = 0; level < CurriculumLevels(settings); ++level)
    {
        ppokit::runtime::OrchestrationStage stage;
        stage.name = "level-" + std::to_string(level);
        stage.payload = nlohmann::json{{"level", level}}.dump();
        stage.promotion.source = ppokit::runtime::PromotionSource::Episode;
        stage.promotion.window = settings.windowEpisodes;
        stage.promotion.threshold = settings.promotionSuccessRate;
        plan.stages.push_back(std::move(stage));
    }
    return plan;
}

} // namespace

FlightCurriculum::FlightCurriculum(CurriculumSettings const& settings)
    : settings_(settings), engine_(FlightPlan(settings))
{
}

double FlightCurriculum::Difficulty() const noexcept
{
    if (Levels() <= 1)
    {
        return 1.0;
    }
    return static_cast<double>(Level()) / static_cast<double>(Levels() - 1);
}

bool FlightCurriculum::Record(bool stopped)
{
    // The full task has no further level.
    if (Level() + 1 >= Levels())
    {
        return false;
    }
    return engine_.RecordEpisode(true, stopped).changed;
}

void FlightCurriculum::Restore(std::size_t level) noexcept
{
    engine_.Restore(std::min(level, Levels() - 1), 0, 0);
}

void FlightCurriculum::Resume(std::size_t level, std::size_t windowEpisodes, std::size_t windowStops)
{
    if (level >= Levels())
    {
        throw std::invalid_argument("FlightControl3D: curriculum level " + std::to_string(level) + " of "
            + std::to_string(Levels()) + " levels");
    }
    engine_.Restore(level, windowEpisodes, windowStops);
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
