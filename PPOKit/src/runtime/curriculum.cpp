/**
 * @file PPOKit/src/runtime/curriculum.cpp
 * @brief Promotion windows and persistence of the curriculum stage.
 */
#include "ppokit/runtime/curriculum.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppokit::runtime {
namespace {

using Json = nlohmann::ordered_json;

} // namespace

Curriculum::Curriculum(schema::CurriculumPlan plan) : plan_(std::move(plan))
{
    if (plan_.stages.empty())
    {
        throw std::invalid_argument("PPOKit curriculum: at least one stage is required");
    }
}

bool Curriculum::Record(ScenarioDraw const& draw, bool success) noexcept
{
    // Rehearsals and episodes drawn before a promotion say nothing about the current stage.
    if (draw.rehearsal || draw.stage != stage_ || AtFinalStage())
    {
        return false;
    }
    auto const& stage = plan_.stages[stage_];
    bool const promoted = window_.Record(success, stage.window, stage.promotionRate);
    if (promoted)
    {
        ++stage_;
    }
    return promoted;
}

void Curriculum::Restore(std::size_t stage) noexcept
{
    stage_ = std::min(stage, plan_.stages.size() - 1);
    window_.Clear();
}

void Curriculum::Restore(std::size_t stage, std::size_t episodes, std::size_t successes)
{
    stage_ = std::min(stage, plan_.stages.size() - 1);
    window_.Restore(episodes, successes);
    if (AtFinalStage() && episodes != 0)
    {
        throw std::invalid_argument("PPOKit curriculum: the final stage cannot have a promotion window");
    }
}

std::optional<std::size_t> Curriculum::StageIndex(std::string const& name) const
{
    for (std::size_t index = 0; index < plan_.stages.size(); ++index)
    {
        if (plan_.stages[index].name == name)
        {
            return index;
        }
    }
    return std::nullopt;
}

void SaveCurriculumStage(std::filesystem::path const& path, Curriculum const& curriculum)
{
    Json const state{{"schema", "ppokit.curriculum-state/1"}, {"stage", curriculum.StageName()},
        {"episodes", curriculum.WindowEpisodes()}, {"successes", curriculum.WindowSuccesses()}};
    artifacts::WriteFileAtomically(path, [&state](std::ostream& stream) { stream << state.dump(2) << '\n'; },
        artifacts::AtomicWriteOptions{}, "PPOKit curriculum state");
}

bool LoadCurriculumStage(std::filesystem::path const& path, Curriculum& curriculum)
{
    std::ifstream stream(path);
    if (!stream)
    {
        return false;
    }
    std::string const text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    std::string name;
    std::size_t episodes = 0;
    std::size_t successes = 0;
    Json const state = Json::parse(text, nullptr, false);
    if (state.is_object())
    {
        if (state.value("schema", std::string()) != "ppokit.curriculum-state/1"
            || !state.contains("stage") || !state.at("stage").is_string()
            || !state.contains("episodes") || !state.at("episodes").is_number_unsigned()
            || !state.contains("successes") || !state.at("successes").is_number_unsigned())
        {
            throw std::runtime_error("PPOKit curriculum: invalid state in '" + path.string() + "'");
        }
        name = state.at("stage").get<std::string>();
        episodes = state.at("episodes").get<std::size_t>();
        successes = state.at("successes").get<std::size_t>();
    }
    else
    {
        std::istringstream legacy(text);
        std::string extra;
        if (!(legacy >> name) || (legacy >> extra))
        {
            throw std::runtime_error("PPOKit curriculum: '" + path.string()
                + "' must hold a curriculum state or exactly one legacy stage name");
        }
    }
    auto const index = curriculum.StageIndex(name);
    if (!index)
    {
        throw std::runtime_error("PPOKit curriculum: saved stage '" + name + "' is not in train.json");
    }
    curriculum.Restore(*index, episodes, successes);
    return true;
}

} // namespace ppokit::runtime
