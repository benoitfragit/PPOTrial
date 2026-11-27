/**
 * @file PPOKit/src/runtime/curriculum.cpp
 * @brief Promotion windows and persistence of the curriculum stage.
 */
#include "ppokit/runtime/curriculum.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace ppokit::runtime {
namespace {

/** @brief Writes the stage name followed by a newline. */
struct StageWriter final
{
    std::string const& name;
    void operator()(std::ostream& stream) const { stream << name << '\n'; }
};

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
    artifacts::WriteFileAtomically(path, StageWriter{curriculum.StageName()}, artifacts::AtomicWriteOptions{},
        "PPOKit curriculum stage");
}

bool LoadCurriculumStage(std::filesystem::path const& path, Curriculum& curriculum)
{
    std::ifstream stream(path);
    if (!stream)
    {
        return false;
    }
    std::string name;
    std::string extra;
    if (!(stream >> name) || (stream >> extra))
    {
        throw std::runtime_error("PPOKit curriculum: '" + path.string() + "' must hold exactly one stage name");
    }
    auto const index = curriculum.StageIndex(name);
    if (!index)
    {
        throw std::runtime_error("PPOKit curriculum: saved stage '" + name + "' is not in train.json");
    }
    curriculum.Restore(*index);
    return true;
}

} // namespace ppokit::runtime
