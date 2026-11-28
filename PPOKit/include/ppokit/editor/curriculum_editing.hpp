/**
 * @file PPOKit/include/ppokit/editor/curriculum_editing.hpp
 * @brief The curriculum of a train.json as the studio edits it (no GUI).
 * @details The JSON section is turned into stages with one draw rule per scenario
 * parameter of task.json, and back. A parameter a stage does not draw takes its task.json
 * default ("Task default"). The editor never decides on its own whether a curriculum is
 * valid: CheckCurriculum renders the file and runs the parser training uses, so the
 * message is the one training would give. PreviewStage draws scenarios of a stage with
 * the curriculum engine itself, for the histograms of the studio.
 */
#pragma once

#include "ppokit/schema/train_schema.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::editor {

/** @brief How a stage sets one parameter. */
enum class SamplerKind : std::uint8_t
{
    /** @brief Not written: the task.json default applies. */
    TaskDefault,
    Fixed,
    Uniform,
    Normal,
    Choice,
};

/** @brief Draw rule of one parameter in one stage, with every field kept so switching kinds loses nothing. */
struct EditableSampler final
{
    SamplerKind kind = SamplerKind::TaskDefault;
    double value = 0;
    double low = 0;
    double high = 0;
    double mean = 0;
    double deviation = 0;
    /** @brief Choice: value and weight (enum index or bool as 0/1 for those types). */
    std::vector<schema::WeightedValue> choices;
};

struct EditableStage final
{
    std::string name;
    /** @brief Overrides of the curriculum window and promotion rate for this stage. */
    std::optional<std::size_t> window;
    std::optional<double> promoteAt;
    /** @brief One per scenario parameter, in task.json order. */
    std::vector<EditableSampler> samplers;
};

struct EditableCurriculum final
{
    std::size_t window = 100;
    double promoteAt = 0.8;
    double rehearse = 0.0;
    std::vector<EditableStage> stages;
};

/** @brief A stage of a fresh curriculum: every parameter at its task default (or its range when it has none). */
EditableStage NewStage(schema::TaskSchema const& task, std::string name);

/** @brief Sensible fields for a parameter that switches to `kind` (its range, its default, every enum value). */
EditableSampler SamplerFor(schema::ScenarioParameter const& parameter, SamplerKind kind);

/** @brief The "curriculum" JSON section as editable stages; validates it with the training parser first. */
EditableCurriculum ReadCurriculum(nlohmann::ordered_json const& section, schema::TaskSchema const& task);

/** @brief The editable curriculum as its JSON section (values as the user writes them: names, booleans). */
nlohmann::ordered_json WriteCurriculum(EditableCurriculum const& curriculum, schema::TaskSchema const& task);

/**
 * @brief The message training would give for this train.json text, or empty when valid.
 * @param trainText A whole train.json text.
 */
std::string CheckCurriculum(std::string const& trainText, schema::TaskSchema const& task);

/** @brief Distribution of one parameter in a stage preview. */
struct ParameterPreview final
{
    std::string parameter;
    /** @brief Bar labels: enum names, false/true, or the centre of each bin for numbers. */
    std::vector<std::string> labels;
    /** @brief Share of the draws in each bar. */
    std::vector<double> shares;
};

/**
 * @brief Draws `draws` scenarios of stage `stage` (no rehearsal) with the curriculum engine.
 * @param trainText A valid train.json text (see CheckCurriculum).
 * @return Empty when there is no such stage or no draws were requested.
 */
std::vector<ParameterPreview> PreviewStage(std::string const& trainText, schema::TaskSchema const& task,
    std::size_t stage, std::size_t draws = 2000, std::uint32_t seed = 1);

} // namespace ppokit::editor
