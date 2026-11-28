/**
 * @file PPOKit/src/editor/curriculum_editing.cpp
 * @brief Curriculum section of train.json <-> editable stages, validation and stage previews.
 */
#include "ppokit/editor/curriculum_editing.hpp"

#include "ppokit/runtime/curriculum.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

namespace ppokit::editor {
namespace {

using Json = nlohmann::ordered_json;
using schema::ParameterType;
using schema::ScenarioParameter;

/** @brief Bars of a numeric preview. */
constexpr std::size_t PreviewBins = 12;

/** @brief A parameter value as the user writes it: a name for an enum, true/false, a whole number, a float. */
Json ValueJson(ScenarioParameter const& parameter, double value)
{
    switch (parameter.type)
    {
    case ParameterType::Bool:
        return Json(value != 0.0);
    case ParameterType::Enum:
        return Json(parameter.values.at(static_cast<std::size_t>(value)));
    case ParameterType::Int:
        return Json(static_cast<std::int64_t>(std::llround(value)));
    case ParameterType::Float:
        break;
    }
    return Json(value);
}

/** @brief A value of the parameter read back from JSON (enum names, booleans, numbers). */
double ValueOf(ScenarioParameter const& parameter, Json const& value)
{
    if (value.is_boolean())
    {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    if (value.is_string())
    {
        return static_cast<double>(parameter.EnumIndex(value.get<std::string>()).value_or(0));
    }
    return value.is_number() ? value.get<double>() : 0.0;
}

/** @brief Key of a choice entry: the enum name, true/false, or the number. */
std::string ChoiceKey(ScenarioParameter const& parameter, double value)
{
    Json const json = ValueJson(parameter, value);
    return json.is_string() ? json.get<std::string>() : json.dump();
}

double ChoiceKeyValue(ScenarioParameter const& parameter, std::string const& key)
{
    if (parameter.type == ParameterType::Enum)
    {
        return static_cast<double>(parameter.EnumIndex(key).value_or(0));
    }
    if (key == "true" || key == "false")
    {
        return key == "true" ? 1.0 : 0.0;
    }
    Json const number = Json::parse(key, nullptr, false);
    return number.is_number() ? number.get<double>() : 0.0;
}

EditableSampler ReadSampler(ScenarioParameter const& parameter, Json const& value)
{
    EditableSampler sampler = SamplerFor(parameter, SamplerKind::TaskDefault);
    if (!value.is_object())
    {
        sampler.kind = SamplerKind::Fixed;
        sampler.value = ValueOf(parameter, value);
        return sampler;
    }
    if (value.contains("uniform") && value.at("uniform").is_array() && value.at("uniform").size() == 2)
    {
        sampler.kind = SamplerKind::Uniform;
        sampler.low = ValueOf(parameter, value.at("uniform")[0]);
        sampler.high = ValueOf(parameter, value.at("uniform")[1]);
    }
    else if (value.contains("normal") && value.at("normal").is_object())
    {
        sampler.kind = SamplerKind::Normal;
        sampler.mean = value.at("normal").value("mean", 0.0);
        sampler.deviation = value.at("normal").value("std", 0.0);
    }
    else if (value.contains("choice") && value.at("choice").is_object())
    {
        sampler.kind = SamplerKind::Choice;
        sampler.choices.clear();
        for (auto const& [key, weight] : value.at("choice").items())
        {
            sampler.choices.push_back({ChoiceKeyValue(parameter, key), weight.is_number() ? weight.get<double>() : 0.0});
        }
    }
    return sampler;
}

/** @brief A sampler as JSON; null for TaskDefault (the key is then left out). */
Json WriteSampler(ScenarioParameter const& parameter, EditableSampler const& sampler)
{
    switch (sampler.kind)
    {
    case SamplerKind::TaskDefault:
        return nullptr;
    case SamplerKind::Fixed:
        return ValueJson(parameter, sampler.value);
    case SamplerKind::Uniform:
        return Json{{"uniform", Json::array({ValueJson(parameter, sampler.low), ValueJson(parameter, sampler.high)})}};
    case SamplerKind::Normal:
        return Json{{"normal", Json{{"mean", sampler.mean}, {"std", sampler.deviation}}}};
    case SamplerKind::Choice:
    {
        Json weights = Json::object();
        for (schema::WeightedValue const& choice : sampler.choices)
        {
            weights[ChoiceKey(parameter, choice.value)] = choice.weight;
        }
        return Json{{"choice", std::move(weights)}};
    }
    }
    return nullptr;
}

std::string Shown(double value)
{
    std::ostringstream text;
    text << std::setprecision(3) << value;
    return text.str();
}

/** @brief Bars of one parameter: one per value for enums and booleans, PreviewBins over the range otherwise. */
ParameterPreview EmptyPreview(ScenarioParameter const& parameter)
{
    ParameterPreview preview;
    preview.parameter = parameter.name;
    if (parameter.type == ParameterType::Enum)
    {
        preview.labels = parameter.values;
    }
    else if (parameter.type == ParameterType::Bool)
    {
        preview.labels = {"false", "true"};
    }
    else
    {
        double const width = (parameter.maximum - parameter.minimum) / static_cast<double>(PreviewBins);
        for (std::size_t bin = 0; bin < PreviewBins; ++bin)
        {
            preview.labels.push_back(Shown(parameter.minimum + (static_cast<double>(bin) + 0.5) * width));
        }
    }
    preview.shares.assign(preview.labels.size(), 0.0);
    return preview;
}

std::size_t BarOf(ScenarioParameter const& parameter, double value, std::size_t bars)
{
    if (parameter.type == ParameterType::Enum || parameter.type == ParameterType::Bool)
    {
        return std::min(static_cast<std::size_t>(value), bars - 1);
    }
    double const span = parameter.maximum - parameter.minimum;
    if (span <= 0.0)
    {
        return 0;
    }
    double const position = (value - parameter.minimum) / span * static_cast<double>(bars);
    return std::min(static_cast<std::size_t>(std::max(position, 0.0)), bars - 1);
}

/** @brief Largest value of a parameter: last enum index, true, or the top of its range. */
double UpperValue(ScenarioParameter const& parameter)
{
    if (parameter.type == ParameterType::Enum)
    {
        return static_cast<double>(parameter.values.size() - 1);
    }
    if (parameter.type == ParameterType::Bool)
    {
        return 1.0;
    }
    return parameter.maximum;
}

} // namespace

EditableSampler SamplerFor(ScenarioParameter const& parameter, SamplerKind kind)
{
    EditableSampler sampler;
    sampler.kind = kind;
    bool const discrete = parameter.type == ParameterType::Enum || parameter.type == ParameterType::Bool;
    double const low = discrete ? 0.0 : parameter.minimum;
    double const high = UpperValue(parameter);
    sampler.value = parameter.defaultValue.value_or(low);
    sampler.low = low;
    sampler.high = high;
    sampler.mean = parameter.defaultValue.value_or((low + high) / 2.0);
    sampler.deviation = (high - low) / 4.0;
    // A choice starts with every discrete value equally likely, or the two ends of a range.
    if (discrete)
    {
        for (double value = low; value <= high; value += 1.0)
        {
            sampler.choices.push_back({value, 1.0});
        }
    }
    else
    {
        sampler.choices = {{low, 1.0}, {high, 1.0}};
    }
    return sampler;
}

EditableStage NewStage(schema::TaskSchema const& task, std::string name)
{
    EditableStage stage;
    stage.name = std::move(name);
    for (ScenarioParameter const& parameter : task.scenario)
    {
        // Without a default, sample the whole domain with a rule the training parser accepts.
        SamplerKind kind = SamplerKind::TaskDefault;
        if (!parameter.defaultValue)
        {
            bool const discrete = parameter.type == ParameterType::Bool || parameter.type == ParameterType::Enum;
            kind = discrete ? SamplerKind::Choice : SamplerKind::Uniform;
        }
        stage.samplers.push_back(SamplerFor(parameter, kind));
    }
    return stage;
}

EditableCurriculum ReadCurriculum(Json const& section, schema::TaskSchema const& task)
{
    // Never silently turn malformed samplers or unknown parameter names into defaults.
    Json const plan = {{"schema", "ppokit.train/1"}, {"curriculum", section}};
    schema::ParseTrainingPlan(plan.dump(), "train.json", task);
    EditableCurriculum curriculum;
    curriculum.window = section.value("window", curriculum.window);
    curriculum.promoteAt = section.value("promote_at", curriculum.promoteAt);
    curriculum.rehearse = section.value("rehearse", curriculum.rehearse);
    if (!section.contains("stages") || !section.at("stages").is_array())
    {
        return curriculum;
    }
    for (Json const& entry : section.at("stages"))
    {
        EditableStage stage = NewStage(task, entry.value("name", std::string("stage")));
        if (entry.contains("window"))
        {
            stage.window = entry.at("window").get<std::size_t>();
        }
        if (entry.contains("promote_at"))
        {
            stage.promoteAt = entry.at("promote_at").get<double>();
        }
        Json const sample = entry.value("sample", Json::object());
        for (std::size_t index = 0; index < task.scenario.size(); ++index)
        {
            ScenarioParameter const& parameter = task.scenario[index];
            stage.samplers[index] = sample.contains(parameter.name)
                ? ReadSampler(parameter, sample.at(parameter.name))
                : SamplerFor(parameter, SamplerKind::TaskDefault);
        }
        curriculum.stages.push_back(std::move(stage));
    }
    return curriculum;
}

Json WriteCurriculum(EditableCurriculum const& curriculum, schema::TaskSchema const& task)
{
    Json section = Json::object();
    section["window"] = curriculum.window;
    section["promote_at"] = curriculum.promoteAt;
    section["rehearse"] = curriculum.rehearse;
    Json stages = Json::array();
    for (EditableStage const& stage : curriculum.stages)
    {
        Json entry = Json::object();
        entry["name"] = stage.name;
        if (stage.window)
        {
            entry["window"] = *stage.window;
        }
        if (stage.promoteAt)
        {
            entry["promote_at"] = *stage.promoteAt;
        }
        Json sample = Json::object();
        for (std::size_t index = 0; index < task.scenario.size() && index < stage.samplers.size(); ++index)
        {
            Json value = WriteSampler(task.scenario[index], stage.samplers[index]);
            if (!value.is_null())
            {
                sample[task.scenario[index].name] = std::move(value);
            }
        }
        entry["sample"] = std::move(sample);
        stages.push_back(std::move(entry));
    }
    section["stages"] = std::move(stages);
    return section;
}

std::string CheckCurriculum(std::string const& trainText, schema::TaskSchema const& task)
{
    try
    {
        schema::ParseTrainingPlan(trainText, "train.json", task);
    }
    catch (schema::SchemaError const& error)
    {
        return error.what();
    }
    return {};
}

std::vector<ParameterPreview> PreviewStage(std::string const& trainText, schema::TaskSchema const& task,
    std::size_t stage, std::size_t draws, std::uint32_t seed)
{
    schema::TrainingPlan plan = schema::ParseTrainingPlan(trainText, "train.json", task);
    std::vector<ParameterPreview> previews;
    if (draws == 0 || !plan.curriculum || stage >= plan.curriculum->stages.size())
    {
        return previews;
    }
    plan.curriculum->rehearsal = 0.0; // the stage alone, without replays of earlier stages
    runtime::Curriculum curriculum(*plan.curriculum);
    curriculum.Restore(stage);
    for (ScenarioParameter const& parameter : task.scenario)
    {
        previews.push_back(EmptyPreview(parameter));
    }
    std::mt19937 rng(seed);
    std::vector<double> values;
    for (std::size_t draw = 0; draw < draws; ++draw)
    {
        curriculum.Sample(rng, values);
        for (std::size_t index = 0; index < task.scenario.size(); ++index)
        {
            ParameterPreview& preview = previews[index];
            preview.shares[BarOf(task.scenario[index], values[index], preview.shares.size())] += 1.0;
        }
    }
    for (ParameterPreview& preview : previews)
    {
        for (double& share : preview.shares)
        {
            share /= static_cast<double>(draws);
        }
    }
    return previews;
}

} // namespace ppokit::editor
