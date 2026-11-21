/**
 * @file PPOKit/src/schema/train_schema.cpp
 * @brief Parsing and validation of `train.json` (curriculum stages and samplers).
 */
#include "ppokit/schema/train_schema.hpp"

#include "json_validation.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace ppokit::schema {
namespace {

using detail::ErrorReporter;
using detail::Json;
using detail::RequireFiniteNumber;
using detail::RequireKnownKeys;
using detail::RequireObject;
using detail::RequirePositiveInteger;

/** @brief Format identifier accepted in the "schema" key. */
constexpr char const* TrainingSchemaVersion = "ppokit.train/1";

/** @brief Default promotion window and threshold when train.json does not set them. */
constexpr std::size_t DefaultWindow = 100;
constexpr double DefaultPromotionRate = 0.8;

/** @brief Checks that `value` is a valid value of `parameter`, and returns it as a double. */
double ParameterValue(ScenarioParameter const& parameter, Json const& value, std::string const& path,
    ErrorReporter const& errors)
{
    if (parameter.type == ParameterType::Bool)
    {
        if (!value.is_boolean())
        {
            errors.Fail(path, "expected true or false for the bool parameter '" + parameter.name + "'");
        }
        return value.get<bool>() ? 1.0 : 0.0;
    }
    if (parameter.type == ParameterType::Enum)
    {
        auto const index = value.is_string() ? parameter.EnumIndex(value.get<std::string>()) : std::nullopt;
        if (!index)
        {
            errors.Fail(path, "'" + (value.is_string() ? value.get<std::string>() : value.dump())
                + "' is not a value of this enum");
        }
        return static_cast<double>(*index);
    }
    double const number = RequireFiniteNumber(value, path, errors);
    if (number < parameter.minimum || number > parameter.maximum)
    {
        errors.Fail(path, "outside the range of '" + parameter.name + "' in task.json");
    }
    if (parameter.type == ParameterType::Int && std::floor(number) != number)
    {
        errors.Fail(path, "expected an integer for the int parameter '" + parameter.name + "'");
    }
    return number;
}

/** @brief Converts the key of a choice ("airborne", "true", "3") into a parameter value. */
double ChoiceKeyValue(ScenarioParameter const& parameter, std::string const& key, std::string const& path,
    ErrorReporter const& errors)
{
    if (parameter.type == ParameterType::Enum)
    {
        return ParameterValue(parameter, Json(key), path, errors);
    }
    if (parameter.type == ParameterType::Bool && (key == "true" || key == "false"))
    {
        return key == "true" ? 1.0 : 0.0;
    }
    Json const number = Json::parse(key, nullptr, false);
    if (number.is_discarded() || !number.is_number())
    {
        errors.Fail(path, "the choice key '" + key + "' is not a value of '" + parameter.name + "'");
    }
    return ParameterValue(parameter, number, path, errors);
}

ParameterSampler ReadUniform(ScenarioParameter const& parameter, Json const& bounds, std::string const& path,
    ErrorReporter const& errors)
{
    if (parameter.type != ParameterType::Float && parameter.type != ParameterType::Int)
    {
        errors.Fail(path, "uniform sampling needs a float or int parameter; use \"choice\"");
    }
    if (!bounds.is_array() || bounds.size() != 2)
    {
        errors.Fail(path, "expected [low, high]");
    }
    ParameterSampler sampler;
    sampler.kind = SamplerKind::Uniform;
    sampler.low = ParameterValue(parameter, bounds[0], path + "[0]", errors);
    sampler.high = ParameterValue(parameter, bounds[1], path + "[1]", errors);
    if (sampler.high < sampler.low)
    {
        errors.Fail(path, "high must not be smaller than low");
    }
    return sampler;
}

ParameterSampler ReadNormal(ScenarioParameter const& parameter, Json const& settings, std::string const& path,
    ErrorReporter const& errors)
{
    if (parameter.type != ParameterType::Float && parameter.type != ParameterType::Int)
    {
        errors.Fail(path, "normal sampling needs a float or int parameter");
    }
    RequireObject(settings, path, errors);
    RequireKnownKeys(settings, {"mean", "std"}, path, errors);
    if (!settings.contains("mean") || !settings.contains("std"))
    {
        errors.Fail(path, "expected { \"mean\": ..., \"std\": ... }");
    }
    ParameterSampler sampler;
    sampler.kind = SamplerKind::Normal;
    sampler.mean = RequireFiniteNumber(settings.at("mean"), path + ".mean", errors);
    sampler.deviation = RequireFiniteNumber(settings.at("std"), path + ".std", errors);
    if (!(sampler.deviation > 0))
    {
        errors.Fail(path + ".std", "expected a positive standard deviation");
    }
    return sampler;
}

ParameterSampler ReadChoice(ScenarioParameter const& parameter, Json const& weights, std::string const& path,
    ErrorReporter const& errors)
{
    RequireObject(weights, path, errors);
    if (weights.empty())
    {
        errors.Fail(path, "expected at least one value");
    }
    ParameterSampler sampler;
    sampler.kind = SamplerKind::Choice;
    for (auto const& item : weights.items())
    {
        std::string const itemPath = path + "." + item.key();
        WeightedValue choice;
        choice.value = ChoiceKeyValue(parameter, item.key(), itemPath, errors);
        choice.weight = RequireFiniteNumber(item.value(), itemPath, errors);
        if (!(choice.weight > 0))
        {
            errors.Fail(itemPath, "expected a positive weight");
        }
        sampler.choices.push_back(choice);
    }
    return sampler;
}

/** @brief Reads one sampler: a plain value (fixed), or { "uniform" | "normal" | "choice": ... }. */
ParameterSampler ReadSampler(ScenarioParameter const& parameter, Json const& value, std::string const& path,
    ErrorReporter const& errors)
{
    ParameterSampler sampler;
    if (!value.is_object())
    {
        sampler.kind = SamplerKind::Fixed;
        sampler.value = ParameterValue(parameter, value, path, errors);
        return sampler;
    }
    RequireKnownKeys(value, {"uniform", "normal", "choice"}, path, errors);
    if (value.size() != 1)
    {
        errors.Fail(path, "expected exactly one of \"uniform\", \"normal\" or \"choice\"");
    }
    if (value.contains("uniform"))
    {
        sampler = ReadUniform(parameter, value.at("uniform"), path + ".uniform", errors);
    }
    else if (value.contains("normal"))
    {
        sampler = ReadNormal(parameter, value.at("normal"), path + ".normal", errors);
    }
    else
    {
        sampler = ReadChoice(parameter, value.at("choice"), path + ".choice", errors);
    }
    return sampler;
}

bool HasScenarioParameter(TaskSchema const& task, std::string const& name) noexcept
{
    for (auto const& parameter : task.scenario)
    {
        if (parameter.name == name)
        {
            return true;
        }
    }
    return false;
}

CurriculumStage ReadStage(Json const& value, TaskSchema const& task, std::size_t defaultWindow,
    double defaultRate, std::string const& path, ErrorReporter const& errors)
{
    RequireObject(value, path, errors);
    RequireKnownKeys(value, {"name", "sample", "window", "promote_at"}, path, errors);
    CurriculumStage stage;
    if (!value.contains("name"))
    {
        errors.Fail(path + ".name", "missing stage name");
    }
    stage.name = detail::RequireIdentifier(value.at("name"), path + ".name", errors);
    stage.window = defaultWindow;
    stage.promotionRate = defaultRate;
    if (value.contains("window"))
    {
        stage.window = RequirePositiveInteger(value.at("window"), path + ".window", errors);
    }
    if (value.contains("promote_at"))
    {
        stage.promotionRate = RequireFiniteNumber(value.at("promote_at"), path + ".promote_at", errors);
    }
    Json const sample = value.contains("sample") ? value.at("sample") : Json::object();
    RequireObject(sample, path + ".sample", errors);
    for (auto const& item : sample.items())
    {
        if (!HasScenarioParameter(task, item.key()))
        {
            errors.Fail(path + ".sample." + item.key(), "not a scenario parameter of task.json");
        }
    }
    for (std::size_t index = 0; index < task.scenario.size(); ++index)
    {
        ScenarioParameter const& parameter = task.scenario[index];
        std::string const samplerPath = path + ".sample." + parameter.name;
        ParameterSampler sampler;
        if (sample.contains(parameter.name))
        {
            sampler = ReadSampler(parameter, sample.at(parameter.name), samplerPath, errors);
        }
        else if (parameter.defaultValue)
        {
            sampler.kind = SamplerKind::Fixed;
            sampler.value = *parameter.defaultValue;
        }
        else
        {
            errors.Fail(samplerPath, "missing: the stage must sample it, or task.json must give a \"default\"");
        }
        sampler.parameter = index;
        sampler.integral = parameter.type != ParameterType::Float;
        sampler.minimum = parameter.minimum;
        sampler.maximum = parameter.maximum;
        stage.samplers.push_back(std::move(sampler));
    }
    return stage;
}

/** @brief A rate in [0, 1], used for promotion thresholds and the rehearsal share. */
double RequireRate(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    double const rate = RequireFiniteNumber(value, path, errors);
    if (rate < 0 || rate > 1)
    {
        errors.Fail(path, "expected a value between 0 and 1");
    }
    return rate;
}

CurriculumPlan ReadCurriculum(Json const& value, TaskSchema const& task, ErrorReporter const& errors)
{
    RequireObject(value, "curriculum", errors);
    RequireKnownKeys(value, {"window", "promote_at", "rehearse", "stages"}, "curriculum", errors);
    std::size_t window = DefaultWindow;
    double rate = DefaultPromotionRate;
    CurriculumPlan plan;
    if (value.contains("window"))
    {
        window = RequirePositiveInteger(value.at("window"), "curriculum.window", errors);
    }
    if (value.contains("promote_at"))
    {
        rate = RequireRate(value.at("promote_at"), "curriculum.promote_at", errors);
    }
    if (value.contains("rehearse"))
    {
        plan.rehearsal = RequireRate(value.at("rehearse"), "curriculum.rehearse", errors);
    }
    if (!value.contains("stages") || !value.at("stages").is_array() || value.at("stages").empty())
    {
        errors.Fail("curriculum.stages", "expected a non-empty list of stages, easiest first");
    }
    Json const& stages = value.at("stages");
    for (std::size_t index = 0; index < stages.size(); ++index)
    {
        std::string const path = "curriculum.stages[" + std::to_string(index) + "]";
        CurriculumStage stage = ReadStage(stages[index], task, window, rate, path, errors);
        if (stage.promotionRate < 0 || stage.promotionRate > 1)
        {
            errors.Fail(path + ".promote_at", "expected a value between 0 and 1");
        }
        for (auto const& previous : plan.stages)
        {
            if (previous.name == stage.name)
            {
                errors.Fail(path + ".name", "duplicate stage name '" + stage.name + "'");
            }
        }
        plan.stages.push_back(std::move(stage));
    }
    return plan;
}

} // namespace

TrainingPlan ParseTrainingPlan(std::string_view text, std::string_view sourceName, TaskSchema const& task)
{
    ErrorReporter const errors(sourceName);
    Json const root = detail::ParseRootObject(text, errors);
    RequireKnownKeys(root, {"schema", "curriculum"}, "", errors);
    if (!root.contains("schema") || root.at("schema") != TrainingSchemaVersion)
    {
        errors.Fail("schema", std::string("expected \"") + TrainingSchemaVersion + "\"");
    }
    TrainingPlan plan;
    if (root.contains("curriculum"))
    {
        plan.curriculum = ReadCurriculum(root.at("curriculum"), task, errors);
    }
    return plan;
}

TrainingPlan LoadTrainingPlan(std::filesystem::path const& path, TaskSchema const& task)
{
    return ParseTrainingPlan(detail::ReadTextFile(path), path.filename().string(), task);
}

} // namespace ppokit::schema
