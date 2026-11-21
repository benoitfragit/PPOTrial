/**
 * @file PPOKit/include/ppokit/schema/train_schema.hpp
 * @brief Training plan read from `train.json`; version 1 holds the curriculum.
 * @details A curriculum is a list of stages, from easy to hard. Each stage says how to
 * draw the scenario parameters declared in task.json. Training moves to the next stage
 * once the success rate over a window of episodes of the current stage reaches a
 * threshold; a small share of episodes can rehearse earlier stages so that the policy
 * does not forget them.
 *
 * @code{.json}
 * {
 *   "schema": "ppokit.train/1",
 *   "curriculum": {
 *     "window": 200, "promote_at": 0.8, "rehearse": 0.1,
 *     "stages": [
 *       { "name": "aligned", "sample": { "wind_kt": 0, "start": "rolling" } },
 *       { "name": "airborne", "sample": { "wind_kt": { "uniform": [0, 10] },
 *                                         "start": { "choice": { "rolling": 0.3, "airborne": 0.7 } } } },
 *       { "name": "full", "window": 400, "sample": { "wind_kt": { "normal": { "mean": 10, "std": 5 } } } }
 *     ]
 *   }
 * }
 * @endcode
 *
 * A parameter a stage does not mention takes its task.json default.
 */
#pragma once

#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::schema {

/** @brief How a stage draws one scenario parameter. */
enum class SamplerKind : std::uint8_t
{
    /** @brief Always the same value. */
    Fixed,
    /** @brief Uniform in [low, high]; integers for an int parameter. */
    Uniform,
    /** @brief Normal(mean, deviation), clamped to the parameter range (and rounded for an int). */
    Normal,
    /** @brief One of weighted values. */
    Choice,
};

/** @brief Weighted value of a Choice sampler. */
struct WeightedValue final
{
    double value = 0;
    double weight = 1;
};

/** @brief Draw rule of one scenario parameter in one stage. */
struct ParameterSampler final
{
    /** @brief Index of the parameter in TaskSchema::scenario. */
    std::size_t parameter = 0;
    SamplerKind kind = SamplerKind::Fixed;
    /** @brief Fixed value. */
    double value = 0;
    /** @brief Uniform bounds. */
    double low = 0;
    double high = 0;
    /** @brief Normal parameters. */
    double mean = 0;
    double deviation = 0;
    /** @brief Choice values and weights. */
    std::vector<WeightedValue> choices;
    /** @brief True for int, bool and enum parameters: draws are whole numbers. */
    bool integral = false;
    /** @brief Range of the parameter, used to clamp normal draws. */
    double minimum = 0;
    double maximum = 0;
};

/** @brief One curriculum stage: a scenario distribution and its promotion rule. */
struct CurriculumStage final
{
    std::string name;
    /** @brief One sampler per scenario parameter, in TaskSchema::scenario order. */
    std::vector<ParameterSampler> samplers;
    /** @brief Episodes of this stage per promotion window. */
    std::size_t window = 100;
    /** @brief Success rate over a full window that promotes to the next stage. */
    double promotionRate = 0.8;
};

/** @brief Ordered stages and the rehearsal share. */
struct CurriculumPlan final
{
    std::vector<CurriculumStage> stages;
    /** @brief Probability that an episode replays an earlier stage (never counted for promotion). */
    double rehearsal = 0;
};

/** @brief Validated content of train.json. */
struct TrainingPlan final
{
    std::optional<CurriculumPlan> curriculum;
};

/**
 * @brief Parses and validates a training plan against its task.
 * @throws SchemaError naming the faulty key, e.g.
 *         "train.json: curriculum.stages[1].sample.start: 'flying' is not a value of this enum".
 */
TrainingPlan ParseTrainingPlan(std::string_view text, std::string_view sourceName, TaskSchema const& task);

/** @brief Reads and parses a training plan file. */
TrainingPlan LoadTrainingPlan(std::filesystem::path const& path, TaskSchema const& task);

} // namespace ppokit::schema
