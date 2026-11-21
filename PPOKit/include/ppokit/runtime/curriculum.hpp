/**
 * @file PPOKit/include/ppokit/runtime/curriculum.hpp
 * @brief Runs a curriculum from train.json: draws episode scenarios and promotes stages.
 * @details
 * @code
 * ppokit::runtime::Curriculum curriculum(*plan.curriculum);
 * std::vector<double> scenario;                       // reused for every episode
 * auto const draw = curriculum.Sample(rng, scenario); // values in task.json order
 * // ... play one episode with this scenario ...
 * if (curriculum.Record(draw, success)) { log("promoted to " + curriculum.StageName()); }
 * @endcode
 * Promotion follows FlightControl3D: once a window of episodes of the current stage is
 * complete, the stage is promoted if their success rate reaches the stage threshold, and a
 * new empty window starts either way. Rehearsal episodes replay an earlier stage and never
 * count for promotion. The last stage is the full task: it is never left.
 */
#pragma once

#include "ppokit/schema/train_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace ppokit::runtime {

/** @brief Which stage produced a scenario, to be handed back to Curriculum::Record. */
struct ScenarioDraw final
{
    std::size_t stage = 0;
    /** @brief True when an earlier stage was replayed; such episodes do not count for promotion. */
    bool rehearsal = false;
};

/** @brief Curriculum state: current stage and promotion window. */
class Curriculum final
{
public:
    /** @throws std::invalid_argument for a plan without stages. */
    explicit Curriculum(schema::CurriculumPlan plan);

    std::size_t Stage() const noexcept { return stage_; }
    std::size_t Stages() const noexcept { return plan_.stages.size(); }
    std::string const& StageName() const noexcept { return plan_.stages[stage_].name; }
    bool AtFinalStage() const noexcept { return stage_ + 1 == plan_.stages.size(); }
    std::size_t WindowEpisodes() const noexcept { return windowEpisodes_; }
    std::size_t WindowSuccesses() const noexcept { return windowSuccesses_; }

    /**
     * @brief Draws the scenario of the next episode.
     * @param values Receives one value per scenario parameter, in task.json order (resized once).
     */
    template<typename Engine>
    ScenarioDraw Sample(Engine& rng, std::vector<double>& values) const
    {
        ScenarioDraw draw;
        draw.stage = stage_;
        if (stage_ > 0 && plan_.rehearsal > 0 && std::bernoulli_distribution(plan_.rehearsal)(rng))
        {
            draw.stage = std::uniform_int_distribution<std::size_t>(0, stage_ - 1)(rng);
            draw.rehearsal = true;
        }
        auto const& samplers = plan_.stages[draw.stage].samplers;
        values.resize(samplers.size());
        for (std::size_t index = 0; index < samplers.size(); ++index)
        {
            values[index] = Draw(samplers[index], rng);
        }
        return draw;
    }

    /**
     * @brief Records the outcome of an episode drawn by Sample.
     * @return True when this episode completes a window that promotes the stage.
     */
    bool Record(ScenarioDraw const& draw, bool success) noexcept;

    /** @brief Restores a stage (e.g. after resuming training), with an empty window; clamped to the last stage. */
    void Restore(std::size_t stage) noexcept;

    /** @brief Index of a stage by name. @return Nothing for an unknown name. */
    std::optional<std::size_t> StageIndex(std::string const& name) const;

private:
    /** @brief One draw of one parameter; whole numbers for int, bool and enum parameters. */
    template<typename Engine>
    static double Draw(schema::ParameterSampler const& sampler, Engine& rng)
    {
        switch (sampler.kind)
        {
        case schema::SamplerKind::Fixed:
            return sampler.value;
        case schema::SamplerKind::Uniform:
            if (sampler.integral)
            {
                auto const low = static_cast<long long>(sampler.low);
                auto const high = static_cast<long long>(sampler.high);
                return static_cast<double>(std::uniform_int_distribution<long long>(low, high)(rng));
            }
            return std::uniform_real_distribution<double>(sampler.low, sampler.high)(rng);
        case schema::SamplerKind::Normal:
        {
            double const drawn = std::normal_distribution<double>(sampler.mean, sampler.deviation)(rng);
            double const clamped = std::clamp(drawn, sampler.minimum, sampler.maximum);
            return sampler.integral ? std::round(clamped) : clamped;
        }
        case schema::SamplerKind::Choice:
            break;
        }
        std::vector<double> weights;
        weights.reserve(sampler.choices.size());
        for (auto const& choice : sampler.choices)
        {
            weights.push_back(choice.weight);
        }
        std::discrete_distribution<std::size_t> pick(weights.begin(), weights.end());
        return sampler.choices[pick(rng)].value;
    }

    schema::CurriculumPlan plan_;
    std::size_t stage_ = 0;
    std::size_t windowEpisodes_ = 0;
    std::size_t windowSuccesses_ = 0;
};

/**
 * @brief Saves the current stage by name (robust to stages added later), atomically.
 * @details Store it beside the training checkpoint so that a resumed run continues at the same stage.
 */
void SaveCurriculumStage(std::filesystem::path const& path, Curriculum const& curriculum);

/**
 * @brief Restores the stage saved by SaveCurriculumStage.
 * @return False when the file does not exist (a new run starts at the first stage).
 * @throws std::runtime_error when the saved stage is not a stage of this curriculum.
 */
bool LoadCurriculumStage(std::filesystem::path const& path, Curriculum& curriculum);

} // namespace ppokit::runtime
