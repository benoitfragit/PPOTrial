/**
 * @file flight_model_store.hpp
 * @brief Validation metrics and the store of latest and BEST checkpoints.
 *
 * Role in the chain: PPO -> **checkpoints** -> session. Two kinds of files live under the
 * model directory:
 * - latest/flight_policy.txt and latest/curriculum_level.txt: mutable training state,
 *   overwritten at checkpoints;
 * - best/flight_<update>_<fnv1a64>.txt and best/current.json: immutable, evaluated BEST
 *   generations and an atomically replaced manifest pointing to the current one.
 *
 * Ranking: a BEST candidate must pass the validation gate (Validated), then beat the
 * incumbent (Better): harder campaign first, then higher stop rate, then shorter mean
 * stop distance. The manifest format, its checksum and the evaluation protocol version
 * are compatibility contracts.
 *
 * Not here: running the validation episodes (Runtime/flight_session.cpp).
 *
 * Read next: flight_model_store.cpp, Runtime/flight_session.cpp (StartValidation).
 */
#pragma once

#include "flight_learner.hpp"
#include <optional>

namespace ppo::examples::flight {

/** @brief Outcome counts of one validation campaign. */
struct ValidationMetrics final
{
    /** @brief PPO update of the frozen candidate. */
    std::uint64_t update{};
    /** @brief Curriculum difficulty of the campaign; a harder campaign always ranks higher. */
    double difficulty{1.0};
    std::size_t episodes{};
    std::size_t touchdowns{};
    std::size_t stops{};
    std::size_t crashes{};
    std::size_t timeouts{};
    /** @brief Sum of the touchdown distances past the threshold, in metres. */
    double touchdownSumMeters{};
    /** @brief Sum of the stop distances past the threshold, in metres. */
    double stopSumMeters{};

    /** @brief Stop rate, in [0, 1]. */
    double SuccessRate() const noexcept
    {
        if (episodes == 0)
        {
            return 0;
        }
        return static_cast<double>(stops) / static_cast<double>(episodes);
    }
    /** @brief Stop rate in percent, as stored in the manifest. */
    double Score() const noexcept
    {
        return 100 * SuccessRate();
    }
    /** @brief Mean stop distance past the threshold, in metres. */
    double MeanStop() const noexcept
    {
        if (stops == 0)
        {
            return 0;
        }
        return stopSumMeters / static_cast<double>(stops);
    }
    /** @brief Adds one finished episode; throws for a Flying mission. */
    void Record(MissionState const& mission);
};

/**
 * @brief Validation gate: a complete, consistent campaign that reaches the minimum success rate.
 * @param config Evaluation budget, success threshold and runway length.
 * @param metrics Campaign to check.
 * @return True when the campaign may be promoted.
 */
bool Validated(Config const& config, ValidationMetrics const& metrics) noexcept;
/**
 * @brief Ranks two validated campaigns for the BEST generation.
 * @param candidate Newly validated campaign.
 * @param incumbent Current BEST campaign.
 * @return True for a harder campaign; at equal difficulty, for a higher stop
 * rate, then for a shorter mean stop distance.
 */
bool Better(ValidationMetrics const& candidate, ValidationMetrics const& incumbent) noexcept;

/** @brief Mutable training state and immutable evaluated BEST generations are separate. */
class FlightModelStore final
{
public:
    FlightModelStore(Config const& config, std::filesystem::path directory);
    /** @brief Replace reward metadata used to qualify future BEST records. */
    void SetRewardContract(RewardSettings const& settings, std::string const& signature);
    /** @brief Path of the latest training checkpoint. */
    std::filesystem::path Latest() const;
    /** @brief Curriculum level saved beside the latest checkpoint. */
    std::filesystem::path CurriculumLevel() const;
    /**
     * @brief Metrics of the current BEST generation.
     * @param allowTrainingChanges Accept a manifest whose contract only differs in warm-startable settings.
     * @return Metrics, or nothing when there is no BEST for the exact current contract and protocol.
     * @throws std::runtime_error for an incompatible or corrupt manifest.
     */
    std::optional<ValidationMetrics> BestMetrics(bool allowTrainingChanges = false) const;
    /** @brief Loads the BEST generation into a learner; throws when there is none. */
    void LoadBest(FlightLearner& learner) const;
    /**
     * @brief Saves the learner as a new BEST generation when it passes the gate and beats the incumbent.
     * @return True when the manifest now points to the candidate.
     */
    bool Promote(FlightLearner const& learner, ValidationMetrics const& metrics);

private:
    Config config_;
    std::filesystem::path directory_;
};

} // namespace ppo::examples::flight
