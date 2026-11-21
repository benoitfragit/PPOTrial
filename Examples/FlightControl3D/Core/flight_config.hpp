/**
 * @file flight_config.hpp
 * @brief Strict, versioned scenario configuration of FlightControl3D.
 *
 * Role in the chain: the scenario JSON (resources/settings/Scenarios/FlightControl3D_*.json)
 * is parsed once by LoadConfig into a Config value that every stage reads: the simulator
 * (time step, aircraft), the mission (limits), the rewards (weights), the spawns (start
 * envelope), the learner (PPO hyperparameters) and the session (evaluation, diagnostics).
 *
 * Not here: no stage mutates Config, except the session, which replaces reward weights at
 * a PPO rollout boundary (see flight_reward_tuning.hpp). The JSON schema is part of the
 * checkpoint contract: field names, order of aggregate initialization and validation
 * rules must not change without a new contract version.
 *
 * Units are encoded in the field names (Meters, Mps, Degrees, Seconds); fields without a
 * suffix are documented individually.
 *
 * Read next: flight_config.cpp (parsing and validation), flight_types.hpp.
 */
#pragma once

#include "flight_aircraft_performance.hpp"
#include "flight_types.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace ppo::examples::flight {

/**
 * @brief Static runway pose and dimensions; the origin is the threshold centre on the deck surface.
 *
 * Read from the "runway" JSON section; origin and headingDegrees are then redrawn for each
 * episode by SampleRunway.
 */
struct Runway final
{
    /** @brief Geodetic latitude of the local tangent frame origin, in degrees, |value| < 85. */
    double latitudeDegrees{};
    /** @brief Geodetic longitude of the local tangent frame origin, in degrees, in [-180, 180]. */
    double longitudeDegrees{};
    /** @brief Height of the local frame origin above the WGS84 ellipsoid, in metres. */
    double elevationMeters{};
    /** @brief Runway true heading, clockwise from north, in degrees, in [0, 360). */
    double headingDegrees{};
    /** @brief Runway length from the threshold to the far end, in metres. */
    double lengthMeters{};
    /** @brief Runway width, in metres; smaller than the length. */
    double widthMeters{};
    /** @brief Height at which the glide plane crosses the threshold, in metres. */
    double thresholdHeightMeters{};
    /** @brief Extra landable surface around each runway edge, in metres. */
    double landingMarginMeters{};
    /** @brief Threshold centre on the deck in the world frame, in metres (set per episode). */
    Vec3 origin{};
    /**
     * @brief World velocity of the deck, in metres per second.
     *
     * Zero for a static deck; a moving-deck stage updates it with the origin at every
     * physics substep, see MovedDeck.
     */
    Vec3 velocity{};
};

/** @brief Closed interval [minimum, maximum], read from a two-element JSON array. */
struct Range final
{
    double minimum{};
    double maximum{};
};

/** @brief Per-episode runway pose randomization ("runwayRandomization" JSON section). */
struct RunwayRandomization final
{
    /** @brief Uniform north offset of the threshold, in metres. */
    Range northMeters{};
    /** @brief Uniform east offset of the threshold, in metres. */
    Range eastMeters{};
    /** @brief Uniform offset added to the runway heading, in degrees, within [-180, 180]. */
    Range headingOffsetDegrees{};
    /** @brief Uniform deck height above the local frame origin, in metres. */
    Range deckHeightMeters{};
    /** @brief Height of the sea surrounding the deck, in metres; below every deck height. */
    double seaHeightMeters{};
};

/** @brief Simulation, phase and landing-limit settings ("mission" JSON section). */
struct MissionSettings final
{
    /** @brief JSBSim integration step, in seconds, in [0.001, 0.02]. */
    double physicsDt{};
    /** @brief First-order continuous control filter time constant, in seconds, in (0, 10]. */
    double controlFilterTimeConstantSeconds{};
    /** @brief Fraction of the requested rudder command available while airborne, in [0, 1]. */
    double airborneRudderAuthority{1.0};
    /** @brief Episode time limit, in seconds; reaching it truncates the episode. */
    double maxSeconds{};
    /** @brief Horizontal distance from the threshold that truncates the episode, in metres. */
    double maxDistanceMeters{};
    /** @brief Height above the deck that truncates the episode, in metres. */
    double maxAltitudeMeters{};
    /** @brief Physics substeps per policy action (10 substeps of 0.01 s = one 0.1 s action). */
    std::size_t actionSubsteps{};
    /** @brief Glide-plane slope, in degrees, in (0, 10). */
    double glideDegrees{};
    /** @brief Calibrated approach speed far out, in metres per second. */
    double approachSpeed{};
    /** @brief Calibrated speed reached on short final and at the flare, in metres per second. */
    double flareSpeed{};
    /** @brief Height below which the commanded descent blends into the contact sink, in metres. */
    double flareHeightMeters{};
    /** @brief Maximum safe sink rate at touchdown, in metres per second. */
    double maxTouchdownSink{};
    /** @brief Maximum safe ground speed at touchdown, in metres per second. */
    double maxTouchdownSpeed{};
    /** @brief Maximum absolute deck-relative bank at touchdown, in degrees. */
    double maxTouchdownRollDegrees{};
    /** @brief Maximum deck-relative pitch at touchdown, in degrees. */
    double maxTouchdownPitchDegrees{};
    /** @brief Minimum deck-relative pitch at touchdown, in degrees. */
    double minTouchdownPitchDegrees{};
    /** @brief Maximum absolute heading error relative to the runway at touchdown, in degrees. */
    double maxTouchdownHeadingDegrees{};
    /** @brief Minimum normalized gear extension accepted at contact, in (0, 1]. */
    double gearDownMinimum{};
    /** @brief Contact height of the geometric fallback detector (no wheel data), in metres. */
    double contactHeightMeters{};
    /** @brief Ground speed below which the aircraft counts as stopped, in metres per second. */
    double stoppedSpeed{};
    /** @brief Time to remain stopped on all wheels before the Stop outcome, in seconds. */
    double stoppedHoldSeconds{};
};

/** @brief Already-landed start used to practise braking on the runway. */
struct RolloutStartSettings final
{
    /** @brief Initial ground speed, in metres per second. */
    Range speedMps{};
    /** @brief Initial distance past the threshold, in metres. */
    Range alongMeters{};
    /** @brief Initial lateral offset, in metres, within a quarter of the runway width. */
    Range crossMeters{};
};

/** @brief Airborne start envelope around the 3-degree approach plane, relative to the runway. */
struct InitialSettings final
{
    /** @brief Start distance from the threshold at difficulty 1, in metres. */
    Range distanceMeters{};
    /** @brief Bearing of the start from the extended centreline at difficulty 1, in degrees. */
    Range bearingDegrees{};
    /** @brief Heading offset from the runway heading at difficulty 1, in degrees. */
    Range headingOffsetDegrees{};
    /** @brief Uniform height offset from the approach plane, in metres. */
    double heightOffsetMeters{};
    /** @brief Uniform airspeed offset around the approach speed, in metres per second. */
    double speedToleranceMps{};
    /** @brief Uniform initial bank amplitude, in degrees. */
    double rollDegrees{};
    /** @brief Uniform flight-path angle offset around the glide slope, in degrees. */
    double slopeToleranceDegrees{};
    /** @brief Initial pitch attitude, in degrees. */
    Range pitchDegrees{};
    /** @brief Envelope of the already-landed starts. */
    RolloutStartSettings rolloutStart{};
    /**
     * @brief Trimmed 3-degree approach controls: the initial airborne commands
     * and the controls a zero actor output requests.
     */
    double throttle{};
    double elevatorTrim{};
    /** @brief Flap command in flight, in [0, 1]. */
    double approachFlap{};
    /** @brief Flap command on the ground, in [0, 1]. */
    double rolloutFlap{};
    /** @brief Propeller advance command, in [0, 1]. */
    double propeller{};
    /** @brief Mixture command, in [0, 1]. */
    double mixture{1.0};
};

/** @brief Reward weights and the Path shaping scale ("reward" JSON section). */
struct RewardSettings final
{
    /** @brief Weight of each channel, in Reward order; Crash is non-positive, the others non-negative. */
    std::array<double, RewardCount> weights{};
    /** @brief Distance at which the Path potential reaches half its weight, in metres. */
    double pathTolerance{};
};

/** @brief Kind of airborne start of a curriculum stage. */
enum class StageKind : std::uint8_t
{
    /** @brief Direct approach from a wider, higher or farther envelope than the approach levels. */
    Approach,
    /** @brief Downwind abeam the runway, then base and final gates. */
    Pattern,
    /** @brief One racetrack lap around a holding fix on the extended centreline, then the approach. */
    Holding
};

/** @brief Pattern route: downwind end, past the threshold (negative), in metres. */
inline constexpr double PatternDownwindEndAlongMeters = -3000.0;
/** @brief Pattern route: base gate, past the threshold (negative), in metres. */
inline constexpr double PatternBaseAlongMeters = -6000.0;
/** @brief Pattern route: lateral offset of the base gate as a fraction of the downwind offset. */
inline constexpr double PatternBaseOffsetFraction = 0.6;
/**
 * @brief Pattern route: final gate on the extended centreline, past the threshold (negative), in metres.
 *
 * Far enough out for the approach to remove the residual lateral offset before the threshold.
 */
inline constexpr double PatternFinalAlongMeters = -7000.0;

/**
 * @brief Deck motion during the episodes of a stage (optional "deckMotion" object of a stage).
 *
 * The deck moves forward along the runway heading at a constant speed, like a ship under
 * way, and heaves vertically in a sinusoidal swell; it stays level. All zero is a static deck.
 */
struct DeckMotionSettings final
{
    /** @brief Forward speed of the deck along the runway heading, in metres per second. */
    double speedMps{};
    /** @brief Amplitude of the vertical swell heave, in metres. */
    double swellAmplitudeMeters{};
    /** @brief Period of the swell heave, in seconds; unused without a swell amplitude. */
    double swellPeriodSeconds{};

    /** @brief True when the deck translates or heaves. */
    bool Moving() const noexcept
    {
        return speedMps > 0 || swellAmplitudeMeters > 0;
    }
};

/**
 * @brief One curriculum stage beyond the approach levels ("curriculum.stages" JSON array).
 *
 * Fields that a kind does not use are left at zero and absent from its JSON object.
 */
struct StageSettings final
{
    /** @brief Log and display name. */
    std::string name;
    StageKind kind{StageKind::Approach};
    /** @brief Approach: start distance from the threshold, in metres. */
    Range distanceMeters{};
    /** @brief Approach: bearing of the start from the extended centreline, in degrees. */
    Range bearingDegrees{};
    /** @brief Pattern and holding: start distance past the threshold, in metres (negative before it). */
    Range startAlongMeters{};
    /** @brief Pattern: downwind offset from the centreline; holding: start offset; in metres. */
    Range lateralMeters{};
    /** @brief Heading offset from the nominal heading of the start, in degrees. */
    Range headingOffsetDegrees{};
    /** @brief Approach: height above the glide plane; pattern and holding: height above the deck; in metres. */
    Range heightMeters{};
    /** @brief Holding: fix position on the extended centreline, past the threshold (negative), in metres. */
    double fixAlongMeters{};
    /** @brief Holding: length of the outbound leg, in metres. */
    double outboundMeters{};
    /** @brief Holding: distance between the inbound and outbound legs, in metres. */
    double holdingWidthMeters{};
    /** @brief Share of the stage starts with the gear retracted, in [0, 1]. */
    double gearUpShare{};
    /** @brief Deck motion of the stage episodes; a static deck when the stage omits "deckMotion". */
    DeckMotionSettings deckMotion{};
};

/**
 * @brief Adaptive start-difficulty curriculum used only by training episodes.
 *
 * The scale runs over the approach levels, from a short aligned final to the full
 * configured start envelope, then over the stages. Difficulty 1 is the last stage.
 * Validations evaluate the mix up to the current level.
 */
struct CurriculumSettings final
{
    /** @brief Start distance range at difficulty 0, in metres. */
    Range firstDistanceMeters{};
    /** @brief Number of approach levels, from the short aligned final to the full start envelope. */
    std::size_t levels{1};
    /** @brief Stop rate over the window that promotes to the next level. */
    double promotionSuccessRate{};
    /** @brief Airborne training episodes per promotion decision. */
    std::size_t windowEpisodes{1};
    /** @brief Share of training episodes that start already landed, to practise braking. */
    double rolloutStartShare{};
    /** @brief Stages after the approach levels, in order of difficulty. */
    std::vector<StageSettings> stages;
};

/** @brief Total curriculum levels: the approach levels followed by one level per stage. */
inline std::size_t CurriculumLevels(CurriculumSettings const& settings) noexcept
{
    return settings.levels + settings.stages.size();
}

/** @brief PPO and checkpoint settings ("training" JSON section). */
struct TrainingSettings final
{
    /** @brief Seed of the learner RNG and base of the training environment seeds. */
    std::uint32_t seed{};
    /** @brief Draw a fresh environment seed for each training session when enabled. */
    bool randomizeEnvironmentSeed{};
    /** @brief Transitions collected per PPO update. */
    std::size_t rolloutSteps{};
    /** @brief Transitions per minibatch; divides rolloutSteps. */
    std::size_t minibatch{};
    /** @brief Passes over the rollout per update. */
    std::size_t epochs{};
    /** @brief Total (or additional, see TrainingBudget) PPO update budget. */
    std::size_t maxUpdates{};
    /** @brief Updates between two saves of the latest checkpoint. */
    std::size_t checkpointEvery{};
    /** @brief Width of each of the two hidden layers: 32, 64, 128 or 256. */
    std::size_t hiddenWidth{};
    /** @brief Adam learning rate of the actor. */
    double actorRate{};
    /** @brief Adam learning rate of the critic. */
    double criticRate{};
    /** @brief Discount factor per action (0.1 s), in (0, 1]. */
    double gamma{};
    /** @brief GAE lambda, in [0, 1]. */
    double lambda{};
    /** @brief PPO ratio clip epsilon, in (0, 1). */
    double clip{};
    /** @brief Entropy bonus coefficient. */
    double entropy{};
    /** @brief Target for the sampled old-to-new minibatch KL; skip above 1.5 times this value. */
    double targetKl{0.02};
    double adamBeta1{};
    double adamBeta2{};
    double adamEpsilon{};
    /** @brief Resume from the latest checkpoint when it exists. */
    bool resume{};
    /** @brief Relative checkpoint directory, without "..". */
    std::string checkpointDirectory;
    CurriculumSettings curriculum{};
};

/** @brief Optional bounded telemetry used to diagnose failed convergence. */
struct DiagnosticsSettings final
{
    bool enabled{};
    /** @brief Updates between two diagnostic log lines. */
    std::size_t logEveryUpdates{1};
    /** @brief Fraction of the bound above which a value counts as saturated, in [0.5, 1]. */
    double saturationThreshold{0.98};
};

/** @brief Desktop execution controls; headless execution is always unthrottled. */
struct RuntimeSettings final
{
    bool maxSpeed{};
};

/** @brief Independent validation campaigns ("evaluation" JSON section). */
struct EvaluationSettings final
{
    /** @brief Seed of the first validation episode; episode i uses seed + i. */
    std::uint32_t seed{};
    /** @brief Episodes per validation campaign. */
    std::size_t episodes{};
    /** @brief Updates between two validation campaigns. */
    std::size_t everyUpdates{};
    /** @brief Stop rate required to qualify as BEST, in [0, 1]. */
    double minimumSuccessRate{};
};

/** @brief Complete validated scenario. */
struct Config final
{
    Runway runway{};
    RunwayRandomization runwayRandomization{};
    MissionSettings mission{};
    InitialSettings initial{};
    RewardSettings reward{};
    TrainingSettings training{};
    EvaluationSettings evaluation{};
    DiagnosticsSettings diagnostics{};
    RuntimeSettings runtime{};
    /** @brief JSBSim aircraft model name. */
    std::string aircraft;
    /** @brief Model-derived performance of the aircraft, resolved from its name when loading. */
    AircraftPerformance performance{};
    /** @brief Canonical JSON of every task-defining setting, stored in checkpoints and BEST manifests. */
    std::string checkpointSignature;

    /** @brief Duration of one policy action, in seconds (physicsDt * actionSubsteps). */
    double StepSeconds() const noexcept
    {
        return mission.physicsDt * static_cast<double>(mission.actionSubsteps);
    }
};

/**
 * @brief Loads and validates a scenario.
 *
 * Strict versioned JSON: missing, unknown and non-finite fields are errors.
 * @param path Scenario JSON file.
 * @return Validated configuration with its checkpoint signature.
 * @throws std::invalid_argument for an invalid document, std::exception for unreadable JSON.
 */
Config LoadConfig(std::filesystem::path const& path);

/**
 * @brief Checks the cross-field invariants of a configuration.
 * @param config Configuration to validate.
 * @throws std::invalid_argument naming the first violated rule.
 */
void Validate(Config const& config);

/**
 * @brief Checks whether a stored learner can warm-start the current training contract.
 * @param stored Serialized checkpoint signature from the stored learner.
 * @param current Serialized checkpoint signature for the requested session.
 * @return True when only reward weights or start ranges differ; BEST still requires an exact contract.
 */
bool CompatibleTrainingSignature(std::string const& stored, std::string const& current);

} // namespace ppo::examples::flight
