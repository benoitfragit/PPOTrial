#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <string_view>

namespace ppo::examples::twovstwo {

/** @brief Half extents of the rectangular simulation world, in metres. */
struct WorldBounds final
{
    float halfNorth{};
    float halfEast{};
};

/** @brief Inclusive floating-point interval used by spawn configuration. */
struct SpawnRange final
{
    float minimum{};
    float maximum{};
};

/** @brief Configurable geometry of the two red attacker spawns. */
struct RedSpawnSettings final
{
    /** @brief Scale relative to the regular rectangular world boundary. */
    SpawnRange worldScale{1.05F, 1.35F};
    /** @brief Bearing around the direction opposite the defended objective. */
    SpawnRange bearingRadians{-1.5707963267948966F, 1.5707963267948966F};
    /** @brief Required initial Euclidean separation between attackers. */
    float minimumSeparation{10000.F};
};

/** @brief Configurable position and activation of the restricted zone. */
struct RestrictedZoneSpawnSettings final
{
    /** @brief Probability that the zone is active in an episode. */
    float enabledProbability{0.75F};
    /** @brief North coordinate divided by the world's north half extent. */
    SpawnRange northFraction{-0.45F, 0.45F};
    /** @brief Side-relative east coordinate divided by the east half extent. */
    SpawnRange eastFraction{0.15F, 0.65F};
};

/**
 * @brief Training-only distribution of hostile activation schedules.
 *
 * The two count probabilities leave the remaining probability to two
 * attackers. Delays are expressed in ECS team steps and do not alter either
 * red RLMission controller.
 */
struct ThreatCurriculumSettings final
{
    /** @brief Enables variable threat counts and delayed activations. */
    bool enabled{false};
    /** @brief Probability that an episode contains no hostile activation. */
    float noThreatProbability{0.20F};
    /** @brief Probability that exactly one hostile is activated. */
    float singleThreatProbability{0.35F};
    /** @brief Probability that the first hostile activation is delayed. */
    float delayedFirstThreatProbability{0.35F};
    /** @brief Probability that a second hostile appears after the first. */
    float delayedSecondThreatProbability{0.70F};
    /** @brief Inclusive delay range for the first hostile. */
    std::array<std::size_t, 2> firstDelaySteps{256U, 768U};
    /** @brief Inclusive additional delay range for the second hostile. */
    std::array<std::size_t, 2> secondDelaySteps{384U, 1024U};
};

/** @brief Complete scenario-level spawn configuration. */
struct SpawnSettings final
{
    /** @brief Probability that red_2 starts destroyed while remaining present. */
    float red2DestroyedProbability{0.15F};
    RedSpawnSettings red{};
    RestrictedZoneSpawnSettings restrictedZone{};
    ThreatCurriculumSettings threatCurriculum{};
};

/** @brief Initial physical state of one scenario entity. */
struct Spawn final
{
    float north{};
    float east{};
    float heading{};
    float speed{};
    bool destroyed{};
};

/**
 * @brief Coordinated random layout generated exactly once per episode.
 *
 * Keeping both attackers in one layout makes their minimum separation a hard
 * invariant instead of an incidental result of independent entity resets.
 */
struct EpisodeSpawnLayout final
{
    /** @brief Sentinel used for an attacker disabled for the whole episode. */
    static constexpr std::size_t NeverActivate =
        std::numeric_limits<std::size_t>::max();

    int side{1};
    std::array<Spawn, 2> red{};
    /** @brief ECS step at which each red becomes alive, or NeverActivate. */
    std::array<std::size_t, 2> redActivationSteps{0U, 0U};
    Spawn restrictedZone{};
};

/**
 * @brief Reports whether a non-activated hostile is scheduled for later.
 * @param layout Current episode layout and activation schedule.
 * @param elapsedSteps Number of already resolved ECS team steps.
 * @return True while at least one finite activation step remains in the future.
 */
inline bool HasPendingThreats(EpisodeSpawnLayout const& layout,
                              std::size_t elapsedSteps) noexcept
{
    return std::any_of(
        layout.redActivationSteps.begin(), layout.redActivationSteps.end(),
        [elapsedSteps](std::size_t activationStep) {
            return activationStep != EpisodeSpawnLayout::NeverActivate
                && activationStep > elapsedSteps;
        });
}

/**
 * @brief Validates all probabilities, ranges and front-arc constraints.
 * @param settings Candidate scenario spawn settings.
 * @return True when the settings are finite and internally consistent.
 */
bool Valid(SpawnSettings const& settings) noexcept;

/**
 * @brief Generates the coordinated random portion of one episode layout.
 * @param side Objective side of the east/west layout; normalized to -1 or +1.
 * @param settings Validated attacker and restricted-zone settings.
 * @param world Rectangular world half extents.
 * @param rng Episode random-number generator.
 * @return Generated layout, or std::nullopt for invalid/impossible geometry.
 */
std::optional<EpisodeSpawnLayout> GenerateEpisodeSpawnLayout(
    int side, SpawnSettings const& settings, WorldBounds const& world,
    std::mt19937& rng);

/**
 * @brief Builds the initial pose of one TwoVsTwo entity.
 * @param name Stable scenario entity name.
 * @param layout Coordinated layout generated for the current episode.
 * @param world Rectangular world half extents.
 * @param rng Episode random-number generator used for blue spawn jitter.
 * @return Spawn pose, or std::nullopt for invalid input or an unknown name.
 */
std::optional<Spawn> MakeSpawn(std::string_view name,
                               EpisodeSpawnLayout const& layout,
                               WorldBounds const& world,
                               std::mt19937& rng);

} // namespace ppo::examples::twovstwo
