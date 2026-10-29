#pragma once

#include "team_capacity.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

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

/**
 * @brief Generalized bounded red-team spawn and activation contract.
 *
 * `activeCount` is the number of red entities created for the episode. Each
 * entry receives either activation step zero (initially active) or one finite
 * positive delayed activation step. No generalized entry uses
 * EpisodeSpawnLayout::NeverActivate; that sentinel remains available only to
 * the archived 2-vs-2 curriculum.
 */
struct RedTeamSpawnSettings final
{
    /** @brief Hard CPU bound accepted for candidates tested per red mobile. */
    static constexpr std::size_t MaximumPlacementAttempts{4096U};
    /** @brief Number of red mobiles, in [0, TeamCapacity::MaximumRedMobiles]. */
    std::size_t activeCount{2U};
    /** @brief Maximum random candidates tested for each red mobile. */
    std::size_t maximumPlacementAttempts{512U};
    /** @brief Independent probability that a red activation is delayed. */
    float delayedActivationProbability{0.F};
    /** @brief Inclusive positive delay range, in ECS team steps. */
    std::array<std::size_t, 2> activationDelaySteps{256U, 768U};
};

/**
 * @brief Additive generalized blue-team contract.
 *
 * The optional contract only selects the bounded number of blue mobiles.
 * Leader identity and formation slots are sampled reproducibly by the episode
 * initializer. When the contract is absent, the archived two-guard reset and
 * its random-number sequence remain unchanged.
 */
struct BlueTeamSpawnSettings final
{
    /** @brief Number of blue mobiles, in [1, TeamCapacity::MaximumBlueMobiles]. */
    std::size_t activeCount{2U};
};

/**
 * @brief Historical friendly zone configuration consumed by red RLMission.
 *
 * Fractions and RNG sampling remain unchanged to preserve red behaviour.
 */
struct FriendlyRestrictedZoneSpawnSettings final
{
    /** @brief Probability that the friendly zone is active in an episode. */
    float enabledProbability{0.75F};
    /** @brief North coordinate divided by the world's north half extent. */
    SpawnRange northFraction{-0.45F, 0.45F};
    /** @brief Side-relative east coordinate divided by the east half extent. */
    SpawnRange eastFraction{0.15F, 0.65F};
};

/** @brief Supported mutually exclusive restricted-zone placement modes. */
enum class RestrictedZonePositionMode
{
    /** @brief Use the exact configured north/east coordinates. */
    Fixed,
    /** @brief Sample north/east independently from configured ranges. */
    Random
};

/** @brief Typed configuration of one hostile circular restricted zone. */
struct RestrictedZoneSpawnSettings final
{
    /** @brief Stable scenario identifier, unique inside the collection. */
    std::string id{};
    /** @brief RestrictedZone platform profile used to obtain its radii. */
    std::string platform{};
    /** @brief Static per-zone switch applied before probability sampling. */
    bool enabled{true};
    /** @brief Probability that the zone is active in an episode. */
    float enabledProbability{1.F};
    /** @brief Selected fixed or random placement contract. */
    RestrictedZonePositionMode positionMode{RestrictedZonePositionMode::Fixed};
    /** @brief Fixed north/east position in metres. */
    std::array<float, 2> fixedPosition{};
    /** @brief Random north interval in metres. */
    SpawnRange randomNorth{};
    /** @brief Random east interval in metres. */
    SpawnRange randomEast{};
    /** @brief Mirrors the east magnitude onto the side opposite the objective. */
    bool oppositeObjective{false};
};

/** @brief Global switch and bounded-at-load collection of hostile zones. */
struct RestrictedZonesSpawnSettings final
{
    /** @brief Global collection switch evaluated before per-zone settings. */
    bool enabled{true};
    /** @brief Validated zone definitions in deterministic JSON order. */
    std::vector<RestrictedZoneSpawnSettings> items{};
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
    float noThreatProbability{0.10F};
    /** @brief Probability that exactly one hostile is activated. */
    float singleThreatProbability{0.30F};
    /** @brief Probability that the first hostile activation is delayed. */
    float delayedFirstThreatProbability{0.20F};
    /** @brief Probability that a second hostile appears after the first. */
    float delayedSecondThreatProbability{0.45F};
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
    /**
     * @brief Generalized team contract, or null for archived 2-vs-2 sampling.
     *
     * Absence deliberately selects the historical RNG sequence, two red
     * names and threat curriculum without changing their checkpoint path.
     */
    std::optional<RedTeamSpawnSettings> redTeam{};
    /** @brief Generalized blue contract, or null for archived two-guard reset. */
    std::optional<BlueTeamSpawnSettings> blueTeam{};
    /** @brief Historical blue zone retained unchanged for red RLMission. */
    FriendlyRestrictedZoneSpawnSettings friendlyRestrictedZone{};
    /** @brief Independently configured hostile restricted zones. */
    RestrictedZonesSpawnSettings restrictedZones{};
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

/** @brief Episode pose associated with one stable restricted-zone id. */
struct RestrictedZoneEpisodeSpawn final
{
    /** @brief Stable scenario identifier copied from configuration. */
    std::string id{};
    /** @brief Sampled pose and active state for this episode. */
    Spawn spawn{};
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
    /** @brief Red poses in stable red_1..red_N order. */
    std::vector<Spawn> red{};
    /**
     * @brief Activation step for each red pose at the same index.
     *
     * At reset, `red[i].destroyed` is true exactly when this value is not
     * zero. Generalized teams use finite delayed steps; NeverActivate is
     * reserved for the archived curriculum.
     */
    std::vector<std::size_t> redActivationSteps{};
    /** @brief Historical friendly zone pose used by the red pseudo-LiDAR. */
    Spawn friendlyRestrictedZone{};
    /** @brief Zone poses sampled once and retained for the whole episode. */
    std::vector<RestrictedZoneEpisodeSpawn> restrictedZones{};
};

/**
 * @brief Reports whether a non-activated hostile is scheduled for later.
 * @param layout Current episode layout and activation schedule.
 * @param elapsedSteps Number of already resolved ECS team steps.
 * @return True while at least one finite delayed activation is still pending.
 *
 * A threat scheduled for the current elapsed step is pending until the next
 * `BeforeStep` call activates it. Step zero denotes an immediately active
 * threat and is therefore never pending.
 */
inline bool HasPendingThreats(EpisodeSpawnLayout const& layout,
                              std::size_t elapsedSteps) noexcept
{
    return std::any_of(
        layout.redActivationSteps.begin(), layout.redActivationSteps.end(),
        [elapsedSteps](std::size_t activationStep) {
            return activationStep != 0U
                && activationStep != EpisodeSpawnLayout::NeverActivate
                && activationStep >= elapsedSteps;
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
 * @param diagnostic Optional destination for validation or placement failure.
 * @return Generated layout, or std::nullopt for invalid/impossible geometry.
 */
std::optional<EpisodeSpawnLayout> GenerateEpisodeSpawnLayout(
    int side, SpawnSettings const& settings, WorldBounds const& world,
    std::mt19937& rng, std::string* diagnostic = nullptr);

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
