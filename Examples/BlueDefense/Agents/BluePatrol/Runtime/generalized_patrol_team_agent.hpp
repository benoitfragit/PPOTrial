/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.hpp
 * @brief The blue team's controller in the ECS world, driven from outside (ppokit-run,
 *        through the blue_environment DLL): it publishes the team's observations, applies
 *        the actions given to it and publishes the step's rewards, outcome and scene.
 * @details The learner, the production actor, the journal, the checkpoints and the
 * evaluation campaigns live outside the world (ppokit_graph_network and ppokit-run).
 */
#pragma once

#include "generalized_evaluation_case.hpp"
#include "generalized_evaluation_metrics.hpp"
#include "generalized_patrol_waypoint_sequencer.hpp"
#include "generalized_runtime_state.hpp"
#include "generalized_cooperative_reward.hpp"
#include "generalized_reward_shaping.hpp"
#include "patrol_training_settings.hpp"
#include "baseentity.hpp"
#include "sharedworldcomponents.hpp"
#include "ppo/runtime/iagent.hpp"

#include <array>
#include <tuple>
#include <cstddef>
#include <cstdint>

namespace ppo::examples::bluedefense::defenders {


/**
 * @brief What a view shows of the team's world at one moment: the state
 *        the observations were built from, the patrol references, the racetrack waypoints.
 */
/** @brief Engagement cone of an armed mobile (sSNAState): no-escape radius and half firing angle. */
struct SceneCone final
{
    float radius{};
    float halfAngle{};
};

/** @brief A blue's pseudo-lidar (sPseudoLidarState): per bin, normalized range and distance. */
struct SceneLidar final
{
    static constexpr std::size_t Bins =
        std::tuple_size_v<decltype(ppo::model::sPseudoLidarState::m_nearestCollisionRanges)>;
    std::array<float, Bins> ranges{};
    std::array<float, Bins> distances{};
    float span{};
    bool valid{};
};

/** @brief A restricted zone: centre, no-escape radius, hostile or not. */
struct SceneZone final
{
    float north{};
    float east{};
    float radius{};
    bool hostile{};
};

struct GeneralizedTeamScene final
{
    /** @brief Restricted zones a scene carries; more are counted, not carried. */
    static constexpr std::size_t ZoneCapacity = 16U;
    /** @brief Role of a blue slot: none, the leader, or teammate slot k (W k). */
    static constexpr int NoRole = -1;
    static constexpr int LeaderRole = 0;

    GeneralizedRuntimeState state{};
    GeneralizedPatrolTargets targets{};
    /** @brief Physical index of the current leader. */
    std::size_t leader{};
    /** @brief The defended area: half the world's bounds. */
    ppo::model::sDefendedArea defendedArea{};
    bool hasDefendedArea{};
    /** @brief The racetrack's waypoints while the team follows them (no red threat). */
    std::array<PatrolCurvePose, GeneralizedPatrolWaypointSequencer::WaypointCount> waypoints{};
    std::size_t activeWaypoint{};
    std::size_t validatedWaypoints{};
    bool followingWaypoints{};
    /** @brief Per blue slot: role (NoRole, LeaderRole, teammate slot + 1), cone and lidar. */
    std::array<int, TeamCapacity::MaximumBlueMobiles> roles{};
    std::array<SceneCone, TeamCapacity::MaximumBlueMobiles> blueCones{};
    std::array<SceneLidar, TeamCapacity::MaximumBlueMobiles> lidars{};
    std::array<SceneCone, TeamCapacity::MaximumRedMobiles> redCones{};
    /** @brief The first ZoneCapacity restricted zones, and how many the world has. */
    std::array<SceneZone, ZoneCapacity> zones{};
    std::size_t zoneCount{};
    /** @brief The outer failure rectangle (half extents). */
    float hardHalfNorth{};
    float hardHalfEast{};
    /** @brief The leader's analytical patrol and the role-ordered references (render data). */
    ppo::model::sPatrolTrackOverlay track{};
    ppo::model::sPatrolReferenceOverlay references{};
};

/**
 * @brief What the team exchanges with a learner outside the world, in
 *        fixed-capacity rows of blue slots (the first P are the episode's blues).
 */
struct GeneralizedTeamExchange final
{
    using Observations = std::array<GeneralizedObservation, TeamCapacity::MaximumBlueMobiles>;
    using Availability = std::array<bool, TeamCapacity::MaximumBlueMobiles>;

    /** @brief The state the team acts on (PrepareExternalStep), and the blues alive in it. */
    Observations observations{};
    Availability available{};
    /** @brief The actions of the step, written by the caller before the world steps. */
    std::array<Action, TeamCapacity::MaximumBlueMobiles> actions{};
    /** @brief After the step: observations and blues alive, as the learner's feedback. */
    Observations next{};
    Availability nextAvailable{};
    /** @brief Each slot's clamped cooperative reward, as the learner receives it. */
    std::array<float, TeamCapacity::MaximumBlueMobiles> rewards{};
    /** @brief Each slot's weighted reward channels (before the clamp). */
    std::array<GeneralizedRewardValues, TeamCapacity::MaximumBlueMobiles> weightedChannels{};
    bool terminated{};
    bool truncated{};
    /** @brief Results of the episode that just ended (valid after a terminated or truncated step). */
    GeneralizedEpisodeEvaluation episode{};
    /** @brief The scene of the latest observations: `observations` after PrepareExternalStep, `next` after a step. */
    GeneralizedTeamScene scene{};
};

/**
 * @brief Cooperative generalized MAPPO controller for P blue ECS mobiles.
 *
 * Each step it publishes the P observations (PrepareExternalStep), applies
 * the P actions written in its exchange before physics, and after world
 * resolution publishes P same-instant rewards, the outcome and the scene. Red
 * mobiles remain owned exclusively by their historical RLMission agents.
 */
class GeneralizedPatrolMappoAgent final : public ppo::model::IAgent
{
public:
    /**
     * @brief Creates the bounded generalized blue team controller.
     * @param guards First @p blueCount entries are blue_1..blue_P.
     * @param blueCount Active blue mobile count P.
     * @param attackers First @p redCount entries are red_1..red_N.
     * @param redCount Active red mobile count N.
     * @param objective Fixed defended objective.
     * @param parameters Patrol, vehicle, and world parameters.
     * @param training Reward, episode and evaluation settings (blue_patrol_training.json).
     */
    GeneralizedPatrolMappoAgent(
        std::array<ppo::model::BaseEntity*,
                   TeamCapacity::MaximumBlueMobiles> const& guards,
        std::size_t blueCount,
        std::array<ppo::model::BaseEntity*,
                   TeamCapacity::MaximumRedMobiles> const& attackers,
        std::size_t redCount,
        ppo::model::BaseEntity& objective,
        Parameters parameters,
        PatrolTrainingSettings training);

    GeneralizedPatrolMappoAgent(GeneralizedPatrolMappoAgent const&) = delete;
    GeneralizedPatrolMappoAgent& operator=(
        GeneralizedPatrolMappoAgent const&) = delete;

/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 * @details Consulte ou modifie des composants ECS.
 */
    void Init() override;
/**
 * @brief Réalise l'opération ChooseAction; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
    void ChooseAction() override;
/**
 * @brief Met à jour ou enregistre les données sous la responsabilité.
 * @details Consulte ou modifie des composants ECS.
 */
    bool SetNewState() override;

    /**
     * @brief After the world began its step (Environment::BeginStep): reads
     *        the team's state and publishes the observations it acts on.
     * @throws The world's errors.
     */
    void PrepareExternalStep();
    /** @brief The exchange: write `actions` between PrepareExternalStep and the step. */
    GeneralizedTeamExchange& Exchange() noexcept { return exchange_; }
    GeneralizedTeamExchange const& Exchange() const noexcept { return exchange_; }
    /**
     * @brief The next episode is an evaluation case (seed, side, racetrack
     *        phase), with the evaluation step limit; the world must reinitialize next.
     * @param test The case.
     * @param caseIndex Its place in the campaign (0 starts a campaign), for the case
     *        numbers of the diagnostic rows ([MAPPO_EVAL_EPISODE], [MAPPO_EVAL_LOSS]).
     */
    void StartExternalEvaluationCase(GeneralizedEvaluationCase const& test, std::size_t caseIndex);
    /** @brief The next episode is a training episode again. */
    void EndExternalEvaluation() noexcept;
    /**
     * @brief Apply a complete weight vector, preserving unchanged configured values above the slider range.
     * @param weights Finite, non-negative weights; changed channels must be in [0, 10].
     */
    void SetRewardWeights(GeneralizedRewardValues const& weights);
    GeneralizedRewardValues const& RewardWeights() const noexcept { return rewardWeights_; }
    /** @brief The episode's P blues and N reds. */
    ActiveTeamCounts Counts() const noexcept { return counts_; }
    /** @brief Steps of the current episode resolved by SetNewState. */
    std::size_t EpisodeStep() const noexcept { return episodeStep_; }

private:
    /**
     * @brief Reads the world before the step: state, targets, overlays, the team's
     *        observations and the blues alive (beforeStep_, beforeTargets_, beforeAvailability_).
     */
    void PrepareStep(std::array<GeneralizedObservation, TeamCapacity::MaximumBlueMobiles>& observations);
    /** @brief Applies lastActions_ to the blue controls; the step then awaits SetNewState. */
    void ApplyActions();
    /** @brief The scene of `state` and `targets` into the exchange. */
    void CaptureScene(GeneralizedRuntimeState const& state, GeneralizedPatrolTargets const& targets);
    /** @brief Whether the episode is an evaluation episode (a case of ppokit-run's evaluation suite). */
    bool EvaluationEpisode() const noexcept;
    /**
     * @brief Publishes the step's feedback in the exchange.
     * @param observations, availability Team state after the step.
     * @param rewards Each slot's clamped reward; @param breakdown Their channels.
     * @param sample The episode's evaluation sample, read when the step ended it.
     */
    void PublishExternalStep(
        std::array<GeneralizedObservation, TeamCapacity::MaximumBlueMobiles> const& observations,
        std::array<bool, TeamCapacity::MaximumBlueMobiles> const& availability,
        std::array<float, TeamCapacity::MaximumBlueMobiles> const& rewards,
        GeneralizedCooperativeRewards const& breakdown, bool terminated, bool truncated,
        GeneralizedEvaluationSample const& sample);

/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    Parameters ReadParameters() const;
/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    GeneralizedRuntimeState ReadState(Parameters const& parameters) const;
/**
 * @brief Réalise l'opération Roles; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    BlueTeamRoles const& Roles() const;
/**
 * @brief Construit, transforme ou échantillonne les données produites par BuildObservations.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void BuildObservations(
        Parameters const& parameters, GeneralizedRuntimeState const& state,
        std::array<GeneralizedObservation,
                   TeamCapacity::MaximumBlueMobiles>& observations,
        GeneralizedPatrolTargets const& targets) const;
/**
 * @brief Construit, transforme ou échantillonne les données produites par BuildEvaluationSample.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    GeneralizedEvaluationSample BuildEvaluationSample(
        Parameters const& parameters, GeneralizedRuntimeState const& state,
        bool missionFailed, bool timeout, bool patrolCompleted,
        GeneralizedPatrolTargets const& targets) const;
    /** @brief The scene's render data read from the world: roles, cones, lidars, zones, track, references. */
    void CaptureSceneOverlays(GeneralizedTeamScene& scene) const;
    /** @brief Where an evaluation episode stands: campaign, case, seed. */
    struct EvaluationCaseLabel final
    {
        std::size_t campaign;
        std::size_t caseNumber;
        std::uint32_t seed;
    };
    /** @brief The label of the current evaluation case. */
    EvaluationCaseLabel ExternalCaseLabel() const noexcept;
    /** @brief Stable terminal label of an evaluation episode. */
    char const* EvaluationTerminal(GeneralizedRuntimeState const& after, bool noThreatRemains,
        bool patrolCompleted, bool truncated) const noexcept;
    /** @brief One [MAPPO_EVAL_LOSS] row per blue lost during this step. */
    void WriteEvaluationLosses(EvaluationCaseLabel const& label,
        GeneralizedRuntimeState const& after) const;
    /** @brief The [MAPPO_EVAL_EPISODE] row of an evaluation episode that just ended. */
    void WriteEvaluationEpisode(EvaluationCaseLabel const& label,
        GeneralizedRuntimeState const& after, char const* terminal,
        GeneralizedEvaluationSample const& sample) const;
    /**
     * @brief Writes one bounded episode geometry and waypoint diagnostic row.
     * @param parameters Current patrol geometry and vehicle limits.
     * @param state Final physical state of the episode.
     * @param targets Final role-ordered measurement and guidance references.
     * @param phase Stable training or evaluation label.
     * @param terminationCause Stable success, timeout or failure label.
     */
    void WritePatrolDiagnostics(Parameters const& parameters,
        GeneralizedRuntimeState const& state,
        GeneralizedPatrolTargets const& targets, char const* phase,
        char const* terminationCause) const;
    /**
     * @brief Logs one RALLY/FOLLOW or waypoint event without per-step output.
     * @param parameters Current geometry and simulated step duration.
     * @param update State-machine event after both progress validators ran.
     */
    void WritePatrolProgress(Parameters const& parameters,
        GeneralizedWaypointUpdate update) const;
    /** @brief Ends an evaluation case's reset overrides: the next reset is a training episode. */
    void RestoreTrainingInitializer() noexcept;

    entt::registry* registry_{};
    std::array<entt::entity, TeamCapacity::MaximumBlueMobiles> guardIds_{};
    std::array<entt::entity, TeamCapacity::MaximumRedMobiles> attackerIds_{};
    entt::entity objectiveId_{entt::null};
    ActiveTeamCounts counts_{};
    Parameters parameters_{};
    PatrolTrainingSettings training_{};
    /** @brief Weights of the reward channels (ppokit-run sets them through the DLL). */
    GeneralizedRewardValues rewardWeights_{};

    GeneralizedRuntimeState beforeStep_{};
    GeneralizedPatrolTargets beforeTargets_{};
    std::array<Action, TeamCapacity::MaximumBlueMobiles> lastActions_{};
    std::array<bool, TeamCapacity::MaximumBlueMobiles> beforeAvailability_{};
    std::size_t episodeStep_{};
    /** @brief Process-local episode identifier used only to group diagnostics. */
    std::uint64_t diagnosticEpisode_{};
    /** @brief Resolved step at RALLY/FOLLOW acquisition; zero until acquired. */
    std::size_t followTransitionStep_{};
    /** @brief Leader-owned, fixed-capacity RALLY/FOLLOW waypoint state. */
    GeneralizedPatrolWaypointSequencer waypointSequencer_{};
    GeneralizedPatrolLapProgress noThreatLap_{};
    std::array<GeneralizedRewardValues, 2U> roleRewardSums_{};
    std::array<std::size_t, 2U> roleRewardSteps_{};
    /** @brief The team's exchange with the learner outside the world. */
    GeneralizedTeamExchange exchange_{};
    /** @brief PrepareExternalStep ran for the step about to be taken. */
    bool externalPrepared_{};
    /** @brief The current episode is an evaluation case. */
    bool externalEvaluation_{};
    /** @brief The External case: its campaign (numbered in the process), place and case. */
    std::size_t externalCampaign_{};
    std::size_t externalCaseIndex_{};
    GeneralizedEvaluationCase externalCase_{};
    bool hasBeforeStep_{};
    bool episodeDone_{};
};

} // namespace ppo::examples::bluedefense::defenders
