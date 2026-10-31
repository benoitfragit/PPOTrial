/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Runtime/generalized_patrol_team_agent.hpp
 * @brief Contrôleur ECS MAPPO partagé des Bleus : états, observations, actions, récompenses, évaluations et checkpoints.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "generalized_evaluation_campaign.hpp"
#include "generalized_runtime_state.hpp"
#include "generalized_cooperative_reward.hpp"
#include "generalized_reward_shaping.hpp"
#include "patrol_training_settings.hpp"
#include "baseentity.hpp"
#include "iagent.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>

namespace ppo::examples::bluedefense::defenders {

/** @brief Déclare le type GeneralizedPatrolLearner, utilisé par Contrôleur ECS MAPPO partagé des Bleus : états, observations, actions, récompenses, évaluations et checkpoints. */
class GeneralizedPatrolLearner;
/** @brief Déclare l'acteur de production déterministe sans apprentissage. */
class GeneralizedPatrolProductionPolicy;

/** @brief Selects the immutable execution contract of the blue controller. */
enum class GeneralizedPatrolExecutionMode : std::uint8_t
{
    Training,
    ProductionInference
};

/**
 * @brief Cooperative generalized MAPPO controller for P blue ECS mobiles.
 *
 * In training mode, one controller samples all P actions before physics and
 * records exactly P same-instant transitions after world resolution. In
 * production-inference mode it evaluates only the promoted deterministic
 * actor. Red mobiles remain owned exclusively by their historical RLMission
 * agents.
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
     * @param training Embedded learner settings.
     * @param executionMode Training or promoted-production inference.
     * @param actorCheckpoint New generalized v8 actor path.
     * @param trainingCheckpoint New generalized v5 full-state path.
     * @param productionCheckpoint Separate promoted generalized v8 actor path.
     * @param legacyTrainingV4 Read-only historical v4 migration source.
     * @param legacyLatestV7 Read-only historical v7 migration source.
     * @param legacyProductionV7 Read-only fallback v7 migration source.
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
        PatrolTrainingSettings training,
        GeneralizedPatrolExecutionMode executionMode,
        std::filesystem::path actorCheckpoint,
        std::filesystem::path trainingCheckpoint,
        std::filesystem::path productionCheckpoint,
        std::filesystem::path legacyTrainingV4,
        std::filesystem::path legacyLatestV7,
        std::filesystem::path legacyProductionV7);
    ~GeneralizedPatrolMappoAgent() override;

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
 * @brief Traite les entrées et applique les effets associés à ProcessCommand.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void ProcessCommand(
        ppo::model::sAgentRealTimeCommand const& command) noexcept override;
/**
 * @brief Retourne une vue ou une valeur calculée depuis l'état courant.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
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
                   TeamCapacity::MaximumBlueMobiles>& observations) const;
/**
 * @brief Construit, transforme ou échantillonne les données produites par BuildEvaluationSample.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    GeneralizedEvaluationSample BuildEvaluationSample(
        Parameters const& parameters, GeneralizedRuntimeState const& state,
        bool missionFailed, bool timeout, bool patrolCompleted) const;
/**
 * @brief Sérialise ou écrit les données associées à SaveAtUpdateBoundary selon les garanties implémentées.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void SaveAtUpdateBoundary(bool force = false) noexcept;
/**
 * @brief Change ou contrôle l'étape de cycle de vie.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void StartEvaluation();
/**
 * @brief Réalise l'opération PrepareEvaluationCase; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void PrepareEvaluationCase();
/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void RestoreTrainingInitializer() noexcept;
/**
 * @brief Réalise l'opération PromoteEvaluatedActor; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void PromoteEvaluatedActor() noexcept;
/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void LoadPromotedScore() noexcept;
/**
 * @brief Sérialise ou écrit les données associées à SavePromotedScore selon les garanties implémentées.
 * @details Consulte ou modifie des composants ECS. Utilise des fichiers ou chemins.
 */
    void SavePromotedScore();

    entt::registry* registry_{};
    std::array<entt::entity, TeamCapacity::MaximumBlueMobiles> guardIds_{};
    std::array<entt::entity, TeamCapacity::MaximumRedMobiles> attackerIds_{};
    entt::entity objectiveId_{entt::null};
    ActiveTeamCounts counts_{};
    Parameters parameters_{};
    PatrolTrainingSettings training_{};
    /** @brief Immutable training or production-inference execution mode. */
    GeneralizedPatrolExecutionMode executionMode_{
        GeneralizedPatrolExecutionMode::Training};
    /** @brief Live independent weights for the generalized UI reward channels. */
    GeneralizedRewardValues rewardWeights_{};
    /** @brief Most recent unweighted team reward channels displayed by the IHM. */
    GeneralizedRewardValues lastRewardChannels_{};
    std::filesystem::path actorCheckpoint_{};
    std::filesystem::path trainingCheckpoint_{};
    std::filesystem::path productionCheckpoint_{};
    std::unique_ptr<GeneralizedPatrolLearner> learner_{};
    std::unique_ptr<GeneralizedPatrolProductionPolicy> productionPolicy_{};
    GeneralizedEvaluationCampaign evaluation_;
    std::mt19937 trainingRng_{};

    GeneralizedRuntimeState beforeStep_{};
    std::array<Action, TeamCapacity::MaximumBlueMobiles> lastActions_{};
    std::array<bool, TeamCapacity::MaximumBlueMobiles> beforeAvailability_{};
    std::size_t episodeStep_{};
    std::uint64_t lastSavedUpdate_{};
    unsigned long long id_{ppo::model::GetNewUniqId()};
    float currentReward_{};
    GeneralizedPatrolLapProgress noThreatLap_{};
    bool hasBeforeStep_{};
    bool episodeDone_{};
    bool finished_{};
    bool trainingRngSaved_{};
};

} // namespace ppo::examples::bluedefense::defenders
