#pragma once

#include "generalized_evaluation_campaign.hpp"
#include "generalized_runtime_state.hpp"
#include "generalized_cooperative_reward.hpp"
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

class GeneralizedPatrolLearner;

/**
 * @brief Cooperative generalized MAPPO controller for P blue ECS mobiles.
 *
 * One controller samples all P actions before physics and records exactly P
 * same-instant transitions after world resolution. Red mobiles remain owned
 * exclusively by their historical RLMission agents.
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

    void Init() override;
    void ChooseAction() override;
    bool SetNewState() override;
    void ProcessCommand(
        ppo::model::sAgentRealTimeCommand const& command) noexcept override;
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
    Parameters ReadParameters() const;
    GeneralizedRuntimeState ReadState(Parameters const& parameters) const;
    BlueTeamRoles const& Roles() const;
    void BuildObservations(
        Parameters const& parameters, GeneralizedRuntimeState const& state,
        std::array<GeneralizedObservation,
                   TeamCapacity::MaximumBlueMobiles>& observations) const;
    GeneralizedEvaluationSample BuildEvaluationSample(
        Parameters const& parameters, GeneralizedRuntimeState const& state,
        bool missionFailed, bool timeout) const;
    void SaveAtUpdateBoundary(bool force = false) noexcept;
    void StartEvaluation();
    void PrepareEvaluationCase();
    void RestoreTrainingInitializer() noexcept;
    void PromoteEvaluatedActor() noexcept;
    void LoadPromotedScore() noexcept;
    void SavePromotedScore();

    entt::registry* registry_{};
    std::array<entt::entity, TeamCapacity::MaximumBlueMobiles> guardIds_{};
    std::array<entt::entity, TeamCapacity::MaximumRedMobiles> attackerIds_{};
    entt::entity objectiveId_{entt::null};
    ActiveTeamCounts counts_{};
    Parameters parameters_{};
    PatrolTrainingSettings training_{};
    /** @brief Live independent generalized weights for the ten UI channels. */
    GeneralizedRewardValues rewardWeights_{};
    /** @brief Most recent unweighted team reward channels displayed by the IHM. */
    GeneralizedRewardValues lastRewardChannels_{};
    std::filesystem::path actorCheckpoint_{};
    std::filesystem::path trainingCheckpoint_{};
    std::filesystem::path productionCheckpoint_{};
    std::unique_ptr<GeneralizedPatrolLearner> learner_{};
    GeneralizedEvaluationCampaign evaluation_;
    std::mt19937 trainingRng_{};

    GeneralizedRuntimeState beforeStep_{};
    std::array<Action, TeamCapacity::MaximumBlueMobiles> lastActions_{};
    std::array<bool, TeamCapacity::MaximumBlueMobiles> beforeAvailability_{};
    std::size_t episodeStep_{};
    std::uint64_t lastSavedUpdate_{};
    unsigned long long id_{ppo::model::GetNewUniqId()};
    float currentReward_{};
    bool hasBeforeStep_{};
    bool episodeDone_{};
    bool finished_{};
    bool trainingRngSaved_{};
};

} // namespace ppo::examples::bluedefense::defenders
