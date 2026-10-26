#pragma once

#include "patrol_training_settings.hpp"
#include "patrol_evaluation_campaign.hpp"
#include "baseentity.hpp"
#include "iagent.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

namespace ppo::examples::twovstwo::defenders {

class PatrolLearner;
class PatrolPolicy;

/**
 * @brief One cooperative MAPPO controller for both blue defenders.
 *
 * A single IAgent guarantees that both actions and the shared reward belong to
 * the same ECS transition. RLtools owns one parameter-shared Actor and one
 * centralized Critic; no red controller is modified or replaced.
 */
class PatrolMappoAgent final : public ppo::model::IAgent
{
public:
    /**
     * @brief Creates the blue team controller.
     * @param guards The two blue mobile entities, in stable role order.
     * @param attackers Optional red mobiles controlled elsewhere by RLMission.
     * @param objective Defended blue objective.
     * @param policy Shared deterministic inference policy.
     * @param parameters Patrol and vehicle parameters.
     * @param training Runtime rollout and reward settings.
     * @param latestCheckpoint Actor warm-start written before each evaluation.
     * @param trainingCheckpoint Complete latest training-state checkpoint.
     * @param productionCheckpoint Best evaluated Actor checkpoint.
     */
    PatrolMappoAgent(std::array<ppo::model::BaseEntity*, 2> const& guards,
                     std::vector<ppo::model::BaseEntity*> const& attackers,
                     ppo::model::BaseEntity& objective,
                     std::shared_ptr<PatrolPolicy> policy,
                     Parameters parameters,
                     PatrolTrainingSettings training,
                     std::filesystem::path latestCheckpoint,
                     std::filesystem::path trainingCheckpoint,
                     std::filesystem::path productionCheckpoint);
    ~PatrolMappoAgent() override;

    PatrolMappoAgent(PatrolMappoAgent const&) = delete;
    PatrolMappoAgent& operator=(PatrolMappoAgent const&) = delete;

    void Init() override;
    void ChooseAction() override;
    bool SetNewState() override;
    void ProcessCommand(ppo::model::sAgentRealTimeCommand const& command) noexcept override;
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
    Parameters ReadParameters() const;
    State ReadState(Parameters const& parameters) const;
    std::array<Observation, 2> BuildObservations(Parameters const& parameters,
                                                State const& state) const;
    void SaveLatest() noexcept;
    void SaveTo(std::filesystem::path const& checkpoint);
    void PromoteEvaluatedActor() noexcept;
    void StartEvaluation(Parameters const& parameters);
    void PrepareEvaluationCase(Parameters const& parameters);
    void RestoreTrainingInitializer() noexcept;
    void LoadPromotedScore() noexcept;
    void SavePromotedScore();
    void PublishActor();

    entt::registry* registry_{};
    std::array<entt::entity, 2> guardIds_{entt::null, entt::null};
    std::array<entt::entity, 2> attackerIds_{entt::null, entt::null};
    entt::entity objectiveId_{entt::null};
    std::shared_ptr<PatrolPolicy> policy_;
    std::unique_ptr<PatrolLearner> learner_;
    Parameters parameters_{};
    PatrolTrainingSettings training_{};
    std::filesystem::path latestCheckpoint_;
    std::filesystem::path trainingCheckpoint_;
    std::filesystem::path productionCheckpoint_;
    PatrolEvaluationCampaign evaluation_;
    std::mt19937 trainingRng_{};
    State beforeStep_{};
    std::array<Action, 2> lastActions_{};
    RewardValues rewardValues_{};
    std::size_t episodeStep_{0U};
    std::uint64_t updates_{0U};
    std::uint64_t lastSavedUpdate_{0U};
    unsigned long long id_{ppo::model::GetNewUniqId()};
    float currentReward_{0.F};
    bool hasBeforeStep_{false};
    bool episodeDone_{false};
    bool finished_{false};
    bool trainingRngSaved_{false};
};

} // namespace ppo::examples::twovstwo::defenders
