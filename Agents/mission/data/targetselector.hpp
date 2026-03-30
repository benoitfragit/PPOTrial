/**
* @file Agents/mission/data/targetselector.hpp
* @brief Declarations for targetselector.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include "aircraft.hpp"
#include "trainingagent.hpp"
#include "prodagent.hpp"
#include "missionlayout.hpp"
#include "actorcriticmission.hpp"
#include "evaluator.hpp"

namespace rl {

/**
* @brief Registers evaluation-time factories required by the mission agent.
*/
void EnsureMissionEvaluationFactoriesRegistered(void);

/**
* @brief State and action adapter used by the mission agents.
*/
struct StatesAndActionsForMission
{
    /**
    * @brief Builds the adapter for one ECS entity.
    * @param entity Controlled entity wrapper.
    */
    explicit StatesAndActionsForMission(ppo::model::BaseEntity& entity)
    {
        m_entity = entity;
    }

    /**
    * @brief Reward channels used by the mission agent.
    */
    enum RewardType
    {
        MainTargetDestruction,
        SecondaryTargetDestruction,
        OwnshipDestruction,
        WingmanDestruction,
        CivilianDestruction,
        MissionFailure,
        Intercept,
        Danger,
        OutOfArea,
        Range,
        Fuel,
        NearDanger,

        _First = MainTargetDestruction,
        _Last = NearDanger,
        _Size = _Last - _First + 1
    };

    /**
    * @brief Human-readable reward channel names.
    */
    constexpr std::array<std::string_view, static_cast<unsigned char>(RewardType::_Size)> const static REWARDS
    {
        "Main target destruction",
        "Secondary target destruction",
        "Ownship destruction",
        "Wingmen destruction",
        "Civilian destruction",
        "MissionFailure",
        "Intercept",
        "Danger",
        "Out Of Area",
        "Range",
        "Fuel",
        "NearDanger"
    };

    /**
    * @brief Maximum number of tracks exposed to the network.
    */
    static constexpr std::size_t MAX_TRACKS{ MissionObservationLayout::kMaxTracks };

    /**
    * @brief Per-track features packed into the mission state vector.
    */
    enum class eFeatures : unsigned int
    {
        NorthDirection,
        EastDirection,
        Range,
        CosBearing,
        SinBearing,
        ClosingSpeed,
        Threat,
        PrimaryTarget,
        Available,

        _First = NorthDirection,
        _Last = Available,
        _Size = _Last - _First + 1
    };

    /**
    * @brief Keeps the semantic track feature list aligned with the shared layout.
    */
    static_assert(
        static_cast<std::size_t>(eFeatures::_Size) == MissionObservationLayout::kTrackFeatureCount,
        "Mission track feature count must stay aligned with MissionObservationLayout");

    /**
    * @brief Flattened mission state vector type.
    */
    using STATES = std::array<float, MissionObservationLayout::kStateDimension>;
    /**
    * @brief Mission action vector type.
    */
    using ACTIONS = std::array<float, MissionObservationLayout::kActionCount>;

    /**
    * @brief Applies the chosen action to the controlled entity.
    * @param action Action chosen by the policy.
    */
    void SetNextAction(ACTIONS const& action) noexcept;
    /**
    * @brief Builds the latest observation vector from the world state.
    * @param state Output state buffer.
    */
    void SetNewState(STATES& state) noexcept;
    /**
    * @brief Resets adapter-side state before a new episode.
    */
    void Init(void);

    /**
    * @brief Controlled entity wrapper.
    */
    ppo::model::BaseEntity m_entity{ nullptr };
};

/**
* @brief Training agent used for the mission task.
*/
class TargetSelectorTrainingAgent :
    public ppo::model::TrainingAgent<ActorCriticTrackAttention, StatesAndActionsForMission>
{
public:
    /**
    * @brief Base training-agent alias.
    */
    using Base = ppo::model::TrainingAgent<ActorCriticTrackAttention, StatesAndActionsForMission>;

    /**
    * @brief Creates the mission training agent for one entity.
    * @param entity Controlled entity wrapper.
    */
    explicit TargetSelectorTrainingAgent(ppo::model::BaseEntity& entity) :
        Base(&(entity.add<StatesAndActionsForMission>(entity)), "RLMission", RESOURCE_RLMISSIONMODEL_PATH, nullptr)
    {
        EnsureMissionEvaluationFactoriesRegistered();
        SetEvaluator(&m_evaluator);
        m_entity = entity;
    }

private:
    /**
    * @brief Resets mission-specific training-side state.
    */
    void Init(void) final;
    /**
    * @brief Computes the mission reward decomposition for one step.
    * @return True when the episode should stop.
    */
    bool RewardFunction(void) noexcept final;

    /**
    * @brief Controlled entity wrapper.
    */
    ppo::model::BaseEntity m_entity{ nullptr };
    /**
    * @brief Evaluator launched periodically by the training flow.
    */
    ppo::evaluation::Evaluator m_evaluator{};
    /**
    * @brief True once previous-step memories have been initialized.
    */
    bool m_stepMemoryInitialized{ false };
    /**
    * @brief True once the intercept reward has already been granted.
    */
    bool m_interceptReward{ false };
    /**
    * @brief True once the threat-margin memory has been initialized.
    */
    bool m_threatMarginInitialized{ false };
    /**
    * @brief Previous distance to the primary target.
    */
    float m_prevTargetRange{ 0.F };
    /**
    * @brief Time spent outside the allowed area.
    */
    float m_timeOutOfArea{ 0.F };
    /**
    * @brief Previous normalized margin to the nearest SAM danger area.
    */
    float m_prevThreatMargin{ 0.F };
};

/**
* @brief Production agent used for mission inference and evaluation.
*/
class TargetSelectorProdAgent :
    public ppo::model::ProdAgent<ActorCriticTrackAttentionProd, StatesAndActionsForMission>
{
public:
    /**
    * @brief Base production-agent alias.
    */
    using Base = ppo::model::ProdAgent<ActorCriticTrackAttentionProd, StatesAndActionsForMission>;

    /**
    * @brief Creates the mission production agent for one entity.
    * @param entity Controlled entity wrapper.
    * @param version Model subfolder to load, usually prod or eval.
    */
    TargetSelectorProdAgent(ppo::model::BaseEntity& entity, std::string const& version = "prod") :
        Base(&(entity.add<StatesAndActionsForMission>(entity)),
             "RLMission",
             std::filesystem::path(RESOURCE_RLMISSIONMODEL_PATH) / version / "model.pt")
    {
        m_entity = entity;
    }

private:
    /**
    * @brief Resets mission-specific production-side state.
    */
    void Init(void) final;

    /**
    * @brief Controlled entity wrapper.
    */
    ppo::model::BaseEntity m_entity{ nullptr };
};

/**
* @brief Registers the production agent factory used by the evaluator.
*/
inline void
EnsureMissionEvaluationFactoriesRegistered(void)
{
    static bool const registered = []()
    {
        ppo::evaluation::RegisterAgentFactory(
            "TargetSelectorProdAgent",
            [](ppo::model::BaseEntity& entity, std::string_view version)
            {
                return std::make_shared<TargetSelectorProdAgent>(entity, std::string(version));
            });
        return true;
    }();

    (void)registered;
}

} /* rl */
