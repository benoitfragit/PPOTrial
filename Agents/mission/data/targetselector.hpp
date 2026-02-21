#pragma once
#include "aircraft.hpp"
#include "trainingagent.hpp"
#include "prodagent.hpp"
#include "actorcriticmission.hpp"

namespace rl {

struct sRealThreat {
    float           m_x{ 0.F };
    float           m_y{ 0.F };
    float           m_noescapeZone{ 0.F };
    std::vector<ppo::model::EntityId> m_sam{};
};

struct StatesAndActionsForMission {
    explicit StatesAndActionsForMission(ppo::model::BaseEntity& entity )
    {
        m_entity = entity;
    };

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

    static constexpr unsigned char MAX_TRACKS{ 10 };

    enum class eFeatures : unsigned int
    {
        NorthRange,
        EastRange,
        Range,
        CosDir,
        SinSin,
        ClosingSpeed,
        Hostile,
        Importancy,
        Available,

        _First = NorthRange,
        _Last = Available,
        _Size = _Last - _First + 1
    };

    using STATES = std::array<float, MAX_TRACKS* static_cast<unsigned int>(eFeatures::_Size) + 62>;
    using ACTIONS = std::array<float, 3>;

    void SetNextAction(ACTIONS const& action) noexcept;
    void SetNewState(STATES& state) noexcept;
    void Init(void);

    ppo::model::BaseEntity m_entity{ nullptr };

    std::vector<sRealThreat> m_threats{};
};

/// <summary>
/// This is the definitionn of the agent used for training
/// </summary>
class TargetSelectorTrainingAgent :
    public ppo::model::TrainingAgent<ActorCriticTrackAttention, StatesAndActionsForMission>
{
public:
    using Base = ppo::model::TrainingAgent<ActorCriticTrackAttention, StatesAndActionsForMission>;

    explicit TargetSelectorTrainingAgent(ppo::model::BaseEntity& entity) :
        Base(&(entity.add<StatesAndActionsForMission>(entity)), "RLMission", RESOURCE_RLMISSIONMODEL_PATH, nullptr)
    {
        m_entity = entity;
    };

private:
    void Init(void) final;

    /// <summary>
    /// this function is used to compute the reward at each step
    /// </summary>
    /// <param name=""></param>
    /// <returns></returns>
    bool RewardFunction(void) noexcept final;

    /// <summary>
    /// this is the entity affected by this agent
    /// </summary>
    ppo::model::BaseEntity m_entity{ nullptr };

    /// <summary>
    /// This is an evaluator periodically launch by the training agent
    /// </summary>
    // Evaluator m_evaluator{};

    bool m_memoryInitialized{ false };
    bool m_interceptReward{ false };
    bool m_firstDanger{ false };
    float m_prevRange{ 0.F };
    float m_timeOutOfArea{ 0.F };
    float m_prevOutRange{ 0.F };
    float m_prevDangerRange{ 0.F };
};

/// <summary>
/// This is the definition of the agent used for production
/// </summary>
class TargetSelectorProdAgent :
    public ppo::model::ProdAgent<ActorCriticTrackAttentionProd, StatesAndActionsForMission>
{
public:
    using Base = ppo::model::ProdAgent<ActorCriticTrackAttentionProd, StatesAndActionsForMission>;

    TargetSelectorProdAgent(ppo::model::BaseEntity& entity, std::string const& version = "prod") :
        Base(&(entity.add<StatesAndActionsForMission>(entity)), "RLMission", std::filesystem::path(RESOURCE_RLMISSIONMODEL_PATH) / version / "model.pt")
    {
        m_entity = entity;
    };

private:
    void Init(void) final;

    /// <summary>
    /// this is the entity affected by this agent
    /// </summary>
    ppo::model::BaseEntity m_entity{ nullptr };
};

} /* ppo::mission */