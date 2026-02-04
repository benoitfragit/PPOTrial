#pragma once

#include <array>
#include <string>
#include <filesystem>

namespace ppo::model {

using EntityId = unsigned long long;

enum class eEnvironmentState : unsigned char {
    INIT = 0,
    RUNNING = 1,
    PAUSED = 2,
    STOPPED = 3
};

enum class eEntitySide : 
    unsigned char 
{
    Unknown = 0,
    Hostile = 1,
    Friend = 2
};

enum class eEntityType :
    unsigned char
{
    Objective = 0,
    Missile = 1,
    Aircraft = 2
};

struct sEnvironmentState
{
    float           m_maxNorth  { 112500.F };
    float           m_maxEast   { 150000.F };
    float           m_dt        { 0.02F };
};

struct sEntityState
{
    EntityId        m_entityId  { ULLONG_MAX };
    float           m_x         { 0.F  };
    float           m_y         { 0.F  };  
    float           m_dx        { 0.F };
    float           m_dy        { 0.F };
    float           m_speed     { 0.F };
    float           m_rotation  { 0.F };
    eEntitySide     m_side      { eEntitySide::Unknown };
    eEntityType     m_type      { eEntityType::Objective };
    bool            m_destroyed { false };
};

struct sSNAState {
    EntityId        m_mainTarget{ ULLONG_MAX };
    EntityId        m_target{ ULLONG_MAX };
    float           m_pk{ 0.F };
    float           m_target_x{ 0.F };
    float           m_target_y{ 0.F };
    float           m_target_dx{ 0.F };
    float           m_target_dy{ 0.F };
    float           m_closingspeed{ 0.F };
    float           m_target_los{ 0.F };
    float           m_target_los_rate{ 0.F };
    float           m_maxNoescapeZone{ 8000.F };
    float           m_escapezone{ 0.F };
    float           m_noescapezone{ 0.F };
    float           m_maxShootAngle{ 0.F };
    float           m_accCmd{ 0.F };
    float           m_rotSpeedCmd{ 0.F };
    unsigned short  m_nbmissiles{ 0 };
    unsigned short  m_maxnbmissiles{ 0 };
};

struct sFlightModelState
{
    float m_acc{ 0.F };
    float m_accpsi{ 0.F };
    float m_maxAcc{ 0.F };
    float m_maxAccRot{ 0.F };
    float m_maxSpeed{ 0.F };
    float m_minSpeed{ 0.F };
    float m_fuel{ 0.F };
    float m_fuelconsumption{ 0.F };
    float m_maxfuel{ 0.F };
};

class ISNAState
{
public:
    virtual                   ~ISNAState(void) = default;
    virtual sSNAState        GetSnapshot(void) const noexcept = 0;
};

class IFlightModelState
{
public:
    virtual                   ~IFlightModelState(void) = default;
    virtual sFlightModelState    GetSnapshot(void) const noexcept = 0;
};

class IEntityState
{
public:
    virtual                    ~IEntityState(void) = default;
    virtual sEntityState       GetSnapshot(void) const noexcept = 0;
    virtual ISNAState*         GetISNA(void) noexcept = 0;
    virtual IFlightModelState* GetIFlightModel(void) noexcept = 0;
};

static constexpr int const CHECKPOINT_DIMENSION = 6;
using CheckPointBuffer = std::array<float, CHECKPOINT_DIMENSION>;
class ICheckPointBuffer
{
public:
    virtual                   ~ICheckPointBuffer(void) = default;
    virtual CheckPointBuffer GetCheckpoint(void) noexcept = 0;
    virtual void             SetCheckpoint(CheckPointBuffer const& buffer) noexcept = 0;
};

struct sEvaluatorState:  public sEnvironmentState
{
    unsigned int m_nbstep{ 0 };
    unsigned int m_nbtests{ 0 };
    float        m_notation{ 0.F };
    float        m_progress{ 0.F };
    bool         m_inprogress{ false };
};

class IEvaluator {
public:
    virtual ~IEvaluator(void) = default;
    virtual void Start(std::string const& name, std::filesystem::path const& modelpath) = 0;
    virtual void Stop(void) = 0;
    virtual sEvaluatorState GetSnapshot(void) const noexcept = 0;
};

class IEntity {
public:
    virtual          ~IEntity(void) = default;
    virtual void     Init       (void) = 0;
    virtual void     Run        (void) = 0;
    virtual EntityId GetEntityId(void) const noexcept = 0;
    virtual void     UpdateEnv  (void) const noexcept = 0;
};

class IAgent {
public:
    virtual                     ~IAgent(void) = default;

    virtual void                Init(void) = 0;
    virtual void                ChooseAction(void) = 0;
    virtual bool                SetNewState(void) = 0;
    virtual bool                Update(void)  noexcept = 0;
};



} /* ppo::model */