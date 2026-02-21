#pragma once

#include <array>
#include <string>
#include <filesystem>

#include "entt/entt.hpp"

namespace ppo::model {

using EntityId = entt::entity;

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
    None = 0,
    Objective = 1,
    Missile = 2,
    Aircraft = 3,
    SAM = 4
};

struct sEnvironmentState
{
    float           m_maxNorth  { 112500.F };
    float           m_maxEast   { 150000.F };
    float           m_dt        { 0.02F };
};

struct sEntityState
{
    EntityId        m_entityId  { entt::null };
    float           m_x         { 0.F  };
    float           m_y         { 0.F  };  
    float           m_dx        { 0.F };
    float           m_dy        { 0.F };
    float           m_speed     { 0.F };
    float           m_rotation  { 0.F };
    eEntitySide     m_side      { eEntitySide::Unknown };
    eEntityType     m_type      { eEntityType::Objective };
    bool            m_prevDestroyed{ false };
    bool            m_destroyed { false };
    EntityId        m_destroyedById{ entt::null };
};

struct sSNAState {
    EntityId        m_target{ entt::null };
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
    unsigned short  m_nbmissiles{ 0 };
    unsigned short  m_maxnbmissiles{ 0 };
};

struct sSNAControlState {
    float m_accCmd{ 0.F };
    float m_rotSpeedCmd{ 0.F };
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

struct sEvaluatorState
{
    unsigned int m_nbstep{ 0 };
    unsigned int m_nbtests{ 0 };
    float        m_notation{ 0.F };
    float        m_progress{ 0.F };
    bool         m_inprogress{ false };
};

} /* ppo::model */