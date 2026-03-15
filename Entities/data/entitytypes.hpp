#pragma once

#include "entt/entity/fwd.hpp"

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

} /* ppo::model */
