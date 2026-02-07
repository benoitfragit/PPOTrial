#pragma once

#include "Entity.hpp"

namespace ppo::model {

class Objective : public Entity
{
public:
    Objective(entt::registry* registry, eEntitySide side) : 
        Entity(registry, side, eEntityType::Objective) {};
};

} /* ppo::model */