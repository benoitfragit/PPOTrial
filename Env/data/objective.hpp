#pragma once

#include "Entity.hpp"
#include <string>

namespace ppo::model {

class Objective : public Entity
{
public:
    template <typename Ty>
    Objective(std::string name, eEntitySide side, Ty& env):
        Entity(name, side, eEntityType::Objective, env)
    {

    };
};

} /* ppo::model */