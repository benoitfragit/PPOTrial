#pragma once

#include "entity.hpp"

namespace ppo::model {

class CheckpointInitializerSystem 
{
public:
    static void Load(Entity& entity);
    static void Init(Entity& entity);
};

} /* ppo::model */