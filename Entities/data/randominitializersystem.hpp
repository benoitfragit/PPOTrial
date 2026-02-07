#pragma once

#include "entity.hpp"

namespace ppo::model {

class RandomInitiliazerSystem 
{
public:
    static void Init(Entity& entity);
};

}