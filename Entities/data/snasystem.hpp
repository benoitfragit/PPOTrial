#pragma once 

#include "serializer.hpp"
#include "interface.hpp"

#include "entity.hpp"

namespace ppo::model {

class SNASystem {
public:
    static void Init  (BaseEntity& entity);                   
    static void Run   (BaseEntity& entity);
    static void Create(BaseEntity& entity, ppo::serialization::json& ar);
};

} /*  ppo::model */