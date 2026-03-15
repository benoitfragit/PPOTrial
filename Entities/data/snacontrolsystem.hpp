#pragma once 

#include <array>

#include "serializer.hpp"
#include "baseentity.hpp"

namespace ppo::model {

class SNAControlSystem {
public:       
    static void Init    (BaseEntity& entity);
    static void Run     (BaseEntity& entity);
    static void Create  (BaseEntity& entity, ppo::serialization::json& ar);
};

} /* ppo::mmodel */