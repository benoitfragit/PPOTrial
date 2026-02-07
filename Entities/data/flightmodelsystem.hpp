#pragma once

#include "serializer.hpp"
#include "interface.hpp"

#include "baseentity.hpp"

namespace ppo::model {

class FlightModelSystem
{
public:
    static void Run   (BaseEntity& entity);
    static void Init  (BaseEntity& entity);
    static void Create(BaseEntity& entity, ppo::serialization::json& ar);
};

}