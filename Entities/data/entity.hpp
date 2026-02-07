#pragma once

#include "baseentity.hpp"

#include "interface.hpp"

namespace ppo::model {

#include <memory>

class Entity: public BaseEntity
{
public:
    Entity(entt::registry* registry, eEntitySide side, eEntityType type);

    using InitializerSystem = void(*)(Entity&);
    void SetInitializerSystem(InitializerSystem system);

    virtual void Init(void) noexcept;
    virtual void Run (void) noexcept;
protected:
    virtual void OnInit(void) noexcept;

    InitializerSystem p_initializer{ nullptr };
};

} /* ppo::model */