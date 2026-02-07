#pragma once

#include "entt/entt.hpp"

namespace ppo::renderer {

class IRenderer
{
public:
    virtual ~IRenderer(void) = default;

    virtual void Run(entt::registry& registry, entt::entity entity) noexcept = 0;
};

} /* ppo::renderer */