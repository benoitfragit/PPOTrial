/**
* @file Renderer/data/irenderer.hpp
* @brief Declarations for irenderer.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entt/entt.hpp"

namespace ppo::renderer {

/**
* @brief Minimal rendering interface for one ECS-backed entity.
*/
class IRenderer
{
public:
    /**
    * @brief Virtual destructor for interface-safe cleanup.
    */
    virtual ~IRenderer(void) = default;

    /**
    * @brief Render one entity from the provided registry.
    * @param registry Snapshot or live registry to read from.
    * @param entity Entity identifier to render.
    */
    virtual void Run(entt::registry& registry, entt::entity entity) noexcept = 0;
};

} /* ppo::renderer */
