#pragma once
#include <SFML/Graphics.hpp>

#include "entity.hpp"
#include "irenderer.hpp"

#include <memory>

namespace ppo::renderer {

constexpr double DEG_TO_RAD = 0.01745329252;
constexpr double RAD_TO_DEG = 57.295779513;

class EntityRenderer :
    public IRenderer,
    public sf::ConvexShape
{
public:
                     EntityRenderer(int height, int width, bool filled = true);

                     EntityRenderer(EntityRenderer&)  = delete;
                     EntityRenderer(EntityRenderer&&) = delete;
    EntityRenderer&  operator=     (EntityRenderer&)  = delete;
    EntityRenderer&  operator=     (EntityRenderer&&) = delete;

    void             Run           (entt::registry& registry, entt::entity entity) noexcept final;

private:
    void             DrawShootingRanges(model::sEntityState const& state, float radius, sf::Color const color, float startangle=-90.F* DEG_TO_RAD, float endangle=90.F * DEG_TO_RAD, unsigned char nbPoint = 10);
    void             DrawDestroySymbol (model::sEntityState const& state);

    float                   m_width { 0.F };
    float                   m_height{ 0.F };
    sf::Color               m_color { sf::Color::Transparent};
    bool                    m_filled{ false };
};


}   /* ppo::renderer */