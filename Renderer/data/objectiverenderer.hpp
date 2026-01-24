#pragma once
#include <SFML/Graphics.hpp>

#include "irenderer.hpp"
#include "interface.hpp"

namespace ppo::renderer {

class ObjectiveRenderer :
    public IRenderer,
    public sf::CircleShape
{
public:
            ObjectiveRenderer(ppo::model::IEntityState* player, int radius, bool filled = true);

    void    Run              (void) noexcept final;

private:
    void    DrawDestroySymbol(model::sEntityState const& state);

    float                m_radius{ 0.F };
    sf::Color            m_color { sf::Color::Transparent };
    bool                 m_filled{ false };
    model::IEntityState* p_player{ nullptr };
};

}