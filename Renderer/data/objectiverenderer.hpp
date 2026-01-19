#pragma once
#include <SFML/Graphics.hpp>

#include "irenderer.hpp"
#include "interface.hpp"

namespace ppo::environment {

class ObjectiveRenderer :
    public IRenderer,
    public sf::CircleShape
{
public:
            ObjectiveRenderer(ppo::model::IPlayerState* player, int radius, bool filled = true);

    void    Run              (void) noexcept final;

private:
    void    DrawDestroySymbol(model::sPlayerState const& state);

    float                m_radius{ 0.F };
    sf::Color            m_color { sf::Color::Transparent };
    bool                 m_filled{ false };
    model::IPlayerState* p_player{ nullptr };
};

}