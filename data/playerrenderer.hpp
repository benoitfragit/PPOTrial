#pragma once
#include <SFML/Graphics.hpp>

#include "player.hpp"

#include <memory>

namespace ppo::environment {

constexpr double DEG_TO_RAD = 0.01745329252;
constexpr double RAD_TO_DEG = 57.295779513;

class PlayerRenderer :
    public sf::ConvexShape
{
public:
                     PlayerRenderer(model::IPlayerState* player, int height, int width, sf::Color const color = sf::Color::Transparent, bool filled = true);

                     PlayerRenderer(PlayerRenderer&)  = delete;
                     PlayerRenderer(PlayerRenderer&&) = delete;
    PlayerRenderer&  operator=     (PlayerRenderer&)  = delete;
    PlayerRenderer&  operator=     (PlayerRenderer&&) = delete;

    void             Run           (void)                   noexcept;

private:
    void             DrawShootingRanges(model::sPlayerState const& state, float radius, sf::Color const color, unsigned char nbPoint=10, float startangle=-90.F* DEG_TO_RAD, float endangle=90.F * DEG_TO_RAD);
    void             DrawDestroySymbol (model::sPlayerState const& state);
    void             DrawPowerLevel    (model::sPlayerState const& state);

    float                   m_width { 0.F };
    float                   m_height{ 0.F };
    sf::Color               m_color { sf::Color::Transparent};
    bool                    m_filled{ false };

    model::IPlayerState*    p_player{ nullptr };
};


}   /* ppo::environment */