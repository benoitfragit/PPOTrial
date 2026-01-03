#include "playerrenderer.hpp"
#include "environmentrenderer.hpp"

namespace ppo::environment {

PlayerRenderer::PlayerRenderer(model::IPlayerState* player, int height, int width, sf::Color const color, bool filled) :
    m_width{ static_cast<float>(width) },
    m_height { static_cast<float>(height) },
    m_color  { color },
    m_filled { filled },
    p_player {player}
{

}

static void
DrawQuadBetweenLine(sf::Vector2f const& a, sf::Vector2f const& b, sf::Color const color, float const linewidth, sf::VertexArray& vertices)
{
    sf::Vector2f d{ b - a };
    float const length{ std::hypot(d.x, d.y) };
    if (length <= std::numeric_limits<float>::epsilon())
        return;

    // normalized direction
    d /= length;
    sf::Vector2f ortho{ -d.y, d.x };
    sf::Vector2f offset{ ortho * linewidth/2.F };

    vertices.append(sf::Vertex(a + offset, color));
    vertices.append(sf::Vertex(a - offset, color));
    vertices.append(sf::Vertex(b - offset, color));
    vertices.append(sf::Vertex(b + offset, color));
}

void 
PlayerRenderer::DrawShootingRanges(model::sPlayerState const& state, float radius, sf::Color const color, float startangle, float endangle, unsigned char nbpoint)
{
    auto&                renderer   { EnvironmentRenderer::GetInstance() };
    auto const&          view       { renderer.GetView() };
    auto const           winSize    { renderer.getSize() };
    auto const           viewSize   { view.getSize() };
    float const          lineWidth  { renderer.GetLineWidth() };
    sf::Vector2f const&  position   { sf::ConvexShape::getPosition() };
    float const          stepangle  { (endangle - startangle) / nbpoint };
    float                angle      { startangle + state.m_rotation};
             
    radius /= renderer.GetScaleWorld();

    // compute all points from the shooting range
    std::vector<sf::Vector2f> points{};
    for (unsigned char i = 0; i < nbpoint+1; ++i)
    {
        points.push_back( {position.x + radius * cos(angle),
                           position.y + radius * sin(angle)});
        angle += stepangle;
    }

    // compute vertices as Quads with thickness
    sf::VertexArray vertices(sf::Quads);
    for (unsigned short idx = 1; idx < points.size(); ++idx)
        DrawQuadBetweenLine(points[idx - 1], points[idx], color, lineWidth, vertices);

    renderer.draw(vertices);
}

void
PlayerRenderer::DrawDestroySymbol(model::sPlayerState const& state)
{
    auto& renderer              { EnvironmentRenderer::GetInstance() };
    auto const& view            { renderer.GetView() };
    auto const  winSize         { renderer.getSize() };
    auto const  viewSize        { view.getSize() };

    float const ratioNorth      { viewSize.x / winSize.y };
    float const ratioEast       { viewSize.y / winSize.x };

    sf::Vector2f const& position{ sf::ConvexShape::getPosition() };
    float const lineWidth       { renderer.GetLineWidth()};

    sf::VertexArray vertices(sf::Quads);

    sf::Vector2f a{ position.x + m_height * ratioNorth / 2.F,
                    position.y - m_width * ratioEast / 2.F };

    sf::Vector2f b{ position.x - m_height * ratioNorth / 2.F,
                    position.y + m_width * ratioEast / 2.F };

    DrawQuadBetweenLine(a, b, sf::Color::White, lineWidth, vertices);

    a = sf::Vector2f(position.x + m_height * ratioNorth / 2.F,
                     position.y + m_width  * ratioEast / 2.F);
    b = sf::Vector2f(position.x - m_height * ratioNorth / 2.F,
                     position.y - m_width  * ratioEast / 2.F);

    DrawQuadBetweenLine(a, b, sf::Color::White, lineWidth, vertices);

    renderer.draw(vertices);
}

void
PlayerRenderer::Run(void) noexcept
{
    if (p_player == nullptr)
        return;

    model::sPlayerState state{ p_player->GetSnapshot() };

    auto&       renderer{ EnvironmentRenderer::GetInstance() };
    auto const& view    { renderer.GetView() };
    auto const  winSize { renderer.getSize() };
    auto const  viewSize{ view.getSize() };
    float const ratioNorth  { viewSize.x / winSize.y };
    float const ratioEast   { viewSize.y / winSize.x };

    sf::ConvexShape::setPointCount(3);

    sf::Vector2f const p1{ -m_height * ratioNorth / 2.F ,  -m_width *ratioEast / 2.F };
    sf::Vector2f const p2{ -m_height * ratioNorth / 2.F,   m_width * ratioEast / 2.F };
    sf::Vector2f const p3{ m_height * ratioNorth / 2.F,                           0 };
                                        
    sf::ConvexShape::setPoint(0, p1);
    sf::ConvexShape::setPoint(1, p2);
    sf::ConvexShape::setPoint(2, p3);

    // position
    setPosition(state.m_x / renderer.GetScaleWorld(), state.m_y / renderer.GetScaleWorld());
    setOrigin(0.F, 0.F);

    sf::ConvexShape::setOutlineColor(m_color);
    if (m_filled)
        sf::ConvexShape::setFillColor(m_color);

    // orientation
    setRotation(static_cast<float>(state.m_rotation * RAD_TO_DEG));

    // shooting ranges
    if (float const escapeZoneValue{ state.m_escapezone };
        std::numeric_limits<float>::epsilon() < escapeZoneValue)
        DrawShootingRanges(state, escapeZoneValue, sf::Color::Magenta, -state.m_maxShootAngle, state.m_maxShootAngle);

    if (float const noEscapeZoneValue{ state.m_noescapezone };
        std::numeric_limits<float>::epsilon() < noEscapeZoneValue)
        DrawShootingRanges(state, noEscapeZoneValue, sf::Color::Red, -state.m_maxShootAngle, state.m_maxShootAngle);

    renderer.draw(*this);

    if (state.m_destroyed)
        DrawDestroySymbol(state);
}

}