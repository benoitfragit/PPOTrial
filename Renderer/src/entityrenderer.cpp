#include "entityrenderer.hpp"
#include "environmentrenderer.hpp"

namespace ppo::renderer {

EntityRenderer::EntityRenderer(int height, int width, bool filled) :
    m_width{ static_cast<float>(width) },
    m_height { static_cast<float>(height) },
    m_filled { filled }{}

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
EntityRenderer::DrawShootingRanges(model::sEntityState const& state, float radius, sf::Color const color, float startangle, float endangle, unsigned char nbpoint)
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
EntityRenderer::DrawDestroySymbol(model::sEntityState const& state)
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
EntityRenderer::Run(entt::registry& registry, entt::entity entity) noexcept
{
    if (!registry.any_of<ppo::model::sEntityState>(entity))
        return;

    auto const [state] = registry.view<ppo::model::sEntityState>().get(entity);

    auto&       renderer{ EnvironmentRenderer::GetInstance() };
    auto const& view    { renderer.GetView() };
    auto const  winSize { renderer.getSize() };
    auto const  viewSize{ view.getSize() };
    float const ratioNorth  { viewSize.x / winSize.y };
    float const ratioEast   { viewSize.y / winSize.x };

    if (registry.any_of<ppo::model::sFlightModelState>(entity)
        && state.m_type != ppo::model::eEntityType::SAM)
    {
        switch (state.m_side)
        {
        case ppo::model::eEntitySide::Friend:  m_color = sf::Color::Blue;           break;
        case ppo::model::eEntitySide::Hostile: m_color = sf::Color::Red;            break;
        case ppo::model::eEntitySide::Unknown: m_color = sf::Color::White;          break;
        default:                               m_color = sf::Color::Transparent;    break;
        }

        sf::ConvexShape::setPointCount(3);

        sf::Vector2f const p1{ -m_height * ratioNorth / 2.F ,  -m_width * ratioEast / 2.F };
        sf::Vector2f const p2{ -m_height * ratioNorth / 2.F,   m_width * ratioEast / 2.F };
        sf::Vector2f const p3{ m_height * ratioNorth / 2.F,                           0 };

        sf::ConvexShape::setPoint(0, p1);
        sf::ConvexShape::setPoint(1, p2);
        sf::ConvexShape::setPoint(2, p3);
        
        sf::ConvexShape::setOutlineColor(m_color);
        if (m_filled)
            sf::ConvexShape::setFillColor(m_color);
    }

    // position
    setPosition(state.m_x / renderer.GetScaleWorld(), state.m_y / renderer.GetScaleWorld());
    setOrigin(0.F, 0.F);

    // orientation
    setRotation(static_cast<float>(state.m_rotation * RAD_TO_DEG));
    
    if (registry.any_of<ppo::model::sMissionLidarState>(entity))
    {
        auto [lidarState] = registry.view<ppo::model::sMissionLidarState>().get(entity);

        // draw only the pseudo-lidar state produced by the agent
        if (float const lidarRadius{ lidarState.m_radius };
            std::numeric_limits<float>::epsilon() < lidarRadius)
            DrawShootingRanges(state, lidarRadius, sf::Color::Magenta, -lidarState.m_halfAperture, lidarState.m_halfAperture);
    }

    renderer.draw(*this);

    if (state.m_destroyed)
        DrawDestroySymbol(state);
}

}
