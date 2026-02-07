#include "objectiverenderer.hpp"

#include "environmentrenderer.hpp"

namespace ppo::renderer {

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
    sf::Vector2f offset{ ortho * linewidth / 2.F };

    vertices.append(sf::Vertex(a + offset, color));
    vertices.append(sf::Vertex(a - offset, color));
    vertices.append(sf::Vertex(b - offset, color));
    vertices.append(sf::Vertex(b + offset, color));
}

ObjectiveRenderer::ObjectiveRenderer(int radius, bool filled) :
    m_radius{ static_cast<float>(radius) },
    m_filled{ filled }
{
}

void
ObjectiveRenderer::Run(entt::registry& registry, entt::entity entity) noexcept
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

    // position
    setRadius(m_radius * ratioNorth / 2.F);
    setPosition(state.m_x / renderer.GetScaleWorld(), state.m_y / renderer.GetScaleWorld());
    setOrigin(0.F, 0.F);

    switch (state.m_side)
    {
        case ppo::model::eEntitySide::Friend:  m_color = sf::Color::Blue;           break;
        case ppo::model::eEntitySide::Hostile: m_color = sf::Color::Red;            break;
        case ppo::model::eEntitySide::Unknown: m_color = sf::Color::White;          break;
        default:                               m_color = sf::Color::Transparent;    break;
    }

    sf::CircleShape::setOutlineColor(m_color);
    if (m_filled)
        sf::CircleShape::setFillColor(m_color);

    renderer.draw(*this);

    if (state.m_destroyed)
        DrawDestroySymbol(state);
}

void
ObjectiveRenderer::DrawDestroySymbol(model::sEntityState const& state)
{
    auto& renderer{ EnvironmentRenderer::GetInstance() };
    auto const& view{ renderer.GetView() };
    auto const  winSize{ renderer.getSize() };
    auto const  viewSize{ view.getSize() };

    float const ratioNorth{ viewSize.x / winSize.y };
    float const ratioEast{ viewSize.y / winSize.x };

    sf::Vector2f const& position{ sf::CircleShape::getPosition() };
    float const lineWidth{ renderer.GetLineWidth() };

    sf::VertexArray vertices(sf::Quads);

    sf::Vector2f a{ position.x + m_radius * ratioNorth / 2.F,
                    position.y - m_radius * ratioEast / 2.F };

    sf::Vector2f b{ position.x - m_radius * ratioNorth / 2.F,
                    position.y + m_radius * ratioEast / 2.F };

    DrawQuadBetweenLine(a, b, sf::Color::White, lineWidth, vertices);

    a = sf::Vector2f(position.x + m_radius * ratioNorth / 2.F,
        position.y + m_radius * ratioEast / 2.F);
    b = sf::Vector2f(position.x - m_radius * ratioNorth / 2.F,
        position.y - m_radius * ratioEast / 2.F);

    DrawQuadBetweenLine(a, b, sf::Color::White, lineWidth, vertices);

    renderer.draw(vertices);
}

} /* ppo::environment */