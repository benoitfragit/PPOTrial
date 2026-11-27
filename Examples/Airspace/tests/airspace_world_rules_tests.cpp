/**
 * @file Examples/Airspace/tests/airspace_world_rules_tests.cpp
 * @brief Tests et assertions déterministes pour airspace world rules tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "airspaceworldrules.hpp"

#include "entity.hpp"
#include "components.hpp"
#include "airspacecomponents.hpp"

#include <entt/entt.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Consulte ou modifie des composants ECS. Peut signaler un état invalide par exception.
 */
void Require(bool condition, char const* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void TestAirspaceWorldRules()
{
    using namespace ppo::model;

    entt::registry world;
    world.ctx().emplace<sEnvironmentState>();
    auto entity = std::make_shared<Entity>(&world, eEntitySide::Friend, eEntityType::Objective);
    WorldEntities entities{entity};
    ppo::examples::airspace::AirspaceWorldRules rules;

    auto [state] = entity->get<sEntityState>();
    state.m_x = 12.0F;
    state.m_y = -4.0F;

    rules.BeforeStep(entities);
    Require(!state.m_prevDestroyed, "unexpected initial destroyed-history flag");
    state.m_destroyed = true;
    rules.Resolve(entities);
    rules.AfterStep();

    // The next step remembers the destruction (no second destruction of the same entity).
    rules.BeforeStep(entities);
    Require(state.m_prevDestroyed, "destroyed-state history was not retained");
    rules.Resolve(entities);
    Require(state.m_destroyed, "a destroyed entity came back during the next step");
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    try
    {
        TestAirspaceWorldRules();
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
