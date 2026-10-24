#include "airspaceworldrules.hpp"

#include "entity.hpp"
#include "components.hpp"

#include <entt/entt.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

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

    // Copy the exact legacy visual components through the example hook.
    auto& sna = world.emplace<sSNAState>(entity->uniqid());
    sna.m_target_x = 142.0F;
    world.emplace<sFlightModelState>(entity->uniqid()).m_acc = 3.0F;
    world.emplace<sPseudoLidarState>(entity->uniqid()).m_angleSpanRadians = 1.5F;

    rules.BeforeStep(entities);
    Require(!state.m_prevDestroyed, "unexpected initial destroyed-history flag");
    state.m_destroyed = true;
    rules.Resolve(entities);
    rules.AfterStep();

    entt::registry snapshot;
    snapshot.create(entity->uniqid());
    rules.CopySnapshot(world, snapshot);

    Require(snapshot.get<sSNAState>(entity->uniqid()).m_target_x == 142.0F,
            "SNA snapshot content changed");
    Require(snapshot.get<sFlightModelState>(entity->uniqid()).m_acc == 3.0F,
            "kinematics snapshot content changed");
    Require(snapshot.get<sPseudoLidarState>(entity->uniqid()).m_angleSpanRadians == 1.5F,
            "lidar snapshot content changed");

    auto const& events = snapshot.ctx().get<sExplosionEventBuffer>().m_events;
    Require(events.size() == 1U, "fresh destruction did not generate exactly one event");
    Require(events.front().m_x == 12.0F && events.front().m_y == -4.0F,
            "destruction event location changed");
    Require(events.front().m_type == eEntityType::Objective,
            "destruction event entity type changed");
    Require(events.front().m_baseRadius == 2.40F
                && events.front().m_heightScale == 2.35F,
            "historical objective explosion appearance changed");

    // A previously destroyed entity must not emit a fresh event on the next step.
    rules.BeforeStep(entities);
    Require(state.m_prevDestroyed, "destroyed-state history was not retained");
    rules.Resolve(entities);
    entt::registry nextSnapshot;
    nextSnapshot.create(entity->uniqid());
    rules.CopySnapshot(world, nextSnapshot);
    Require(nextSnapshot.ctx().get<sExplosionEventBuffer>().m_events.size() == 1U,
            "previous destruction generated a duplicate event");
}

} // namespace

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
