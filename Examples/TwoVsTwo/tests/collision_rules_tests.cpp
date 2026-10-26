#include "twovstworules.hpp"
#include "initializer.hpp"

#include "components.hpp"
#include "entity.hpp"

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

void TestSweptBlueCollision()
{
    using namespace ppo::model;

    entt::registry world;
    world.ctx().emplace<sEnvironmentState>().m_dt = 0.2F;
    auto first = std::make_shared<Entity>(
        &world, eEntitySide::Friend, eEntityType::Mobile);
    auto second = std::make_shared<Entity>(
        &world, eEntitySide::Friend, eEntityType::Mobile);
    WorldEntities entities{first, second};
    ppo::examples::twovstwo::TwoVsTwoWorldRules rules;

    auto [firstState] = first->get<sEntityState>();
    auto [secondState] = second->get<sEntityState>();
    firstState.m_x = -40.F;
    firstState.m_y = 0.F;
    firstState.m_dx = 400.F;
    secondState.m_x = 40.F;
    secondState.m_y = 0.F;
    secondState.m_dx = -400.F;
    rules.BeforeStep(entities);

    // Endpoints are 80 m apart, but the cars cross halfway through the step.
    firstState.m_x += firstState.m_dx * 0.2F;
    secondState.m_x += secondState.m_dx * 0.2F;
    rules.Resolve(entities);

    Require(firstState.m_destroyed && secondState.m_destroyed,
            "swept blue collision did not destroy both mobiles");
    Require(firstState.m_destroyedById == second->uniqid()
                && secondState.m_destroyedById == first->uniqid(),
            "blue collision attribution is incorrect");

    entt::registry snapshot;
    auto const snapshotFirst = snapshot.create(first->uniqid());
    auto const snapshotSecond = snapshot.create(second->uniqid());
    Require(snapshotFirst == first->uniqid()
                && snapshotSecond == second->uniqid(),
            "snapshot entity identifiers changed");
    rules.CopySnapshot(world, snapshot);
    auto const& events = snapshot.ctx().get<sExplosionEventBuffer>().m_events;
    Require(events.size() == 2U,
            "blue collision did not publish both explosion events");
}

void TestBlueNearMiss()
{
    using namespace ppo::model;

    entt::registry world;
    world.ctx().emplace<sEnvironmentState>().m_dt = 0.2F;
    auto first = std::make_shared<Entity>(
        &world, eEntitySide::Friend, eEntityType::Mobile);
    auto second = std::make_shared<Entity>(
        &world, eEntitySide::Friend, eEntityType::Mobile);
    WorldEntities entities{first, second};
    ppo::examples::twovstwo::TwoVsTwoWorldRules rules;

    auto [firstState] = first->get<sEntityState>();
    auto [secondState] = second->get<sEntityState>();
    firstState.m_x = -40.F;
    firstState.m_y = -600.F;
    firstState.m_dx = 400.F;
    secondState.m_x = 40.F;
    secondState.m_y = 600.F;
    secondState.m_dx = -400.F;
    rules.BeforeStep(entities);
    firstState.m_x += firstState.m_dx * 0.2F;
    secondState.m_x += secondState.m_dx * 0.2F;
    rules.Resolve(entities);

    Require(!firstState.m_destroyed && !secondState.m_destroyed,
            "1.2 km blue near miss was treated as a collision");
}

void TestDelayedThreatActivation()
{
    using namespace ppo::model;
    using ppo::examples::twovstwo::EpisodeSpawnLayout;
    using ppo::examples::twovstwo::InitializerContext;

    entt::registry world;
    world.ctx().emplace<sEnvironmentState>().m_dt = 0.2F;
    auto& context = world.ctx().emplace<InitializerContext>();
    context.episodeLayout.emplace();
    context.episodeLayout->redActivationSteps = {
        1U, EpisodeSpawnLayout::NeverActivate};

    auto delayed = std::make_shared<Entity>(
        &world, eEntitySide::Hostile, eEntityType::Mobile);
    delayed->add<sEntityMetadata>(sEntityMetadata{"red_1", "attacker_car"});
    auto disabled = std::make_shared<Entity>(
        &world, eEntitySide::Hostile, eEntityType::Mobile);
    disabled->add<sEntityMetadata>(sEntityMetadata{"red_2", "attacker_car"});
    auto [delayedState] = delayed->get<sEntityState>();
    auto [disabledState] = disabled->get<sEntityState>();
    delayedState.m_destroyed = true;
    delayedState.m_prevDestroyed = true;
    disabledState.m_destroyed = true;
    disabledState.m_prevDestroyed = true;

    WorldEntities entities{delayed, disabled};
    ppo::examples::twovstwo::TwoVsTwoWorldRules rules;
    rules.BeforeStep(entities);
    Require(delayedState.m_destroyed && disabledState.m_destroyed,
            "a delayed threat activated before its scheduled step");
    rules.Resolve(entities);
    Require(context.elapsedSteps == 1U,
            "the threat curriculum clock did not follow the ECS step");
    rules.BeforeStep(entities);
    Require(!delayedState.m_destroyed && !delayedState.m_prevDestroyed,
            "the delayed threat was not activated on schedule");
    Require(disabledState.m_destroyed,
            "a never-scheduled threat was activated");
}

} // namespace

int main()
{
    try
    {
        TestSweptBlueCollision();
        TestBlueNearMiss();
        TestDelayedThreatActivation();
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
