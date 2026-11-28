/**
 * @file Framework/Simulation/Meta/tests/meta_contract_tests.cpp
 * @brief The factory tables: unknown names create nothing, a registered name creates its type.
 */
#include "metaregistry.hpp"
#include "environment.hpp"

#include <stdexcept>

namespace {

/** @brief A plain entity, for the registered factory. */
std::shared_ptr<ppo::model::Entity>
CreatePlain(ppo::model::Environment& environment, std::string_view, ppo::model::eEntitySide side)
{
    return environment.CreateEntity<ppo::model::Entity>(side, ppo::model::eEntityType::Objective);
}

} // namespace

int main()
{
    using namespace ppo::reflection;
    ppo::model::Environment world;
    if (CreateEntity("Aircraft", world, "", ppo::model::eEntitySide::Unknown))
        throw std::runtime_error("Unknown entity unexpectedly registered");
    ppo::model::BaseEntity entity{nullptr, entt::null};
    if (CreateAgent("SNATrainingAgent", world, entity))
        throw std::runtime_error("Airspace agents leaked into the generic tables");

    RegisterEntityFactory("PlainObjective", &CreatePlain);
    auto const created = CreateEntity("PlainObjective", world, "", ppo::model::eEntitySide::Friend);
    if (created == nullptr || !created->has<ppo::model::sEntityState>())
        throw std::runtime_error("A registered entity factory did not create its entity");
    return 0;
}
