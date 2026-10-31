/**
 * @file Examples/Airspace/tests/airspace_meta_tests.cpp
 * @brief Tests et assertions déterministes pour airspace meta tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "airspacemeta.hpp"
#include "metaregistry.hpp"
#include "environment.hpp"
#include "components.hpp"
#include "airspacecomponents.hpp"
#include "mobile.hpp"
#include "restrictedzone.hpp"
#include <stdexcept>
#include <initializer_list>

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 * @details Peut signaler un état invalide par exception.
 */
int main()
{
    using namespace ppo::reflection;
    ppo::examples::airspace::RegisterAirspaceMeta();
    ppo::examples::airspace::RegisterAirspaceMeta();

    for (auto const* name : {"Aircraft", "SAM", "Mobile", "RestrictedZone", "Objective",
                             "SNATrainingAgent", "SNAProdAgent",
                             "TargetSelectorTrainingAgent", "TargetSelectorProdAgent",
                             "SNAState", "FlightModelState", "PseudoLidarState",
                             "InterceptionStateModel", "MissionStateModel"})
    {
        if (!ResolveType(name))
            throw std::runtime_error(name);
    }


    // Each pair of scenario names must resolve to the same reflected C++ class.
    if (ResolveType("Aircraft") != ResolveType("Mobile")
        || ResolveType("SAM") != ResolveType("RestrictedZone"))
        throw std::runtime_error("Legacy reflection aliases do not match");

    ppo::model::Environment env;
    auto objective = CreateEntity("Objective", env, "", ppo::model::eEntitySide::Friend);
    if (objective == nullptr || !objective->has<ppo::model::sEntityState>())
        throw std::runtime_error("Airspace objective factory failed");

    for (auto const* name : {"Aircraft", "Mobile"})
    {
        auto entity = CreateEntity(name, env, "aircraft", ppo::model::eEntitySide::Friend);
        if (!entity || dynamic_cast<ppo::model::Mobile*>(entity.get()) == nullptr
            || !entity->has<ppo::model::sEntityState>())
            throw std::runtime_error("Mobile factory failed");
        auto [state] = entity->get<ppo::model::sEntityState>();
        if (state.m_type != ppo::model::eEntityType::Mobile
            || !entity->has<ppo::model::sFlightModelState, ppo::model::sSNAState>())
            throw std::runtime_error("Mobile factory category/components mismatch");
    }
    for (auto const* name : {"SAM", "RestrictedZone"})
    {
        auto entity = CreateEntity(name, env, "sam", ppo::model::eEntitySide::Hostile);
        if (!entity || dynamic_cast<ppo::model::RestrictedZone*>(entity.get()) == nullptr)
            throw std::runtime_error("RestrictedZone factory failed");
        auto [state] = entity->get<ppo::model::sEntityState>();
        if (state.m_type != ppo::model::eEntityType::RestrictedZone
            || !entity->has<ppo::model::sSNAState>())
            throw std::runtime_error("RestrictedZone factory category/components mismatch");
    }
    if (CreateEntity("NoSuchEntity", env, "", ppo::model::eEntitySide::Friend))
        throw std::runtime_error("Unknown entity should not be created");
    return 0;
}
