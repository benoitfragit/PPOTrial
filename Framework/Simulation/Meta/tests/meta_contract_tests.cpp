/**
 * @file Framework/Simulation/Meta/tests/meta_contract_tests.cpp
 * @brief Tests et assertions déterministes pour meta contract tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "metaregistry.hpp"
#include "environment.hpp"
#include <stdexcept>

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 * @details Peut signaler un état invalide par exception.
 */
int main()
{
    using namespace ppo::reflection;
    RegisterAllMeta();
    RegisterAllMeta();

    if (!IsMetaRegistered() || !ResolveType("BaseEntity") ||
        !ResolveType("Entity") || !ResolveType("IAgent") ||
        !ResolveType("EntityState"))
        throw std::runtime_error("Missing shared reflected type");

    if (ResolveType("Aircraft") || ResolveType("SNATrainingAgent"))
        throw std::runtime_error("Airspace types leaked into MetaReflection");

    ppo::model::Environment world;
    if (CreateEntity("Aircraft", world, "", ppo::model::eEntitySide::Unknown))
        throw std::runtime_error("Unknown entity unexpectedly registered");
    return 0;
}
