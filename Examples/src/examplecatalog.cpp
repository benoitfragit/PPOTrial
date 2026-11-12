/**
 * @file Examples/src/examplecatalog.cpp
 * @brief Interface ou implémentation du composant examplecatalog.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "examplecatalog.hpp"
#include "airspaceexample.hpp"
#include "bluedefenseexample.hpp"
#ifdef PPO_WITH_FLIGHT_CONTROL
#include "flight_example.hpp"
#endif

namespace ppo::app {

/**
 * @brief Construit, transforme ou échantillonne les données produites par CreateExampleCatalog.
 */
ExampleCatalog CreateExampleCatalog()
{
    // Keep Airspace first so the existing no-argument startup is unchanged.
    return ExampleCatalog{ppo::examples::airspace::Definition(),
                          ppo::examples::bluedefense::Definition(),
                          ppo::examples::bluedefense::InferenceDefinition(),
                          ppo::examples::bluedefense::EvaluationDefinition()
#ifdef PPO_WITH_FLIGHT_CONTROL
                          , ppo::examples::flight::Definition()
                          , ppo::examples::flight::InferenceDefinition()
#endif
                          };
}

} // namespace ppo::app
