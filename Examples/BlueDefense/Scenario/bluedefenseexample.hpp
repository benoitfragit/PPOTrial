/**
 * @file Examples/BlueDefense/Scenario/bluedefenseexample.hpp
 * @brief Chargement des modes entraînement et inférence de BlueDefense.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once
#include "example.hpp"
namespace ppo::examples::bluedefense {
/**
 * @brief Réalise l'opération Definition; voir les préconditions et effets dans son corps C++.
 */
ppo::app::ExampleDefinition Definition();
/**
 * @brief Définit le scénario BlueDefense exécuté avec la politique de production.
 */
ppo::app::ExampleDefinition InferenceDefinition();
}
