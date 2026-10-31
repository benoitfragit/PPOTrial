/**
 * @file Examples/BlueDefense/Scenario/bluedefenseexample.hpp
 * @brief Chargement BlueDefense et BluePatrol et composition des entités et agents.
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
 * @brief Réalise l'opération BluePatrolDefinition; voir les préconditions et effets dans son corps C++.
 */
ppo::app::ExampleDefinition BluePatrolDefinition();
}
