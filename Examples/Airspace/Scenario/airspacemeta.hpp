/**
 * @file Examples/Airspace/Scenario/airspacemeta.hpp
 * @brief Construction, données et arbitrage du scénario airspacemeta.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

namespace ppo::examples::airspace {
// Register Airspace's concrete types and factories once, on the application
// thread, before creating agents or starting evaluation workers.
/**
 * @brief Met à jour ou enregistre les données sous la responsabilité.
 */
void RegisterAirspaceMeta();
} // namespace ppo::examples::airspace
