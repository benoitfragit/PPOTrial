/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_advantage.hpp
 * @brief Implémentation des calculs et contrats d'entraînement generalized advantage.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief One GAE step for an episode participant, alive or destroyed.
 *
 * Only a real mission terminal removes bootstrap. A time-limit truncation
 * keeps bootstrap but stops the trace at the reset. Physical availability
 * belongs to the actor mask and deliberately does not enter this recursion.
 */
inline float GeneralizedMissionAdvantage(
    float const reward, float const value, float const nextValue,
    float const followingAdvantage, float const gamma, float const lambda,
    bool const terminated, bool const truncated) noexcept
{
    float const bootstrap = terminated ? 0.F : nextValue;
    float const trace = terminated || truncated ? 0.F : followingAdvantage;
    return reward + gamma * bootstrap - value + gamma * lambda * trace;
}

} // namespace ppo::examples::bluedefense::defenders
