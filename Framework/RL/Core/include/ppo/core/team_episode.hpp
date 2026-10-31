/**
 * @file Framework/RL/Core/include/ppo/core/team_episode.hpp
 * @brief Interface ou implémentation du composant team episode.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ppo/core/environment.hpp"

namespace ppo::core {

// A domain-independent contract for parameter-sharing PPO and future MAPPO.
// Actors consume only their local observation. A centralized critic may consume
// the separate global state without changing an actor or the environment.
template<std::size_t N>
/** @brief Déclare le type TeamStep, utilisé par Interface ou implémentation du composant team episode. */
struct TeamStep
{
    static_assert(N > 0, "An episode must contain at least one agent");
    std::array<Observation, N> observations{};
    std::array<float, N> individualRewards{};
    float teamReward = 0.F;
    bool terminated = false;
    bool truncated = false;
/**
 * @brief Réalise l'opération Done; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
    bool Done() const noexcept { return terminated || truncated; }
};

template<std::size_t N>
/** @brief Déclare le type ITeamEnvironment, utilisé par Interface ou implémentation du composant team episode. */
class ITeamEnvironment
{
public:
/**
 * @brief Libère les ressources détenues par l'objet.
 * @details Peut signaler un état invalide par exception.
 */
    virtual ~ITeamEnvironment() = default;
/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 * @details Peut signaler un état invalide par exception.
 */
    virtual TeamStep<N> Reset() = 0;
/**
 * @brief Exécute une étape d'évolution ou de calcul du composant.
 * @details Peut signaler un état invalide par exception.
 */
    virtual TeamStep<N> Step(std::array<Action, N> const& actions) = 0;
/**
 * @brief Réalise l'opération LocalObservationSize; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
    virtual std::size_t LocalObservationSize() const noexcept = 0;
/**
 * @brief Réalise l'opération LocalActionSize; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
    virtual std::size_t LocalActionSize() const noexcept = 0;
    // Reserved for a centralized critic; local PPO can ignore it.
/**
 * @brief Réalise l'opération GlobalState; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
    virtual Observation GlobalState() const = 0;
};

template<std::size_t N>
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void ValidateTeamStep(ITeamEnvironment<N> const& environment,
                      TeamStep<N> const& step)
{
    for (auto const& obs : step.observations)
        if (obs.size() != environment.LocalObservationSize())
            throw std::invalid_argument("Team environment returned invalid observation size");
}
} // namespace ppo::core
