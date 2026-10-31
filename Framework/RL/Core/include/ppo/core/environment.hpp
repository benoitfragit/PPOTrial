/**
 * @file Framework/RL/Core/include/ppo/core/environment.hpp
 * @brief Thread ECS, ordonnanceur d'agents, règles de monde et snapshots synchronisés.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <cstddef>
#include <vector>

namespace ppo::core {

// Domain-independent, owned observation/action values. No tensor or ECS lifetime escapes.
using Observation = std::vector<float>;
using Action = std::vector<float>;

/** @brief Déclare le type Transition, utilisé par Thread ECS, ordonnanceur d'agents, règles de monde et snapshots synchronisés. */
struct Transition {
    Observation observation;
    float reward = 0.F;
    bool done = false;
};

// Reset and Step are executed by the caller on a single thread; no implicit reset.
/** @brief Déclare le type IEnvironment, utilisé par Thread ECS, ordonnanceur d'agents, règles de monde et snapshots synchronisés. */
class IEnvironment {
public:
/**
 * @brief Libère les ressources détenues par l'objet.
 */
    virtual ~IEnvironment() = default;
/**
 * @brief Réalise l'opération ObservationSize; voir les préconditions et effets dans son corps C++.
 */
    virtual std::size_t ObservationSize() const noexcept = 0;
/**
 * @brief Réalise l'opération ActionSize; voir les préconditions et effets dans son corps C++.
 */
    virtual std::size_t ActionSize() const noexcept = 0;
/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 */
    virtual Observation Reset() = 0;
/**
 * @brief Exécute une étape d'évolution ou de calcul du composant.
 */
    virtual Transition Step(Action const& action) = 0;
};

/** @brief Déclare le type IPolicy, utilisé par Thread ECS, ordonnanceur d'agents, règles de monde et snapshots synchronisés. */
class IPolicy {
public:
/**
 * @brief Libère les ressources détenues par l'objet.
 */
    virtual ~IPolicy() = default;
/**
 * @brief Réalise l'opération ChooseAction; voir les préconditions et effets dans son corps C++.
 */
    virtual Action ChooseAction(Observation const& observation) = 0;
};

// Optional observer; no assumptions about the optimization algorithm or rewards.
/** @brief Déclare le type ILearner, utilisé par Thread ECS, ordonnanceur d'agents, règles de monde et snapshots synchronisés. */
class ILearner : public IPolicy {
public:
/**
 * @brief Réalise l'opération Observe; voir les préconditions et effets dans son corps C++.
 */
    virtual void Observe(Transition const& transition) = 0;
};

} // namespace ppo::core
