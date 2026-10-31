/**
 * @file Framework/Simulation/Env/data/worldrules.hpp
 * @brief Interface ou implémentation du composant worldrules.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <vector>

namespace ppo::model {

/** @brief Déclare le type Entity, utilisé par Interface ou implémentation du composant worldrules. */
class Entity;
using WorldEntities = std::vector<std::shared_ptr<Entity>>;

// Scenario extension point; no policy, learning, reward or physics assumptions.
// Called only on the simulation thread (except initial registration).
/** @brief Déclare le type IWorldRules, utilisé par Interface ou implémentation du composant worldrules. */
class IWorldRules
{
public:
/**
 * @brief Libère les ressources détenues par l'objet.
 * @details Consulte ou modifie des composants ECS.
 */
    virtual ~IWorldRules() = default;

/**
 * @brief Réalise l'opération BeforeStep; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
    virtual void BeforeStep(WorldEntities& entities) = 0;
/**
 * @brief Traite les entrées et applique les effets associés à Resolve.
 * @details Consulte ou modifie des composants ECS.
 */
    virtual void Resolve(WorldEntities& entities) = 0;
/**
 * @brief Réalise l'opération AfterStep; voir les préconditions et effets dans son corps C++.
 * @details Consulte ou modifie des composants ECS.
 */
    virtual void AfterStep() = 0;
/**
 * @brief Réalise l'opération CopySnapshot; voir les préconditions et effets dans son corps C++.
 */
    virtual void CopySnapshot(entt::registry& source, entt::registry& destination) const = 0;
};

} // namespace ppo::model
