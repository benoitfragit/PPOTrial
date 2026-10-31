/**
 * @file Examples/data/examplecatalog.hpp
 * @brief Interface ou implémentation du composant examplecatalog.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "example.hpp"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ppo::app {

// A catalog is composed explicitly by the application, never by static
// initialization. Entries are validated before any example is started.
/** @brief Déclare le type ExampleCatalog, utilisé par Interface ou implémentation du composant examplecatalog. */
class ExampleCatalog final
{
public:
/**
 * @brief Réalise l'opération ExampleCatalog; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
    explicit ExampleCatalog(std::initializer_list<ExampleDefinition> definitions)
    {
        for (auto const& definition : definitions)
        {
            if (!IsValid(definition))
                throw std::invalid_argument("ExampleCatalog: invalid example definition");
            if (Find(definition.id) != nullptr)
                throw std::invalid_argument("ExampleCatalog: duplicate example id");
            m_definitions.push_back(definition);
        }

        if (m_definitions.empty())
            throw std::invalid_argument("ExampleCatalog: at least one example is required");
    }

    // Preserve the first compiled example as the default CLI selection.
/**
 * @brief Réalise l'opération Default; voir les préconditions et effets dans son corps C++.
 */
    ExampleDefinition const& Default() const noexcept { return m_definitions.front(); }

/**
 * @brief Retourne une vue ou une valeur calculée depuis l'état courant.
 */
    ExampleDefinition const* Find(std::string_view id) const noexcept
    {
        auto const it = std::find_if(m_definitions.begin(), m_definitions.end(),
            [id](ExampleDefinition const& definition) { return definition.id == id; });
        return it == m_definitions.end() ? nullptr : &*it;
    }

/**
 * @brief Réalise l'opération Entries; voir les préconditions et effets dans son corps C++.
 */
    std::vector<ExampleDefinition> const& Entries() const noexcept { return m_definitions; }

private:
    std::vector<ExampleDefinition> m_definitions{};
};

// Composition root for compiled examples. Main does not depend on concrete
// example headers; each future example is added here and in Examples/CMakeLists.txt.
/**
 * @brief Construit, transforme ou échantillonne les données produites par CreateExampleCatalog.
 */
ExampleCatalog CreateExampleCatalog();

} // namespace ppo::app
