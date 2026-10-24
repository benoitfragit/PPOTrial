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
class ExampleCatalog final
{
public:
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
    ExampleDefinition const& Default() const noexcept { return m_definitions.front(); }

    ExampleDefinition const* Find(std::string_view id) const noexcept
    {
        auto const it = std::find_if(m_definitions.begin(), m_definitions.end(),
            [id](ExampleDefinition const& definition) { return definition.id == id; });
        return it == m_definitions.end() ? nullptr : &*it;
    }

    std::vector<ExampleDefinition> const& Entries() const noexcept { return m_definitions; }

private:
    std::vector<ExampleDefinition> m_definitions{};
};

// Composition root for compiled examples. Main does not depend on concrete
// example headers; each future example is added here and in Examples/CMakeLists.txt.
ExampleCatalog CreateExampleCatalog();

} // namespace ppo::app
