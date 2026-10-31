/**
 * @file Examples/tests/examplecatalog_tests.cpp
 * @brief Tests et assertions déterministes pour examplecatalog tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "examplecatalog.hpp"

#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace {

/**
 * @brief Met à jour ou enregistre les données sous la responsabilité.
 * @details Utilise des fichiers ou chemins. Peut signaler un état invalide par exception.
 */
void RegisterTypes() {}

/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées.
 * @details Peut signaler un état invalide par exception.
 */
bool LoadScenario(std::filesystem::path const&, ppo::model::Environment&)
{
    return true;
}

ppo::app::ExampleDefinition MakeDefinition(std::string_view id)
{
    return {id, "Test", "Scenarios/Test.json", &RegisterTypes, &LoadScenario};
}

template<typename Function>
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void RequireInvalid(Function&& function)
{
    try
    {
        function();
    }
    catch (std::invalid_argument const&)
    {
        return;
    }
    throw std::runtime_error("Invalid example catalog was accepted");
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 * @details Peut signaler un état invalide par exception.
 */
int main()
{
    using ppo::app::ExampleCatalog;

    ExampleCatalog const catalog{MakeDefinition("airspace"), MakeDefinition("second")};
    if (catalog.Entries().size() != 2U
        || catalog.Default().id != "airspace"
        || catalog.Find("airspace") == nullptr
        || catalog.Find("second") == nullptr
        || catalog.Find("unknown") != nullptr)
        throw std::runtime_error("Example lookup or ordering failed");

    RequireInvalid([] { ExampleCatalog const empty{}; });
    RequireInvalid([] {
        ExampleCatalog const duplicates{MakeDefinition("airspace"), MakeDefinition("airspace")};
    });
    RequireInvalid([] {
        ExampleCatalog const emptyId{MakeDefinition("")};
    });
    RequireInvalid([] {
        auto broken = MakeDefinition("broken");
        broken.registerTypes = nullptr;
        ExampleCatalog const missingCallback{broken};
    });
    RequireInvalid([] {
        auto broken = MakeDefinition("broken");
        broken.loadScenario = nullptr;
        ExampleCatalog const missingLoader{broken};
    });
    RequireInvalid([] {
        auto broken = MakeDefinition("broken");
        broken.defaultScenario.clear();
        ExampleCatalog const missingScenario{broken};
    });
    return 0;
}
