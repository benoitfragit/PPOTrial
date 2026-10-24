/**
* @file Entities/src/aircraft.cpp
* @brief Implementation for aircraft.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "mobile.hpp"
#include "sharedworldcomponents.hpp"

#include "snasystem.hpp"
#include "flightmodelsystem.hpp"
#include "snacontrolsystem.hpp"

#include "serializer.hpp"

#include <stdexcept>

namespace ppo::model {

/**
* @brief Builds one aircraft entity from its platform description file.
* @param registry Registry that owns the entity.
* @param name Platform file name without extension.
* @param side Tactical side assigned to the aircraft.
*/
Mobile::Mobile(entt::registry* registry, std::string const& name, eEntitySide side):
    Entity(registry, side, eEntityType::Mobile)
{
    std::filesystem::path filepath =
        std::filesystem::path{ RESOURCE_PATH } / "Platforms" / "Mobile" / name;
    filepath.replace_extension(".json");
    filepath.make_preferred();

    // Existing user-defined platform files under the historical directory remain valid.
    if (!std::filesystem::exists(filepath))
        filepath = std::filesystem::path{ RESOURCE_PATH } / "Platforms" / "Aircraft" / name;
    filepath.replace_extension(".json");

    if (std::filesystem::exists(filepath))
    {
        auto content = ppo::serialization::Serializer::Parse(filepath);

        if (!content.is_object() || content.value("formatVersion", 0) != 2
            || !content.contains("parameters") || !content["parameters"].is_object())
            throw std::runtime_error("invalid platform configuration: " + filepath.string());

        // Component factories currently accept a mutable JSON archive.
        // Keep the original by-value extraction to preserve their API.
        auto section = content["parameters"];

        FlightModelSystem::Create(*this, section);
        SNASystem::Create(*this, section);
        SNAControlSystem::Create(*this, section);
    }
}

/**
* @brief Initializes control, flight, and weapon subsystems in dependency order.
*/
void
Mobile::OnInit(void) noexcept
{
    if (has<sSNAControlState>())
        SNAControlSystem::Init(*this);

    if (has<sFlightModelState>())
        FlightModelSystem::Init(*this);

    if (has<sSNAState>())
        SNASystem::Init(*this);
}

/**
* @brief Runs control, flight, kinematics, and weapon systems for one step.
*/
void
Mobile::Run(void) noexcept
{
    if (has<sSNAControlState>())
        SNAControlSystem::Run(*this);

    if (has<sFlightModelState>())
        FlightModelSystem::Run(*this);

    Entity::Run();

    if (has<sSNAState>())
        SNASystem::Run(*this);
}

}
