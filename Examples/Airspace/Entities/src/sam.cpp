/**
* @file Entities/src/sam.cpp
* @brief Implementation for sam.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "sam.hpp"

#include "snasystem.hpp"

#include "serializer.hpp"

#include <stdexcept>

namespace ppo::model {

/**
* @brief Builds one SAM entity from its platform description file.
* @param registry Registry that owns the entity.
* @param name Platform file name without extension.
* @param side Tactical side assigned to the SAM.
*/
SAM::SAM(entt::registry* registry, std::string const& name, eEntitySide side):
    Entity(registry, side, eEntityType::SAM)
{
    std::filesystem::path filepath =
        std::filesystem::path{ RESOURCE_PATH } / "Platforms" / "Ground" / name;
    filepath.replace_extension(".json");
    filepath.make_preferred();

    if (std::filesystem::exists(filepath))
    {
        auto content = ppo::serialization::Serializer::Parse(filepath);

        if (!content.is_object() || content.value("formatVersion", 0) != 2
            || !content.contains("parameters") || !content["parameters"].is_object())
            throw std::runtime_error("invalid platform configuration: " + filepath.string());

        // Component factories currently accept a mutable JSON archive.
        // Keep the original by-value extraction to preserve their API.
        auto section = content["parameters"];

        SNASystem::Create(*this, section);
    }
}

/**
* @brief Initializes the SAM weapon state after base entity initialization.
*/
void
SAM::OnInit(void) noexcept
{
    if (has<sSNAState>())
        SNASystem::Init(*this);
}

/**
* @brief Advances the SAM simulation for one environment step.
*/
void
SAM::Run(void) noexcept
{
    Entity::Run();

    if (has<sSNAState>())
        SNASystem::Run(*this);
}

}
