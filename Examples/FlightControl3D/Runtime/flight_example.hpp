#pragma once
#include "example.hpp"
#include <filesystem>
namespace ppo::model { class Environment; }
namespace ppo::examples::flight {
ppo::app::ExampleDefinition Definition();
ppo::app::ExampleDefinition InferenceDefinition();
/** @brief Loads the dedicated desktop in training or validated-BEST inference mode. */
bool LoadDesktop(std::filesystem::path const& path,ppo::model::Environment& environment,bool inference);
}
