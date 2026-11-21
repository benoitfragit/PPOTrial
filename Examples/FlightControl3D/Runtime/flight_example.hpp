/**
 * @file flight_example.hpp
 * @brief Registration of FlightControl3D in the desktop example framework (ECS environment).
 *
 * Role in the chain: session -> **desktop bridge** -> ppo::model::Environment -> renderer.
 * The definitions expose a training and an inference entry to the desktop application;
 * LoadDesktop builds the session for the dedicated FlightControl3D executable.
 *
 * Not here: the ECS integration details (flight_example.cpp) and drawing (Renderer/).
 *
 * Read next: flight_example.cpp, flight_desktop.cpp, Renderer/flight_view.hpp.
 */
#pragma once
#include "example.hpp"
#include <filesystem>
namespace ppo::model { class Environment; }
namespace ppo::examples::flight {
/** @brief Desktop entry that trains and validates a policy. */
ppo::app::ExampleDefinition Definition();
/** @brief Desktop entry that evaluates the validated BEST policy without training. */
ppo::app::ExampleDefinition InferenceDefinition();
/** @brief Loads the dedicated desktop in training or validated-BEST inference mode. */
bool LoadDesktop(std::filesystem::path const& path,ppo::model::Environment& environment,bool inference);
}
