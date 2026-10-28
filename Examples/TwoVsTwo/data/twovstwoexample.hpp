#pragma once
#include "example.hpp"
namespace ppo::examples::twovstwo {
ppo::app::ExampleDefinition Definition();
ppo::app::ExampleDefinition BluePatrolDefinition();
/** @brief Returns the isolated generalized P-vs-N training example. */
ppo::app::ExampleDefinition GeneralizedPvsNDefinition();
}
