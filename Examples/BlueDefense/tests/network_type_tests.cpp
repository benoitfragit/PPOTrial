/**
 * @file Examples/BlueDefense/tests/network_type_tests.cpp
 * @brief The critic generated from BluePatrol/graph.json is exactly the hand-written MAPPO critic type.
 *
 * Identical rl_tools types guarantee identical layers, initialization and arithmetic. The
 * attention actor stays hand-written ("model": "external" in graph.json).
 */
#include "generalized_critic_network.hpp"

#include "blue_patrol_networks.hpp"

#include <iostream>
#include <type_traits>

namespace blue = ppo::examples::bluedefense::defenders;
namespace generated = blue::task::networks;

using GeneratedCritic = generated::Critic::Network<blue::Index, blue::TrainingParameters::BATCH_SIZE>;
static_assert(std::is_same_v<GeneratedCritic, blue::GeneralizedCriticConfig::Network>,
    "graph.json critic differs from GeneralizedCriticConfig::Network");
static_assert(generated::Critic::Inputs == blue::GeneralizedCriticLayout::Size, "centralized critic input");

int main()
{
    std::cout << "BluePatrol graph.json critic is the hand-written MAPPO critic type\n";
    return 0;
}
