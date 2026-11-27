/**
 * @file Examples/BlueDefense/tests/network_type_tests.cpp
 * @brief The critic generated from BluePatrol/graph.json keeps the shape that the existing
 *        training checkpoints were written with.
 *
 * GeneralizedCriticConfig::Network is the generated type. The training checkpoint header
 * records its hidden width and layer count, so changing graph.json would make every saved
 * MAPPO training state unreadable: this test makes such a change explicit.
 */
#include "generalized_critic_network.hpp"

#include <iostream>
#include <type_traits>

namespace blue = ppo::examples::bluedefense::defenders;
using Config = blue::GeneralizedCriticConfig;

static_assert(std::is_same_v<Config::Network,
    blue::task::networks::Critic::Network<blue::Index, blue::TrainingParameters::BATCH_SIZE>>,
    "the MAPPO critic is the type generated from graph.json");
static_assert(Config::HiddenDim == 256 && Config::Layers == 3,
    "graph.json changed the critic shape recorded in existing training checkpoints (256 wide, 3 layers)");
static_assert(Config::Network::SPEC::INPUT_DIM == blue::GeneralizedCriticLayout::Size, "centralized critic input");

int main()
{
    std::cout << "BluePatrol critic generated from graph.json keeps the checkpoint shape\n";
    return 0;
}
