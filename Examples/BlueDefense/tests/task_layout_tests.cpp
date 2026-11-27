/**
 * @file Examples/BlueDefense/tests/task_layout_tests.cpp
 * @brief The PPOKit layout generated from BluePatrol/task.json matches the v10 observation,
 *        the centralized critic input, the actions and the 15 reward channels of the code.
 *
 * Until the MAPPO runtime reads its layout from the generated header, both descriptions
 * coexist: this test fails as soon as one of them changes without the other.
 */
#include "blue_patrol_task_layout.hpp"
#include "generalized_cooperative_reward.hpp"
#include "generalized_critic_observation.hpp"
#include "generalized_observation.hpp"

#include <cstring>
#include <iostream>

namespace blue = ppo::examples::bluedefense::defenders;
using Layout = blue::task::TaskLayout;
using Observation = blue::GeneralizedObservationLayout;
using Critic = blue::GeneralizedCriticLayout;

// Actor observation v10, group by group.
static_assert(Layout::Observation::Size == Observation::Size);
static_assert(Layout::Observation::Own::Offset == Observation::OwnOffset);
static_assert(Layout::Observation::Lidar::Offset == Observation::LidarOffset);
static_assert(Layout::Observation::Role::Offset == Observation::RoleOffset);
static_assert(Layout::Observation::RedTracks::Offset == Observation::RedOffset);
static_assert(Layout::Observation::RedTracks::Capacity == Observation::RedTracks);
static_assert(Layout::Observation::RedTracks::Features == Observation::RedFeatures);
static_assert(Layout::Observation::BlueTracks::Offset == Observation::BlueOffset);
static_assert(Layout::Observation::BlueTracks::Capacity == Observation::BlueTracks);
static_assert(Layout::Observation::BlueTracks::Features == Observation::FriendlyFeatures);
static_assert(Layout::Observation::Objective::Offset == Observation::ObjectiveOffset);
static_assert(Layout::Observation::Objective::Capacity == Observation::ObjectiveTracks);
static_assert(Layout::Observation::Patrol::Offset == Observation::PatrolOffset);
static_assert(Layout::Observation::Global::Offset == Observation::GlobalOffset);
static_assert(Layout::Observation::TeamCounts::Offset == Observation::TeamCountsOffset);
// Every track carries its availability mask in its final feature.
static_assert(Layout::Observation::RedTracks::MaskFeature == Observation::RedFeatures - 1);
static_assert(Layout::Observation::BlueTracks::MaskFeature == Observation::FriendlyFeatures - 1);

// Centralized critic input conditioned on each participant.
static_assert(Layout::Critic::Size == Critic::Size);
static_assert(Layout::Critic::TeamMean::Offset == Critic::MeanOffset);
static_assert(Layout::Critic::TeamMax::Offset == Critic::MaxOffset);
static_assert(Layout::Critic::LivingFraction::Offset == Critic::LivingFractionOffset);
static_assert(Layout::Critic::TeamFraction::Offset == Critic::TeamFractionOffset);

// Team capacity and actions.
static_assert(Layout::MinimumAgents == 1 && Layout::MaximumAgents == blue::TeamCapacity::MaximumBlueMobiles);
static_assert(Layout::Action::Size == Observation::ActionSize);

// Reward channels, in GeneralizedRewardChannel order.
static_assert(Layout::Reward::Count == blue::kGeneralizedRewardChannelCount);
static_assert(static_cast<std::size_t>(Layout::Reward::Channel::OwnLoss)
    == static_cast<std::size_t>(blue::GeneralizedRewardChannel::OwnLoss));

int main()
{
    for (std::size_t channel = 0; channel < Layout::Reward::Count; ++channel)
    {
        if (std::strcmp(Layout::Reward::Labels[channel], blue::kGeneralizedRewardChannelNames[channel]) != 0)
        {
            std::cerr << "FAILED: reward channel " << channel << " is '" << Layout::Reward::Labels[channel]
                      << "' in task.json and '" << blue::kGeneralizedRewardChannelNames[channel]
                      << "' in the code\n";
            return 1;
        }
    }
    std::cout << "BluePatrol task.json matches the code contract\n";
    return 0;
}
