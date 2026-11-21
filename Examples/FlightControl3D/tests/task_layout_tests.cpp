/**
 * @file task_layout_tests.cpp
 * @brief The PPOKit layout generated from task.json matches the hand-written FlightControl3D contract.
 *
 * task.json describes the observation, action and reward shapes for PPOKit. Until the example
 * reads its layout from the generated header, both descriptions coexist: this test fails as soon
 * as one of them changes without the other.
 */
#include "flight_learner.hpp"
#include "flight_observation.hpp"
#include "flight_task_layout.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ppo::examples::flight;
using Layout = task::TaskLayout;

static_assert(Layout::Observation::Size == ObservationSize, "observation size");
static_assert(static_cast<std::size_t>(Layout::Observation::Slot::Rollout) == RolloutObservation, "rollout slot");
static_assert(Layout::Action::Size == ContinuousActions, "action size");
static_assert(static_cast<std::size_t>(Layout::Action::Component::Brake) == BrakeAction, "brake action");
static_assert(static_cast<std::size_t>(Layout::Action::Component::Throttle) == ThrottleAction, "throttle action");
static_assert(Layout::Reward::Count == RewardCount, "reward channel count");
static_assert(Layout::MaximumAgents == 1, "single aircraft");

namespace {

/** @brief Compares two name tables of the same size, reporting the first difference. */
template<typename Left, typename Right>
void CheckSameNames(Left const& generated, Right const& handWritten, char const* table)
{
    for (std::size_t index = 0; index < generated.size(); ++index)
    {
        if (std::strcmp(generated[index], handWritten[index]) != 0)
        {
            throw std::runtime_error(std::string(table) + " differs at index " + std::to_string(index) + ": '"
                + generated[index] + "' in task.json, '" + handWritten[index] + "' in the code");
        }
    }
}

} // namespace

int main()
{
    try
    {
        CheckSameNames(Layout::Observation::SlotNames, ObservationNames, "observation slot names");
        CheckSameNames(Layout::Action::ComponentNames, ActionNames, "action names");
        CheckSameNames(Layout::Reward::Names, RewardNames, "reward channel names");
        std::cout << "FlightControl3D task.json matches the code contract\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
