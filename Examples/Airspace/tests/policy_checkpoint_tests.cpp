/**
 * @file Examples/Airspace/tests/policy_checkpoint_tests.cpp
 * @brief The red agents' production policies and training networks load from the repository.
 * @details Read-only: the policies of prod/ and eval/ run through the same helpers as the
 * production agents (MakeMissionPolicy, MakeInterceptionPolicy), and the training networks
 * of task.json and graph.json build and decide one action. Nothing is trained or saved.
 */
#include "interception.hpp"
#include "missionlayout.hpp"
#include "targetselector.hpp"

#include "ppo/core/agent_contract.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/**
 * @brief Distinct values catch a reordered observation; RLMission tracks get a binary
 *        presence, with present and absent slots.
 */
ppo::core::Observation MakeObservation(std::size_t dimension)
{
    ppo::core::Observation observation(dimension);
    for (std::size_t i = 0; i < dimension; ++i)
        observation[i] = 0.1F + static_cast<float>(i) * 0.001F;
    if (dimension == rl::MissionObservationLayout::kStateDimension)
    {
        constexpr auto stride = rl::MissionObservationLayout::kTrackFeatureCount;
        for (std::size_t track = 0; track < rl::MissionObservationLayout::kMaxTracks; ++track)
            observation[track * stride + stride - 1] = track % 3 == 0 ? 1.F : 0.F;
    }
    return observation;
}

/** @brief The policy exists, gives finite actions of the right size, and twice the same. */
void CheckPolicy(std::unique_ptr<ppo::core::IPolicy> const& policy, std::string const& label,
    std::size_t observationSize, std::size_t actionSize)
{
    if (policy == nullptr)
        throw std::runtime_error(label + ": no production policy (policy.ppokit)");
    ppo::core::Observation const observation = MakeObservation(observationSize);
    ppo::core::Action const first = policy->ChooseAction(observation);
    ppo::core::Action const second = policy->ChooseAction(observation);
    if (first.size() != actionSize || first != second)
        throw std::runtime_error(label + ": inference is not deterministic or has the wrong size");
    for (float const value : first)
        if (!std::isfinite(value))
            throw std::runtime_error(label + ": non-finite action");
    std::cout << "Production policy OK: " << label << std::endl;
}

/** @brief The PPOKit training networks of task.json and graph.json build and decide. */
template<std::size_t ObservationSize, std::size_t ActionSize>
void CheckTrainingNetwork(ppo::model::GraphNetworkFiles const& files, std::string const& label)
{
    // The critic shares the actor's trunk: one Adam, one learning rate (as the training agent sets).
    ppokit::core::PpoSettings settings;
    settings.criticLearningRate = settings.actorLearningRate;
    auto network = ppo::model::LoadGraphNetwork<ObservationSize, ActionSize>(files, settings);
    typename decltype(network)::Learner::Observation observation{};
    ppo::core::Observation const values = MakeObservation(ObservationSize);
    std::copy(values.begin(), values.end(), observation.begin());
    auto const decision = network.learner->Decide(observation);
    if (!std::isfinite(decision.logProbability) || !std::isfinite(decision.value))
        throw std::runtime_error(label + ": training network decides non-finite values");
    std::cout << "Training network OK: " << label << std::endl;
}

} // namespace

/** @brief Training and production agents keep the historical sizes and layouts. */
void CheckExistingAgentContracts()
{
    constexpr auto snaTraining = rl::SNATrainingAgent::Contract();
    constexpr auto snaProduction = rl::SNAProdAgent::Contract();
    constexpr auto missionTraining = rl::TargetSelectorTrainingAgent::Contract();
    constexpr auto missionProduction = rl::TargetSelectorProdAgent::Contract();

    static_assert(ppo::core::SameContract(snaTraining, snaProduction),
                  "RLSNA training and inference contracts differ");
    static_assert(ppo::core::SameContract(missionTraining, missionProduction),
                  "RLMission training and inference contracts differ");
    static_assert(snaTraining.observationSize == 6U && snaTraining.actionSize == 2U,
                  "RLSNA historical neural dimensions changed");
    static_assert(missionTraining.observationSize == 152U && missionTraining.actionSize == 3U,
                  "RLMission historical neural dimensions changed");
    static_assert(snaTraining.observationLayout == "rlsna.observation.v1",
                  "RLSNA observation layout changed");
    static_assert(snaTraining.actionLayout == "rlsna.action.v1",
                  "RLSNA action layout changed");
    static_assert(missionTraining.observationLayout == "rlmission.observation.v1",
                  "RLMission observation layout changed");
    static_assert(missionTraining.actionLayout == "rlmission.action.v1",
                  "RLMission action layout changed");
}

int main()
{
    try
    {
        CheckExistingAgentContracts();
        constexpr std::size_t missionObservation = rl::MissionObservationLayout::kStateDimension;
        constexpr std::size_t missionActions = rl::MissionObservationLayout::kActionCount;
        for (char const* version : {"prod", "eval"})
        {
            CheckPolicy(rl::MakeInterceptionPolicy(version), std::string("RLSNA/") + version, 6, 2);
            CheckPolicy(rl::MakeMissionPolicy(version), std::string("RLMission/") + version, missionObservation,
                missionActions);
        }
        CheckTrainingNetwork<6, 2>(rl::InterceptionNetworkFiles(), "RLSNA");
        CheckTrainingNetwork<missionObservation, missionActions>(rl::MissionNetworkFiles(), "RLMission");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
