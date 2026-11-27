/**
 * @file Examples/Airspace/tests/converted_policy_tests.cpp
 * @brief The PPOKit policies converted from the LibTorch agents (policy.ppokit) act as the
 *        legacy networks (policy.txt), on the reference observations of the legacy fixture.
 * @details The legacy policies return the Gaussian means and the raw fire logit; the graph
 * policies return the means and fire = 1 when the logit is positive. RLMission's graph masks
 * like the LibTorch network (mask_fill -1e9, absent tracks encoded): every reference case is
 * compared, those with no present track included (the -1e9 propagates to means of order
 * 1e8). The one left out is a fractional presence: the legacy mean pool weighted the tracks
 * by their presence, PPOKit's mask is a flag (presence > 0.5); the environment writes a
 * presence of 0 or 1 only (targetselector.cpp clears the observation, then sets 1 per track).
 */
#include "interception.hpp"
#include "interception_policy.hpp"
#include "legacy_policy_reference.hpp"
#include "mission_policy.hpp"
#include "missionlayout.hpp"
#include "targetselector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

int failures = 0;

/** @brief Relative tolerance of a converted mean: float32 summation order only. */
constexpr double MeanTolerance = 1e-4;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::filesystem::path ModelFile(std::string const& model, char const* file)
{
    return std::filesystem::path(PPO_REPOSITORY) / "Examples/Airspace/models" / model / file;
}

/** @brief True when a track presence is neither 0 nor 1. */
bool HasFractionalPresence(ppo::core::Observation const& observation)
{
    constexpr std::size_t stride = rl::MissionObservationLayout::kTrackFeatureCount;
    for (std::size_t track = 0; track < rl::MissionObservationLayout::kMaxTracks; ++track)
    {
        float const presence = observation[track * stride + stride - 1];
        if (presence != 0.F && presence != 1.F)
        {
            return true;
        }
    }
    return false;
}

/** @brief True when no track is present. */
bool HasNoTrack(ppo::core::Observation const& observation)
{
    constexpr std::size_t stride = rl::MissionObservationLayout::kTrackFeatureCount;
    for (std::size_t track = 0; track < rl::MissionObservationLayout::kMaxTracks; ++track)
    {
        if (observation[track * stride + stride - 1] != 0.F)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief The graph policy's means equal the legacy means, and it fires exactly when the
 *        legacy fire logit is positive.
 * @param gaussian Number of Gaussian components before the Bernoulli ones.
 */
void Compare(ppo::core::IPolicy& converted, ppo::core::IPolicy& legacy, std::string const& model,
    std::vector<rl::test::ReferenceCase> const& cases, std::size_t gaussian, bool mission)
{
    std::size_t compared = 0;
    std::size_t withoutTrack = 0;
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        ppo::core::Observation const& observation = cases[index].observation;
        if (mission && HasFractionalPresence(observation))
        {
            continue;
        }
        if (mission && HasNoTrack(observation))
        {
            ++withoutTrack;
        }
        ppo::core::Action const graph = converted.ChooseAction(observation);
        ppo::core::Action const reference = legacy.ChooseAction(observation);
        bool same = graph.size() == reference.size();
        for (std::size_t component = 0; same && component < graph.size(); ++component)
        {
            if (component < gaussian)
            {
                double const scale = std::max(1.0, std::abs(static_cast<double>(reference[component])));
                same = std::abs(graph[component] - reference[component]) <= MeanTolerance * scale;
            }
            else
            {
                same = (graph[component] == 1.F) == (reference[component] > 0.F);
            }
        }
        Check(same, model + " case " + std::to_string(index) + ": the converted policy acts as the legacy one");
        ++compared;
    }
    Check(compared > 0, model + ": reference cases compared");
    Check(!mission || withoutTrack > 0, model + ": a case with no present track is compared");
}

} // namespace

int main()
{
    try
    {
        rl::test::ReferenceCases const cases = rl::test::ReadReferenceCases(PPO_LEGACY_POLICY_REFERENCE);
        constexpr std::size_t missionObservation = rl::MissionObservationLayout::kStateDimension;
        constexpr std::size_t missionActions = rl::MissionObservationLayout::kActionCount;
        for (char const* version : {"prod", "eval"})
        {
            std::string const mission = std::string("RLMission/") + version;
            ppo::model::GraphPolicy<missionObservation, missionActions> missionGraph(
                rl::MissionNetworkFiles(), ModelFile(mission, "policy.ppokit"));
            rl::MissionPolicy missionLegacy(ModelFile(mission, "policy.txt"));
            Compare(missionGraph, missionLegacy, mission, cases.at(mission), 2, true);
            std::string const sna = std::string("RLSNA/") + version;
            ppo::model::GraphPolicy<6, 2> snaGraph(rl::InterceptionNetworkFiles(), ModelFile(sna, "policy.ppokit"));
            rl::InterceptionPolicy snaLegacy(ModelFile(sna, "policy.txt"), 6, 2);
            Compare(snaGraph, snaLegacy, sna, cases.at(sna), 2, false);
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " converted policy check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "converted red policy checks passed\n";
    return EXIT_SUCCESS;
}
