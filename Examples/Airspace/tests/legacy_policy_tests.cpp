/**
 * @file Examples/Airspace/tests/legacy_policy_tests.cpp
 * @brief The red policies read from policy.txt give the reference actions, without LibTorch.
 */
#include "interception_policy.hpp"
#include "legacy_policy_reference.hpp"
#include "mission_policy.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::filesystem::path PolicyFile(std::string const& model)
{
    return std::filesystem::path(PPO_REPOSITORY) / "Examples/Airspace/models" / model / "policy.txt";
}

/** @brief Every reference case of `model` gives its action, twice (no hidden state). */
void CheckModel(ppo::core::IPolicy& policy, std::string const& model, std::vector<rl::test::ReferenceCase> const& cases)
{
    Check(!cases.empty(), model + " has reference cases");
    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            ppo::core::Action const action = policy.ChooseAction(cases[index].observation);
            bool same = action.size() == cases[index].action.size();
            for (std::size_t component = 0; same && component < action.size(); ++component)
            {
                same = rl::test::SameValue(action[component], cases[index].action[component]);
            }
            Check(same, model + " case " + std::to_string(index) + " gives the reference action");
        }
    }
}

/** @brief Malformed files and wrong sizes are refused with a reason. */
void RefusesWrongInput()
{
    bool refused = false;
    try
    {
        rl::MissionPolicy policy(PolicyFile("RLSNA/prod"));
    }
    catch (std::runtime_error const& error)
    {
        refused = std::string(error.what()).find("expected mission_track_attention") != std::string::npos;
    }
    Check(refused, "an RLSNA file is not taken for an RLMission policy");
    rl::InterceptionPolicy sna(PolicyFile("RLSNA/prod"), 6, 2);
    refused = false;
    try
    {
        sna.ChooseAction(ppo::core::Observation(5, 0.F));
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "an observation of the wrong size is refused");
    std::filesystem::path const broken = std::filesystem::temp_directory_path() / "broken_legacy_policy.txt";
    std::ofstream(broken) << "ppokit.legacy_policy/1\narchitecture interception_mlp\nparameter fc1.weight 2 2\n1 2\n3\nend\n";
    refused = false;
    try
    {
        rl::LegacyPolicyFile file(broken);
    }
    catch (std::runtime_error const& error)
    {
        refused = std::string(error.what()).find(":5: expected 2 finite values") != std::string::npos;
    }
    Check(refused, "a short row is refused with its line");
    std::filesystem::remove(broken);
}

} // namespace

int main()
{
    try
    {
        rl::test::ReferenceCases const cases = rl::test::ReadReferenceCases(PPO_LEGACY_POLICY_REFERENCE);
        for (char const* version : {"prod", "eval"})
        {
            std::string const mission = std::string("RLMission/") + version;
            rl::MissionPolicy missionPolicy(PolicyFile(mission));
            CheckModel(missionPolicy, mission, cases.at(mission));
            std::string const sna = std::string("RLSNA/") + version;
            rl::InterceptionPolicy snaPolicy(PolicyFile(sna), 6, 2);
            CheckModel(snaPolicy, sna, cases.at(sna));
        }
        RefusesWrongInput();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " legacy policy check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "legacy red policy checks passed\n";
    return EXIT_SUCCESS;
}
