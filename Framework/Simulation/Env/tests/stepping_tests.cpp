/**
 * @file stepping_tests.cpp
 * @brief The world driven step by step by its caller (Environment::BeginStep / EndStep), as
 *        an environment DLL drives it: agents choose then read the new state in each step,
 *        an episode ended by an agent re-initializes the agents at the next step, and the
 *        seed given to the world draws the initializations of its entities.
 */
#include "environment.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

/** @brief Counts its calls and ends the episode every `episodeSteps` steps. */
class CountingAgent final : public ppo::model::IAgent
{
public:
    explicit CountingAgent(int episodeSteps) : episodeSteps_(episodeSteps) {}

    void Init() override
    {
        ++inits;
        step = 0;
    }
    void ChooseAction() override { ++choices; }
    bool SetNewState() override
    {
        ++feedbacks;
        ++step;
        return step == episodeSteps_;
    }

    int inits = 0;
    int choices = 0;
    int feedbacks = 0;
    int step = 0;

private:
    int episodeSteps_;
};

/** @brief The draws of DrawFromWorld, in the order the entities were initialized. */
std::vector<std::uint64_t> draws;

/** @brief Entity initializer: one draw of the world's initialization engine. */
void DrawFromWorld(ppo::model::Entity& entity)
{
    draws.push_back(ppo::model::InitializationEngine(*entity.registry())());
}

/** @brief The draws of two initializations of one entity, in a world seeded `seed`. */
std::vector<std::uint64_t> DrawsOfSeed(std::uint64_t seed)
{
    draws.clear();
    ppo::model::Environment environment;
    environment.SeedInitialization(seed);
    environment.SetEntityInitializer(&DrawFromWorld);
    environment.CreateEntity<ppo::model::Entity>(ppo::model::eEntitySide::Friend, ppo::model::eEntityType::Mobile);
    for (int episode = 0; episode < 2; ++episode)
    {
        environment.RequestReinitialization();
        environment.BeginStep();
    }
    return draws;
}

/** @brief The same seed gives the same initializations, another seed other ones. */
void TestSeedInitialization()
{
    std::vector<std::uint64_t> const first = DrawsOfSeed(7);
    Check(first.size() == 2 && first[0] != first[1], "each initialization draws anew from the world's engine");
    Check(DrawsOfSeed(7) == first, "the same seed gives the same initializations");
    Check(DrawsOfSeed(8) != first, "another seed gives other initializations");
}

} // namespace

int main()
{
    try
    {
        ppo::model::Environment environment;
        auto agent = environment.CreateAgent<CountingAgent>(3);

        environment.BeginStep();
        Check(agent->choices == 0 && agent->inits == 0, "BeginStep prepares the step without choosing");
        Check(!environment.EndStep() && agent->choices == 1 && agent->feedbacks == 1,
            "EndStep runs the choice and the feedback of the step");
        environment.BeginStep();
        Check(!environment.EndStep(), "the second step of the episode goes on");
        environment.BeginStep();
        Check(environment.EndStep(), "the agent ends the episode at its third step");
        Check(agent->inits == 0, "the end of an episode re-initializes nothing yet");
        environment.BeginStep();
        Check(agent->inits == 1 && agent->step == 0, "the next BeginStep re-initializes the agents");
        Check(!environment.EndStep() && agent->choices == 4, "and the new episode steps");
        environment.RequestReinitialization();
        environment.BeginStep();
        Check(agent->inits == 2 && agent->step == 0, "a requested re-initialization starts a new episode");
        environment.EndStep();
        TestSeedInitialization();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " stepping check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "environment stepping tests passed\n";
    return EXIT_SUCCESS;
}
