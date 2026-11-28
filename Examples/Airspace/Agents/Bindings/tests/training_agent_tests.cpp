/**
 * @file Examples/Airspace/Agents/Bindings/tests/training_agent_tests.cpp
 * @brief TrainingAgent driven from outside (as rlmission_environment drives it) and ProdAgent.
 * @details A one-dimensional model: the agent reads its position and pushes it; an episode
 * terminates beyond |2|. The agent's settings come from Training/Line.json.
 */
#include "prodagent.hpp"
#include "trainingagent.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

void Write(fs::path const& path, std::string const& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

/** @brief Position on a line; the action moves it. */
struct LineModel final
{
    using STATES = std::array<float, 2>;
    using ACTIONS = std::array<float, 1>;
    static constexpr std::array<std::string_view, 2> REWARDS{"Centre", "Exit"};
    static constexpr ppo::core::AgentContract kAgentContract{"Line", "line.observation.v1", 2, "line.action.v1", 1};

    float position = 0.5F;
    float lastAction = 0.F;

    void SetNewState(STATES& state) const noexcept { state = {position, 1.F}; }
    void SetNextAction(ACTIONS const& action) noexcept
    {
        lastAction = action[0];
        position += 0.3F * std::clamp(action[0], -1.F, 1.F);
    }
};

/** @brief Rewards the distance to the centre; exits beyond |2| (penalized, and the last step would be). */
class LineAgent final : public ppo::model::TrainingAgent<LineModel>
{
public:
    explicit LineAgent(LineModel& model) : TrainingAgent(&model, "Line"), model_(model) {}

    bool RewardFunction() noexcept override
    {
        AddReward(0, -std::abs(model_.position));
        bool const exited = std::abs(model_.position) > 2.F;
        if (exited || IsLastStep())
        {
            AddReward(1, -1.F);
        }
        SetDone(exited);
        return exited;
    }

private:
    LineModel& model_;
};

/** @brief One step as the environment DLL takes it: the caller's action, then the world's step. */
bool Step(LineAgent& agent, float action)
{
    agent.External().action[0] = action;
    agent.ChooseAction();
    return agent.SetNewState();
}

/** @brief A constant policy, for the production agent. */
class ConstantPolicy final : public ppo::core::IPolicy
{
public:
    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override
    {
        Check(observation.size() == 2, "the production agent hands its observation to its policy");
        return {0.75F};
    }
};

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path()
        / ("ppo_training_agent_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        Write(root / "Training" / "Line.json", R"({"formatVersion": 2,
            "rewards": {"clamp": 1.0, "weights": {"Centre": 0.5, "Exit": 2.0}},
            "training": {"rolloutSteps": 3}})");
        auto& serializer = ppo::serialization::Serializer::GetInstance();
        serializer.SetSearchPaths({root});
        LineModel model;
        LineAgent agent(model);
        serializer.Load();
        Check(agent.RolloutSteps() == 3, "the step limit is the rollout size of the settings");
        double weights[2]{};
        agent.RewardWeights(weights);
        Check(weights[0] == 0.5 && weights[1] == 2.0, "the channel weights are those of the settings");

        agent.Init();
        ppo::model::ExternalStep const& exchange = agent.External();
        Check(exchange.observation == std::vector<float>{0.5F, 1.F}, "Init publishes the first observation");

        // Step 1: the caller's action moves the model; weighted channel, reward under the clamp.
        Check(!Step(agent, 1.F), "the first step goes on");
        Check(model.lastAction == 1.F && std::abs(model.position - 0.8F) < 1e-6F, "the caller's action is applied");
        Check(std::abs(exchange.channels[0] + 0.5 * 0.8) < 1e-6 && exchange.channels[1] == 0.0,
            "the channels are weighted");
        Check(std::abs(exchange.reward + 0.4) < 1e-6 && !exchange.terminated && !exchange.ended,
            "the reward is the channels' sum");
        Check(std::abs(exchange.observation[0] - 0.8F) < 1e-6F, "the next observation is published");

        // Step 3 of 3 is the step limit's last: IsLastStep stays false, no penalty is paid.
        Step(agent, 0.F);
        Check(!Step(agent, 0.F), "the step limit does not end the episode in the agent (the caller truncates)");
        Check(exchange.channels[1] == 0.0 && std::abs(exchange.reward + 0.4) < 1e-6,
            "the step limit's last step is not penalized");

        // A new episode: an exit terminates, its reward clamped.
        model.position = 1.9F;
        agent.Init();
        Check(Step(agent, 1.F), "an exit ends the episode");
        Check(exchange.terminated && exchange.ended && exchange.channels[1] == -2.0, "an exit is terminal");
        Check(exchange.reward == -1.0, "the reward is clamped to the settings' clamp");

        double const live[2]{1.0, 0.0};
        agent.SetRewardWeights(live);
        model.position = 0.5F;
        agent.Init();
        Step(agent, 0.F);
        Check(std::abs(exchange.channels[0] + 0.5) < 1e-6, "new weights apply from the next step");

        // The production agent plays its policy.
        LineModel productionModel;
        ppo::model::ProdAgent<LineModel> production(&productionModel, "LineProd", std::make_unique<ConstantPolicy>());
        production.Init();
        production.ChooseAction();
        Check(productionModel.lastAction == 0.75F, "a production agent plays its policy");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " training agent check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "training agent checks passed\n";
    return EXIT_SUCCESS;
}
