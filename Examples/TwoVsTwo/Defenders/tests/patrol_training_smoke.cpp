#include "patrol_learner.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace patrol = ppo::examples::twovstwo::defenders;

int main()
{
    patrol::Parameters params{};
    // Exercise two finite-horizon boundaries inside one rollout so GAE must
    // retain the final transition and cut bootstrapping across each reset.
    params.episodeSteps = 64U;
    patrol::PatrolLearner trainer{params, 7U, 128U};
    auto state = patrol::InitialState(params, 1, 0U, 0.F);

    // One full real-transition PPO rollout; there must be no simulator-driven
    // training update before all feedback samples have arrived.
    for (std::size_t step = 0; step < trainer.RolloutSize(); ++step)
    {
        patrol::PatrolLearner::Observations observations{};
        for (std::size_t role = 0; role < observations.size(); ++role)
        {
            state.controlledRole = role;
            observations[role] = patrol::Observe(params, state);
        }
        auto const actions = trainer.ChooseActions(observations);
        if (!trainer.HasPendingAction())
            return 1;
        auto after = patrol::AdvanceBoth(params, state, actions);
        auto nextObservations = observations;
        for (std::size_t role = 0; role < nextObservations.size(); ++role)
        {
            after.controlledRole = role;
            nextObservations[role] = patrol::Observe(params, after);
        }
        patrol::RewardValues weights{};
        weights.fill(1.F);
        float const reward = patrol::TeamRewardBreakdown(
            params, state, actions, after).Total(weights, 100.F);
        bool const truncated = after.step >= params.episodeSteps;
        bool const updated = trainer.Observe(nextObservations, reward,
                                             after.failed, truncated);
        if (trainer.HasPendingAction() ||
            updated != (step + 1U == trainer.RolloutSize()))
            return 2;
        state = after.failed || truncated
            ? patrol::InitialState(params, 1, 0U, 0.F) : after;
    }
    if (trainer.StepCount() != 1U || trainer.CollectedSteps() != 0U)
        return 3;
    auto const diagnostics = trainer.Diagnostics();
    if (!std::isfinite(diagnostics.totalLoss)
        || !std::isfinite(diagnostics.actorLoss)
        || !std::isfinite(diagnostics.valueLoss)
        || !std::isfinite(diagnostics.entropyLoss)
        || diagnostics.valueLoss <= 0.F)
        return 4;

    // A complete training checkpoint restores the centralized Critic, both
    // Adam states, update counter and RNG, but deliberately not the rollout.
    auto const unique = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    auto const trainingCheckpoint = std::filesystem::temp_directory_path()
        / ("ppotrial_blue_training_" + std::to_string(unique) + ".txt");
    auto const repeatedCheckpoint = std::filesystem::temp_directory_path()
        / ("ppotrial_blue_training_repeat_" + std::to_string(unique) + ".txt");
    trainer.SaveTrainingState(trainingCheckpoint);
    patrol::PatrolLearner resumedTraining{params, 19U, 256U};
    resumedTraining.LoadTrainingState(trainingCheckpoint);
    if (resumedTraining.StepCount() != trainer.StepCount()
        || resumedTraining.CollectedSteps() != 0U
        || resumedTraining.HasPendingAction())
        return 5;
    resumedTraining.SaveTrainingState(repeatedCheckpoint);
    if (!std::filesystem::exists(trainingCheckpoint)
        || !std::filesystem::exists(repeatedCheckpoint)
        || std::filesystem::file_size(trainingCheckpoint) == 0U
        || std::filesystem::file_size(repeatedCheckpoint) == 0U)
        return 6;
    auto readFile = [](std::filesystem::path const& path) {
        std::ifstream stream{path, std::ios::binary};
        return std::string{std::istreambuf_iterator<char>{stream},
                           std::istreambuf_iterator<char>{}};
    };
    bool const sameTrainingState =
        readFile(trainingCheckpoint) == readFile(repeatedCheckpoint);
    std::filesystem::remove(trainingCheckpoint);
    std::filesystem::remove(repeatedCheckpoint);
    if (!sameTrainingState)
        return 7;

    // A restarted learner must resume the saved Actor instead of silently
    // replacing it with new random weights.
    auto const checkpoint = std::filesystem::temp_directory_path()
        / ("ppotrial_blue_mappo_" + std::to_string(unique) + ".txt");
    std::ostringstream expected;
    trainer.WithActor([&](auto const& actor) {
        patrol::WritePatrolPolicy(expected, actor);
        patrol::SavePatrolPolicy(checkpoint, actor);
    });
    patrol::PatrolLearner resumed{params, 19U, 128U};
    resumed.LoadActor(checkpoint);
    std::ostringstream actual;
    resumed.WithActor([&](auto const& actor) {
        patrol::WritePatrolPolicy(actual, actor);
    });
    std::filesystem::remove(checkpoint);
    if (actual.str() != expected.str())
        return 8;

    // The headless adapter is still available as a separate optional tool.
    patrol::PatrolLearner headless{params, 11U, 128U};
    headless.Step();
    return headless.StepCount() == 1U ? 0 : 9;
}
