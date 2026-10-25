#include "patrol_learner.hpp"

#include <cmath>

namespace patrol = ppo::examples::twovstwo::defenders;

int main()
{
    patrol::Parameters params{};
    patrol::PatrolLearner trainer{params, 7U};
    auto state = patrol::InitialState(params, 1, 0U, 0.F);

    // One full real-transition PPO rollout; there must be no simulator-driven
    // training update before all feedback samples have arrived.
    for (std::size_t step = 0; step < patrol::PatrolLearner::RolloutSteps; ++step)
    {
        auto const observation = patrol::Observe(params, state);
        auto const action = trainer.ChooseAction(observation);
        if (!trainer.HasPendingAction())
            return 1;
        auto after = patrol::Advance(params, state, action);
        float const reward = patrol::TeamReward(params, state, action, after);
        bool const truncated = after.step >= params.episodeSteps;
        bool const updated = trainer.Observe(patrol::Observe(params, after),
                                             reward, after.failed, truncated);
        if (trainer.HasPendingAction() ||
            updated != (step + 1U == patrol::PatrolLearner::RolloutSteps))
            return 2;
        state = after.failed || truncated
            ? patrol::InitialState(params, 1, 0U, 0.F) : after;
    }
    if (trainer.StepCount() != 1U || trainer.CollectedSteps() != 0U)
        return 3;

    // The headless adapter is still available as a separate optional tool.
    patrol::PatrolLearner headless{params, 11U};
    headless.Step();
    return headless.StepCount() == 1U ? 0 : 4;
}
