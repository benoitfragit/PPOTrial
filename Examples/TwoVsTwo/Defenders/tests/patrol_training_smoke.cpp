#include "patrol_learner.hpp"

namespace patrol = ppo::examples::twovstwo::defenders;
int main()
{
    patrol::Parameters params{};
    patrol::PatrolLearner trainer{params, 7U};
    auto const initial = trainer.StepCount();
    trainer.Step();
    return trainer.StepCount() > initial ? 0 : 1;
}
