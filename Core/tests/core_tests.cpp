#include "ppo/core/runner.hpp"
#include "ppo/core/evaluator_state.hpp"

#include <stdexcept>
#include <vector>

namespace {
void Check(bool condition) {
    if (!condition) throw std::runtime_error("core regression test failed");
}

class FakeEnvironment final : public ppo::core::IEnvironment {
public:
    std::size_t ObservationSize() const noexcept override { return 1U; }
    std::size_t ActionSize() const noexcept override { return 1U; }
    ppo::core::Observation Reset() override { position_ = 0.F; return { position_ }; }
    ppo::core::Transition Step(ppo::core::Action const& action) override {
        position_ += action.at(0);
        if (invalid_observation) return {{}, position_, false};
        return {{position_}, position_, position_ >= 3.F};
    }
    bool invalid_observation = false;
private:
    float position_ = 0.F;
};

class FakeLearner final : public ppo::core::ILearner {
public:
    ppo::core::Action ChooseAction(ppo::core::Observation const& obs) override {
        Check(obs.size() == 1U);
        return invalid_action ? ppo::core::Action{} : ppo::core::Action{1.F};
    }
    bool invalid_action = false;
    void Observe(ppo::core::Transition const& t) override {
        ++observed;
        last = t;
    }
    int observed = 0;
    ppo::core::Transition last;
};
} // namespace

int main() {
    FakeEnvironment env;
    FakeLearner learner;
    ppo::core::EpisodeRunner runner(env, learner);
    bool rejected = false;
    try { runner.Advance(); }
    catch (std::logic_error const&) { rejected = true; }
    Check(rejected);
    Check(runner.Reset() == ppo::core::Observation{0.F});
    Check(runner.Advance().reward == 1.F);
    Check(runner.Advance().reward == 2.F);
    Check(!runner.Finished());
    Check(runner.Advance().done);
    Check(learner.observed == 3);
    Check(learner.last.observation == ppo::core::Observation{3.F});
    rejected = false;
    try { runner.Advance(); }
    catch (std::logic_error const&) { rejected = true; }
    Check(rejected);
    Check(runner.Reset().at(0) == 0.F);
    learner.invalid_action = true;
    rejected = false;
    try { runner.Advance(); }
    catch (std::invalid_argument const&) { rejected = true; }
    Check(rejected);
    learner.invalid_action = false;
    rejected = false;
    try { runner.Advance(); }
    catch (std::logic_error const&) { rejected = true; }
    Check(rejected);
    Check(runner.Reset().at(0) == 0.F);

    env.invalid_observation = true;
    rejected = false;
    try { runner.Advance(); }
    catch (std::invalid_argument const&) { rejected = true; }
    Check(rejected);
    env.invalid_observation = false;
    // The world was stepped before malformed output was discovered: the old
    // observation must not be reused without an explicit reset.
    rejected = false;
    try { runner.Advance(); }
    catch (std::logic_error const&) { rejected = true; }
    Check(rejected);
    Check(runner.Reset().at(0) == 0.F);

    // A policy-only runner is inference: no learner feedback is generated.
    class ConstantPolicy final : public ppo::core::IPolicy {
    public:
        ppo::core::Action ChooseAction(ppo::core::Observation const&) override {
            return {1.F};
        }
    } policy;
    ppo::core::EpisodeRunner inference(env, policy);
    Check(inference.Reset().at(0) == 0.F);
    Check(inference.Advance().reward == 1.F);
    Check(learner.observed == 3);

    std::vector<int> steps;
    ppo::core::StepWorld(
        [&] { steps.push_back(1); steps.push_back(1); },
        [&] { steps.push_back(2); },
        [&] { steps.push_back(3); },
        [&] { steps.push_back(4); steps.push_back(4); });
    Check(steps == std::vector<int>({1, 1, 2, 3, 4, 4}));
    ppo::model::sEvaluatorState state;
    Check(state.m_nbstep == 0U && !state.m_inprogress);
}
