/**
 * @file Framework/RL/Core/tests/core_tests.cpp
 * @brief Tests et assertions déterministes pour core tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "ppo/core/runner.hpp"
#include "ppo/core/agent_contract.hpp"
#include "ppo/core/buffers.hpp"
#include "ppo/runtime/iagent.hpp"
#include "ppo/core/multi_agent.hpp"
#include "ppo/core/evaluation.hpp"
#include "ppo/core/evaluator_state.hpp"

#include <array>
#include <stdexcept>
#include <memory>
#include <vector>

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void Check(bool condition) {
    if (!condition) throw std::runtime_error("core regression test failed");
}

/** @brief Déclare le type FakeEnvironment, utilisé par Tests et assertions déterministes pour core tests. */
class FakeEnvironment final : public ppo::core::IEnvironment {
public:
    std::size_t ObservationSize() const noexcept override { return 1U; }
    std::size_t ActionSize() const noexcept override { return 1U; }
    ppo::core::Observation Reset() override {
        position_ = 0.F;
        return invalid_reset ? ppo::core::Observation{} : ppo::core::Observation{position_};
    }
    bool invalid_reset = false;
    ppo::core::Transition Step(ppo::core::Action const& action) override {
        position_ += action.at(0);
        if (invalid_observation) return {{}, position_, false};
        return {{position_}, position_, position_ >= 3.F};
    }
    bool invalid_observation = false;
private:
    float position_ = 0.F;
};

/** @brief Déclare le type FakeLearner, utilisé par Tests et assertions déterministes pour core tests. */
class FakeLearner final : public ppo::core::ILearner {
public:
/**
 * @brief Réalise l'opération ChooseAction; voir les préconditions et effets dans son corps C++.
 */
    ppo::core::Action ChooseAction(ppo::core::Observation const& obs) override {
        Check(obs.size() == 1U);
        return invalid_action ? ppo::core::Action{} : ppo::core::Action{1.F};
    }
    bool invalid_action = false;
/**
 * @brief Réalise l'opération Observe; voir les préconditions et effets dans son corps C++.
 */
    void Observe(ppo::core::Transition const& t) override {
        ++observed;
        last = t;
    }
    int observed = 0;
    ppo::core::Transition last;
};
} // namespace


void TestMultiAgentPhases() {
/** @brief Déclare le type Agent, utilisé par Tests et assertions déterministes pour core tests. */
    struct Agent { int id; };
    auto a = std::make_shared<Agent>(Agent{1});
    auto b = std::make_shared<Agent>(Agent{2});
    std::vector<std::shared_ptr<Agent>> agents{a, nullptr, b};
    std::vector<int> events{};

    ppo::core::ResetAgents(agents, [&](Agent& agent) { events.push_back(10 + agent.id); });
    Check(events == std::vector<int>({11, 12}));
    events.clear();

    bool const reset = ppo::core::StepAgents(
        agents,
        [&](Agent& agent) { events.push_back(10 + agent.id); },
        [&]() { events.push_back(20); },
        [&]() { events.push_back(30); },
        [&](Agent& agent) {
            events.push_back(40 + agent.id);
            return agent.id == 1;
        });
    Check(reset);
    Check(events == std::vector<int>({11, 12, 20, 30, 41, 42}));

    std::vector<std::shared_ptr<Agent>> empty{};
    events.clear();
    bool const noReset = ppo::core::StepAgents(
        empty,
        [&](Agent&) { events.push_back(-1); },
        [&]() { events.push_back(20); },
        [&]() { events.push_back(30); },
        [&](Agent&) { events.push_back(-2); return true; });
    Check(!noReset);
    Check(events == std::vector<int>({20, 30}));
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
void TestEvaluationMonitoring()
{
    using ppo::core::EvaluationProgress;
    Check(EvaluationProgress(0, 0, 100, 1024) == 0.F);
    Check(EvaluationProgress(100, 0, 100, 1024) == 1.F);
    Check(EvaluationProgress(1, 0, 100, 1024) == 0.01F);
    Check(EvaluationProgress(0, 512, 100, 1024) == 0.005F);
    Check(EvaluationProgress(-5, -5, 100, 1024) == 0.F);
    Check(EvaluationProgress(0, 9999, 100, 1024) == 0.01F);
    Check(EvaluationProgress(10, 100, 0, 1024) == 0.F);
    Check(EvaluationProgress(10, 100, 100, 0) == 0.F);

    // The generic scheduler exposes the actual index at which a test stops.
    // Exhausting the step budget is NOT synonymous with a task failure.
    int stepLimitExhausted = 0;
    int finished = 0;
    auto const outcome = ppo::core::RunEvaluation(
        2, 3,
        [] { return true; },
        [](int) { return true; },
        [](int scenario, int step) { return scenario == 0 && step == 1; },
        [&](int, int step)
        {
            ++finished;
            if (step == 3)
                ++stepLimitExhausted;
        });
    Check(outcome.completed && finished == 2 && stepLimitExhausted == 1);
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
void TestEvaluationScheduling() {
    std::vector<int> events{};
    // Early termination at step 1 of test 0, normal exhaustion for test 1.
    ppo::core::RunEvaluation(
        2, 3,
        [] { return true; },
        [&](int test) { events.push_back(10 + test); return true; },
        [&](int test, int step) {
            events.push_back(100 + test * 10 + step);
            return test == 0 && step == 1;
        },
        [&](int test, int step) { events.push_back(200 + test * 10 + step); });
    Check(events == std::vector<int>({10, 100, 101, 201,
                                      11, 110, 111, 112, 213}));

    // Cancel mid-test: the completion callback still receives the current
    // zero-based step index, matching the legacy evaluator.
    events.clear();
    bool running = true;
    ppo::core::RunEvaluation(
        3, 4,
        [&] { return running; },
        [&](int test) { events.push_back(test); return true; },
        [&](int, int step) { events.push_back(10 + step); running = false; return false; },
        [&](int test, int step) { events.push_back(100 + test * 10 + step); });
    Check(events == std::vector<int>({0, 10, 101}));

    // Failed reset aborts without counting or completing that test.
    events.clear();
    ppo::core::RunEvaluation(
        3, 2,
        [] { return true; },
        [&](int test) { events.push_back(test); return test == 0; },
        [&](int, int) { return false; },
        [&](int, int) { events.push_back(99); });
    Check(events == std::vector<int>({0, 99, 1}));

    // Promotion is only permitted when the entire campaign is valid.
    auto complete = ppo::core::RunEvaluation(
        2, 2, [] { return true; }, [](int) { return true; },
        [](int, int) { return false; }, [](int, int) {});
    Check(complete.completed && complete.finishedTests == 2);

    bool cancellationRunning = true;
    auto cancelled = ppo::core::RunEvaluation(
        2, 3, [&] { return cancellationRunning; }, [](int) { return true; },
        [&](int, int) { cancellationRunning = false; return false; }, [](int, int) {});
    Check(!cancelled.completed && cancelled.finishedTests == 1);

    auto failedReset = ppo::core::RunEvaluation(
        2, 2, [] { return true; }, [](int index) { return index == 0; },
        [](int, int) { return false; }, [](int, int) {});
    Check(!failedReset.completed && failedReset.finishedTests == 1);

    auto empty = ppo::core::RunEvaluation(
        0, 2, [] { return true; }, [](int) { return true; },
        [](int, int) { return false; }, [](int, int) {});
    Check(!empty.completed && empty.finishedTests == 0);
}

// The helper must also work in the dependency-free build, for a domain
// that is entirely unrelated to either current neural policy.
/** @brief Déclare le type ExampleAgentModel, utilisé par Tests et assertions déterministes pour core tests. */
struct ExampleAgentModel
{
    using STATES = std::array<float, 4>;
    using ACTIONS = std::array<float, 2>;
    inline static constexpr ppo::core::AgentContract kAgentContract =
        ppo::core::MakeAgentContract<STATES, ACTIONS>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
};

static_assert(ppo::core::SameContract(
                  ppo::core::AgentContractOf<ExampleAgentModel>(),
                  ExampleAgentModel::kAgentContract),
              "Model and generic agent contract must agree");

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
void TestAgentContracts()
{
    using ppo::core::AgentContract;
    using ppo::core::MakeAgentContract;
    using ppo::core::IsValid;
    using ppo::core::SameContract;

    // An unrelated new agent type can declare its own contract without
    // modifying the generic core or existing agent implementations.
    constexpr AgentContract example = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract same = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract reordered = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v2", "navigation2d.action.v1");
    constexpr AgentContract differentAction = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v2");
    constexpr AgentContract differentSize = MakeAgentContract<
        std::array<float, 5>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract differentProfile = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "DifferentAgent", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract invalid = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract noObservationLayout = { "Navigation2D", "", 4U,
                                                     "navigation2d.action.v1", 2U };
    constexpr AgentContract noActionLayout = { "Navigation2D", "navigation2d.observation.v1",
                                                4U, "", 2U };
    constexpr AgentContract zeroObservation = { "Navigation2D", "navigation2d.observation.v1",
                                                 0U, "navigation2d.action.v1", 2U };
    constexpr AgentContract zeroAction = { "Navigation2D", "navigation2d.observation.v1",
                                            4U, "navigation2d.action.v1", 0U };



    static_assert(IsValid(example), "Example contract must be valid");
    static_assert(SameContract(example, same), "Identical layouts must match");
    static_assert(!SameContract(example, reordered), "Ordering is part of compatibility");
    static_assert(!SameContract(example, differentAction), "Action semantics must match");
    static_assert(!SameContract(example, differentSize), "Dimensions must match");
    static_assert(!SameContract(example, differentProfile), "Profile identity must match");
    static_assert(!IsValid(invalid), "Unnamed agent contracts are invalid");
    static_assert(!IsValid(noObservationLayout), "Missing observation ID is invalid");
    static_assert(!IsValid(noActionLayout), "Missing action ID is invalid");
    static_assert(!IsValid(zeroObservation), "Zero-length observation is invalid");
    static_assert(!IsValid(zeroAction), "Zero-length action is invalid");
    static_assert(!SameContract(invalid, invalid), "Invalid contracts cannot match");
    Check(SameContract(example, same));
    Check(!SameContract(example, reordered));
}

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main() {
    // The legacy agent ABI/telemetry contract is standalone C++17:
    // no LibTorch or ECS header is required by the headless build.
    ppo::model::sAgentStateSnaphot legacySnapshot{};
    Check(legacySnapshot.m_common.m_gamma == 0.99F);
    Check(legacySnapshot.m_common.m_batch_size == 64);
    Check(legacySnapshot.m_id == 0ULL);
    // Existing fixed-size domain buffers can be bridged without exposing
    // their lifetime or changing their numerical ordering.
    std::array<float, 3> domainObservation{0.25F, -1.F, 2.F};
    Check(ppo::core::CopyObservation(domainObservation) ==
          ppo::core::Observation({0.25F, -1.F, 2.F}));
    std::array<float, 2> domainAction{};
    ppo::core::CopyAction(domainAction, ppo::core::Action({0.5F, -0.75F}));
    Check(domainAction == std::array<float, 2>{0.5F, -0.75F});
    bool copyRejected = false;
    try { ppo::core::CopyAction(domainAction, ppo::core::Action{1.F}); }
    catch (std::invalid_argument const&) { copyRejected = true; }
    Check(copyRejected);
    Check(domainAction == std::array<float, 2>{0.5F, -0.75F});
    TestAgentContracts();
    TestMultiAgentPhases();
    TestEvaluationScheduling();
    TestEvaluationMonitoring();
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
/** @brief Déclare le type ConstantPolicy, utilisé par Tests et assertions déterministes pour core tests. */
    class ConstantPolicy final : public ppo::core::IPolicy {
    public:
/**
 * @brief Réalise l'opération ChooseAction; voir les préconditions et effets dans son corps C++.
 */
        ppo::core::Action ChooseAction(ppo::core::Observation const&) override {
            return {1.F};
        }
    } policy;
    ppo::core::EpisodeRunner inference(env, policy);
    Check(inference.Reset().at(0) == 0.F);
    Check(inference.Advance().reward == 1.F);
    Check(learner.observed == 3);
    env.invalid_reset = true;
    rejected = false;
    try { inference.Reset(); }
    catch (std::invalid_argument const&) { rejected = true; }
    Check(rejected);
    env.invalid_reset = false;
    rejected = false;
    try { inference.Advance(); }
    catch (std::logic_error const&) { rejected = true; }
    Check(rejected);
    Check(inference.Reset().at(0) == 0.F);

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
