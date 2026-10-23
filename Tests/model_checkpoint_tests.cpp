// Smoke test of the two *existing* production checkpoints. This test must
// never train or serialize a model; it only restores and evaluates its policy.
#include "actorcriticnet.hpp"
#include "actorcriticmission.hpp"
#include "missionlayout.hpp"
#include "torchproductionpolicy.hpp"
#include "torchlearner.hpp"
#include "rlagent.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckActions(torch::Tensor const& actions, int64_t expectedDim)
{
    if (!actions.defined() || actions.dim() != 2 ||
        actions.size(0) != 1 || actions.size(1) != expectedDim ||
        actions.scalar_type() != torch::kFloat ||
        !torch::isfinite(actions).all().item<bool>())
        throw std::runtime_error("Checkpoint inference returned invalid actions");
}

template<typename Policy>
void CheckCheckpoint(std::filesystem::path const& path,
                     Policy& policy,
                     int64_t inputDim,
                     int64_t actionDim)
{
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("Missing checkpoint: " + path.string());

    std::cout << "Checking existing checkpoint: " << path.string() << std::endl;
    policy.load(path);
    torch::NoGradGuard guard;
    // Distinct feature values catch accidental reordering of observation
    // slots, which a constant vector would not detect.
    std::vector<float> features(static_cast<std::size_t>(inputDim));
    for (std::size_t i = 0; i < features.size(); ++i)
        features[i] = 0.1F + static_cast<float>(i) * 0.001F;
    auto input = torch::from_blob(features.data(), {1, inputDim}, torch::kFloat);
    auto first = policy.forward(input);
    auto second = policy.forward(input);

    CheckActions(first, actionDim);
    CheckActions(second, actionDim);
    if (!torch::equal(first, second))
        throw std::runtime_error("Production inference is not deterministic");
    // The generic IPolicy adapter must return the exact same output as the
    // original production wrapper using the existing trained weights.
    ppo::model::TorchProductionPolicy<Policy> generic(
        static_cast<std::size_t>(inputDim), static_cast<std::size_t>(actionDim));
    generic.Load(path);
    auto const genericAction = generic.ChooseAction(
        ppo::core::Observation(features.begin(), features.end()));
    if (genericAction.size() != static_cast<std::size_t>(actionDim))
        throw std::runtime_error("Generic production policy action dimension mismatch");
    auto const expected = first.contiguous();
    auto const* values = expected.data_ptr<float>();
    for (std::size_t index = 0; index < genericAction.size(); ++index)
        if (genericAction[index] != values[index])
            throw std::runtime_error("Generic IPolicy adapter changed checkpoint inference");
    std::cout << "Checkpoint inference OK: " << path.string() << std::endl;
}

// Compare the actual pre-update PPO transition path, not merely neural
// inference. This is read-only: a partial rollout never calls UpdatePolicy,
// Save or any evaluator, so existing archives and weights remain unchanged.
template<typename TrainingNet>
void CheckTrainingTransitionParity(std::filesystem::path const& modelRoot,
                                   int64_t inputDim,
                                   int64_t actionDim)
{
    using namespace ppo::model;
    if (!std::filesystem::is_regular_file(modelRoot / "prod" / "model.pt"))
        throw std::runtime_error("Missing training checkpoint: " + modelRoot.string());

    std::vector<float> legacyState(static_cast<std::size_t>(inputDim));
    for (std::size_t i = 0; i < legacyState.size(); ++i)
        legacyState[i] = 0.1F + static_cast<float>(i) * 0.001F;
    std::vector<float> legacyAction(static_cast<std::size_t>(actionDim), 0.F);
    PPOParamters const parameters{};

    auto oldNetwork = std::make_shared<TrainingNet>(inputDim, actionDim);
    RLAgent legacy(oldNetwork, modelRoot.string(), inputDim, actionDim,
                   8, parameters, legacyState.data(), legacyAction.data(), true);
    legacy.Init();

    auto newNetwork = std::make_shared<TrainingNet>(inputDim, actionDim);
    TorchLearner learner(newNetwork, modelRoot.string(),
                         static_cast<std::size_t>(inputDim),
                         static_cast<std::size_t>(actionDim),
                         8, parameters, true);

    // Both restored policies receive identical inputs and Torch RNG state.
    torch::manual_seed(20261023);
    legacy.ChooseAction();
    torch::manual_seed(20261023);
    auto const newAction = learner.ChooseAction(
        ppo::core::Observation(legacyState.begin(), legacyState.end()));

    if (newAction.size() != legacyAction.size())
        throw std::runtime_error("Training adapter changed action dimensions");
    for (std::size_t i = 0; i < newAction.size(); ++i)
    {
        if (!std::isfinite(newAction[i]) || legacyAction[i] != newAction[i])
            throw std::runtime_error("Training adapter changed PPO sampled action");
    }

    // A second action is prohibited until the preceding transition arrives.
    bool refused = false;
    try { (void)learner.ChooseAction(ppo::core::Observation(legacyState.begin(), legacyState.end())); }
    catch (std::logic_error const&) { refused = true; }
    if (!refused)
        throw std::runtime_error("Training adapter permitted an action without feedback");

    // The next observation is deliberately changed before transition feedback.
    // RLAgent's original from_blob path and TorchLearner's owned copy must
    // advance to the same rollout index without invoking any optimizer.
    std::fill(legacyState.begin(), legacyState.end(), 0.2F);
    legacy.SetNewState(0.25F, false);
    learner.Observe(ppo::core::Transition{
        ppo::core::Observation(legacyState.begin(), legacyState.end()),
        0.25F, false
    });
    if (legacy.GetCurrentStep() != 1 || learner.RolloutStep() != 1 ||
        !legacy.IsTraining() || learner.RolloutComplete())
        throw std::runtime_error("Training adapter changed rollout progression");

    std::cout << "Training transition parity OK: " << modelRoot.string() << std::endl;
}

} // namespace

int main()
{
    const std::filesystem::path root{PPO_MODEL_RESOURCE_DIR};
    // Verify both production and candidate-evaluation archives. Loading either
    // must not modify weights or run an optimizer.
    constexpr auto missionDim = rl::MissionObservationLayout::kStateDimension;
    constexpr auto missionActions = rl::MissionObservationLayout::kActionCount;

    for (auto const* version : {"prod", "eval"})
    {
        rl::ActorCriticNetProd interception{6, 2};
        CheckCheckpoint(root / "RLSNA" / version / "model.pt",
                        interception, 6, 2);

        rl::ActorCriticTrackAttentionProd mission{
            static_cast<int64_t>(missionDim),
            static_cast<int64_t>(missionActions)
        };
        CheckCheckpoint(root / "RLMission" / version / "model.pt",
                        mission,
                        static_cast<int64_t>(missionDim),
                        static_cast<int64_t>(missionActions));
    }

    // Both current PPO-compatible architectures: loaded weights, action
    // sampling and transition ingestion, without retraining or saving.
    CheckTrainingTransitionParity<rl::ActorCriticNet>(
        root / "RLSNA", 6, 2);
    CheckTrainingTransitionParity<rl::ActorCriticTrackAttention>(
        root / "RLMission",
        static_cast<int64_t>(missionDim),
        static_cast<int64_t>(missionActions));
}
