// Smoke test of the two *existing* production checkpoints. This test must
// never train or serialize a model; it only restores and evaluates its policy.
#include "actorcriticnet.hpp"
#include "actorcriticmission.hpp"
#include "missionlayout.hpp"

#include <torch/torch.h>
#include <filesystem>
#include <stdexcept>
#include <string>

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

    policy.load(path);
    torch::NoGradGuard guard;
    auto input = torch::full({1, inputDim}, 0.1F, torch::kFloat);
    auto first = policy.forward(input);
    auto second = policy.forward(input);

    CheckActions(first, actionDim);
    CheckActions(second, actionDim);
    if (!torch::equal(first, second))
        throw std::runtime_error("Production inference is not deterministic");
}

} // namespace

int main()
{
    const std::filesystem::path root{PPO_MODEL_RESOURCE_DIR};
    // Dimensions match the committed legacy model layouts; no weights change.
    rl::ActorCriticNetProd interception{6, 2};
    CheckCheckpoint(root / "RLSNA" / "prod" / "model.pt", interception, 6, 2);

    constexpr auto missionDim = rl::MissionObservationLayout::kStateDimension;
    constexpr auto missionActions = rl::MissionObservationLayout::kActionCount;
    rl::ActorCriticTrackAttentionProd mission{
        static_cast<int64_t>(missionDim),
        static_cast<int64_t>(missionActions)
    };
    CheckCheckpoint(root / "RLMission" / "prod" / "model.pt", mission,
                    static_cast<int64_t>(missionDim),
                    static_cast<int64_t>(missionActions));
}
