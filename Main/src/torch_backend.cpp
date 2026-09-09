#include "policybackend.hpp"
#include "torch_policy.hpp"

namespace navigation {
std::unique_ptr<ppo::rl::ILearner> TorchBackend::create_learner(
    std::size_t observations, std::size_t actions, std::size_t rollout,
    const std::filesystem::path& root)
{
    return std::make_unique<ppo::rl::TorchLearner>(observations, actions, rollout, root);
}
std::unique_ptr<ppo::rl::IPolicy> TorchBackend::load_policy(
    std::size_t observations, std::size_t actions, const std::filesystem::path& checkpoint)
{
    return std::make_unique<ppo::rl::FrozenTorchPolicy>(observations, actions, checkpoint);
}
BackendTelemetry TorchBackend::telemetry(const ppo::rl::ILearner& learner) const
{
    const auto& backend = dynamic_cast<const ppo::rl::TorchLearner&>(learner);
    const auto outputs = backend.outputs();
    return {backend.rollout_step(), outputs.m_loss, outputs.m_actorloss,
            outputs.m_valueloss, outputs.m_entropyloss};
}
void TorchBackend::save(ppo::rl::ILearner& learner, const std::filesystem::path& candidate)
{
    dynamic_cast<ppo::rl::TorchLearner&>(learner).save_candidate(candidate);
}
}
