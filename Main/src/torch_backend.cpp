#include "policybackend.hpp"
#include "torch_policy.hpp"

namespace navigation {
std::unique_ptr<ppo::rl::ILearner> TorchBackend::create_learner(
    std::size_t observations, std::size_t actions, std::size_t rollout,
    const std::filesystem::path& root)
{
    return create_learner_configured(observations, actions, rollout, root, TrainingParameters{});
}
std::unique_ptr<ppo::rl::ILearner> TorchBackend::create_learner_configured(
    std::size_t observations, std::size_t actions, std::size_t rollout,
    const std::filesystem::path& root, const TrainingParameters& settings)
{
    validate_training_parameters(settings);
    ppo::model::PPOParamters parameters;
    parameters.m_gamma = settings.gamma;
    parameters.m_lambda_gae = settings.lambda_gae;
    parameters.m_clip_epsilon = settings.clip_epsilon;
    parameters.m_value_coef = settings.value_coefficient;
    parameters.m_entropy_coef = settings.entropy_coefficient;
    parameters.m_learningReate = settings.learning_rate;
    parameters.m_K_epochs = settings.epochs;
    parameters.m_batch_size = settings.batch_size;
    parameters.m_target_kl = settings.target_kl;
    return std::make_unique<ppo::rl::TorchLearner>(
        observations, actions, rollout, root, parameters);
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
