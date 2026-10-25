#include "patrol_training_ecs_agent.hpp"
#include "patrol_learner.hpp"
#include "patrol_checkpoint.hpp"
#include "patrol_policy.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppo::examples::twovstwo::defenders {
PatrolTrainingAgent::PatrolTrainingAgent(
    ppo::model::BaseEntity& own, ppo::model::BaseEntity& teammate,
    std::size_t role, std::shared_ptr<PatrolPolicy> policy,
    Parameters parameters, std::filesystem::path checkpoint,
    std::uint32_t seed, std::uint64_t maxUpdates)
    : PatrolProdAgent(own, teammate, role, policy),
      policy_(std::move(policy)), parameters_(parameters),
      checkpoint_(std::move(checkpoint)), seed_(seed), maxUpdates_(maxUpdates)
{
    if (!Valid(parameters_) || !policy_ || checkpoint_.empty() || maxUpdates_ == 0)
        throw std::invalid_argument("Invalid patrol PPO training agent configuration");
}
PatrolTrainingAgent::~PatrolTrainingAgent()
{
    Save();
}
void PatrolTrainingAgent::Init()
{
    // A world episode reset only resets car state; never the PPO optimizer.
    PatrolProdAgent::Init();
}
void PatrolTrainingAgent::Save() noexcept
{
    if (!learner_ || updates_ == 0)
        return;
    try
    {
        learner_->WithActor([&](auto const& actor) {
            SavePatrolPolicy(checkpoint_, actor);
        });
    }
    catch (std::exception const& e)
    {
        std::cerr << "[PATROL] Checkpoint save failed: " << e.what() << '\n';
    }
}
bool PatrolTrainingAgent::SetNewState()
{
    // Called by Environment after the two cars' physics and world rules.
    bool const worldReset = PatrolProdAgent::SetNewState();
    if (finished_)
        return worldReset;

    ++ticks_;
    // RLtools performs one complete PPO optimizer update per Step.
    // Amortize this cost over simulation steps to avoid running an optimizer
    // on every rendered frame; all work remains on the existing ECS thread.
    if (ticks_ % 128U != 0U)
        return worldReset;

    try
    {
        if (!learner_)
            learner_ = std::make_unique<PatrolLearner>(parameters_, seed_);
        bool const done = learner_->Step();
        updates_ = learner_->StepCount();
        if (updates_ > 0)
        {
            std::ostringstream serialized;
            learner_->WithActor([&](auto const& actor) {
                WritePatrolPolicy(serialized, actor);
                if (updates_ % 10U == 0U || done || updates_ >= maxUpdates_)
                    SavePatrolPolicy(checkpoint_, actor);
            });
            policy_->ReloadFromText(serialized.str());
        }
        finished_ = done || updates_ >= maxUpdates_;
        if (finished_)
            std::cout << "[PATROL] PPO training completed after "
                      << updates_ << " updates\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << "[PATROL] PPO training failed: " << e.what() << '\n';
        finished_ = true;
    }
    return worldReset;
}
ppo::model::sAgentStateSnaphot PatrolTrainingAgent::GetSnapshot() const noexcept
{
    auto snap = PatrolProdAgent::GetSnapshot();
    snap.m_name = "Patrol PPO (training)";
    snap.m_common.m_step = static_cast<long long>(updates_);
    snap.m_common.m_maxStep = static_cast<long long>(maxUpdates_);
    snap.m_common.m_training = !finished_;
    snap.m_common.m_done = finished_;
    return snap;
}
} // namespace ppo::examples::twovstwo::defenders
