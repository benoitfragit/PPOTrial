#pragma once
// PPO-driven car agent. Environment schedules its action and reward feedback
// just as for the legacy agents. No extra thread or auxiliary IAgent.
#include "patrol_agent.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ppo::examples::twovstwo::defenders {
class PatrolLearner;
class PatrolTrainingAgent final : public PatrolProdAgent
{
public:
    PatrolTrainingAgent(ppo::model::BaseEntity& own,
                        ppo::model::BaseEntity& teammate,
                        std::size_t role,
                        std::shared_ptr<PatrolPolicy> policy,
                        Parameters parameters,
                        std::filesystem::path checkpoint,
                        std::uint32_t seed = 42U,
                        std::uint64_t maxUpdates = 300U);
    ~PatrolTrainingAgent() override;
    void Init() override;
    bool SetNewState() override;
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
    void Save() noexcept;
    std::shared_ptr<PatrolPolicy> policy_;
    std::unique_ptr<PatrolLearner> learner_;
    Parameters parameters_;
    std::filesystem::path checkpoint_;
    std::uint32_t seed_;
    std::uint64_t maxUpdates_;
    std::uint64_t ticks_ = 0;
    std::uint64_t updates_ = 0;
    bool finished_ = false;
};
} // namespace ppo::examples::twovstwo::defenders
