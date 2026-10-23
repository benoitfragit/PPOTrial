#pragma once

#include <cstdint>

namespace ppo::model {

// Retains the legacy public type/fields used by ECS, evaluator and UI.
struct sEvaluatorState {
    unsigned int m_nbstep{ 0 };
    unsigned int m_nbtests{ 0 };
    float m_notation{ 0.F };
    float m_progress{ 0.F };
    bool m_inprogress{ false };
    // Latest COMPLETED campaign, as opposed to m_notation (best promoted score).
    float m_latestNotation{ 0.F };
    std::uint64_t m_completedCampaigns{ 0 };
    bool m_hasLatestNotation{ false };
};

} // namespace ppo::model
