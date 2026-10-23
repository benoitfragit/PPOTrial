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
    // Raw task outcomes in the fixed evaluation suite. Values are in [0,1].
    float m_latestSuccessRate{ 0.F };
    float m_latestSurvivalRate{ 0.F };
    unsigned int m_lastCompletedTestCount{ 0U };
    // Live performance counters for the running (or most recent) campaign.
    unsigned int m_testsFinished{ 0U };
    unsigned long long m_stepsExecuted{ 0ULL };
    float m_stepsPerSecond{ 0.F };
    float m_elapsedSeconds{ 0.F };
};

} // namespace ppo::model
