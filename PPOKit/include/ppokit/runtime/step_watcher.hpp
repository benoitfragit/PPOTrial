/**
 * @file PPOKit/include/ppokit/runtime/step_watcher.hpp
 * @brief Called after every environment step of a collection: live control, observers.
 */
#pragma once

namespace ppokit::runtime {

/** @brief Watches a collection step by step; may pause it (by blocking) or stop it. */
class StepWatcher
{
public:
    virtual ~StepWatcher() = default;
    /**
     * @brief Called after each step. Blocks while the run is paused.
     * @return False to stop the collection now (the run was asked to stop).
     */
    virtual bool AfterStep() = 0;
};

} // namespace ppokit::runtime
