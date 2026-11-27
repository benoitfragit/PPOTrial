/** @file step_control.hpp
 * @brief Adapt the local session to standalone collectors and deterministic evaluators.
 */
#pragma once

#include "ppokit/ipc/control.hpp"
#include "ppokit/runtime/step_watcher.hpp"

namespace ppokit::ipc {
class StepControl final : public runtime::StepWatcher
{
public:
    explicit StepControl(Worker& worker) noexcept : worker_(worker) {}
    bool AfterStep() override { return worker_.Continue(); }
private:
    Worker& worker_;
};
} // namespace ppokit::ipc
