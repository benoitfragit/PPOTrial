#pragma once
#include <algorithm>
#include <cstdint>

namespace ppo::renderer {
/** @brief Keeps a user's slider edit stable until its worker acknowledgement arrives. */
struct RewardSliderDraft final
{
    float value{};
    bool editing{},awaiting{};
    std::uint64_t expectedRevision{};

    /** @brief Reconcile a worker snapshot without discarding in-progress edits. */
    void Sync(float authoritative,std::uint64_t revision) noexcept
    {
        if(awaiting && revision>=expectedRevision) awaiting=false;
        if(!editing && !awaiting) value=authoritative;
    }

    /** @brief Record an emitted command, including multiple edits before acknowledgement. */
    void Submitted(std::uint64_t revision) noexcept
    {
        expectedRevision=std::max(awaiting ? expectedRevision : revision,revision)+1;
        awaiting=true;
    }
};
} // namespace ppo::renderer
