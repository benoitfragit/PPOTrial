#pragma once

#include "ppo/core/environment.hpp"

#include <algorithm>
#include <stdexcept>

namespace ppo::core {

// Copy between fixed-size domain buffers and owned, model-independent values.
// Never expose a borrowed model/ECS buffer to a policy.
template<typename Buffer>
Observation CopyObservation(Buffer const& buffer)
{
    return Observation(buffer.begin(), buffer.end());
}

template<typename Buffer>
void CopyAction(Buffer& destination, Action const& action)
{
    if (destination.size() != action.size())
        throw std::invalid_argument("CopyAction: action dimension mismatch");
    std::copy(action.begin(), action.end(), destination.begin());
}

} // namespace ppo::core
