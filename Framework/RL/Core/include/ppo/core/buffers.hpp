/**
 * @file Framework/RL/Core/include/ppo/core/buffers.hpp
 * @brief Interface ou implémentation du composant buffers.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "ppo/core/environment.hpp"

#include <algorithm>
#include <stdexcept>

namespace ppo::core {

// Copy between fixed-size domain buffers and owned, model-independent values.
// Never expose a borrowed model/ECS buffer to a policy.
template<typename Buffer>
/**
 * @brief Réalise l'opération CopyObservation; voir les préconditions et effets dans son corps C++.
 * @details Peut signaler un état invalide par exception.
 */
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
