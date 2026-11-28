/**
 * @file PPOKit/include/ppokit/core/gae.hpp
 * @brief Generalized Advantage Estimation (GAE), shared by every PPOKit learner.
 * @details GAE is computed backwards over a trajectory:
 *
 *     delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
 *     A_t     = delta_t + gamma * lambda * A_{t+1} * (1 - terminated_t) * (1 - truncated_t)
 *
 * The two end-of-episode flags have different meanings, and confusing them
 * biases learning:
 * - `terminated`: the episode really ended (crash, landing, mission outcome).
 *   The future is worth zero, so the bootstrap V(s_{t+1}) is dropped.
 * - `truncated`: the episode was stopped from outside (time limit). The future
 *   still had value, so the bootstrap is kept, but the trace stops because
 *   the next transition belongs to another episode.
 *
 * The expressions are written in a fixed evaluation order so that a learner
 * migrated to these helpers stays bit-identical: callers must not reorder them.
 */
#pragma once

namespace ppokit::core {

/**
 * @brief One-step temporal difference error `r + gamma * V(s') - V(s)`.
 * @param reward Reward of the transition.
 * @param value Critic estimate V(s) of the state where the action was taken.
 * @param nextValue Critic estimate V(s') of the following state; ignored when `terminated`.
 * @param terminated True when the episode really ended at this transition.
 * @param gamma Discount factor, in [0, 1].
 */
template<typename Scalar>
constexpr Scalar TemporalDifference(Scalar reward, Scalar value, Scalar nextValue, bool terminated,
    Scalar gamma) noexcept
{
    return reward + (terminated ? Scalar{0} : gamma * nextValue) - value;
}

/**
 * @brief Advantage of one transition given the advantage of the next one.
 * @param followingAdvantage Advantage of the next transition in the same trajectory;
 *        ignored when this transition ends the episode (terminated or truncated).
 * @param lambda GAE bias/variance trade-off, in [0, 1].
 * @param truncated True when the episode was stopped from outside at this transition.
 * @return A_t; the critic regression target is `value + A_t`.
 */
template<typename Scalar>
constexpr Scalar GaeStep(Scalar reward, Scalar value, Scalar nextValue, Scalar followingAdvantage,
    Scalar gamma, Scalar lambda, bool terminated, bool truncated) noexcept
{
    Scalar const trace = terminated || truncated ? Scalar{0} : followingAdvantage;
    return TemporalDifference(reward, value, nextValue, terminated, gamma) + gamma * lambda * trace;
}

} // namespace ppokit::core
