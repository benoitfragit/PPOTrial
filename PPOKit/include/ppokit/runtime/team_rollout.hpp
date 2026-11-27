/**
 * @file PPOKit/include/ppokit/runtime/team_rollout.hpp
 * @brief Rollout of synchronized agents for MAPPO: one row per agent and team step.
 * @details A team of `agents` acts together: each team step stores one transition per
 * agent, in agent order, and the rollout holds `teamSteps` such steps. Rows are laid out
 * team step by team step (index = step * agents + agent), the order in which a shuffled
 * minibatch draws them.
 *
 * Each agent keeps its own trajectory, so the GAE runs per agent backwards along the team
 * steps. An agent that is not available (destroyed) at a step keeps its row: its
 * advantage and value target still train the critic on the team's mission return, but
 * the row is not learnable, so it never trains the actor. A row can also hold no agent of
 * the episode at all (a team smaller than the rollout's rows): it is not a member, and the
 * update ignores it entirely (no baseline, no loss, no statistic). Members keep their
 * step-major order, so a rollout of P members in larger rows updates exactly as a rollout
 * of P rows.
 *
 * Values are not known at collection time: the trainer evaluates the critic on `critic`
 * and `nextCritic` once the rollout is full (SetValues), with the critic of this update.
 * Every buffer is allocated at construction: recording a team step never allocates.
 */
#pragma once

#include "ppokit/runtime/rollout.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::runtime {

/**
 * @brief One agent at one team step, with the dimensions of the loaded task.
 * @details Its vectors are sized once by TeamRollout; recording a step copies into them.
 */
template<typename Scalar>
struct TeamTransition final
{
    using ScalarType = Scalar;

    std::vector<float> observation;
    /** @brief Centralized critic input at this step and at the next one (for the bootstrap value). */
    std::vector<float> critic;
    std::vector<float> nextCritic;
    /** @brief Raw sampled action. */
    std::vector<float> action;
    /** @brief log pi_old(action | observation) at collection time. */
    Scalar logProbability{};
    Scalar reward{};
    StepEnd end = StepEnd::Running;
    /** @brief An agent of the episode, alive or destroyed: the row trains the critic. */
    bool member = true;
    /** @brief Available member: the row trains the actor too. */
    bool learnable = true;
    /** @brief Critic estimates, filled by SetValues before the advantages. */
    Scalar value{};
    Scalar nextValue{};
    /** @brief Filled by ComputeAdvantages. */
    Scalar advantage{};
    Scalar valueTarget{};
};

/** @brief Fixed-capacity rollout of `teamSteps` synchronized steps of `agents` agents. */
template<typename Scalar>
class TeamRollout final
{
public:
    using Row = TeamTransition<Scalar>;

    /**
     * @param observationSize, criticSize, actionSize Values of one agent's observation,
     *        centralized critic input and action.
     * @throws std::invalid_argument when a dimension is zero.
     */
    TeamRollout(std::size_t agents, std::size_t teamSteps, std::size_t observationSize, std::size_t criticSize,
        std::size_t actionSize)
        : rows_(agents, teamSteps, MakeRow(observationSize, criticSize, actionSize)), members_(rows_.Capacity())
    {
    }

    std::size_t Agents() const noexcept { return rows_.Streams(); }
    std::size_t TeamSteps() const noexcept { return rows_.Steps(); }
    /** @brief Team steps recorded so far. */
    std::size_t Collected() const noexcept { return rows_.Collected(); }
    bool Full() const noexcept { return rows_.Full(); }
    /** @brief Total row capacity, including steps not yet committed. */
    std::size_t Size() const noexcept { return rows_.Capacity(); }

    /** @brief Row of `agent` at the team step being recorded (fill it, then EndTeamStep). */
    Row& Current(std::size_t agent) { return rows_.Current(agent); }

    /**
     * @brief Closes the team step being recorded.
     * @throws std::invalid_argument when a reward or log-probability is not finite: a NaN
     *         would silently corrupt every advantage of the agent.
     */
    void EndTeamStep()
    {
        for (std::size_t agent = 0; agent < Agents(); ++agent)
        {
            Row const& row = Current(agent);
            if (!std::isfinite(row.reward) || !std::isfinite(row.logProbability))
            {
                throw std::invalid_argument("PPOKit team rollout: non-finite reward or log-probability for agent "
                    + std::to_string(agent));
            }
        }
        rows_.Commit();
    }

    void Clear() noexcept { rows_.Clear(); }

    Row& operator[](std::size_t index) noexcept { return rows_[index]; }
    Row const& operator[](std::size_t index) const noexcept { return rows_[index]; }
    Row& At(std::size_t teamStep, std::size_t agent) { return rows_.At(teamStep, agent); }

    /**
     * @brief GAE of each agent, backwards along the team steps of the full rollout.
     * @details Restarts at every episode end. Unavailable rows take part in the recursion:
     * a destroyed agent still shares the mission return. Also lists the members (CollectMembers).
     * @return Number of learnable rows.
     */
    std::size_t ComputeAdvantages(Scalar gamma, Scalar lambda)
    {
        CollectMembers();
        return ComputeTrajectoryAdvantages(rows_.Data(), TeamSteps(), Agents(), gamma, lambda);
    }

    /** @brief Zero mean, unit variance over the learnable rows; returns their number. */
    std::size_t NormalizeAdvantages() noexcept { return NormalizeLearnableAdvantages(rows_.Data(), Size()); }

    /**
     * @brief Lists the member rows of the full rollout, in step-major order (see Member).
     * @return Their number.
     */
    std::size_t CollectMembers()
    {
        RequireFull();
        memberCount_ = 0;
        for (std::size_t index = 0; index < Size(); ++index)
        {
            if (rows_[index].member)
            {
                members_[memberCount_] = index;
                ++memberCount_;
            }
        }
        return memberCount_;
    }

    /** @brief Member rows found by the last CollectMembers. */
    std::size_t MemberCount() const noexcept { return memberCount_; }
    /** @brief Row index of member `position` (below MemberCount). */
    std::size_t Member(std::size_t position) const noexcept { return members_[position]; }

    /** @brief Critic quality before the update, over the members (see ExplainedVarianceOver). */
    Scalar ExplainedVariance() const noexcept
    {
        return ExplainedVarianceOver<Scalar>(
            memberCount_, [this](std::size_t position) -> Row const& { return rows_[members_[position]]; });
    }

    /** @brief Mean |advantage| of the learnable rows, before normalization (diagnostic). */
    Scalar MeanAbsoluteLearnableAdvantage() const noexcept
    {
        Scalar sum{};
        std::size_t learnable = 0;
        for (std::size_t agent = 0; agent < Agents(); ++agent)
        {
            for (std::size_t step = TeamSteps(); step > 0; --step)
            {
                Row const& row = rows_[(step - 1) * Agents() + agent];
                if (row.learnable)
                {
                    sum += std::abs(row.advantage);
                    ++learnable;
                }
            }
        }
        return learnable == 0 ? Scalar(0) : sum / static_cast<Scalar>(learnable);
    }

private:
    static Row MakeRow(std::size_t observationSize, std::size_t criticSize, std::size_t actionSize)
    {
        if (observationSize == 0 || criticSize == 0 || actionSize == 0)
        {
            throw std::invalid_argument("PPOKit team rollout: dimensions must be positive");
        }
        Row row;
        row.observation.resize(observationSize);
        row.critic.resize(criticSize);
        row.nextCritic.resize(criticSize);
        row.action.resize(actionSize);
        return row;
    }

    void RequireFull() const
    {
        if (!Full())
        {
            throw std::logic_error("PPOKit team rollout: advantages need a full rollout");
        }
    }

    detail::RolloutRows<Row> rows_;
    std::vector<std::size_t> members_;
    std::size_t memberCount_ = 0;
};

} // namespace ppokit::runtime
