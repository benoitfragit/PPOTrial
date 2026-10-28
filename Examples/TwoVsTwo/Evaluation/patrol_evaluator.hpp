#pragma once

// Deterministic headless evaluation of *two* concurrently controlled patrol
// defenders. Independent of RLtools: the caller supplies its prediction adapter.
#include "patrol_core.hpp"
#include "patrol_evaluation_settings.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ppo::examples::twovstwo::defenders {

enum class PatrolOutcome : std::uint8_t { Success, Failure, Truncated };
enum class PatrolFailure : std::uint8_t {
    None, WorldBoundary, Collision, RestrictedZone, InvalidState, TrackingQuality
};

struct PatrolEpisodeResult
{
    PatrolOutcome outcome = PatrolOutcome::Truncated;
    PatrolFailure failure = PatrolFailure::None;
    std::size_t steps = 0U;
    float trackingFraction = 0.F;
    float meanNormalizedError = 0.F;
    std::array<float, 2> accumulatedTurnRadians{};
    std::array<float, 2> accumulatedProgressMeters{};
    int side = 1;
    float initialPhase = 0.F;
};

struct PatrolEvaluationReport
{
    std::vector<PatrolEpisodeResult> episodes{};
    std::size_t successes = 0U;
    std::size_t failures = 0U;
    std::size_t truncations = 0U;

    bool Passed() const noexcept
    {
        return !episodes.empty() && successes == episodes.size();
    }
};

inline float WrapAngle(float angle) noexcept
{
    return std::remainder(angle, kTwoPi);
}

inline PatrolFailure SafetyFailure(Parameters const& p, State const& state,
                                  PatrolEvaluationSettings const& settings) noexcept
{
    for (auto const& guard : state.guards)
    {
        if (!guard.available || !std::isfinite(guard.north)
            || !std::isfinite(guard.east) || !std::isfinite(guard.heading)
            || !std::isfinite(guard.speed) || !std::isfinite(guard.turnRate))
            return PatrolFailure::InvalidState;
        if (std::abs(guard.north) >= p.hardBoundaryScale * p.halfNorth
            || std::abs(guard.east) >= p.hardBoundaryScale * p.halfEast)
            return PatrolFailure::WorldBoundary;
        if (settings.restrictedRadius > 0.F
            && std::hypot(guard.north - settings.restrictedNorth,
                          guard.east - settings.restrictedEast) <= settings.restrictedRadius)
            return PatrolFailure::RestrictedZone;
    }
    if (std::hypot(state.guards[0].north - state.guards[1].north,
                   state.guards[0].east - state.guards[1].east)
        < settings.collisionDistance)
        return PatrolFailure::Collision;
    return PatrolFailure::None;
}

// Callback called once per defender, per simulation step. It must yield a
// deterministic policy mean, not a stochastic exploratory action.
using PatrolPredictor = std::function<Action(Observation const&)>;

class PatrolEvaluator final
{
public:
    PatrolEvaluator(Parameters parameters, PatrolEvaluationSettings settings)
        : parameters_(parameters), settings_(settings)
    {
        if (!Valid(parameters_) || !Valid(settings_))
            throw std::invalid_argument("Invalid patrol evaluation configuration");
    }

    PatrolEvaluationReport Evaluate(PatrolPredictor const& predict) const
    {
        if (!predict)
            throw std::invalid_argument("Patrol evaluator requires a predictor");
        PatrolEvaluationReport report{};
        report.episodes.reserve(settings_.episodes);
        for (std::size_t episode = 0; episode < settings_.episodes; ++episode)
        {
            auto const result = RunEpisode(episode, predict);
            report.episodes.push_back(result);
            switch (result.outcome)
            {
            case PatrolOutcome::Success: ++report.successes; break;
            case PatrolOutcome::Failure: ++report.failures; break;
            case PatrolOutcome::Truncated: ++report.truncations; break;
            }
        }
        return report;
    }

private:
    PatrolEpisodeResult RunEpisode(std::size_t episode,
                                  PatrolPredictor const& predict) const
    {
        PatrolEpisodeResult result{};
        result.side = episode % 2U == 0U ? 1 : -1;
        result.initialPhase = CircuitLength(parameters_)
            * static_cast<float>((episode / 2U) % 4U) / 4.F;
        State state = InitialState(parameters_, result.side, 0U, result.initialPhase);

        std::size_t trackingSamples = 0U;
        float summedError = 0.F;
        for (std::size_t step = 0; step < settings_.maxSteps; ++step)
        {
            auto const preFailure = SafetyFailure(parameters_, state, settings_);
            if (preFailure != PatrolFailure::None)
            {
                result.outcome = PatrolOutcome::Failure;
                result.failure = preFailure;
                break;
            }

            std::array<Action, 2> actions{};
            for (std::size_t role = 0; role < 2; ++role)
            {
                auto local = state;
                local.controlledRole = role;
                actions[role] = predict(Observe(parameters_, local));
                for (float const v : actions[role])
                    if (!std::isfinite(v))
                    {
                        result.outcome = PatrolOutcome::Failure;
                        result.failure = PatrolFailure::InvalidState;
                        return result;
                    }
            }
            State const next = AdvanceBoth(parameters_, state, actions);
            ++result.steps;
            // Includes collisions between snapshots (swept relative geometry).
            if (next.collided)
            {
                result.outcome = PatrolOutcome::Failure;
                result.failure = PatrolFailure::Collision;
                break;
            }
            auto const safety = SafetyFailure(parameters_, next, settings_);
            if (safety != PatrolFailure::None)
            {
                result.outcome = PatrolOutcome::Failure;
                result.failure = safety;
                break;
            }

            PatrolProjection const projection = ProjectOnPatrolCurve(
                parameters_.geometry, next.guards[0].north,
                next.guards[0].east);
            float const leaderError = projection.distance
                / std::hypot(parameters_.halfNorth, parameters_.halfEast);
            FormationErrors const formation = MeasureFormationErrors(
                parameters_, next);
            float const wingmanError = std::hypot(
                formation.longitudinal, formation.inward);
            summedError += leaderError + wingmanError;
            float const leaderHeading = std::cos(
                next.guards[0].heading - projection.pose.heading);
            float const wingmanHeading = std::cos(
                next.guards[1].heading - next.guards[0].heading);
            if (leaderError <= settings_.trackingTolerance
                && leaderHeading >= settings_.minimumHeadingCosine
                && next.guards[0].speed >= settings_.minimumSpeedFraction
                    * parameters_.cruiseSpeed)
                ++trackingSamples;
            if (wingmanError <= settings_.trackingTolerance
                && wingmanHeading >= settings_.minimumHeadingCosine
                && next.guards[1].speed >= settings_.minimumSpeedFraction
                    * parameters_.cruiseSpeed)
                ++trackingSamples;
            for (std::size_t role = 0; role < 2; ++role)
            {
                Mobile const& car = next.guards[role];
                result.accumulatedTurnRadians[role] +=
                    std::abs(WrapAngle(car.heading - state.guards[role].heading));
            }
            result.accumulatedProgressMeters[0] += std::max(
                PatrolProgressDelta(parameters_, state, next), 0.F);
            state = next;
        }

        if (result.steps > 0U)
        {
            result.trackingFraction = static_cast<float>(trackingSamples)
                / (2.F * static_cast<float>(result.steps));
            result.meanNormalizedError = summedError
                / (2.F * static_cast<float>(result.steps));
        }

        if (result.outcome == PatrolOutcome::Failure)
            return result;

        // A short horizon is a truncation, not proof of a good/bad patrol.
        float const travelled = result.accumulatedProgressMeters[0];
        if (travelled < settings_.minimumCircuits * CircuitLength(parameters_))
        {
            result.outcome = PatrolOutcome::Truncated;
            return result;
        }
        bool const turns = result.accumulatedTurnRadians[0] >=
                settings_.minimumCircuits * kTwoPi * 0.75F
            && result.accumulatedTurnRadians[1] >=
                settings_.minimumCircuits * kTwoPi * 0.75F;
        if (result.trackingFraction >= settings_.minimumTrackingFraction && turns)
            result.outcome = PatrolOutcome::Success;
        else
        {
            result.outcome = PatrolOutcome::Failure;
            result.failure = PatrolFailure::TrackingQuality;
        }
        return result;
    }

    Parameters parameters_{};
    PatrolEvaluationSettings settings_{};
};

} // namespace ppo::examples::twovstwo::defenders
