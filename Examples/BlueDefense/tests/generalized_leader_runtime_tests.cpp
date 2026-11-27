/** @file
 * @brief Exercises leader loss through the real ECS controller, driven as ppokit-run drives
 *        it (PrepareExternalStep, the actions in the exchange, the world's step).
 */
#include "generalized_patrol_team_agent.hpp"
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace blue = ppo::examples::bluedefense::defenders;
namespace scenario = ppo::examples::bluedefense;
namespace model = ppo::model;
namespace {

void Require(bool const condition, char const* const message)
{
    if (!condition)
        throw std::runtime_error(message);
}

/** @brief Publishes the team's observations and writes a finite action for every slot. */
void ActOnce(blue::GeneralizedPatrolMappoAgent& agent)
{
    agent.PrepareExternalStep();
    for (auto& action : agent.Exchange().actions)
        action = {0.25F, -0.5F};
    agent.ChooseAction();
}

/** @brief Whether the published observations of the available slots are finite. */
bool FiniteObservations(blue::GeneralizedPatrolMappoAgent const& agent, std::size_t const count)
{
    for (std::size_t index = 0U; index < count; ++index)
    {
        if (!agent.Exchange().available[index])
            continue;
        for (float const value : agent.Exchange().observations[index].values)
            if (!std::isfinite(value))
                return false;
    }
    return true;
}

/** @brief Makes a minimal finite ECS mobile controlled by the real actor. */
std::unique_ptr<model::BaseEntity> Mobile(entt::registry& registry,
    float const north, bool const destroyed)
{
    auto entity = std::make_unique<model::BaseEntity>(&registry);
    auto& state = entity->add<model::sEntityState>();
    state.m_x = north;
    state.m_speed = 300.F;
    state.m_destroyed = destroyed;
    entity->add<model::sSNAControlState>();
    entity->add<model::sPseudoLidarState>().m_nearestCollisionRanges.fill(1.F);
    return entity;
}

/** @brief Loss before delayed spawn and in combat, with repeated promotions. */
void RunCase(std::size_t const count, std::size_t const redCount, bool const activeRed)
{
    entt::registry registry;
    auto& context = registry.ctx().emplace<scenario::InitializerContext>();
    context.patrolGeometryReady = true;
    context.blueTeamRoles = blue::BlueTeamRoles::Create(count, count - 1U);
    context.episodeLayout.emplace();
    if (redCount != 0U)
        context.episodeLayout->redActivationSteps.push_back(activeRed ? 0U : 100U);
    std::array<std::unique_ptr<model::BaseEntity>, 9U> bodies{};
    std::array<model::BaseEntity*, 9U> guards{};
    for (std::size_t index = 0U; index < count; ++index)
    {
        bodies[index] = Mobile(registry, 3000.F * static_cast<float>(index), false);
        guards[index] = bodies[index].get();
    }
    auto red = Mobile(registry, 40000.F, !activeRed);
    std::array<model::BaseEntity*, 10U> attackers{};
    attackers[0] = red.get();
    auto objective = Mobile(registry, -10000.F, false);
    blue::GeneralizedPatrolMappoAgent agent{guards, count, attackers, redCount,
        *objective, {}, {}};
    agent.Init();
    std::size_t visibilitySteps = 0U;
    if (redCount != 0U && !activeRed)
    {
        ActOnce(agent);
        Require(context.patrolReferenceOverlay.Visible(),
            "configured but inactive Red hid patrol references");
        auto [redState] = red->get<model::sEntityState>();
        redState.m_destroyed = false;
        Require(!agent.SetNewState() && !context.patrolReferenceOverlay.Visible(),
            "Red activation retained visible references in the resolved snapshot");
        Require(context.patrolReferenceOverlay.m_references[0].m_valid,
            "render-only hiding changed policy guidance validity");
        ActOnce(agent);
        redState.m_destroyed = true;
        Require(!agent.SetNewState() && context.patrolReferenceOverlay.Visible(),
            "patrol references did not return after the active threat disappeared");
        visibilitySteps = 2U;
    }
    for (std::size_t loss = 0U; loss < count; ++loss)
    {
        std::size_t const oldLeader = context.blueTeamRoles->LeaderPhysicalIndex();
        ActOnce(agent);
        Require(context.patrolReferenceOverlay.Visible() == !activeRed,
            "patrol reference visibility disagrees with actual Red activity");
        auto [state] = guards[oldLeader]->get<model::sEntityState>();
        state.m_destroyed = true; // Physics/world resolution is authoritative.
        bool const done = agent.SetNewState();
        bool const expectedEnd = redCount == 0U || loss + 1U == count;
        Require(done == expectedEnd, "leader loss changed mission termination");
        if (expectedEnd)
        {
            Require(!context.patrolReferenceOverlay.Visible(),
                "leader loss left a ghost patrol marker");
            break;
        }
        Require(context.blueTeamRoles->LeaderPhysicalIndex() == loss,
            "controller did not promote the first physical survivor");
        Require(context.patrolReferenceOverlay.m_references[0].m_valid,
            "new leader guidance was not published");
        Require(context.patrolReferenceOverlay.Visible() == !activeRed,
            "leader relief retained stale patrol visibility");
        ActOnce(agent);
        Require(FiniteObservations(agent, count), "nonfinite post-relief observation");
        Require(!agent.SetNewState(), "post-relief step ended an ongoing mission");
        Require(agent.EpisodeStep() == visibilitySteps + 2U * (loss + 1U),
            "relief reset or dropped the physical trajectory step");
    }
}

/** @brief The ECS reward must use live pre-loss kinematics, including the objective. */
void NeutralizationCredit(float const distance, float const risk)
{
    entt::registry registry;
    auto& context = registry.ctx().emplace<scenario::InitializerContext>();
    context.patrolGeometryReady = true;
    context.blueTeamRoles = blue::BlueTeamRoles::Create(1U, 0U);
    auto guard = Mobile(registry, -20000.F, false);
    auto red = Mobile(registry, distance, false);
    auto objective = Mobile(registry, 0.F, false);
    auto [redState] = red->get<model::sEntityState>();
    redState.m_speed = 1.F;
    redState.m_rotation = blue::kPi;
    std::array<model::BaseEntity*, 9U> guards{};
    std::array<model::BaseEntity*, 10U> attackers{};
    guards[0] = guard.get();
    attackers[0] = red.get();
    blue::GeneralizedPatrolMappoAgent agent{guards, 1U, attackers, 1U,
        *objective, {}, {}};
    blue::GeneralizedRewardValues weights{};
    weights.fill(1.F); // The published channels are then the unweighted ones.
    agent.SetRewardWeights(weights);
    agent.Init();
    ActOnce(agent);
    redState.m_destroyed = true;
    redState.m_x = 500000.F; // Post-loss data must not erase the pre-loss risk.
    redState.m_speed = -1.F;
    Require(agent.SetNewState(), "last neutralized Red did not end the mission");
    // Shared channels: the single blue's are the team's.
    auto const& channels = agent.Exchange().weightedChannels[0];
    Require(std::abs(channels[static_cast<std::size_t>(
                blue::GeneralizedRewardChannel::RedNeutralization)] - (1.F + risk))
            < 1.e-6F, "runtime neutralization did not use pre-loss objective risk");
    Require(std::abs(channels[static_cast<std::size_t>(
                blue::GeneralizedRewardChannel::ThreatContainment)] - risk)
            < 1.e-6F, "existing threat containment changed during neutralization");
}

/** @brief A live edit preserves the configured own-loss weight and rejects invalid edits atomically. */
void LiveRewardWeights()
{
    entt::registry registry;
    auto guard = Mobile(registry, 0.F, false);
    auto objective = Mobile(registry, 10000.F, false);
    std::array<model::BaseEntity*, 9U> guards{};
    std::array<model::BaseEntity*, 10U> attackers{};
    guards[0] = guard.get();
    blue::GeneralizedPatrolMappoAgent agent{guards, 1U, attackers, 0U, *objective, {}, {}};
    auto const own = static_cast<std::size_t>(blue::GeneralizedRewardChannel::OwnLoss);
    auto const patrol = static_cast<std::size_t>(blue::GeneralizedRewardChannel::PatrolTracking);
    auto weights = agent.RewardWeights();
    Require(weights[own] == 24.F, "configured own-loss default changed");
    weights[patrol] = 2.F;
    agent.SetRewardWeights(weights);
    Require(agent.RewardWeights()[own] == 24.F && agent.RewardWeights()[patrol] == 2.F,
        "editing patrol must preserve the configured own-loss weight");
    auto const accepted = agent.RewardWeights();
    for (float invalid : {-1.F, 11.F, std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN()})
    {
        auto rejected = accepted;
        rejected[patrol] = invalid;
        bool failed = false;
        try { agent.SetRewardWeights(rejected); }
        catch (std::invalid_argument const&) { failed = true; }
        Require(failed && agent.RewardWeights() == accepted, "invalid edit changed the active reward weights");
    }
}

} // namespace

int main()
{
    try
    {
        LiveRewardWeights();
        NeutralizationCredit(120000.F, 0.F);
        NeutralizationCredit(63333.333F, 0.5F);
        NeutralizationCredit(9999.F, 1.F);
        for (std::size_t count : {1U, 2U, 3U, 4U, 9U})
        {
            RunCase(count, 0U, false);
            RunCase(count, 1U, false);
            RunCase(count, 1U, true);
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
