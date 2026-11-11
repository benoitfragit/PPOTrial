/** @file
 * @brief Exercises leader loss through the real ECS training/inference controller.
 */
#include "generalized_patrol_team_agent.hpp"
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace blue = ppo::examples::bluedefense::defenders;
namespace scenario = ppo::examples::bluedefense;
namespace model = ppo::model;
namespace {

/** @brief Owns only disposable test checkpoints, never latest or prod. */
struct Checkpoints final
{
    std::filesystem::path directory = std::filesystem::temp_directory_path()
        / ("PPOTrial_leader_relief_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Checkpoints() { std::filesystem::create_directory(directory); }
    ~Checkpoints()
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

void Require(bool const condition, char const* const message)
{
    if (!condition)
        throw std::runtime_error(message);
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
void RunCase(std::size_t const count, std::size_t const redCount,
    bool const activeRed, blue::GeneralizedPatrolExecutionMode const mode,
    Checkpoints const& checkpoints, std::filesystem::path const& actor)
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
        *objective, {}, {}, mode, checkpoints.directory / "latest.txt",
        checkpoints.directory / "training.txt", actor};
    agent.Init();
    std::size_t visibilitySteps = 0U;
    if (redCount != 0U && !activeRed)
    {
        agent.ChooseAction();
        Require(context.patrolReferenceOverlay.Visible(),
            "configured but inactive Red hid patrol references");
        auto [redState] = red->get<model::sEntityState>();
        redState.m_destroyed = false;
        Require(!agent.SetNewState() && !context.patrolReferenceOverlay.Visible(),
            "Red activation retained visible references in the resolved snapshot");
        Require(context.patrolReferenceOverlay.m_references[0].m_valid,
            "render-only hiding changed policy guidance validity");
        agent.ChooseAction();
        redState.m_destroyed = true;
        Require(!agent.SetNewState() && context.patrolReferenceOverlay.Visible(),
            "patrol references did not return after the active threat disappeared");
        visibilitySteps = 2U;
    }
    for (std::size_t loss = 0U; loss < count; ++loss)
    {
        std::size_t const oldLeader = context.blueTeamRoles->LeaderPhysicalIndex();
        agent.ChooseAction();
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
        agent.ChooseAction();
        for (std::size_t index = 0U; index < count; ++index)
        {
            auto const [control] = guards[index]->get<model::sSNAControlState>();
            Require(std::isfinite(control.m_rotSpeedCmd)
                    && std::isfinite(control.m_accCmd), "nonfinite post-relief action");
            auto const [body] = guards[index]->get<model::sEntityState>();
            if (body.m_destroyed
                && mode == blue::GeneralizedPatrolExecutionMode::ProductionInference)
                Require(control.m_rotSpeedCmd == 0.F && control.m_accCmd == 0.F,
                    "destroyed physical participant received an action");
        }
        Require(!agent.SetNewState(), "post-relief step ended an ongoing mission");
        auto const snapshot = agent.GetSnapshot();
        Require(snapshot.m_common.m_completedEvaluations == 0U,
            "relief unexpectedly started an evaluation");
        Require(snapshot.m_common.m_step
                == static_cast<long long>(visibilitySteps + 2U * (loss + 1U)),
            "relief reset or dropped the physical trajectory step");
    }
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    try
    {
        Checkpoints checkpoints;
        for (auto mode : {blue::GeneralizedPatrolExecutionMode::Training,
                blue::GeneralizedPatrolExecutionMode::ProductionInference})
            for (std::size_t count : {1U, 2U, 3U, 4U, 9U})
            {
                RunCase(count, 0U, false, mode, checkpoints, argv[1]);
                RunCase(count, 1U, false, mode, checkpoints, argv[1]);
                RunCase(count, 1U, true, mode, checkpoints, argv[1]);
            }
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
