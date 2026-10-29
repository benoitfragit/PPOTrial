#include "twovstwoexample.hpp"
#include "patrol_config.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>

namespace {
struct Sample
{
    std::array<float, 2> north{}, east{};
    std::array<bool, 2> present{}, destroyed{};
    std::array<bool, 2> alignedBluePlatform{};
    std::array<float, 2> speed{};
    unsigned controllers = 0U;
    unsigned redEntities = 0U;
    unsigned finiteReds = 0U;
    unsigned activeReds = 0U;
    unsigned unmodifiedRedPlatforms = 0U;
    unsigned redsTargetingObjective = 0U;
    bool trainingPresent = false;
    bool rewardsPresent = false;
    bool evaluationInProgress = false;
    std::uint64_t completedEvaluations = 0U;
    long long rollout = 0;
};

Sample Capture(ppo::model::Environment const& environment)
{
    Sample result{};
    environment.ReadSnapshot([&](entt::registry const& world) {
        entt::entity objective = entt::null;
        world.view<ppo::model::sEntityMetadata>().each(
            [&](entt::entity entity, auto const& metadata) {
                if (metadata.m_name == "blue_objective")
                    objective = entity;
            });
        world.view<ppo::model::sEntityState,
                   ppo::model::sEntityMetadata>().each(
            [&](entt::entity entity, auto const& state, auto const& metadata) {
                int const index = metadata.m_name == "blue_guard_1" ? 0
                    : metadata.m_name == "blue_guard_2" ? 1 : -1;
                if (index >= 0)
                {
                    auto const i = static_cast<std::size_t>(index);
                    result.present[i] = true;
                    result.destroyed[i] = state.m_destroyed;
                    result.north[i] = state.m_x;
                    result.east[i] = state.m_y;
                    result.speed[i] = state.m_speed;
                    auto const* flight =
                        world.try_get<ppo::model::sFlightModelState>(entity);
                    auto const* sna = world.try_get<ppo::model::sSNAState>(entity);
                    result.alignedBluePlatform[i] = flight && sna
                        && std::abs(flight->m_maxAcc - 600.F) < 1e-3F
                        && std::abs(flight->m_maxAccRot - 0.70000059604645F) < 1e-5F
                        && std::abs(flight->m_maxSpeed - 700.F) < 1e-3F
                        && std::abs(flight->m_minSpeed - 10.F) < 1e-3F
                        && std::abs(sna->m_maxNoescapeZone - 5000.F) < 1e-3F
                        && std::abs(sna->m_noescapezone - 5000.F) < 1e-3F
                        && std::abs(sna->m_escapezone - 10000.F) < 1e-3F
                        && std::abs(sna->m_maxShootAngle
                                    - 3.1471980571746827F) < 1e-5F;
                }
                if (metadata.m_name == "red_1" || metadata.m_name == "red_2")
                {
                    ++result.redEntities;
                    result.activeReds += state.m_destroyed ? 0U : 1U;
                    if (std::isfinite(state.m_x) && std::isfinite(state.m_y)
                        && std::isfinite(state.m_rotation)
                        && std::isfinite(state.m_speed))
                        ++result.finiteReds;
                    auto const* flight =
                        world.try_get<ppo::model::sFlightModelState>(entity);
                    auto const* sna = world.try_get<ppo::model::sSNAState>(entity);
                    if (flight && sna
                        && std::abs(flight->m_maxAcc - 100.F) < 1e-3F
                        && std::abs(flight->m_maxAccRot - 0.4F) < 1e-5F
                        && std::abs(flight->m_maxSpeed - 400.F) < 1e-3F
                        && std::abs(sna->m_maxNoescapeZone - 4000.F) < 1e-3F
                        && std::abs(sna->m_maxShootAngle - 1.047198F) < 1e-5F)
                        ++result.unmodifiedRedPlatforms;
                }
            });
        world.view<ppo::model::sSNAState,
                   ppo::model::sEntityMetadata>().each(
            [&](auto const& sna, auto const& metadata) {
                if ((metadata.m_name == "red_1" || metadata.m_name == "red_2")
                    && objective != entt::null && sna.m_target == objective)
                    ++result.redsTargetingObjective;
            });
        world.view<ppo::model::sAgentStateSnaphot>().each(
            [&](auto const& snapshot) {
                std::string_view const name{snapshot.m_name};
                if (name.find("Blue MAPPO") == 0U)
                {
                    ++result.controllers;
                    result.trainingPresent |= snapshot.m_common.m_training;
                    result.rewardsPresent |= snapshot.m_rewards.size() == 7U
                        && snapshot.m_rewardnames.size() == 7U;
                    result.evaluationInProgress |=
                        snapshot.m_common.m_evaluationinprogress;
                    result.completedEvaluations = std::max(
                        result.completedEvaluations,
                        snapshot.m_common.m_completedEvaluations);
                    result.rollout = snapshot.m_common.m_maxStep;
                }
            });
    });
    return result;
}
bool Ready(Sample const& s, bool combat)
{
    bool const redSetup = combat
        ? s.redEntities == 2U && s.finiteReds == 2U
            && s.redsTargetingObjective == 2U && s.unmodifiedRedPlatforms == 2U
        : s.redEntities == 0U;
    return redSetup && s.present[0] && s.present[1] && !s.destroyed[0]
        && !s.destroyed[1] && s.speed[0] > 0.F && s.speed[1] > 0.F
        && s.alignedBluePlatform[0] && s.alignedBluePlatform[1]
        && s.controllers == 1U && s.trainingPresent && s.rewardsPresent
        && s.rollout == 2048;
}
}

int main(int argc, char** argv)
{
    bool const combat = argc >= 2 && std::string{argv[1]} == "two-vs-two";
    bool const evaluationSmoke = argc == 3
        && std::string{argv[2]} == "evaluation-smoke";
    if (argc > 3 || (argc >= 2 && !combat) || (argc == 3 && !evaluationSmoke))
        return 3;
    auto const patrolParameters =
        ppo::examples::twovstwo::defenders::LoadPatrolParameters(RESOURCE_PATH);
    if (std::abs(patrolParameters.cruiseSpeed - 700.F) >= 1e-3F
        || std::abs(patrolParameters.maxAcceleration - 600.F) >= 1e-3F
        || std::abs(patrolParameters.maxRotationRate - 0.70000059604645F) >= 1e-5F
        || std::abs(patrolParameters.maxSpeed - 700.F) >= 1e-3F)
        return 5;
    auto const definition = combat
        ? ppo::examples::twovstwo::Definition()
        : ppo::examples::twovstwo::BluePatrolDefinition();
    definition.registerTypes();
    ppo::serialization::Serializer::GetInstance().SetPath(
        std::filesystem::path{RESOURCE_PATH});
    ppo::model::Environment environment;
    if (!definition.loadScenario(
            std::filesystem::path{RESOURCE_PATH} / definition.defaultScenario,
            environment))
        return 1;
    ppo::serialization::Serializer::GetInstance().Load();
    environment.Play();

    if (evaluationSmoke)
    {
        bool evaluationSeen = false;
        bool evaluationCompleted = false;
        for (unsigned attempt = 0U; attempt < 12000U; ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto const current = Capture(environment);
            evaluationSeen |= current.evaluationInProgress;
            evaluationCompleted |= current.completedEvaluations > 0U;
            if (evaluationCompleted)
                break;
        }
        environment.Stop();
        return evaluationSeen && evaluationCompleted ? 0 : 4;
    }

    Sample first{}, later{};
    bool foundFirst = false, moved = false;
    // Poll published ECS snapshots, not wall-clock positions; the worker may
    // start slowly under instrumented/loaded CI runners.
    for (unsigned attempt = 0U; attempt < 300U; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto const current = Capture(environment);
        if (!foundFirst && Ready(current, combat))
        {
            first = current;
            foundFirst = true;
            continue;
        }
        if (!foundFirst || !Ready(current, combat))
            continue;
        later = current;
        moved = true;
        for (std::size_t role = 0; role < 2U; ++role)
            moved &= std::hypot(later.north[role] - first.north[role],
                                later.east[role] - first.east[role]) > 32.F;
        if (moved)
            break;
    }
    environment.Stop();
    if (!foundFirst || !moved)
    {
        std::cerr << "[PATROL SMOKE] initialized=" << foundFirst
                  << " moved=" << moved << " combat=" << combat
                  << " guards=" << later.present[0] << '/' << later.present[1]
                  << " destroyed=" << later.destroyed[0] << '/'
                  << later.destroyed[1]
                  << " reds=" << later.redEntities
                  << " controllers=" << later.controllers
                  << " training=" << later.trainingPresent
                  << " rewards=" << later.rewardsPresent
                  << " rollout=" << later.rollout << '\n';
    }
    return foundFirst && moved ? 0 : 2;
}
