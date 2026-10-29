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
    static constexpr std::size_t kBlueCapacity = 9U;
    std::array<float, kBlueCapacity> north{}, east{};
    std::array<bool, kBlueCapacity> present{}, destroyed{};
    std::array<bool, kBlueCapacity> alignedBluePlatform{};
    std::array<float, kBlueCapacity> speed{};
    unsigned blueEntities = 0U;
    unsigned controllers = 0U;
    unsigned redEntities = 0U;
    unsigned finiteReds = 0U;
    unsigned activeReds = 0U;
    unsigned unmodifiedRedPlatforms = 0U;
    unsigned redsTargetingObjective = 0U;
    bool trainingPresent = false;
    bool rewardsPresent = false;
    bool generalizedController = false;
    bool evaluationInProgress = false;
    std::uint64_t completedEvaluations = 0U;
    long long rollout = 0;
};

struct ExpectedTeam final
{
    std::size_t blueCount{};
    std::size_t redCount{};
    bool generalized{};
};

// Resolve dynamic entity names without depending on fixed 2v2 slot names.
int NumberedIndex(std::string_view const name, std::string_view const prefix,
                  std::size_t const maximum) noexcept
{
    if (name.size() <= prefix.size()
        || name.substr(0U, prefix.size()) != prefix)
        return -1;
    std::size_t number = 0U;
    for (char const digit : name.substr(prefix.size()))
    {
        if (digit < '0' || digit > '9')
            return -1;
        number = number * 10U + static_cast<std::size_t>(digit - '0');
        if (number > maximum)
            return -1;
    }
    return number > 0U ? static_cast<int>(number - 1U) : -1;
}

ExpectedTeam ScenarioTeam(std::filesystem::path const& scenarioPath)
{
    auto const document = ppo::serialization::Serializer::Parse(scenarioPath);
    auto const& settings = document.at("twoVsTwo");
    ExpectedTeam result{};
    result.generalized = settings.contains("blueTeam");
    result.blueCount = result.generalized
        ? settings.at("blueTeam").at("activeCount").get<std::size_t>() : 2U;
    result.redCount = settings.value("patrolOnly", false) ? 0U
        : settings.contains("redTeam")
        ? settings.at("redTeam").at("activeCount").get<std::size_t>() : 2U;
    return result;
}

Sample Capture(
    ppo::model::Environment const& environment,
    ppo::examples::twovstwo::defenders::Parameters const& parameters)
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
                std::string_view const name{metadata.m_name};
                int const index = name == "blue_guard_1" ? 0
                    : name == "blue_guard_2" ? 1
                    : NumberedIndex(name, "blue_", Sample::kBlueCapacity);
                if (index >= 0)
                {
                    auto const i = static_cast<std::size_t>(index);
                    result.present[i] = true;
                    ++result.blueEntities;
                    result.destroyed[i] = state.m_destroyed;
                    result.north[i] = state.m_x;
                    result.east[i] = state.m_y;
                    result.speed[i] = state.m_speed;
                    auto const* flight =
                        world.try_get<ppo::model::sFlightModelState>(entity);
                    auto const* sna = world.try_get<ppo::model::sSNAState>(entity);
                    result.alignedBluePlatform[i] = flight && sna
                        && std::abs(flight->m_maxAcc - parameters.maxAcceleration) < 1e-3F
                        && std::abs(flight->m_maxAccRot - parameters.maxRotationRate) < 1e-5F
                        && std::abs(flight->m_maxSpeed - parameters.maxSpeed) < 1e-3F
                        && std::abs(flight->m_minSpeed - 10.F) < 1e-3F
                        && std::abs(sna->m_maxNoescapeZone - 5000.F) < 1e-3F
                        && std::abs(sna->m_noescapezone - 5000.F) < 1e-3F
                        && std::abs(sna->m_escapezone - 10000.F) < 1e-3F
                        && std::abs(sna->m_maxShootAngle
                                    - 3.1471980571746827F) < 1e-5F;
                }
                if (NumberedIndex(name, "red_", 10U) >= 0)
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
                if (NumberedIndex(metadata.m_name, "red_", 10U) >= 0
                    && objective != entt::null && sna.m_target == objective)
                    ++result.redsTargetingObjective;
            });
        world.view<ppo::model::sAgentStateSnaphot>().each(
            [&](auto const& snapshot) {
                std::string_view const name{snapshot.m_name};
                bool const legacy = name == "Blue MAPPO (RLTools)";
                bool const generalized =
                    name == "Blue generalized shared MAPPO";
                if (legacy || generalized)
                {
                    ++result.controllers;
                    result.generalizedController |= generalized;
                    result.trainingPresent |= snapshot.m_common.m_training;
                    result.rewardsPresent |= generalized
                        ? snapshot.m_rewards.size() == 8U
                            && snapshot.m_rewardnames.size() == 8U
                            && snapshot.m_rewardnames[4] == "Threat containment"
                        : snapshot.m_rewards.size() == 7U
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
bool Ready(Sample const& s, ExpectedTeam const expected)
{
    if (expected.blueCount == 0U || expected.blueCount > s.present.size()
        || s.blueEntities != expected.blueCount
        || s.redEntities != expected.redCount
        || s.finiteReds != expected.redCount
        || s.redsTargetingObjective != expected.redCount
        || s.unmodifiedRedPlatforms != expected.redCount
        || s.controllers != 1U || !s.trainingPresent || !s.rewardsPresent
        || s.generalizedController != expected.generalized
        || s.rollout != 2048)
        return false;
    for (std::size_t i = 0U; i < expected.blueCount; ++i)
    {
        if (!s.present[i] || s.destroyed[i] || s.speed[i] <= 0.F
            || !s.alignedBluePlatform[i])
            return false;
    }
    return true;
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
    // Verify the real configured platform contract, not stale numerical
    // constants from earlier defender_car.json revisions.
    if (!ppo::examples::twovstwo::defenders::Valid(patrolParameters))
        return 5;
    auto const definition = combat
        ? ppo::examples::twovstwo::Definition()
        : ppo::examples::twovstwo::BluePatrolDefinition();
    ExpectedTeam const expected = ScenarioTeam(
        std::filesystem::path{RESOURCE_PATH} / definition.defaultScenario);
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
            auto const current = Capture(environment, patrolParameters);
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
        auto const current = Capture(environment, patrolParameters);
        if (!foundFirst && Ready(current, expected))
        {
            first = current;
            foundFirst = true;
            continue;
        }
        if (!foundFirst || !Ready(current, expected))
            continue;
        later = current;
        moved = true;
        for (std::size_t role = 0; role < expected.blueCount; ++role)
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
                  << " blues=" << later.blueEntities << '/'
                  << expected.blueCount
                  << " reds=" << later.redEntities << '/'
                  << expected.redCount
                  << " generalized=" << later.generalizedController
                  << " controllers=" << later.controllers
                  << " training=" << later.trainingPresent
                  << " rewards=" << later.rewardsPresent
                  << " rollout=" << later.rollout << '\n';
    }
    return foundFirst && moved ? 0 : 2;
}
