#include "twovstwoexample.hpp"
#include "environment.hpp"
#include "serializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string_view>
#include <thread>

namespace {
struct Sample
{
    std::array<float, 2> north{}, east{};
    std::array<bool, 2> present{}, destroyed{};
    std::array<float, 2> speed{};
    unsigned controllers = 0U;
};

Sample Capture(ppo::model::Environment const& environment)
{
    Sample result{};
    environment.ReadSnapshot([&](entt::registry const& world) {
        world.view<ppo::model::sEntityState,
                   ppo::model::sEntityMetadata>().each(
            [&](entt::entity, auto const& state, auto const& metadata) {
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
                }
            });
        world.view<ppo::model::sAgentStateSnaphot>().each(
            [&](auto const& snapshot) {
                std::string_view const name{snapshot.m_name};
                if (name.find("Patrol ") == 0U)
                    ++result.controllers;
            });
    });
    return result;
}
bool Ready(Sample const& s)
{
    return s.present[0] && s.present[1] && !s.destroyed[0]
        && !s.destroyed[1] && s.speed[0] > 0.F && s.speed[1] > 0.F
        && s.controllers == 2U;
}
}

int main()
{
    auto const definition = ppo::examples::twovstwo::BluePatrolDefinition();
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

    Sample first{}, later{};
    bool foundFirst = false, moved = false;
    // Poll published ECS snapshots, not wall-clock positions; the worker may
    // start slowly under instrumented/loaded CI runners.
    for (unsigned attempt = 0U; attempt < 300U; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto const current = Capture(environment);
        if (!foundFirst && Ready(current))
        {
            first = current;
            foundFirst = true;
            continue;
        }
        if (!foundFirst || !Ready(current))
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
    return foundFirst && moved ? 0 : 2;
}
