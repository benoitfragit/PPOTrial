# Dependencies of the ECS world of the 2D examples (Framework/Simulation, Examples/Airspace,
# Examples/BlueDefense): EnTT only. The desktop build adds its graphics in dependencies.cmake;
# a build with PPO_BUILD_WORLD and without the desktop uses this file alone (environment DLLs).
include(FetchContent)

if (NOT TARGET EnTT::EnTT)
    FetchContent_Declare(
        entt
        GIT_REPOSITORY https://github.com/skypjack/entt.git
        GIT_TAG        v3.16.0
    )
    FetchContent_MakeAvailable(entt)
endif()
