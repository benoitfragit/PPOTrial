cmake_minimum_required(VERSION 3.19)
project(TopPPO LANGUAGES CXX)

# Example libraries also go into DLLs (environment, network and Visualiseur libraries).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()
# Floating point as written: GNU mode lets GCC and Clang fuse a multiply and an add into one
# FMA (-ffp-contract=fast) wherever the target has FMA, differently in each copy of a formula.
# Bit-for-bit checks (GAE, Adam, the Blue MAPPO update on the shared graph backend) then fail
# on an FMA runner only. MSVC does not contract by default; GCC and Clang must not either.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-ffp-contract=off)
endif()

option(PPO_BUILD_DESKTOP "Build the desktop programs (raylib: ppokit-studio, ppokit-viewer) and the examples" ON)
# Tests are opt-in for local development; CI workflows enable them explicitly.
option(PPO_BUILD_CORE_TESTS "Build dependency-free RL core tests" OFF)
option(PPO_BUILD_CHECKPOINT_TESTS "Verify existing policy checkpoints without training" OFF)
option(PPO_BUILD_FLIGHT_CONTROL "Build the native JSBSim FlightControl3D example" ${PPO_BUILD_DESKTOP})
# The ECS world of the 2D examples without graphics (EnTT only): the environment and network
# DLLs of BlueDefense for ppokit-run. The desktop build always has it.
option(PPO_BUILD_WORLD "Build the ECS world of the 2D examples without graphics (BlueDefense DLLs)" OFF)

if (PPO_BUILD_CORE_TESTS)
    enable_testing()
endif()

# Repository layout. Each example keeps its settings, models and projects in
# Examples/<Name>; the 2D world layers they share live in Examples/Shared.
add_library(PPOLayout INTERFACE)
target_compile_definitions(PPOLayout INTERFACE
    PPO_SHARED_SETTINGS="${CMAKE_SOURCE_DIR}/Examples/Shared/settings"
    PPO_AIRSPACE_SETTINGS="${CMAKE_SOURCE_DIR}/Examples/Airspace/settings"
    PPO_BLUEDEFENSE_PROJECT="${CMAKE_SOURCE_DIR}/Examples/BlueDefense/projects/BluePatrol"
    PPO_FLIGHT_T6_PROJECT="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/T6"
    PPO_FLIGHT_PC7_PROJECT="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/PC7")

# PPOKit (only third-party dependency: nlohmann_json) is shared by every build flavour.
add_subdirectory(PPOKit PPOKit)
add_subdirectory(Framework/RL/Core Core)
add_subdirectory(Framework/RL/Runtime Runtime)

# The ECS world (EnTT) and the 2D examples on it, without graphics.
macro(ppo_add_world)
    add_subdirectory(Framework/Support/Serialization Serialization)
    add_subdirectory(Framework/Simulation/Entities Entities)
    add_subdirectory(Examples/Shared/World2D SharedWorld)
    add_subdirectory(Examples/Airspace/Agents/Bindings Bindings)
    add_subdirectory(Framework/Simulation/Env Env)
    add_subdirectory(Framework/Simulation/Meta Meta)
endmacro()

# Core contracts deliberately compile independently of EnTT and graphics.
if (NOT PPO_BUILD_DESKTOP)
    if (PPO_BUILD_FLIGHT_CONTROL)
        add_subdirectory(Examples/FlightControl3D)
    endif()
    if (PPO_BUILD_WORLD)
        include(${CMAKE_SOURCE_DIR}/world_dependencies.cmake)
        ppo_add_world()
        # Examples/ without the desktop programs.
        add_subdirectory(Examples)
    endif()
    if (PPO_BUILD_CORE_TESTS)
        # The world's examples already define the shared regressions; this adds the others.
        add_subdirectory(Examples/BlueDefense/tests BlueDefenseCoreTests)
    endif()
    ppokit_write_program_registry()
    return()
endif()

# For Visual Studio 2019 x64 (legacy setting retained for desktop).
set(CMAKE_GENERATOR_PLATFORM x64)

include(${CMAKE_SOURCE_DIR}/dependencies.cmake)

ppo_add_world()
add_subdirectory(Examples)
if (PPO_BUILD_CORE_TESTS)
    # The shared contract tests also apply to the desktop build; existing world tests are skipped.
    add_subdirectory(Examples/BlueDefense/tests BlueDefenseCoreTests)
endif()
if (PPO_BUILD_FLIGHT_CONTROL)
    # Runs under ppokit-run (environment, network and Visualiseur DLLs): no desktop part.
    add_subdirectory(Examples/FlightControl3D)
endif()
add_subdirectory(PPOKit/studio PPOKitStudio)
add_subdirectory(PPOKit/viewer PPOKitViewer)
if (PPO_BUILD_CHECKPOINT_TESTS)
    enable_testing()
endif()

# Where ppokit-studio finds the programs of the projects it opens.
ppokit_write_program_registry()
