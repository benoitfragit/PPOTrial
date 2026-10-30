cmake_minimum_required(VERSION 3.16)
project(TopPPO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

option(PPO_BUILD_DESKTOP "Build existing LibTorch desktop application" ON)
# Tests are opt-in for local development; CI workflows enable them explicitly.
option(PPO_BUILD_CORE_TESTS "Build dependency-free RL core tests" OFF)
option(PPO_BUILD_CHECKPOINT_TESTS "Verify existing policy checkpoints without training" OFF)

if (PPO_BUILD_CORE_TESTS)
    enable_testing()
endif()
add_subdirectory(Framework/RL/Core Core)
add_subdirectory(Framework/RL/Runtime Runtime)
if (PPO_BUILD_CORE_TESTS)
    target_link_libraries(GenericRLCoreTests PRIVATE LegacyAgentAPI)
endif()

# Core contracts deliberately compile independently of Torch, EnTT and graphics.
if (NOT PPO_BUILD_DESKTOP)
    if (PPO_BUILD_CORE_TESTS)
        add_subdirectory(Examples/BlueDefense/tests BlueDefenseCoreTests)
    endif()
    return()
endif()

# For Visual Studio 2019 x64 (legacy setting retained for desktop).
set(CMAKE_GENERATOR_PLATFORM x64)

include(${CMAKE_SOURCE_DIR}/dependencies.cmake)

add_subdirectory(Framework/Support/Serialization Serialization)
add_subdirectory(Framework/Simulation/Entities Entities)
add_subdirectory(Framework/Simulation/World2D Examples/Shared/World)
add_subdirectory(Framework/RL/PPO PPO)
add_subdirectory(Framework/RL/Bindings Bindings)
add_subdirectory(Framework/Simulation/Env Env)
add_subdirectory(Framework/Simulation/Meta Meta)
add_subdirectory(Framework/Support/Renderer Renderer)
add_subdirectory(Examples)
add_subdirectory(Main)
if (PPO_BUILD_CHECKPOINT_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
