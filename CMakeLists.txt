cmake_minimum_required(VERSION 3.15)
project(TopPPO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

option(PPO_BUILD_DESKTOP "Build existing LibTorch desktop application" ON)
option(PPO_BUILD_CORE_TESTS "Build dependency-free RL core tests" ON)
option(PPO_BUILD_CHECKPOINT_TESTS "Verify existing policy checkpoints without training" OFF)

if (PPO_BUILD_CORE_TESTS)
    enable_testing()
endif()
add_subdirectory(Core)
add_subdirectory(Runtime)
if (PPO_BUILD_CORE_TESTS)
    target_link_libraries(GenericRLCoreTests PRIVATE LegacyAgentAPI)
endif()

# Core contracts deliberately compile independently of Torch, EnTT and graphics.
if (NOT PPO_BUILD_DESKTOP)
    return()
endif()

# For Visual Studio 2019 x64 (legacy setting retained for desktop).
set(CMAKE_GENERATOR_PLATFORM x64)

include(${CMAKE_SOURCE_DIR}/dependencies.cmake)

add_subdirectory(Serialization)
add_subdirectory(Entities)
add_subdirectory(PPO)
add_subdirectory(Bindings)
add_subdirectory(Env)
add_subdirectory(Evaluation)
add_subdirectory(Renderer)
add_subdirectory(Examples)
add_subdirectory(Meta)
add_subdirectory(Main)
if (PPO_BUILD_CHECKPOINT_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
