# PPOKit: framework shared by the PPOTrial examples.
# Design and migration plan: docs/architecture/ppokit-framework-design.md
# Progress journal for resuming the work: PPOKit/PROGRESS.md
#
# The public headers live under include/ppokit and never expose a backend
# (rl_tools, libtorch), a simulator or a GUI library.

# --- ppokit::core and ppokit::heads : algorithmic building blocks (header-only)
add_library(ppokit_core INTERFACE)
add_library(PPOKit::Core ALIAS ppokit_core)
target_include_directories(ppokit_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_core INTERFACE cxx_std_17)

# --- ppokit::schema and ppokit::eval : task, graph and evaluation files ------
include(FetchContent)
if (NOT TARGET nlohmann_json::nlohmann_json)
    # Same declaration as dependencies.cmake and FlightControl3D: the first one wins.
    FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0)
    FetchContent_MakeAvailable(json)
endif()

add_library(ppokit_schema STATIC src/schema/json_validation.cpp src/schema/task_schema.cpp
    src/schema/task_codegen.cpp src/schema/network_schema.cpp src/eval/evaluation.cpp)
add_library(PPOKit::Schema ALIAS ppokit_schema)
# Evaluation suites share the JSON validation of the schema files.
add_library(PPOKit::Evaluation ALIAS ppokit_schema)
target_include_directories(ppokit_schema PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_schema PUBLIC cxx_std_17)
target_link_libraries(ppokit_schema PRIVATE nlohmann_json::nlohmann_json)

add_executable(ppokit-codegen codegen/main.cpp)
target_link_libraries(ppokit-codegen PRIVATE PPOKit::Schema PPOKit::Artifacts)

# ppokit_generate_task_layout(TARGET ... TASK ... HEADER ...)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PPOKitTask.cmake")

# --- ppokit::backend::rltools : learner on generated networks ---------------
# PPOKit owns the rl_tools dependency; the examples and dependencies.cmake reuse it.
if (NOT TARGET RLtools::Core)
    set(RL_TOOLS_ENABLE_TARGETS OFF CACHE BOOL "Build upstream RLtools examples" FORCE)
    set(RL_TOOLS_ENABLE_TESTS OFF CACHE BOOL "Build upstream RLtools tests" FORCE)
    set(RL_TOOLS_ENABLE_HDF5 OFF CACHE BOOL "Use RLtools HDF5" FORCE)
    set(RL_TOOLS_ENABLE_TENSORBOARD OFF CACHE BOOL "Use RLtools TensorBoard" FORCE)
    set(RL_TOOLS_BACKEND_ENABLE_CUDA OFF CACHE BOOL "Use RLtools CUDA" FORCE)
    # Same declaration as dependencies.cmake and FlightControl3D: header-only, no submodules.
    FetchContent_Declare(rltools
        GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
        GIT_TAG 63a7520b2d0d967920ff14c7336ce4f7d682c21c
        GIT_SUBMODULES "" GIT_SUBMODULES_RECURSE FALSE)
    FetchContent_MakeAvailable(rltools)
endif()
add_library(ppokit_backend_rltools INTERFACE)
add_library(PPOKit::BackendRltools ALIAS ppokit_backend_rltools)
target_link_libraries(ppokit_backend_rltools INTERFACE PPOKit::Core RLtools::Core)

# --- ppokit::artifacts : transactional artifact files ------------------------
add_library(ppokit_artifacts STATIC src/artifacts/atomic_file.cpp)
add_library(PPOKit::Artifacts ALIAS ppokit_artifacts)
target_include_directories(ppokit_artifacts PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_artifacts PUBLIC cxx_std_17)

if (PPO_BUILD_CORE_TESTS)
    add_executable(PPOKitAtomicFileTests tests/atomic_file_tests.cpp)
    target_link_libraries(PPOKitAtomicFileTests PRIVATE PPOKit::Artifacts)
    add_test(NAME PPOKitAtomicFileTests COMMAND PPOKitAtomicFileTests)

    add_executable(PPOKitGaeTests tests/gae_tests.cpp)
    target_link_libraries(PPOKitGaeTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitGaeTests COMMAND PPOKitGaeTests)

    add_executable(PPOKitRuntimeTests tests/runtime_tests.cpp)
    target_link_libraries(PPOKitRuntimeTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitRuntimeTests COMMAND PPOKitRuntimeTests)

    add_executable(PPOKitHeadsTests tests/heads_tests.cpp)
    target_link_libraries(PPOKitHeadsTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitHeadsTests COMMAND PPOKitHeadsTests)

    add_executable(PPOKitTaskSchemaTests tests/task_schema_tests.cpp)
    target_link_libraries(PPOKitTaskSchemaTests PRIVATE PPOKit::Schema)
    target_compile_definitions(PPOKitTaskSchemaTests PRIVATE
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json")
    ppokit_generate_task_layout(TARGET PPOKitTaskSchemaTests
        TASK tests/fixtures/sample_task.json HEADER sample_task_layout.hpp)
    add_test(NAME PPOKitTaskSchemaTests COMMAND PPOKitTaskSchemaTests)

    add_executable(PPOKitEvaluationTests tests/evaluation_tests.cpp)
    target_link_libraries(PPOKitEvaluationTests PRIVATE PPOKit::Evaluation PPOKit::Core)
    add_test(NAME PPOKitEvaluationTests COMMAND PPOKitEvaluationTests)

    add_executable(PPOKitBackendLearnerTests tests/backend_learner_tests.cpp)
    target_link_libraries(PPOKitBackendLearnerTests PRIVATE PPOKit::BackendRltools PPOKit::Evaluation)
    target_compile_definitions(PPOKitBackendLearnerTests PRIVATE
        PPOKIT_CORRIDOR_SMOKE_SUITE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/eval/smoke.json")
    ppokit_generate_networks(TARGET PPOKitBackendLearnerTests TASK tests/fixtures/corridor/task.json
        GRAPH tests/fixtures/corridor/graph.json HEADER corridor_networks.hpp)
    add_test(NAME PPOKitBackendLearnerTests COMMAND PPOKitBackendLearnerTests)

    add_executable(PPOKitNetworkSchemaTests tests/network_schema_tests.cpp)
    target_link_libraries(PPOKitNetworkSchemaTests PRIVATE PPOKit::Schema)
    add_test(NAME PPOKitNetworkSchemaTests COMMAND PPOKitNetworkSchemaTests)
endif()
