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

# --- ppokit::schema : task.json and generated layout headers ----------------
include(FetchContent)
if (NOT TARGET nlohmann_json::nlohmann_json)
    # Same declaration as dependencies.cmake and FlightControl3D: the first one wins.
    FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0)
    FetchContent_MakeAvailable(json)
endif()

add_library(ppokit_schema STATIC src/schema/json_validation.cpp src/schema/task_schema.cpp
    src/schema/task_codegen.cpp src/schema/network_schema.cpp)
add_library(PPOKit::Schema ALIAS ppokit_schema)
target_include_directories(ppokit_schema PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_schema PUBLIC cxx_std_17)
target_link_libraries(ppokit_schema PRIVATE nlohmann_json::nlohmann_json)

add_executable(ppokit-codegen codegen/main.cpp)
target_link_libraries(ppokit-codegen PRIVATE PPOKit::Schema PPOKit::Artifacts)

# ppokit_generate_task_layout(TARGET ... TASK ... HEADER ...)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PPOKitTask.cmake")

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

    add_executable(PPOKitNetworkSchemaTests tests/network_schema_tests.cpp)
    target_link_libraries(PPOKitNetworkSchemaTests PRIVATE PPOKit::Schema)
    add_test(NAME PPOKitNetworkSchemaTests COMMAND PPOKitNetworkSchemaTests)
endif()
