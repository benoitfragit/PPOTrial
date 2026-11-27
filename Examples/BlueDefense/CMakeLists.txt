add_subdirectory(Agents/BluePatrol Defenders)

add_library(PPOBlueDefenseSpawn STATIC
    Scenario/spawn.hpp
    Scenario/spawn.cpp
    Scenario/spawn_config.hpp
    Scenario/spawn_config.cpp)
target_compile_features(PPOBlueDefenseSpawn PUBLIC cxx_std_17)
target_include_directories(PPOBlueDefenseSpawn PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Scenario"
    "${CMAKE_CURRENT_SOURCE_DIR}/Agents/BluePatrol/Core")
target_include_directories(PPOBlueDefenseSpawn PRIVATE
    "${CMAKE_SOURCE_DIR}/Examples/Shared/World2D/data")
target_link_libraries(PPOBlueDefenseSpawn PUBLIC nlohmann_json::nlohmann_json)

add_library(PPOBlueDefenseExample STATIC
    Scenario/bluedefenseexample.hpp
    Scenario/bluedefenseexample.cpp
    Scenario/initializer.hpp
    Scenario/initializer.cpp
    Scenario/bluedefenserules.hpp
    Scenario/bluedefenserules.cpp)
target_compile_features(PPOBlueDefenseExample PUBLIC cxx_std_17)
target_include_directories(PPOBlueDefenseExample PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
target_link_libraries(PPOBlueDefenseExample PUBLIC
    PPOAirspaceExample PPOBlueDefenseSpawn
    BlueDefenseDefenderBackend BlueDefensePatrolRuntime
    PRIVATE PPOKit::Schema)

# BluePatrol for ppokit-run: the world (P blues against N RLMission reds, the blue team
# driven from outside) and the team's networks (the generalized MAPPO learner and the
# production actor, whose V7/V10 checkpoint files they keep).
add_library(blue_environment SHARED Environment/blue_environment.cpp)
target_link_libraries(blue_environment PRIVATE PPOBlueDefenseExample PPOKit::Core PPOKit::Schema)
set_target_properties(blue_environment PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    PREFIX "")
ppokit_bind_task_contract(TARGET blue_environment TASK Agents/BluePatrol/task.json)
ppokit_register_program(blue_environment)
add_library(blue_network SHARED Network/blue_network.cpp)
target_include_directories(blue_network PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
target_link_libraries(blue_network PRIVATE BlueDefenseDefenderBackend PPOBlueDefenseSpawn PPOLayout
    PPOKit::Schema nlohmann_json::nlohmann_json)
set_target_properties(blue_network PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "")
ppokit_bind_task_contract(TARGET blue_network TASK Agents/BluePatrol/task.json)
ppokit_register_program(blue_network)

# Former V7/V10 checkpoints -> PPOKit artifacts, checked by an identical round trip.
add_executable(blue_checkpoint_import Tools/blue_checkpoint_import.cpp)
target_link_libraries(blue_checkpoint_import PRIVATE BlueDefenseDefenderBackend)
# Blue MAPPO artifacts (rl_tools backend) -> shared graph backend, checked by reload and actions.
add_executable(blue_shared_backend_import Tools/blue_shared_backend_import.cpp)
target_link_libraries(blue_shared_backend_import PRIVATE BlueDefenseDefenderBackend PPOKit::Schema PPOKit::Training
    PPOKit::Artifacts)
target_compile_definitions(blue_shared_backend_import PRIVATE
    PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
# The BlueDefense Visualiseur: an observer block of network.json and a 2D view for ppokit-viewer.
add_library(blue_visualiser MODULE Visualiser/blue_visualiser.cpp)
target_link_libraries(blue_visualiser PRIVATE PPOKit::ComponentSdk PPOKit::Live)
set_target_properties(blue_visualiser PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "")
ppokit_register_program(blue_visualiser)
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_dependencies(blue_shared_backend_import blue_visualiser)
endif()
if (TARGET PPOKitRepositoryProjectsTests)
    # BluePatrol's network.json names the Visualiseur: the repository check opens it.
    add_dependencies(PPOKitRepositoryProjectsTests blue_visualiser)
    target_compile_definitions(PPOKitRepositoryProjectsTests PRIVATE PPOKIT_BLUE_LIBRARIES=1)
endif()

if (TARGET PPOKitPublishedModelsTests)
    add_dependencies(PPOKitPublishedModelsTests blue_environment blue_network blue_visualiser)
    add_test(NAME PPOKitPublishedModels_BluePatrol COMMAND PPOKitPublishedModelsTests
        "${CMAKE_CURRENT_SOURCE_DIR}/projects/BluePatrol" "${CMAKE_CURRENT_BINARY_DIR}/published_models_BluePatrol" resume)
endif()

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(BlueDefenseGeneralizedLeaderRuntimeTests
        tests/generalized_leader_runtime_tests.cpp)
    target_compile_features(BlueDefenseGeneralizedLeaderRuntimeTests PRIVATE cxx_std_17)
    target_include_directories(BlueDefenseGeneralizedLeaderRuntimeTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
    target_link_libraries(BlueDefenseGeneralizedLeaderRuntimeTests PRIVATE
        BlueDefensePatrolRuntime)
    add_test(NAME BlueDefenseGeneralizedLeaderRuntimeTests
        COMMAND BlueDefenseGeneralizedLeaderRuntimeTests)

    # The DLL's world steps with the time step and limits of Environment.json.
    add_executable(BlueDefenseWorldSettingsTests tests/world_settings_tests.cpp)
    target_compile_features(BlueDefenseWorldSettingsTests PRIVATE cxx_std_17)
    target_link_libraries(BlueDefenseWorldSettingsTests PRIVATE PPOBlueDefenseExample GenericEnv PPOLayout
        nlohmann_json::nlohmann_json)
    add_test(NAME BlueDefenseWorldSettingsTests COMMAND BlueDefenseWorldSettingsTests)

    # The Visualiseur end to end: BluePatrol play and training send frames to BlueView.
    add_executable(BlueDefenseVisualiserTests tests/visualiser_tests.cpp)
    target_compile_features(BlueDefenseVisualiserTests PRIVATE cxx_std_17)
    target_link_libraries(BlueDefenseVisualiserTests PRIVATE PPOKit::View PPOLayout nlohmann_json::nlohmann_json)
    add_dependencies(BlueDefenseVisualiserTests ppokit-run blue_environment blue_network blue_visualiser ppokit_graph_network)
    target_compile_definitions(BlueDefenseVisualiserTests PRIVATE
        PPOKIT_RUN="$<TARGET_FILE:ppokit-run>"
        BLUE_ENVIRONMENT_LIBRARY="$<TARGET_FILE:blue_environment>"
        BLUE_NETWORK_LIBRARY="$<TARGET_FILE:blue_network>"
        BLUE_GRAPH_NETWORK_LIBRARY="$<TARGET_FILE:ppokit_graph_network>"
        BLUE_VISUALISER_LIBRARY="$<TARGET_FILE:blue_visualiser>")
    add_test(NAME BlueDefenseVisualiserTests COMMAND BlueDefenseVisualiserTests)

    add_executable(BlueDefenseSpawnTests tests/spawn_tests.cpp)
    target_compile_features(BlueDefenseSpawnTests PRIVATE cxx_std_17)
    target_include_directories(BlueDefenseSpawnTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
    target_link_libraries(BlueDefenseSpawnTests PRIVATE
        PPOBlueDefenseSpawn BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseSpawnTests COMMAND BlueDefenseSpawnTests)

    add_executable(BlueDefenseCollisionRulesTests tests/collision_rules_tests.cpp)
    target_compile_features(BlueDefenseCollisionRulesTests PRIVATE cxx_std_17)
    target_link_libraries(BlueDefenseCollisionRulesTests PRIVATE PPOBlueDefenseExample)
    add_test(NAME BlueDefenseCollisionRulesTests COMMAND BlueDefenseCollisionRulesTests)

    add_executable(BlueDefenseGeneralizedInitializerTests
        tests/generalized_initializer_tests.cpp)
    target_compile_features(BlueDefenseGeneralizedInitializerTests
        PRIVATE cxx_std_17)
    target_link_libraries(BlueDefenseGeneralizedInitializerTests
        PRIVATE PPOBlueDefenseExample)
    add_test(NAME BlueDefenseGeneralizedInitializerTests
        COMMAND BlueDefenseGeneralizedInitializerTests)
endif()
