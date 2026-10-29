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
    BlueDefenseDefenderBackend BlueDefensePatrolRuntime)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
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

    # Regression: Play advances every P/N agent through the actual desktop ECS
    # without requiring a pre-existing trained actor checkpoint.
    add_executable(BlueDefensePatrolRuntimeSmokeTest tests/patrol_runtime_smoke.cpp)
    target_compile_features(BlueDefensePatrolRuntimeSmokeTest PRIVATE cxx_std_17)
    target_link_libraries(BlueDefensePatrolRuntimeSmokeTest PRIVATE
        PPOBlueDefenseExample GenericSerializer GenericEnv)
    # The smoke executable links the legacy LibTorch desktop libraries;
    # provide their runtime DLLs alongside the test binary on Windows.
    if (WIN32)
        add_custom_command(TARGET BlueDefensePatrolRuntimeSmokeTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_BINARY_DIR}/libtorch/lib"
                "$<TARGET_FILE_DIR:BlueDefensePatrolRuntimeSmokeTest>")
    endif()
    add_test(NAME BlueDefensePatrolRuntimeSmoke COMMAND BlueDefensePatrolRuntimeSmokeTest)
    add_test(NAME BlueDefenseCombatRuntimeSmoke
        COMMAND BlueDefensePatrolRuntimeSmokeTest blue-defense)

    # Renderer contract: both enabled and disabled hostile zones must remain
    # identifiable through the public ECS snapshot and m_destroyed flag.
    add_executable(BlueDefenseRestrictedZoneSnapshotTests
        tests/restricted_zone_snapshot_tests.cpp)
    target_compile_features(BlueDefenseRestrictedZoneSnapshotTests PRIVATE cxx_std_17)
    target_link_libraries(BlueDefenseRestrictedZoneSnapshotTests PRIVATE
        PPOBlueDefenseExample GenericEnv)
    add_test(NAME BlueDefenseRestrictedZoneSnapshotTests
        COMMAND BlueDefenseRestrictedZoneSnapshotTests)
endif()
