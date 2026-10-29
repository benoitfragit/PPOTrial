add_subdirectory(Defenders)

add_library(PPOTwoVsTwoSpawn STATIC
    data/spawn.hpp
    src/spawn.cpp
    data/spawn_config.hpp
    src/spawn_config.cpp)
target_compile_features(PPOTwoVsTwoSpawn PUBLIC cxx_std_17)
target_include_directories(PPOTwoVsTwoSpawn PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Defenders/Core/data")
target_link_libraries(PPOTwoVsTwoSpawn PUBLIC nlohmann_json::nlohmann_json)

add_library(PPOTwoVsTwoExample STATIC
    data/twovstwoexample.hpp
    src/twovstwoexample.cpp
    data/initializer.hpp
    src/initializer.cpp
    data/twovstworules.hpp
    src/twovstworules.cpp)
target_compile_features(PPOTwoVsTwoExample PUBLIC cxx_std_17)
target_include_directories(PPOTwoVsTwoExample PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOTwoVsTwoExample PUBLIC
    PPOAirspaceExample PPOTwoVsTwoSpawn
    TwoVsTwoDefenderBackend TwoVsTwoPatrolRuntime)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoSpawnTests tests/spawn_tests.cpp)
    target_compile_features(TwoVsTwoSpawnTests PRIVATE cxx_std_17)
    target_include_directories(TwoVsTwoSpawnTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/data")
    target_link_libraries(TwoVsTwoSpawnTests PRIVATE
        PPOTwoVsTwoSpawn TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoSpawnTests COMMAND TwoVsTwoSpawnTests)

    add_executable(TwoVsTwoCollisionRulesTests tests/collision_rules_tests.cpp)
    target_compile_features(TwoVsTwoCollisionRulesTests PRIVATE cxx_std_17)
    target_link_libraries(TwoVsTwoCollisionRulesTests PRIVATE PPOTwoVsTwoExample)
    add_test(NAME TwoVsTwoCollisionRulesTests COMMAND TwoVsTwoCollisionRulesTests)

    add_executable(TwoVsTwoGeneralizedInitializerTests
        tests/generalized_initializer_tests.cpp)
    target_compile_features(TwoVsTwoGeneralizedInitializerTests
        PRIVATE cxx_std_17)
    target_link_libraries(TwoVsTwoGeneralizedInitializerTests
        PRIVATE PPOTwoVsTwoExample)
    add_test(NAME TwoVsTwoGeneralizedInitializerTests
        COMMAND TwoVsTwoGeneralizedInitializerTests)

    # Regression: Play advances every P/N agent through the actual desktop ECS
    # without requiring a pre-existing trained actor checkpoint.
    add_executable(TwoVsTwoPatrolRuntimeSmokeTest tests/patrol_runtime_smoke.cpp)
    target_compile_features(TwoVsTwoPatrolRuntimeSmokeTest PRIVATE cxx_std_17)
    target_link_libraries(TwoVsTwoPatrolRuntimeSmokeTest PRIVATE
        PPOTwoVsTwoExample GenericSerializer GenericEnv)
    # The smoke executable links the legacy LibTorch desktop libraries;
    # provide their runtime DLLs alongside the test binary on Windows.
    if (WIN32)
        add_custom_command(TARGET TwoVsTwoPatrolRuntimeSmokeTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_BINARY_DIR}/libtorch/lib"
                "$<TARGET_FILE_DIR:TwoVsTwoPatrolRuntimeSmokeTest>")
    endif()
    add_test(NAME TwoVsTwoPatrolRuntimeSmoke COMMAND TwoVsTwoPatrolRuntimeSmokeTest)
    add_test(NAME TwoVsTwoCombatRuntimeSmoke
        COMMAND TwoVsTwoPatrolRuntimeSmokeTest blue-defense)

    # Renderer contract: both enabled and disabled hostile zones must remain
    # identifiable through the public ECS snapshot and m_destroyed flag.
    add_executable(TwoVsTwoRestrictedZoneSnapshotTests
        tests/restricted_zone_snapshot_tests.cpp)
    target_compile_features(TwoVsTwoRestrictedZoneSnapshotTests PRIVATE cxx_std_17)
    target_link_libraries(TwoVsTwoRestrictedZoneSnapshotTests PRIVATE
        PPOTwoVsTwoExample GenericEnv)
    add_test(NAME TwoVsTwoRestrictedZoneSnapshotTests
        COMMAND TwoVsTwoRestrictedZoneSnapshotTests)
endif()
