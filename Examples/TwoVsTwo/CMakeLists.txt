add_subdirectory(Agents/BluePatrol Defenders)

add_library(PPOTwoVsTwoSpawn STATIC
    Scenario/spawn.hpp
    Scenario/spawn.cpp
    Scenario/spawn_config.hpp
    Scenario/spawn_config.cpp)
target_compile_features(PPOTwoVsTwoSpawn PUBLIC cxx_std_17)
target_include_directories(PPOTwoVsTwoSpawn PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
target_link_libraries(PPOTwoVsTwoSpawn PUBLIC nlohmann_json::nlohmann_json)

add_library(PPOTwoVsTwoExample STATIC
    Scenario/twovstwoexample.hpp
    Scenario/twovstwoexample.cpp
    Scenario/initializer.hpp
    Scenario/initializer.cpp
    Scenario/twovstworules.hpp
    Scenario/twovstworules.cpp)
target_compile_features(PPOTwoVsTwoExample PUBLIC cxx_std_17)
target_include_directories(PPOTwoVsTwoExample PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
target_link_libraries(PPOTwoVsTwoExample PUBLIC
    PPOAirspaceExample PPOTwoVsTwoSpawn
    TwoVsTwoDefenderBackend TwoVsTwoPatrolRuntime)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoSpawnTests tests/spawn_tests.cpp)
    target_compile_features(TwoVsTwoSpawnTests PRIVATE cxx_std_17)
    target_include_directories(TwoVsTwoSpawnTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
    target_link_libraries(TwoVsTwoSpawnTests PRIVATE
        PPOTwoVsTwoSpawn TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoSpawnTests COMMAND TwoVsTwoSpawnTests)

    add_executable(TwoVsTwoCollisionRulesTests tests/collision_rules_tests.cpp)
    target_compile_features(TwoVsTwoCollisionRulesTests PRIVATE cxx_std_17)
    target_link_libraries(TwoVsTwoCollisionRulesTests PRIVATE PPOTwoVsTwoExample)
    add_test(NAME TwoVsTwoCollisionRulesTests COMMAND TwoVsTwoCollisionRulesTests)

    # Regression: Play must advance both guards in the GUI ECS even when
    # neither one has a trained PPO checkpoint available.
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
        COMMAND TwoVsTwoPatrolRuntimeSmokeTest two-vs-two)
endif()
