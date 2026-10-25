add_subdirectory(Defenders)

add_library(PPOTwoVsTwoExample STATIC
    data/twovstwoexample.hpp
    src/twovstwoexample.cpp
    data/initializer.hpp
    src/initializer.cpp
    data/spawn.hpp
    data/twovstworules.hpp
    src/twovstworules.cpp)
target_compile_features(PPOTwoVsTwoExample PUBLIC cxx_std_17)
target_include_directories(PPOTwoVsTwoExample PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOTwoVsTwoExample PUBLIC
    PPOAirspaceExample TwoVsTwoDefenderBackend TwoVsTwoPatrolRuntime)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoSpawnTests tests/spawn_tests.cpp)
    target_compile_features(TwoVsTwoSpawnTests PRIVATE cxx_std_17)
    target_include_directories(TwoVsTwoSpawnTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/data")
    target_link_libraries(TwoVsTwoSpawnTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoSpawnTests COMMAND TwoVsTwoSpawnTests)

    # Regression: Play must advance both guards in the GUI ECS even when
    # neither one has a trained PPO checkpoint available.
    add_executable(TwoVsTwoPatrolRuntimeSmoke tests/patrol_runtime_smoke.cpp)
    target_compile_features(TwoVsTwoPatrolRuntimeSmoke PRIVATE cxx_std_17)
    target_link_libraries(TwoVsTwoPatrolRuntimeSmoke PRIVATE
        PPOTwoVsTwoExample GenericSerializer GenericEnv)
    add_test(NAME TwoVsTwoPatrolRuntimeSmoke COMMAND TwoVsTwoPatrolRuntimeSmoke)
endif()
