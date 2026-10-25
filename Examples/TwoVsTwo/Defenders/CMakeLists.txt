add_library(TwoVsTwoDefenderBackend INTERFACE)
target_compile_features(TwoVsTwoDefenderBackend INTERFACE cxx_std_17)
target_include_directories(TwoVsTwoDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(TwoVsTwoDefenderBackend INTERFACE RLtools::Core GenericRLCore)

# Separate CPU learner. PPO.exe remains the historical desktop application.
add_executable(PPOBlueTrain src/patrol_train.cpp data/patrol_core.hpp data/patrol_rltools.hpp)
target_link_libraries(PPOBlueTrain PRIVATE TwoVsTwoDefenderBackend)
target_compile_features(PPOBlueTrain PRIVATE cxx_std_17)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoPatrolTests tests/patrol_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolTests COMMAND TwoVsTwoPatrolTests)
endif()
