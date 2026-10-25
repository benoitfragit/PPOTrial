add_library(TwoVsTwoDefenderBackend INTERFACE)
target_compile_features(TwoVsTwoDefenderBackend INTERFACE cxx_std_17)
target_include_directories(TwoVsTwoDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(TwoVsTwoDefenderBackend INTERFACE RLtools::Core GenericRLCore)

# Separate CPU learner. PPO.exe remains the historical desktop application.
add_executable(PPOBlueTrain src/patrol_train.cpp data/patrol_core.hpp data/patrol_rltools.hpp data/attention_actor.hpp)
target_link_libraries(PPOBlueTrain PRIVATE TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
target_compile_definitions(PPOBlueTrain PRIVATE
    PPO_BLUE_SETTINGS_DIR=\"${CMAKE_SOURCE_DIR}/resources/settings\")
target_compile_definitions(PPOBlueTrain PRIVATE
    PPO_BLUE_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/resources/model/BluePatrol/prod/blue_patrol_v2.txt\")
target_compile_features(PPOBlueTrain PRIVATE cxx_std_17)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoPatrolTests tests/patrol_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolTests COMMAND TwoVsTwoPatrolTests)
    add_executable(TwoVsTwoAttentionTests tests/attention_tests.cpp)
    target_link_libraries(TwoVsTwoAttentionTests PRIVATE TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoAttentionTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoAttentionTests COMMAND TwoVsTwoAttentionTests)
    add_test(NAME TwoVsTwoPPOTrainingSmoke
        COMMAND $<TARGET_FILE:PPOBlueTrain> 7
                "${CMAKE_CURRENT_BINARY_DIR}/blue_ppo_smoke.txt" 1)
endif()

# The desktop uses this inference-only actor, independently of the trainer.
add_library(TwoVsTwoBlueRuntime STATIC
    data/blue_agent.hpp data/patrol_policy.hpp data/attention_actor.hpp src/blue_agent.cpp)
target_compile_features(TwoVsTwoBlueRuntime PUBLIC cxx_std_17)
target_include_directories(TwoVsTwoBlueRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data")
target_link_libraries(TwoVsTwoBlueRuntime PUBLIC
    TwoVsTwoDefenderBackend GenericEnv SharedWorld LegacyAgentAPI)
