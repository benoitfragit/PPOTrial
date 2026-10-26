add_library(TwoVsTwoDefenderBackend INTERFACE)
target_compile_features(TwoVsTwoDefenderBackend INTERFACE cxx_std_17)
target_include_directories(TwoVsTwoDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(TwoVsTwoDefenderBackend INTERFACE RLtools::Core GenericRLCore)

# PPO.exe is the default training application; optional console utilities.
option(PPO_BUILD_PATROL_TOOLS "Build auxiliary patrol CLI executables" OFF)
if (PPO_BUILD_PATROL_TOOLS)
    # Separate CPU learner. PPO.exe remains the historical desktop application.
    add_executable(PPOBlueTrain src/patrol_train.cpp data/patrol_learner.hpp data/patrol_config.hpp data/patrol_core.hpp data/patrol_rltools.hpp data/attention_actor.hpp)
    target_link_libraries(PPOBlueTrain PRIVATE TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
    target_compile_definitions(PPOBlueTrain PRIVATE
        PPO_BLUE_SETTINGS_DIR=\"${CMAKE_SOURCE_DIR}/resources/settings\")
    target_compile_definitions(PPOBlueTrain PRIVATE
        PPO_BLUE_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/resources/model/BluePatrol/latest/blue_patrol_mappo_v4.txt\")
    target_compile_features(PPOBlueTrain PRIVATE cxx_std_17)

endif()

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoPatrolTests tests/patrol_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolTests COMMAND TwoVsTwoPatrolTests)
    add_executable(TwoVsTwoAttentionTests tests/attention_tests.cpp)
    target_link_libraries(TwoVsTwoAttentionTests PRIVATE TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoAttentionTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoAttentionTests COMMAND TwoVsTwoAttentionTests)
    add_executable(TwoVsTwoEmbeddedLearnerSmoke tests/patrol_training_smoke.cpp)
    target_link_libraries(TwoVsTwoEmbeddedLearnerSmoke PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoEmbeddedLearnerSmoke COMMAND TwoVsTwoEmbeddedLearnerSmoke)
    if (PPO_BUILD_PATROL_TOOLS)
        add_test(NAME TwoVsTwoPPOTrainingCLI
            COMMAND $<TARGET_FILE:PPOBlueTrain> 7
                    "${CMAKE_CURRENT_BINARY_DIR}/blue_ppo_smoke.txt" 1)
    endif()
endif()

# One cooperative MAPPO IAgent controls both blue defenders in one ECS step.
add_library(TwoVsTwoPatrolRuntime STATIC
    data/patrol_agent.hpp data/patrol_policy.hpp data/attention_actor.hpp
    data/patrol_reference_agent.hpp data/patrol_reference_controller.hpp
    data/patrol_team_agent.hpp data/patrol_training_settings.hpp
    data/patrol_evaluation_settings.hpp data/patrol_evaluation_campaign.hpp
    data/patrol_learner.hpp data/patrol_config.hpp
    data/patrol_checkpoint.hpp data/patrol_training_checkpoint.hpp
    src/patrol_reference_agent.cpp src/patrol_team_agent.cpp)
target_compile_features(TwoVsTwoPatrolRuntime PUBLIC cxx_std_17)
target_include_directories(TwoVsTwoPatrolRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data")
target_link_libraries(TwoVsTwoPatrolRuntime PUBLIC
    TwoVsTwoDefenderBackend GenericEnv SharedWorld LegacyAgentAPI
    nlohmann_json::nlohmann_json)

# Optional deterministic headless evaluator.
if (PPO_BUILD_PATROL_TOOLS)
add_executable(PPOPatrolEvaluate src/patrol_evaluate.cpp data/patrol_evaluator.hpp)
target_link_libraries(PPOPatrolEvaluate PRIVATE TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
target_compile_features(PPOPatrolEvaluate PRIVATE cxx_std_17)
target_compile_definitions(PPOPatrolEvaluate PRIVATE
    PPO_BLUE_SETTINGS_DIR=\"${CMAKE_SOURCE_DIR}/resources/settings\"
    PPO_BLUE_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/resources/model/BluePatrol/prod/blue_patrol_mappo_v4.txt\")

endif()

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoPatrolEvaluationTests tests/patrol_evaluation_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolEvaluationTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolEvaluationTests COMMAND TwoVsTwoPatrolEvaluationTests)
endif()
