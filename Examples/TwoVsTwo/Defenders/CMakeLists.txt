add_library(TwoVsTwoDefenderBackend INTERFACE)
target_compile_features(TwoVsTwoDefenderBackend INTERFACE cxx_std_17)
target_include_directories(TwoVsTwoDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Evaluation/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Runtime/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Training/data")
target_link_libraries(TwoVsTwoDefenderBackend INTERFACE RLtools::Core GenericRLCore)

# PPO.exe is the default training application; optional console utilities.
option(PPO_BUILD_PATROL_TOOLS "Build auxiliary patrol CLI executables" OFF)
if (PPO_BUILD_PATROL_TOOLS)
    # Separate CPU learner. PPO.exe remains the historical desktop application.
    add_executable(PPOBlueTrain
        Training/src/patrol_train.cpp
        Training/data/patrol_learner.hpp
        Core/data/patrol_config.hpp
        Core/data/patrol_core.hpp
        Training/data/patrol_rltools.hpp
        Training/data/attention_actor.hpp)
    target_link_libraries(PPOBlueTrain PRIVATE TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
    target_compile_definitions(PPOBlueTrain PRIVATE
        PPO_BLUE_SETTINGS_DIR=\"${CMAKE_SOURCE_DIR}/resources/settings\")
    target_compile_definitions(PPOBlueTrain PRIVATE
        PPO_BLUE_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/resources/model/BluePatrol/latest/blue_patrol_mappo_v5.txt\")
    target_compile_features(PPOBlueTrain PRIVATE cxx_std_17)

endif()

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoPatrolDefenseTests Tests/patrol_defense_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolDefenseTests PRIVATE
        TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolDefenseTests COMMAND TwoVsTwoPatrolDefenseTests)
    add_executable(TwoVsTwoPatrolGeometryTests Tests/patrol_geometry_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolGeometryTests PRIVATE
        TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
    add_test(NAME TwoVsTwoPatrolGeometryTests COMMAND TwoVsTwoPatrolGeometryTests)
    add_executable(TwoVsTwoPatrolTests Tests/patrol_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolTests COMMAND TwoVsTwoPatrolTests)
    add_executable(TwoVsTwoAttentionTests Tests/attention_tests.cpp)
    target_link_libraries(TwoVsTwoAttentionTests PRIVATE TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoAttentionTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoAttentionTests COMMAND TwoVsTwoAttentionTests)
    add_executable(TwoVsTwoEmbeddedLearnerSmoke Tests/patrol_training_smoke.cpp)
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
    Core/data/patrol_agent.hpp
    Core/data/patrol_config.hpp
    Core/data/patrol_defense.hpp
    Core/data/patrol_geometry.hpp
    Evaluation/data/patrol_evaluation_campaign.hpp
    Evaluation/data/patrol_evaluation_settings.hpp
    Runtime/data/patrol_policy.hpp
    Runtime/data/patrol_reference_agent.hpp
    Runtime/data/patrol_reference_controller.hpp
    Runtime/data/patrol_team_agent.hpp
    Runtime/src/patrol_reference_agent.cpp
    Runtime/src/patrol_team_agent.cpp
    Training/data/attention_actor.hpp
    Training/data/patrol_checkpoint.hpp
    Training/data/patrol_learner.hpp
    Training/data/patrol_training_checkpoint.hpp
    Training/data/patrol_training_settings.hpp)
target_compile_features(TwoVsTwoPatrolRuntime PUBLIC cxx_std_17)
target_include_directories(TwoVsTwoPatrolRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data")
target_link_libraries(TwoVsTwoPatrolRuntime PUBLIC
    TwoVsTwoDefenderBackend GenericEnv SharedWorld LegacyAgentAPI
    nlohmann_json::nlohmann_json)

# Optional deterministic headless evaluator.
if (PPO_BUILD_PATROL_TOOLS)
add_executable(PPOPatrolEvaluate
    Evaluation/src/patrol_evaluate.cpp
    Evaluation/data/patrol_evaluator.hpp)
target_link_libraries(PPOPatrolEvaluate PRIVATE TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
target_compile_features(PPOPatrolEvaluate PRIVATE cxx_std_17)
target_compile_definitions(PPOPatrolEvaluate PRIVATE
    PPO_BLUE_SETTINGS_DIR=\"${CMAKE_SOURCE_DIR}/resources/settings\"
    PPO_BLUE_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/resources/model/BluePatrol/prod/blue_patrol_mappo_v5.txt\")

endif()

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoPatrolEvaluationTests Tests/patrol_evaluation_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolEvaluationTests PRIVATE TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolEvaluationTests COMMAND TwoVsTwoPatrolEvaluationTests)
endif()
