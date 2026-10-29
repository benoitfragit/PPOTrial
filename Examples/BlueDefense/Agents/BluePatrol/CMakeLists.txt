add_library(BlueDefenseDefenderBackend INTERFACE)
target_compile_features(BlueDefenseDefenderBackend INTERFACE cxx_std_17)
target_include_directories(BlueDefenseDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/Core"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Evaluation"
    "${CMAKE_CURRENT_SOURCE_DIR}/Runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/Training")
target_link_libraries(BlueDefenseDefenderBackend INTERFACE RLtools::Core GenericRLCore)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(BlueDefenseTeamDomainTests ../../tests/team_domain_tests.cpp)
    target_link_libraries(BlueDefenseTeamDomainTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseTeamDomainTests COMMAND BlueDefenseTeamDomainTests)
    add_executable(BlueDefenseTeamDynamicsTests ../../tests/team_dynamics_tests.cpp)
    target_link_libraries(BlueDefenseTeamDynamicsTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseTeamDynamicsTests COMMAND BlueDefenseTeamDynamicsTests)
    add_executable(BlueDefenseGeneralizedRewardEvaluationTests
        ../../tests/generalized_reward_evaluation_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedRewardEvaluationTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedRewardEvaluationTests PRIVATE
        cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedRewardEvaluationTests
        COMMAND BlueDefenseGeneralizedRewardEvaluationTests)
    add_executable(BlueDefensePatrolDefenseTests ../../tests/patrol_defense_tests.cpp)
    target_link_libraries(BlueDefensePatrolDefenseTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefensePatrolDefenseTests COMMAND BlueDefensePatrolDefenseTests)
    add_executable(BlueDefensePatrolGeometryTests ../../tests/patrol_geometry_tests.cpp)
    target_link_libraries(BlueDefensePatrolGeometryTests PRIVATE
        BlueDefenseDefenderBackend nlohmann_json::nlohmann_json)
    add_test(NAME BlueDefensePatrolGeometryTests COMMAND BlueDefensePatrolGeometryTests)
    add_executable(BlueDefenseAttentionTests ../../tests/attention_tests.cpp)
    target_link_libraries(BlueDefenseAttentionTests PRIVATE BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseAttentionTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseAttentionTests COMMAND BlueDefenseAttentionTests)
    add_executable(BlueDefenseGeneralizedActorTests
        ../../tests/generalized_actor_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedActorTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedActorTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedActorTests
        COMMAND BlueDefenseGeneralizedActorTests)
    add_executable(BlueDefenseGeneralizedRuntimeStateTests
        ../../tests/generalized_runtime_state_tests.cpp)
    add_executable(BlueDefenseGeneralizedRewardShapingTests
        ../../tests/generalized_reward_shaping_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedRewardShapingTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedRewardShapingTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedRewardShapingTests
        COMMAND BlueDefenseGeneralizedRewardShapingTests)
    target_link_libraries(BlueDefenseGeneralizedRuntimeStateTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedRuntimeStateTests PRIVATE
        cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedRuntimeStateTests
        COMMAND BlueDefenseGeneralizedRuntimeStateTests)
    add_executable(BlueDefenseGeneralizedLearnerTests
        ../../tests/generalized_learner_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedLearnerTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedLearnerTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedLearnerTests
        COMMAND BlueDefenseGeneralizedLearnerTests)
endif()

# One generalized MAPPO IAgent owns all P blue ECS mobiles per step.
add_library(BlueDefensePatrolRuntime STATIC
    Core/blue_team_roles.hpp
    Core/generalized_observation.hpp
    Core/generalized_cooperative_reward.hpp
    Core/generalized_team_dynamics.hpp
    Core/patrol_config.hpp
    Core/patrol_defense.hpp
    Core/patrol_geometry.hpp
    Core/team_capacity.hpp
    ../../Evaluation/generalized_evaluation_campaign.hpp
    ../../Evaluation/generalized_evaluation_metrics.hpp
    ../../Evaluation/patrol_evaluation_settings.hpp
    Runtime/generalized_patrol_team_agent.hpp
    Runtime/generalized_runtime_state.hpp
    Runtime/generalized_reward_shaping.hpp
    Runtime/generalized_patrol_team_agent.cpp
    Training/attention_actor.hpp
    Training/generalized_attention_actor.hpp
    Training/generalized_critic_observation.hpp
    Training/generalized_critic_network.hpp
    Training/generalized_checkpoint.hpp
    Training/generalized_patrol_learner.hpp
    Training/generalized_training_checkpoint.hpp
    Training/patrol_checkpoint.hpp
    Training/patrol_training_checkpoint.hpp
    Training/patrol_training_settings.hpp)
target_compile_features(BlueDefensePatrolRuntime PUBLIC cxx_std_17)
target_include_directories(BlueDefensePatrolRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Scenario")
target_link_libraries(BlueDefensePatrolRuntime PUBLIC
    BlueDefenseDefenderBackend GenericEnv SharedWorld LegacyAgentAPI
    nlohmann_json::nlohmann_json)

