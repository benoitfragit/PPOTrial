add_library(BlueDefenseDefenderBackend INTERFACE)
target_compile_features(BlueDefenseDefenderBackend INTERFACE cxx_std_17)
target_include_directories(BlueDefenseDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Evaluation/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Runtime/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Training/data")
target_link_libraries(BlueDefenseDefenderBackend INTERFACE RLtools::Core GenericRLCore)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(BlueDefenseTeamDomainTests Tests/team_domain_tests.cpp)
    target_link_libraries(BlueDefenseTeamDomainTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseTeamDomainTests COMMAND BlueDefenseTeamDomainTests)
    add_executable(BlueDefenseTeamDynamicsTests Tests/team_dynamics_tests.cpp)
    target_link_libraries(BlueDefenseTeamDynamicsTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseTeamDynamicsTests COMMAND BlueDefenseTeamDynamicsTests)
    add_executable(BlueDefenseGeneralizedRewardEvaluationTests
        Tests/generalized_reward_evaluation_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedRewardEvaluationTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedRewardEvaluationTests PRIVATE
        cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedRewardEvaluationTests
        COMMAND BlueDefenseGeneralizedRewardEvaluationTests)
    add_executable(BlueDefensePatrolDefenseTests Tests/patrol_defense_tests.cpp)
    target_link_libraries(BlueDefensePatrolDefenseTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefensePatrolDefenseTests COMMAND BlueDefensePatrolDefenseTests)
    add_executable(BlueDefensePatrolGeometryTests Tests/patrol_geometry_tests.cpp)
    target_link_libraries(BlueDefensePatrolGeometryTests PRIVATE
        BlueDefenseDefenderBackend nlohmann_json::nlohmann_json)
    add_test(NAME BlueDefensePatrolGeometryTests COMMAND BlueDefensePatrolGeometryTests)
    add_executable(BlueDefenseAttentionTests Tests/attention_tests.cpp)
    target_link_libraries(BlueDefenseAttentionTests PRIVATE BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseAttentionTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseAttentionTests COMMAND BlueDefenseAttentionTests)
    add_executable(BlueDefenseGeneralizedActorTests
        Tests/generalized_actor_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedActorTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedActorTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedActorTests
        COMMAND BlueDefenseGeneralizedActorTests)
    add_executable(BlueDefenseGeneralizedRuntimeStateTests
        Tests/generalized_runtime_state_tests.cpp)
    add_executable(BlueDefenseGeneralizedRewardShapingTests
        Tests/generalized_reward_shaping_tests.cpp)
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
        Tests/generalized_learner_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedLearnerTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedLearnerTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedLearnerTests
        COMMAND BlueDefenseGeneralizedLearnerTests)
endif()

# One generalized MAPPO IAgent owns all P blue ECS mobiles per step.
add_library(BlueDefensePatrolRuntime STATIC
    Core/data/blue_team_roles.hpp
    Core/data/generalized_observation.hpp
    Core/data/generalized_cooperative_reward.hpp
    Core/data/generalized_team_dynamics.hpp
    Core/data/patrol_config.hpp
    Core/data/patrol_defense.hpp
    Core/data/patrol_geometry.hpp
    Core/data/team_capacity.hpp
    Evaluation/data/generalized_evaluation_campaign.hpp
    Evaluation/data/generalized_evaluation_metrics.hpp
    Evaluation/data/patrol_evaluation_settings.hpp
    Runtime/data/generalized_patrol_team_agent.hpp
    Runtime/data/generalized_runtime_state.hpp
    Runtime/data/generalized_reward_shaping.hpp
    Runtime/src/generalized_patrol_team_agent.cpp
    Training/data/attention_actor.hpp
    Training/data/generalized_attention_actor.hpp
    Training/data/generalized_critic_observation.hpp
    Training/data/generalized_critic_network.hpp
    Training/data/generalized_checkpoint.hpp
    Training/data/generalized_patrol_learner.hpp
    Training/data/generalized_training_checkpoint.hpp
    Training/data/patrol_checkpoint.hpp
    Training/data/patrol_training_checkpoint.hpp
    Training/data/patrol_training_settings.hpp)
target_compile_features(BlueDefensePatrolRuntime PUBLIC cxx_std_17)
target_include_directories(BlueDefensePatrolRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data")
target_link_libraries(BlueDefensePatrolRuntime PUBLIC
    BlueDefenseDefenderBackend GenericEnv SharedWorld LegacyAgentAPI
    nlohmann_json::nlohmann_json)

