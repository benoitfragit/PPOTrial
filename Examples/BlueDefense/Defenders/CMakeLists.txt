add_library(TwoVsTwoDefenderBackend INTERFACE)
target_compile_features(TwoVsTwoDefenderBackend INTERFACE cxx_std_17)
target_include_directories(TwoVsTwoDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Evaluation/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Runtime/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/Training/data")
target_link_libraries(TwoVsTwoDefenderBackend INTERFACE RLtools::Core GenericRLCore)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TwoVsTwoTeamDomainTests Tests/team_domain_tests.cpp)
    target_link_libraries(TwoVsTwoTeamDomainTests PRIVATE
        TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoTeamDomainTests COMMAND TwoVsTwoTeamDomainTests)
    add_executable(TwoVsTwoTeamDynamicsTests Tests/team_dynamics_tests.cpp)
    target_link_libraries(TwoVsTwoTeamDynamicsTests PRIVATE
        TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoTeamDynamicsTests COMMAND TwoVsTwoTeamDynamicsTests)
    add_executable(TwoVsTwoGeneralizedRewardEvaluationTests
        Tests/generalized_reward_evaluation_tests.cpp)
    target_link_libraries(TwoVsTwoGeneralizedRewardEvaluationTests PRIVATE
        TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoGeneralizedRewardEvaluationTests PRIVATE
        cxx_std_17)
    add_test(NAME TwoVsTwoGeneralizedRewardEvaluationTests
        COMMAND TwoVsTwoGeneralizedRewardEvaluationTests)
    add_executable(TwoVsTwoPatrolDefenseTests Tests/patrol_defense_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolDefenseTests PRIVATE
        TwoVsTwoDefenderBackend)
    add_test(NAME TwoVsTwoPatrolDefenseTests COMMAND TwoVsTwoPatrolDefenseTests)
    add_executable(TwoVsTwoPatrolGeometryTests Tests/patrol_geometry_tests.cpp)
    target_link_libraries(TwoVsTwoPatrolGeometryTests PRIVATE
        TwoVsTwoDefenderBackend nlohmann_json::nlohmann_json)
    add_test(NAME TwoVsTwoPatrolGeometryTests COMMAND TwoVsTwoPatrolGeometryTests)
    add_executable(TwoVsTwoAttentionTests Tests/attention_tests.cpp)
    target_link_libraries(TwoVsTwoAttentionTests PRIVATE TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoAttentionTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoAttentionTests COMMAND TwoVsTwoAttentionTests)
    add_executable(TwoVsTwoGeneralizedActorTests
        Tests/generalized_actor_tests.cpp)
    target_link_libraries(TwoVsTwoGeneralizedActorTests PRIVATE
        TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoGeneralizedActorTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoGeneralizedActorTests
        COMMAND TwoVsTwoGeneralizedActorTests)
    add_executable(TwoVsTwoGeneralizedRuntimeStateTests
        Tests/generalized_runtime_state_tests.cpp)
    add_executable(TwoVsTwoGeneralizedRewardShapingTests
        Tests/generalized_reward_shaping_tests.cpp)
    target_link_libraries(TwoVsTwoGeneralizedRewardShapingTests PRIVATE
        TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoGeneralizedRewardShapingTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoGeneralizedRewardShapingTests
        COMMAND TwoVsTwoGeneralizedRewardShapingTests)
    target_link_libraries(TwoVsTwoGeneralizedRuntimeStateTests PRIVATE
        TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoGeneralizedRuntimeStateTests PRIVATE
        cxx_std_17)
    add_test(NAME TwoVsTwoGeneralizedRuntimeStateTests
        COMMAND TwoVsTwoGeneralizedRuntimeStateTests)
    add_executable(TwoVsTwoGeneralizedLearnerTests
        Tests/generalized_learner_tests.cpp)
    target_link_libraries(TwoVsTwoGeneralizedLearnerTests PRIVATE
        TwoVsTwoDefenderBackend)
    target_compile_features(TwoVsTwoGeneralizedLearnerTests PRIVATE cxx_std_17)
    add_test(NAME TwoVsTwoGeneralizedLearnerTests
        COMMAND TwoVsTwoGeneralizedLearnerTests)
endif()

# One generalized MAPPO IAgent owns all P blue ECS mobiles per step.
add_library(TwoVsTwoPatrolRuntime STATIC
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
target_compile_features(TwoVsTwoPatrolRuntime PUBLIC cxx_std_17)
target_include_directories(TwoVsTwoPatrolRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data")
target_link_libraries(TwoVsTwoPatrolRuntime PUBLIC
    TwoVsTwoDefenderBackend GenericEnv SharedWorld LegacyAgentAPI
    nlohmann_json::nlohmann_json)

