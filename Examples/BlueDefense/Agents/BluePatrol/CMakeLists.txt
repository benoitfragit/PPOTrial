add_library(BlueDefenseDefenderBackend INTERFACE)
target_compile_features(BlueDefenseDefenderBackend INTERFACE cxx_std_17)
target_include_directories(BlueDefenseDefenderBackend INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/Core"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Evaluation"
    "${CMAKE_CURRENT_SOURCE_DIR}/Runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/Training")
target_link_libraries(BlueDefenseDefenderBackend INTERFACE
    RLtools::Core PPOKit::Core PPOKit::Artifacts PPOKit::Schema)
# The centralized critic is generated from graph.json (PPOKit).
ppokit_generate_networks(TARGET BlueDefenseDefenderBackend
    TASK task.json GRAPH graph.json HEADER blue_patrol_networks.hpp)
# The attention actor is the block graph of graph_blocks.json, embedded with its task so
# that loading a checkpoint never depends on the working directory.
ppokit_embed_files(TARGET BlueDefenseDefenderBackend HEADER blue_actor_graph_files.hpp
    NAMESPACE ppo::examples::bluedefense::defenders::embedded
    FILES BlueTaskJson task.json BlueGraphBlocksJson graph_blocks.json)

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
        BlueDefenseDefenderBackend nlohmann_json::nlohmann_json PPOLayout)
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
    # The actor (graph_blocks.json) must compute what the former hand-written actor did.
    add_executable(BlueDefenseGraphActorParityTests
        ../../tests/graph_actor_parity_tests.cpp)
    target_link_libraries(BlueDefenseGraphActorParityTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGraphActorParityTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGraphActorParityTests
        COMMAND BlueDefenseGraphActorParityTests)
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
    add_executable(BlueDefenseGeneralizedPatrolReferenceTests
        ../../tests/generalized_patrol_reference_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedPatrolReferenceTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseGeneralizedPatrolReferenceTests
        COMMAND BlueDefenseGeneralizedPatrolReferenceTests)
    add_executable(BlueDefenseGeneralizedPatrolWaypointTests
        ../../tests/generalized_patrol_waypoint_sequencer_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedPatrolWaypointTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedPatrolWaypointTests PRIVATE
        cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedPatrolWaypointTests
        COMMAND BlueDefenseGeneralizedPatrolWaypointTests)
    add_executable(BlueDefenseGeneralizedEpisodeOutcomeTests
        ../../tests/generalized_episode_outcome_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedEpisodeOutcomeTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedEpisodeOutcomeTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedEpisodeOutcomeTests
        COMMAND BlueDefenseGeneralizedEpisodeOutcomeTests)
    add_executable(BlueDefenseGeneralizedAdvantageTests
        ../../tests/generalized_advantage_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedAdvantageTests PRIVATE
        BlueDefenseDefenderBackend)
    add_test(NAME BlueDefenseGeneralizedAdvantageTests
        COMMAND BlueDefenseGeneralizedAdvantageTests)
    add_executable(BlueDefenseGeneralizedLearnerTests
        ../../tests/generalized_learner_tests.cpp)
    target_link_libraries(BlueDefenseGeneralizedLearnerTests PRIVATE
        BlueDefenseDefenderBackend)
    target_compile_features(BlueDefenseGeneralizedLearnerTests PRIVATE cxx_std_17)
    add_test(NAME BlueDefenseGeneralizedLearnerTests
        COMMAND BlueDefenseGeneralizedLearnerTests)
    # Blue MAPPO on the shared graph backend updates bit for bit as the Blue learner.
    add_executable(BlueDefenseSharedBackendParityTests
        ../../tests/shared_backend_parity_tests.cpp)
    target_link_libraries(BlueDefenseSharedBackendParityTests PRIVATE
        BlueDefenseDefenderBackend PPOKit::Schema PPOKit::Training PPOKit::Artifacts nlohmann_json::nlohmann_json)
    target_compile_definitions(BlueDefenseSharedBackendParityTests PRIVATE
        BLUE_SHARED_BACKEND_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/shared_backend")
    add_test(NAME BlueDefenseSharedBackendParityTests
        COMMAND BlueDefenseSharedBackendParityTests)
    add_test(NAME BlueDefenseSharedBackendImportSmoke
        COMMAND ${CMAKE_COMMAND} -DIMPORT=$<TARGET_FILE:blue_shared_backend_import>
            -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/shared_backend
            -DSHIPPED=${CMAKE_CURRENT_SOURCE_DIR}/../../projects/BluePatrol
            -DCHECKPOINTS=${CMAKE_CURRENT_SOURCE_DIR}/../../projects/BluePatrol/checkpoints
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/shared_backend_import
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/shared_backend_import_smoke.cmake)
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
    ../../Evaluation/generalized_evaluation_case.hpp
    ../../Evaluation/generalized_evaluation_metrics.hpp
    ../../Evaluation/patrol_evaluation_settings.hpp
    Runtime/generalized_patrol_team_agent.hpp
    Runtime/generalized_patrol_production_policy.hpp
    Runtime/generalized_runtime_state.hpp
    Runtime/generalized_episode_outcome.hpp
    Runtime/generalized_reward_shaping.hpp
    Runtime/generalized_patrol_team_agent.cpp
    Training/attention_actor.hpp
    Training/generalized_attention_actor.hpp
    Training/generalized_critic_observation.hpp
    Training/generalized_critic_network.hpp
    Training/generalized_checkpoint.hpp
    Training/generalized_patrol_learner.hpp
    Training/generalized_advantage.hpp
    Training/generalized_training_checkpoint.hpp
    Training/patrol_checkpoint.hpp
    Training/patrol_training_checkpoint.hpp
    Training/patrol_training_settings.hpp)
target_compile_features(BlueDefensePatrolRuntime PUBLIC cxx_std_17)
target_include_directories(BlueDefensePatrolRuntime PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Scenario")
target_link_libraries(BlueDefensePatrolRuntime PUBLIC
    PPOKit::Training BlueDefenseDefenderBackend GenericEnv SharedWorld LegacyAgentAPI
    nlohmann_json::nlohmann_json)


