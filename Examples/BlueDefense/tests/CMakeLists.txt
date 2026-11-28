# Dependency-free generalized MAPPO regressions for the headless core CI.
foreach(test IN ITEMS RewardEvaluation RewardShaping RuntimeState Advantage PatrolReference PatrolWaypoint EpisodeOutcome UnifiedGuidance)
    if(test STREQUAL "RewardEvaluation")
        set(source generalized_reward_evaluation_tests.cpp)
    elseif(test STREQUAL "RewardShaping")
        set(source generalized_reward_shaping_tests.cpp)
    elseif(test STREQUAL "RuntimeState")
        set(source generalized_runtime_state_tests.cpp)
    elseif(test STREQUAL "PatrolReference")
        set(source generalized_patrol_reference_tests.cpp)
    elseif(test STREQUAL "PatrolWaypoint")
        set(source generalized_patrol_waypoint_sequencer_tests.cpp)
    elseif(test STREQUAL "UnifiedGuidance")
        set(source generalized_unified_guidance_tests.cpp)
    elseif(test STREQUAL "EpisodeOutcome")
        set(source generalized_episode_outcome_tests.cpp)
    else()
        set(source generalized_advantage_tests.cpp)
    endif()
    # A build with the ECS world (PPO_BUILD_WORLD) already has them from Agents/BluePatrol.
    if (TARGET BlueDefenseGeneralized${test}Tests)
        continue()
    endif()
    add_executable(BlueDefenseGeneralized${test}Tests ${source})
    target_compile_features(BlueDefenseGeneralized${test}Tests PRIVATE cxx_std_17)
    target_include_directories(BlueDefenseGeneralized${test}Tests PRIVATE
        ../Agents/BluePatrol/Core ../Agents/BluePatrol/Runtime
        ../Agents/BluePatrol/Training ../Evaluation)
    target_link_libraries(BlueDefenseGeneralized${test}Tests PRIVATE PPOKit::Core)
    add_test(NAME BlueDefenseGeneralized${test}Tests
        COMMAND BlueDefenseGeneralized${test}Tests)
endforeach()

# BluePatrol/task.json (PPOKit) must describe the same layouts and channels as the code.
add_executable(BlueDefenseTaskLayoutTests task_layout_tests.cpp)
target_compile_features(BlueDefenseTaskLayoutTests PRIVATE cxx_std_17)
target_include_directories(BlueDefenseTaskLayoutTests PRIVATE
    ../Agents/BluePatrol/Core ../Agents/BluePatrol/Training)
ppokit_generate_task_layout(TARGET BlueDefenseTaskLayoutTests
    TASK ../Agents/BluePatrol/task.json HEADER blue_patrol_task_layout.hpp)
add_test(NAME BlueDefenseTaskLayoutTests COMMAND BlueDefenseTaskLayoutTests)

# BluePatrol/graph.json (PPOKit) generates the hand-written critic type. It needs rl_tools,
# which the headless build fetches with FlightControl3D.
if (TARGET RLtools::Core)
    add_executable(BlueDefenseNetworkTypeTests network_type_tests.cpp)
    target_compile_features(BlueDefenseNetworkTypeTests PRIVATE cxx_std_17)
    target_include_directories(BlueDefenseNetworkTypeTests PRIVATE
        ../Agents/BluePatrol/Core ../Agents/BluePatrol/Runtime
        ../Agents/BluePatrol/Training ../Evaluation)
    target_link_libraries(BlueDefenseNetworkTypeTests PRIVATE RLtools::Core PPOKit::Core)
    ppokit_generate_networks(TARGET BlueDefenseNetworkTypeTests
        TASK ../Agents/BluePatrol/task.json GRAPH ../Agents/BluePatrol/graph.json
        HEADER blue_patrol_networks.hpp)
    add_test(NAME BlueDefenseNetworkTypeTests COMMAND BlueDefenseNetworkTypeTests)
endif()
