# Dependency-free generalized MAPPO regressions for the headless core CI.
foreach(test IN ITEMS RewardEvaluation RewardShaping RuntimeState Advantage)
    if(test STREQUAL "RewardEvaluation")
        set(source generalized_reward_evaluation_tests.cpp)
    elseif(test STREQUAL "RewardShaping")
        set(source generalized_reward_shaping_tests.cpp)
    elseif(test STREQUAL "RuntimeState")
        set(source generalized_runtime_state_tests.cpp)
    else()
        set(source generalized_advantage_tests.cpp)
    endif()
    add_executable(BlueDefenseGeneralized${test}Tests ${source})
    target_compile_features(BlueDefenseGeneralized${test}Tests PRIVATE cxx_std_17)
    target_include_directories(BlueDefenseGeneralized${test}Tests PRIVATE
        ../Agents/BluePatrol/Core ../Agents/BluePatrol/Runtime
        ../Agents/BluePatrol/Training ../Evaluation)
    add_test(NAME BlueDefenseGeneralized${test}Tests
        COMMAND BlueDefenseGeneralized${test}Tests)
endforeach()
