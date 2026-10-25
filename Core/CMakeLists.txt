add_library(GenericRLCore INTERFACE)
target_sources(GenericRLCore INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/core/agent_contract.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/core/buffers.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/core/multi_agent.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/core/team_episode.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/core/evaluation.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/core/learning_monitor.hpp")
target_include_directories(GenericRLCore INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(GenericRLCore INTERFACE cxx_std_17)

if (PPO_BUILD_CORE_TESTS)
    add_executable(GenericRLCoreTests tests/core_tests.cpp)
    target_link_libraries(GenericRLCoreTests PRIVATE GenericRLCore)
    add_test(NAME GenericRLCoreTests COMMAND GenericRLCoreTests)

    add_executable(GenericRLLearningMonitorTests tests/learning_monitor_tests.cpp)
    target_link_libraries(GenericRLLearningMonitorTests PRIVATE GenericRLCore)
    add_test(NAME GenericRLLearningMonitorTests COMMAND GenericRLLearningMonitorTests)
endif()
