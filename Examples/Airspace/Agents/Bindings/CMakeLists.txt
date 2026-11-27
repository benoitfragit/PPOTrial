# Domain-to-policy templates stay at the integration edge: training agents are driven by
# the environment DLL (ppokit-run learns), production agents run a policy given by their domain.
add_library(GenericRLBindings INTERFACE)
target_sources(GenericRLBindings INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/data/graph_network.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/data/prodagent.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/data/trainingagent.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/data/trainingagent_def.hpp")
target_include_directories(GenericRLBindings INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(GenericRLBindings INTERFACE
    GenericSerializer LegacyAgentAPI GenericRLCore PPOKit::Training PPOKit::Schema PPOKit::Artifacts)
target_compile_features(GenericRLBindings INTERFACE cxx_std_17)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TrainingAgentTests "${CMAKE_CURRENT_SOURCE_DIR}/tests/training_agent_tests.cpp")
    target_link_libraries(TrainingAgentTests PRIVATE GenericRLBindings)
    add_test(NAME TrainingAgentTests COMMAND TrainingAgentTests)
endif()
