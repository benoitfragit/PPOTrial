# The agent contract of the ECS world (IAgent); no optimizer or ECS dependencies.
add_library(LegacyAgentAPI INTERFACE)
target_sources(LegacyAgentAPI INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/runtime/iagent.hpp")
target_include_directories(LegacyAgentAPI INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(LegacyAgentAPI INTERFACE cxx_std_17)
