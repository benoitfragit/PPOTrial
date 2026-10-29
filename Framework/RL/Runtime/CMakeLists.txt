# Legacy runtime agent/UI contract; no Torch, optimizer or ECS dependencies.
add_library(LegacyAgentAPI INTERFACE)
target_sources(LegacyAgentAPI INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/runtime/iagent.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ppo/runtime/ppoparameters.hpp")
target_include_directories(LegacyAgentAPI INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(LegacyAgentAPI INTERFACE GenericRLCore)
target_compile_features(LegacyAgentAPI INTERFACE cxx_std_17)
