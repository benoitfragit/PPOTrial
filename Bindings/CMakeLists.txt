# Domain-to-policy templates stay at the integration edge; the PPO engine
# itself has no dependency on domain models, entities, or JSON.
add_library(GenericRLBindings INTERFACE)
target_sources(GenericRLBindings INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/data/torchproductionpolicy.hpp")
target_include_directories(GenericRLBindings INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(GenericRLBindings INTERFACE GenericPPO GenericSerializer)
target_compile_features(GenericRLBindings INTERFACE cxx_std_17)
