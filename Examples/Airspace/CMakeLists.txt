add_subdirectory(Initializer)
add_subdirectory(Agents/Interception)
add_subdirectory(Agents/mission)

add_library(PPOAirspaceExample STATIC
    data/airspaceexample.hpp
    src/airspaceexample.cpp
    data/airspaceworldrules.hpp
    src/airspaceworldrules.cpp)

target_compile_features(PPOAirspaceExample PUBLIC cxx_std_17)
target_include_directories(PPOAirspaceExample PUBLIC
    "${CMAKE_SOURCE_DIR}/Examples/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOAirspaceExample PUBLIC
    GenericEntities
    GenericEnv
    EntityInitializer
    MetaReflection)

# Validate that historical outcomes and published components remain unchanged.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(AirspaceWorldRulesTests
        tests/airspace_world_rules_tests.cpp)
    target_compile_features(AirspaceWorldRulesTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceWorldRulesTests PRIVATE PPOAirspaceExample)
    add_test(NAME AirspaceWorldRulesTests COMMAND AirspaceWorldRulesTests)
endif()
