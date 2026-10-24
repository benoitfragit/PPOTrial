add_subdirectory(Agents/Interception)
add_subdirectory(Agents/mission)

add_library(PPOExistingExample STATIC
    data/existingexample.hpp
    src/existingexample.cpp
    data/existingworldrules.hpp
    src/existingworldrules.cpp)

target_compile_features(PPOExistingExample PUBLIC cxx_std_17)
target_include_directories(PPOExistingExample PUBLIC
    "${CMAKE_SOURCE_DIR}/Examples/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOExistingExample PUBLIC
    GenericEntities
    GenericEnv
    EntityInitializer
    MetaReflection)

# Validate that historical outcomes and published components remain unchanged.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(ExistingWorldRulesTests
        tests/existing_world_rules_tests.cpp)
    target_compile_features(ExistingWorldRulesTests PRIVATE cxx_std_17)
    target_link_libraries(ExistingWorldRulesTests PRIVATE PPOExistingExample)
    add_test(NAME ExistingWorldRulesTests COMMAND ExistingWorldRulesTests)
endif()
