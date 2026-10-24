add_subdirectory(Entities)
add_subdirectory(Initializer)
add_subdirectory(Evaluation)
add_subdirectory(Agents/Interception)
add_subdirectory(Agents/mission)

add_library(PPOAirspaceExample STATIC
    data/airspaceexample.hpp
    src/airspaceexample.cpp
    data/airspaceworldrules.hpp
    src/airspaceworldrules.cpp
    data/airspacemeta.hpp
    src/airspacemeta.cpp)

target_compile_features(PPOAirspaceExample PUBLIC cxx_std_17)
target_include_directories(PPOAirspaceExample PUBLIC
    "${CMAKE_SOURCE_DIR}/Examples/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOAirspaceExample PUBLIC
    GenericEntities
    AirspaceEntities
    GenericEnv
    EntityInitializer
    MetaReflection
    AirspaceEvaluation
    Interception
    Mission)

# Validate that historical outcomes and published components remain unchanged.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(AirspaceWorldRulesTests
        tests/airspace_world_rules_tests.cpp)
    target_compile_features(AirspaceWorldRulesTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceWorldRulesTests PRIVATE PPOAirspaceExample)
    add_test(NAME AirspaceWorldRulesTests COMMAND AirspaceWorldRulesTests)
endif()

# Registration must preserve the same concrete factory and reflection names.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(AirspaceMetaTests tests/airspace_meta_tests.cpp)
    target_compile_features(AirspaceMetaTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceMetaTests PRIVATE PPOAirspaceExample)
    # The test links legacy Torch-backed agent factories, so copy the same
    # configuration-matched runtime DLLs as PPO.exe and PolicyCheckpointTests.
    add_custom_command(TARGET AirspaceMetaTests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/libtorch/lib"
            "$<TARGET_FILE_DIR:AirspaceMetaTests>")
    add_test(NAME AirspaceMetaTests COMMAND AirspaceMetaTests)
endif()
