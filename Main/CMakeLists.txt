cmake_minimum_required(VERSION 3.15)
project(PPO LANGUAGES CXX)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/exampleselector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/exampleselector.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GenericEntities
		GenericSerializer
		GenericPPO
        GenericEnv
		AirspaceRenderer
		PPOExampleCatalog
        MetaReflection)
		
add_dependencies(${PROJECT_NAME} GenericEntities 
								 GenericSerializer 
								 GenericPPO
								 GenericEnv
								 AirspaceRenderer
								 PPOExampleCatalog
                                 MetaReflection)

# The launcher uses raylib/rlImGui directly before the simulation window opens.
target_link_libraries(${PROJECT_NAME} PRIVATE raylib rlImGui imgui)

# Copier les DLL automatiquement
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/libtorch/lib"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)

# Headless desktop smoke tests: exercise the actual executable's example
# selection on the server without creating a graphics window.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_test(NAME AirspaceExampleList
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --list-examples)
    set_tests_properties(AirspaceExampleList PROPERTIES
        PASS_REGULAR_EXPRESSION "airspace")
    add_test(NAME AirspaceExampleBootstrap
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --check-example)
    set_tests_properties(AirspaceExampleBootstrap PROPERTIES
        PASS_REGULAR_EXPRESSION "Bootstrapped example: airspace")
    add_test(NAME TwoVsTwoExampleBootstrap
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --example two-vs-two --check-example)
    set_tests_properties(TwoVsTwoExampleBootstrap PROPERTIES
        PASS_REGULAR_EXPRESSION "Bootstrapped example: two-vs-two")

    add_test(NAME UnknownExampleRejected
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --example unknown-example)
    set_tests_properties(UnknownExampleRejected PROPERTIES WILL_FAIL TRUE)

    # Missing --example argument must remain an error (exit status 2).
    add_test(NAME MissingExampleIdRejected
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --example)
    set_tests_properties(MissingExampleIdRejected PROPERTIES WILL_FAIL TRUE)

    # Preserve CLI handling for an explicitly selected compiled example.
    add_test(NAME ExplicitAirspaceExampleBootstrap
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --example airspace --check-example)
    set_tests_properties(ExplicitAirspaceExampleBootstrap PROPERTIES
        PASS_REGULAR_EXPRESSION "Bootstrapped example: airspace")
endif()

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(ExampleSelectorPolicyTests tests/example_selector_tests.cpp)
    target_compile_features(ExampleSelectorPolicyTests PRIVATE cxx_std_17)
    target_include_directories(ExampleSelectorPolicyTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "${CMAKE_SOURCE_DIR}/Examples/data")
    add_test(NAME ExampleSelectorPolicyTests COMMAND ExampleSelectorPolicyTests)
endif()
