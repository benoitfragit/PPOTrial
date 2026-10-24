cmake_minimum_required(VERSION 3.15)
project(PPO LANGUAGES CXX)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GenericEntities
		GenericSerializer
		GenericPPO
        GenericEnv
		AirspaceRenderer
		PPOAirspaceExample
        MetaReflection)
		
add_dependencies(${PROJECT_NAME} GenericEntities 
								 GenericSerializer 
								 GenericPPO
								 GenericEnv
								 AirspaceRenderer
								 PPOAirspaceExample
                                 MetaReflection)

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

    add_test(NAME UnknownExampleRejected
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> --example unknown-example)
    set_tests_properties(UnknownExampleRejected PROPERTIES WILL_FAIL TRUE)
endif()
