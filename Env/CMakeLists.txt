cmake_minimum_required(VERSION 3.15)
project(GenericEnv LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/scenarioconfig.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/worldrules.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scenarioconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} PUBLIC GenericEntities 
											 
                                             LegacyAgentAPI
                                             GenericSerializer 
											 EnTT::EnTT)

add_dependencies(${PROJECT_NAME} GenericEntities 
								 
                                 LegacyAgentAPI
                                 GenericSerializer)

# Validate the current scenario, profile and evaluation contracts.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(ScenarioConfigurationTests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/scenario_config_tests.cpp)
    target_compile_features(ScenarioConfigurationTests PRIVATE cxx_std_17)
    target_link_libraries(ScenarioConfigurationTests PRIVATE GenericEnv)
    target_compile_definitions(ScenarioConfigurationTests PRIVATE
        PPO_SETTINGS_ROOT="${CMAKE_SOURCE_DIR}/resources/settings")
    add_test(NAME ScenarioConfigurationTests
        COMMAND ScenarioConfigurationTests)
endif()
