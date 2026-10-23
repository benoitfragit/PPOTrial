cmake_minimum_required(VERSION 3.15)
project(GenericEnv LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/scenarioconfig.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scenarioconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} PUBLIC GenericEntities 
											 EntityInitializer
                                             LegacyAgentAPI
                                             GenericSerializer 
											 EnTT::EnTT)

add_dependencies(${PROJECT_NAME} GenericEntities 
								 EntityInitializer
                                 LegacyAgentAPI
                                 GenericSerializer)
