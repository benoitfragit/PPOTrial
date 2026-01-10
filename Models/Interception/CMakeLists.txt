cmake_minimum_required(VERSION 3.15)
project(Interception LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/actorcriticnet.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/interception.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/evaluator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/actorcriticnet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/interception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/evaluator.cpp
	)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} 
						PUBLIC GenericEnv GenericPPO)

set(RESOURCE_MISSILEMODEL_PATH "${CMAKE_SOURCE_DIR}/resources/model/missile")
target_compile_definitions(${PROJECT_NAME} PUBLIC RESOURCE_MISSILEMODEL_PATH="${RESOURCE_MISSILEMODEL_PATH}")
