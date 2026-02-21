cmake_minimum_required(VERSION 3.15)
project(GenericEntities LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/baseentity.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/entity.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/interface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/objective.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/snasystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/flightmodelsystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/aircraft.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/snacontrolsystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/sam.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sam.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/snacontrolsystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/aircraft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flightmodelsystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/snasystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/entity.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} PUBLIC GenericSerializer EnTT::EnTT)

set(RESOURCE_PATH "${CMAKE_SOURCE_DIR}/resources/settings")
target_compile_definitions(${PROJECT_NAME} PUBLIC RESOURCE_PATH="${RESOURCE_PATH}")

add_dependencies(${PROJECT_NAME} GenericSerializer)