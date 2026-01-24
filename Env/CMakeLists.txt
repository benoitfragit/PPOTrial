cmake_minimum_required(VERSION 3.15)
project(GenericEnv LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/entity.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/interface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/objective.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/sna.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/flightmodel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/aircraft.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/aircraft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flightmodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sna.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/entity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} PUBLIC GenericSerializer)