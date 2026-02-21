cmake_minimum_required(VERSION 3.15)
project(EntityInitializer LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/randominitializersystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/checkpointinitializersystem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/randominitializersystem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/checkpointinitializersystem.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} PUBLIC GenericSerializer GenericEntities EnTT::EnTT)

add_dependencies(${PROJECT_NAME} GenericSerializer GenericEntities)