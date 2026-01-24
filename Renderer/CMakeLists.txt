cmake_minimum_required(VERSION 3.15)
project(GenericRenderer LANGUAGES CXX)

add_library(${PROJECT_NAME}	
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/entityrenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environmentrenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/objectiverenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/irenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/objectiverenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/entityrenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environmentrenderer.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        imgui
        imgui-sfml
        sfml-system 
        sfml-window 
        sfml-graphics
        opengl32
        glu32
        GenericEnv
        GenericPPO)

