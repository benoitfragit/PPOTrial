cmake_minimum_required(VERSION 3.15)
project(GenericRenderer LANGUAGES CXX)

# Legacy SFML renderers were removed; GenericRenderer is raylib/imgui based.

add_library(${PROJECT_NAME}	
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environmentrenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environmentrenderer.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        imgui
        implot
        raylib
        rlImGui
        opengl32
        GenericEntities
        GenericEnv
        GenericPPO
        MetaReflection
    PRIVATE
        AirspaceEntities)

