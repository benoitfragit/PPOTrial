cmake_minimum_required(VERSION 3.15)
project(GenericSerializer LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/serializer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serializer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        tomlplusplus::tomlplusplus
)