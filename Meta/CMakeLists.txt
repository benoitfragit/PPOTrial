cmake_minimum_required(VERSION 3.15)
project(MetaReflection LANGUAGES CXX)

add_library(${PROJECT_NAME}
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/metaregistry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metaregistry.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GenericEntities
        AirspaceEntities
        GenericEnv
        GenericPPO
        Interception
        Mission
        EnTT::EnTT)

add_dependencies(${PROJECT_NAME}
    GenericEntities
    GenericEnv
    GenericPPO
    Interception
    Mission)
