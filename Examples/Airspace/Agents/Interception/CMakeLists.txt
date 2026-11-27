cmake_minimum_required(VERSION 3.15)
project(Interception LANGUAGES CXX)

# RLSNA red agent: production policy from AirspacePolicies (or a PPOKit-trained one),
# training on the PPOKit graph of task.json and graph.json.
add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/interception.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interception.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
    PUBLIC GenericEntities AirspaceEntities GenericEnv AirspaceEvaluation EntityInitializer GenericSerializer
        GenericRLBindings AirspacePolicies)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    RESOURCE_RLSNAMODEL_PATH="${CMAKE_SOURCE_DIR}/Examples/Airspace/models/RLSNA"
    RL_SNA_NETWORK_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
