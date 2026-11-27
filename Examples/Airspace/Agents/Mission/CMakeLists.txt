cmake_minimum_required(VERSION 3.15)
project(Mission LANGUAGES CXX)

# RLMission red agent: production policy from AirspacePolicies (or a PPOKit-trained one),
# training on the PPOKit graph of task.json and graph.json.
add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/missionlayout.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/targetselector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/missiontrackfilter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/targetselector.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
    PUBLIC GenericEntities AirspaceEntities AirspaceEvaluation EntityInitializer GenericSerializer
        GenericRLBindings AirspacePolicies)

target_compile_definitions(${PROJECT_NAME} PUBLIC
    RESOURCE_RLMISSIONMODEL_PATH="${CMAKE_SOURCE_DIR}/Examples/Airspace/models/RLMission"
    RL_MISSION_NETWORK_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
