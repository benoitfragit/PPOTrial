cmake_minimum_required(VERSION 3.15)
project(Mission LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/missionlayout.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/actorcriticmission.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/actorcriticmission.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/targetselector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/missiontrackfilter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/targetselector.cpp
	)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} 
						PUBLIC	GenericEntities
                                AirspaceEntities
								AirspaceEvaluation
								EntityInitializer
								GenericSerializer 
								GenericPPO GenericRLBindings)

add_dependencies(${PROJECT_NAME} GenericEntities 
								 AirspaceEvaluation
								 EntityInitializer
								 GenericSerializer 
								 GenericPPO)

set(RESOURCE_RLMISSIONMODEL_PATH "${CMAKE_SOURCE_DIR}/resources/model/RLMission")
target_compile_definitions(${PROJECT_NAME} PUBLIC RESOURCE_RLMISSIONMODEL_PATH="${RESOURCE_RLMISSIONMODEL_PATH}")
