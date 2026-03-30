cmake_minimum_required(VERSION 3.15)
project(Mission LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/missionlayout.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/actorcriticmission.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/actorcriticmission.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/data/targetselector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/targetselector.cpp
	)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} 
						PUBLIC	GenericEntities 
								Evaluation
								EntityInitializer
								GenericSerializer 
								GenericPPO)

add_dependencies(${PROJECT_NAME} GenericEntities 
								 Evaluation
								 EntityInitializer
								 GenericSerializer 
								 GenericPPO)

set(RESOURCE_RLMISSIONMODEL_PATH "${CMAKE_SOURCE_DIR}/resources/model/RLMission")
target_compile_definitions(${PROJECT_NAME} PUBLIC RESOURCE_RLMISSIONMODEL_PATH="${RESOURCE_RLMISSIONMODEL_PATH}")
