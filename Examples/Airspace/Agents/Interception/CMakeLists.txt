cmake_minimum_required(VERSION 3.15)
project(Interception LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/actorcriticnet.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interception.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/actorcriticnet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interception.cpp
	)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} 
						PUBLIC 	GenericEntities
                                AirspaceEntities
								GenericEnv
								AirspaceEvaluation
								EntityInitializer
								GenericPPO GenericRLBindings)

set(RESOURCE_RLSNAMODEL_PATH "${CMAKE_SOURCE_DIR}/resources/model/RLSNA")
target_compile_definitions(${PROJECT_NAME} PUBLIC RESOURCE_RLSNAMODEL_PATH="${RESOURCE_RLSNAMODEL_PATH}")

add_dependencies(${PROJECT_NAME} GenericEntities 
								 GenericEnv
								 AirspaceEvaluation
								 EntityInitializer
								 GenericSerializer 
								 GenericPPO)
