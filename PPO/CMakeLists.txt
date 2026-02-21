cmake_minimum_required(VERSION 3.15)
project(GenericPPO LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/iagent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ppo.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/trainingagent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/trainingagent_def.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/prodagent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/prodagent_def.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/rlagent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/trainingnet.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rlagent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppo.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}" GenericEntities)