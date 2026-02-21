cmake_minimum_required(VERSION 3.15)
project(TopPPO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
	add_compile_options(/Zc:__cplusplus)
endif()

# For Visual Studio 2019 x64
set(CMAKE_GENERATOR_PLATFORM x64)

include(${CMAKE_SOURCE_DIR}/dependencies.cmake)

add_subdirectory(Serialization)
add_subdirectory(Entities)
add_subdirectory(Initializer)
add_subdirectory(PPO)
add_subdirectory(Env)
add_subdirectory(Renderer)
add_subdirectory(Agents)
add_subdirectory(Main)

