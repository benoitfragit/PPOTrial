cmake_minimum_required(VERSION 3.15)
project(PPO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
	add_compile_options(/Zc:__cplusplus)
endif()

# For Visual Studio 2019 x64
set(CMAKE_GENERATOR_PLATFORM x64)

include(FetchContent)

# --- IMGUI ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.0
)
FetchContent_MakeAvailable(imgui)

# GLOBBING AUTOMATIQUE DES SOURCES IMGUI
file(GLOB IMGUI_SRC
    ${imgui_SOURCE_DIR}/*.cpp
)

add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})

# --- LIBTORCH ---
# ZIP LibTorch précompilé (CPU + MSVC)
set (DEBUG_VERSION_LIBTORCH "libtorch-win-shared-with-deps-debug-2.9.1%2Bcpu.zip")
set(LIBTORCH_URL
    "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.9.1%2Bcpu.zip"
)

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    SOURCE_DIR ${CMAKE_BINARY_DIR}/libtorch
)

FetchContent_MakeAvailable(libtorch)

# Important : le find_package regarde dans <build>/libtorch/libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/libtorch")

find_package(Torch REQUIRED)

# --- SFML ---
FetchContent_Declare(
    sfml
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.1
)
FetchContent_MakeAvailable(sfml)

# ---------------------------------------------------------
#                Fetch ImGui-SFML (wrapper)
# ---------------------------------------------------------
set(IMGUI_SFML_FIND_SFML OFF CACHE BOOL "" FORCE)
set(IMGUI_DIR ${imgui_SOURCE_DIR} CACHE PATH "" FORCE)
FetchContent_Declare(
    imgui_sfml
    GIT_REPOSITORY https://github.com/eliasdaler/imgui-sfml.git
    GIT_TAG v2.6
)
FetchContent_MakeAvailable(imgui_sfml)

# Construire ImGui-SFML
add_library(imgui-sfml STATIC
    ${imgui_sfml_SOURCE_DIR}/imgui-SFML.cpp
)

target_include_directories(imgui-sfml PUBLIC
    ${imgui_sfml_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
)

target_link_libraries(imgui-sfml PUBLIC
    imgui
    sfml-system 
    sfml-window 
    sfml-graphics
)

# ---------------------------------------------------------
#                Fetch TOML++
# ---------------------------------------------------------
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(tomlplusplus)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/data/player.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/interface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/playerrenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/environmentrenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/ppo.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/agent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/agent_def.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/rlagent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rlagent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/player.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playerrenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environmentrenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        imgui
        imgui-sfml
        sfml-system 
        sfml-window 
        sfml-graphics
        tomlplusplus::tomlplusplus
        opengl32
        glu32
        "${TORCH_LIBRARIES}"
)

set(RESOURCE_TOML_PATH "${CMAKE_SOURCE_DIR}/resources/toml")
target_compile_definitions(${PROJECT_NAME} PRIVATE RESOURCE_TOML_PATH="${RESOURCE_TOML_PATH}")

# For Windows: ensure DLLs are copied
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    $<TARGET_FILE_DIR:sfml-graphics>
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# Copier les DLL automatiquement
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/libtorch/lib"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)