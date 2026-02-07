cmake_minimum_required(VERSION 3.15)
project(TopPPO LANGUAGES CXX)

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

FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG v0.16
)
FetchContent_MakeAvailable(implot)

add_library(implot STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)

target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})

target_link_libraries(implot PUBLIC imgui)

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
#                Fetch JSON
# ---------------------------------------------------------
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

# ---------------------------------------------------------
#                Fetch ENTT
# ---------------------------------------------------------
FetchContent_Declare(
    entt
    GIT_REPOSITORY https://github.com/skypjack/entt.git
    GIT_TAG        v3.16.0
)

FetchContent_MakeAvailable(entt)

add_subdirectory(Serialization)
add_subdirectory(Entities)
add_subdirectory(PPO)
add_subdirectory(Env)
add_subdirectory(Renderer)
add_subdirectory(Models)
add_subdirectory(Main)

