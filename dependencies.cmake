cmake_minimum_required(VERSION 3.15)

include(FetchContent)

# --- IMGUI ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.1-docking
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
    GIT_TAG v0.17
)
FetchContent_MakeAvailable(implot)

add_library(implot STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)

target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})

target_link_libraries(implot PUBLIC imgui)

# --- IMNODES (node editor of PPOKit studio) ---
# Pinned commit: it compiles against the Dear ImGui 1.92.1 used above.
# Only the sources are used: SOURCE_SUBDIR names a folder without CMakeLists.txt, so
# imnodes' own project (which expects other ImGui targets) is not added.
FetchContent_Declare(
    imnodes
    GIT_REPOSITORY https://github.com/Nelarius/imnodes.git
    GIT_TAG 2d0c280b07b2f62b120034811cc6e41251673961
    SOURCE_SUBDIR sources-only
)
FetchContent_MakeAvailable(imnodes)
add_library(imnodes STATIC ${imnodes_SOURCE_DIR}/imnodes.cpp)
target_include_directories(imnodes PUBLIC ${imnodes_SOURCE_DIR})
target_link_libraries(imnodes PUBLIC imgui)

# --- RAYLIB ---
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

# ---------------------------------------------------------
#                Fetch rlImGui (raylib + ImGui)
# ---------------------------------------------------------
FetchContent_Declare(
    rlimgui
    GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
    # Frozen at the rlImGui implementation for Dear ImGui 1.92.1.
    # Do not follow main: it now requires newer ImFontConfig APIs.
    GIT_TAG 4d8a61842903978bc42adf3347cd34f4e6524efc
)
FetchContent_MakeAvailable(rlimgui)

add_library(rlImGui STATIC
    ${rlimgui_SOURCE_DIR}/rlImGui.cpp
)

target_include_directories(rlImGui PUBLIC
    ${rlimgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
)

target_compile_definitions(rlImGui PUBLIC NO_FONT_AWESOME)

target_link_libraries(rlImGui PUBLIC
    imgui
    raylib
    opengl32
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
#                ENTT (the ECS world, shared with the world-only build)
# ---------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/world_dependencies.cmake)


# RLtools (CPU, C++17): network backend of the PPOKit learners.
set(RL_TOOLS_ENABLE_TARGETS OFF CACHE BOOL "Build upstream RLtools examples" FORCE)
set(RL_TOOLS_ENABLE_TESTS OFF CACHE BOOL "Build upstream RLtools tests" FORCE)
set(RL_TOOLS_ENABLE_HDF5 OFF CACHE BOOL "Use RLtools HDF5" FORCE)
set(RL_TOOLS_ENABLE_TENSORBOARD OFF CACHE BOOL "Use RLtools TensorBoard" FORCE)
set(RL_TOOLS_BACKEND_ENABLE_CUDA OFF CACHE BOOL "Use RLtools CUDA" FORCE)

FetchContent_Declare(
    rltools
    GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
    GIT_TAG 63a7520b2d0d967920ff14c7336ce4f7d682c21c
    # RLtools::Core is header-only. The upstream optional submodules include
    # unavailable commits; do not fetch them (CMP0097 NEW, CMake >= 3.16).
    GIT_SUBMODULES ""
    GIT_SUBMODULES_RECURSE FALSE
)
FetchContent_MakeAvailable(rltools)
if (NOT TARGET RLtools::Core)
    message(FATAL_ERROR "RLtools::Core is missing after FetchContent")
endif()
