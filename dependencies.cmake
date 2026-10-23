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

# --- LIBTORCH ---
# Windows LibTorch binaries use different CRTs for Debug and Release.
# Each build directory must use the archive matching its MSVC configuration.
# Keep Debug as the default to preserve the historical desktop setup.
set(PPO_LIBTORCH_VARIANT "Debug" CACHE STRING "LibTorch binary variant (Debug or Release)")
set_property(CACHE PPO_LIBTORCH_VARIANT PROPERTY STRINGS Debug Release)

if(PPO_LIBTORCH_VARIANT STREQUAL "Debug")
    set(PPO_LIBTORCH_ARCHIVE "libtorch-win-shared-with-deps-debug-2.9.1%2Bcpu.zip")
elseif(PPO_LIBTORCH_VARIANT STREQUAL "Release")
    set(PPO_LIBTORCH_ARCHIVE "libtorch-win-shared-with-deps-2.9.1%2Bcpu.zip")
else()
    message(FATAL_ERROR "PPO_LIBTORCH_VARIANT must be Debug or Release")
endif()

# Visual Studio is multi-configuration; avoid silently testing a Release binary
# against a Debug LibTorch distribution (or the reverse). Use separate build
# directories with CMAKE_CONFIGURATION_TYPES set to the selected variant.
if(MSVC AND CMAKE_CONFIGURATION_TYPES AND PPO_BUILD_CHECKPOINT_TESTS)
    foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
        if(NOT config STREQUAL PPO_LIBTORCH_VARIANT)
            message(FATAL_ERROR
                "LibTorch ${PPO_LIBTORCH_VARIANT} cannot be used with ${config}. "
                "Configure a separate build directory with "
                "-DCMAKE_CONFIGURATION_TYPES=${PPO_LIBTORCH_VARIANT} "
                "-DPPO_LIBTORCH_VARIANT=${PPO_LIBTORCH_VARIANT}, "
                "or disable checkpoint tests for a legacy mixed-config build.")
        endif()
    endforeach()
endif()

set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/${PPO_LIBTORCH_ARCHIVE}")

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    SOURCE_DIR ${CMAKE_BINARY_DIR}/libtorch
)

FetchContent_MakeAvailable(libtorch)

# Important : le find_package regarde dans <build>/libtorch/libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/libtorch")

find_package(Torch REQUIRED)

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
#                Fetch ENTT
# ---------------------------------------------------------
FetchContent_Declare(
    entt
    GIT_REPOSITORY https://github.com/skypjack/entt.git
    GIT_TAG        v3.16.0
)

FetchContent_MakeAvailable(entt)
