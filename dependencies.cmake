include(FetchContent)
# Preserve the existing desktop library versions. LibTorch is supplied via Torch_DIR.
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.92.1-docking)
FetchContent_MakeAvailable(imgui)
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR})

FetchContent_Declare(implot
    GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG v0.17)
FetchContent_MakeAvailable(implot)
add_library(implot STATIC ${implot_SOURCE_DIR}/implot.cpp ${implot_SOURCE_DIR}/implot_items.cpp)
target_include_directories(implot SYSTEM PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot PUBLIC imgui)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git GIT_TAG 5.5)
FetchContent_MakeAvailable(raylib)

# Pin the pre-existing ImGui 1.92.1-compatible integration; newer main
# requires ImFontConfig::ExtraSizeScale, absent from the retained ImGui version.
FetchContent_Declare(rlimgui
    GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
    GIT_TAG 286e11acd6c785004c9550c7ed3762add2ae3d47)
FetchContent_MakeAvailable(rlimgui)
add_library(rlImGui STATIC ${rlimgui_SOURCE_DIR}/rlImGui.cpp)
target_include_directories(rlImGui SYSTEM PUBLIC ${rlimgui_SOURCE_DIR})
target_compile_definitions(rlImGui PUBLIC NO_FONT_AWESOME)
target_link_libraries(rlImGui PUBLIC imgui raylib)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0)
FetchContent_MakeAvailable(json)
