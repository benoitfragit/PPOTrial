# Historical desktop rendering stays with the Airspace example.
# Keep the UI, model assets and simulation controls unchanged.
add_library(AirspaceRenderer STATIC
    tactical_view_layout.hpp
    environmentrenderer.hpp
    environmentrenderer.cpp)

target_compile_features(AirspaceRenderer PUBLIC cxx_std_17)
target_include_directories(AirspaceRenderer PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(AirspaceRenderer
    PUBLIC
        GenericRenderer
        imgui
        implot
        raylib
        rlImGui
        opengl32
        GenericEntities
        GenericEnv
        GenericPPO
        MetaReflection
    PRIVATE
        AirspaceEntities)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TacticalViewLayoutTests
        ../tests/tactical_view_layout_tests.cpp)
    target_compile_features(TacticalViewLayoutTests PRIVATE cxx_std_17)
    target_include_directories(TacticalViewLayoutTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME TacticalViewLayoutTests COMMAND TacticalViewLayoutTests)
endif()
