# Historical desktop rendering stays with the Airspace example.
# Keep the UI, model assets and simulation controls unchanged.
add_library(AirspaceRenderer STATIC
    data/environmentrenderer.hpp
    src/environmentrenderer.cpp)

target_compile_features(AirspaceRenderer PUBLIC cxx_std_17)
target_include_directories(AirspaceRenderer PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
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
