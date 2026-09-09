add_library(GenericRenderer STATIC src/environmentrenderer.cpp)
target_include_directories(GenericRenderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)
target_link_libraries(GenericRenderer PUBLIC navigation_application raylib imgui
    PRIVATE implot rlImGui nlohmann_json::nlohmann_json)
set_target_properties(GenericRenderer PROPERTIES CXX_EXTENSIONS OFF)
