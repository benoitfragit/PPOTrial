add_library(navigation_application STATIC src/application.cpp)
target_include_directories(navigation_application PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)
target_link_libraries(navigation_application PUBLIC navigation_environment PRIVATE ppo_torch)
set_target_properties(navigation_application PROPERTIES CXX_EXTENSIONS OFF)

# One application, with the original entry point and renderer.
add_executable(PPO src/main.cpp)
target_link_libraries(PPO PRIVATE navigation_application GenericRenderer)
target_compile_definitions(PPO PRIVATE
    NAVIGATION_RESOURCE_PATH="${PROJECT_SOURCE_DIR}/Examples/Navigation2D/scenarios")
set_target_properties(PPO PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(PPO PRIVATE /Zc:__cplusplus)
    # Torch_DIR selects the build/ABI; do not silently download a debug-only archive.
    file(GLOB PPO_TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(PPO_TORCH_DLLS)
        add_custom_command(TARGET PPO POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PPO_TORCH_DLLS} "$<TARGET_FILE_DIR:PPO>"
            COMMAND_EXPAND_LISTS)
    endif()
endif()
