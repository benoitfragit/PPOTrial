cmake_minimum_required(VERSION 3.15)
project(PPO LANGUAGES CXX)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
		GenericEnv
		GenericSerializer
		GenericPPO
		GenericRenderer
		Interception)

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