cmake_minimum_required(VERSION 3.15)
project(PPO LANGUAGES CXX)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GenericEntities
		GenericSerializer
		GenericPPO
        GenericEnv
		GenericRenderer
		Interception
        Mission
        MetaReflection)
		
add_dependencies(${PROJECT_NAME} GenericEntities 
								 GenericSerializer 
								 GenericPPO
								 GenericEnv
								 GenericRenderer
								 Interception
                                 Mission
                                 MetaReflection)

# Copier les DLL automatiquement
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/libtorch/lib"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
)
