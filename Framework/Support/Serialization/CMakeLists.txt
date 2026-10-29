cmake_minimum_required(VERSION 3.15)
project(GenericSerializer LANGUAGES CXX)

add_library(${PROJECT_NAME}
	STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/serializer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data/training_settings.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/training_settings.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        nlohmann_json::nlohmann_json
)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(TrainingSettingsTests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/training_settings_tests.cpp)
    target_compile_features(TrainingSettingsTests PRIVATE cxx_std_17)
    target_link_libraries(TrainingSettingsTests PRIVATE GenericSerializer)
    target_compile_definitions(TrainingSettingsTests PRIVATE
        PPO_SETTINGS_ROOT="${CMAKE_SOURCE_DIR}/resources/settings")
    add_test(NAME TrainingSettingsTests
        COMMAND TrainingSettingsTests)
    add_executable(SerializerSaveSafetyTests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/serializer_safety_tests.cpp)
    target_compile_features(SerializerSaveSafetyTests PRIVATE cxx_std_17)
    target_link_libraries(SerializerSaveSafetyTests PRIVATE GenericSerializer)
    add_test(NAME SerializerSaveSafetyTests
        COMMAND SerializerSaveSafetyTests)
endif()
