include(FetchContent)

if (NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0)
    FetchContent_MakeAvailable(json)
endif()
if (NOT TARGET RLtools::Core)
    set(RL_TOOLS_ENABLE_TARGETS OFF CACHE BOOL "" FORCE)
    set(RL_TOOLS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(RL_TOOLS_ENABLE_HDF5 OFF CACHE BOOL "" FORCE)
    set(RL_TOOLS_ENABLE_TENSORBOARD OFF CACHE BOOL "" FORCE)
    set(RL_TOOLS_BACKEND_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(rltools
        GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
        GIT_TAG 63a7520b2d0d967920ff14c7336ce4f7d682c21c
        GIT_SUBMODULES "" GIT_SUBMODULES_RECURSE FALSE)
    FetchContent_MakeAvailable(rltools)
endif()

# JSBSim 1.3.3, immutable release commit and native T-6 and PC-7 data.
set(BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(BUILD_JULIA_PACKAGE OFF CACHE BOOL "" FORCE)
set(BUILD_MATLAB_SFUNCTION OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(jsbsim
    GIT_REPOSITORY https://github.com/JSBSim-Team/jsbsim.git
    GIT_TAG 66f7fafc07fe13e7ede9db20873030fed2881713
    GIT_SUBMODULES "")
FetchContent_MakeAvailable(jsbsim)

set(PPO_JSBSIM_DATA_ROOT "${CMAKE_BINARY_DIR}/resources/FlightControl3D/jsbsim")
file(MAKE_DIRECTORY "${PPO_JSBSIM_DATA_ROOT}/aircraft" "${PPO_JSBSIM_DATA_ROOT}/engine" "${PPO_JSBSIM_DATA_ROOT}/systems")
file(COPY "${jsbsim_SOURCE_DIR}/aircraft/t6texan2" "${jsbsim_SOURCE_DIR}/aircraft/pc7"
    DESTINATION "${PPO_JSBSIM_DATA_ROOT}/aircraft")
file(COPY "${jsbsim_SOURCE_DIR}/engine/PT6A-68.xml" "${jsbsim_SOURCE_DIR}/engine/direct.xml"
    DESTINATION "${PPO_JSBSIM_DATA_ROOT}/engine")
file(COPY "${jsbsim_SOURCE_DIR}/systems/GNCUtilities.xml" DESTINATION "${PPO_JSBSIM_DATA_ROOT}/systems")
file(COPY "${jsbsim_SOURCE_DIR}/COPYING" DESTINATION "${PPO_JSBSIM_DATA_ROOT}")

add_library(FlightControlCore STATIC Core/flight_config.cpp Core/runway_geometry.cpp Core/flight_spawn.cpp
    Core/flight_guidance.cpp Core/flight_landing.cpp Core/flight_reward.cpp Core/flight_observation.cpp
    Core/flight_mission.cpp)
target_compile_features(FlightControlCore PUBLIC cxx_std_17)
target_include_directories(FlightControlCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Core" "${CMAKE_CURRENT_SOURCE_DIR}/Training")
target_link_libraries(FlightControlCore PUBLIC nlohmann_json::nlohmann_json)

add_library(FlightControlSimulator STATIC Simulator/jsbsim_aircraft.cpp Simulator/flight_environment.cpp
    Simulator/flight_trim.cpp)
target_include_directories(FlightControlSimulator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Simulator")
target_compile_definitions(FlightControlSimulator PRIVATE PPO_JSBSIM_DATA_ROOT="${PPO_JSBSIM_DATA_ROOT}")
target_link_libraries(FlightControlSimulator PUBLIC FlightControlCore PRIVATE libJSBSim)

add_library(FlightControlLearner STATIC Training/flight_learner.cpp Training/flight_model_store.cpp
    Training/flight_curriculum.cpp)
target_link_libraries(FlightControlLearner PUBLIC FlightControlCore PRIVATE RLtools::Core)

add_library(FlightControlRuntime STATIC Runtime/flight_session.cpp Runtime/flight_reward_tuning.cpp)
target_include_directories(FlightControlRuntime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")
target_link_libraries(FlightControlRuntime PUBLIC FlightControlSimulator FlightControlLearner)
add_executable(FlightControlHeadless Runtime/flight_headless.cpp)
target_link_libraries(FlightControlHeadless PRIVATE FlightControlRuntime)
target_compile_definitions(FlightControlHeadless PRIVATE
    PPO_FLIGHT_BUILD_CONFIGURATION="$<CONFIG>"
    PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/resources/settings/Scenarios/FlightControl3D_T6_Joint.json")

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    enable_testing()
    foreach(test_name IN ITEMS core reward simulator learner session golden_trace)
        add_executable(FlightControl_${test_name}_tests tests/${test_name}_tests.cpp)
        target_link_libraries(FlightControl_${test_name}_tests PRIVATE FlightControlRuntime)
        target_compile_definitions(FlightControl_${test_name}_tests PRIVATE
            PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/resources/settings/Scenarios/FlightControl3D_T6_Joint.json"
            PPO_FLIGHT_PC7_CONFIG="${CMAKE_SOURCE_DIR}/resources/settings/Scenarios/FlightControl3D_PC7_Joint.json")
        add_test(NAME FlightControl_${test_name}_tests COMMAND FlightControl_${test_name}_tests)
    endforeach()
    # The golden trace is only frozen on its reference platform; elsewhere it reports a skip.
    set_tests_properties(FlightControl_golden_trace_tests PROPERTIES SKIP_RETURN_CODE 77)
    target_sources(FlightControl_learner_tests PRIVATE tests/other_rltools_backend.cpp)
    target_link_libraries(FlightControl_learner_tests PRIVATE RLtools::Core)
endif()

if (PPO_BUILD_DESKTOP)
    add_library(FlightControlRenderer STATIC Renderer/flight_view.cpp Renderer/flight_profile_view.cpp Renderer/flight_reward_view.cpp)
    target_include_directories(FlightControlRenderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Renderer")
    target_link_libraries(FlightControlRenderer PUBLIC FlightControlRuntime raylib imgui)
    if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
        add_executable(FlightControl_display_tests tests/display_tests.cpp)
        target_link_libraries(FlightControl_display_tests PRIVATE FlightControlRenderer)
        add_test(NAME FlightControl_display_tests COMMAND FlightControl_display_tests)
    endif()
    add_library(PPOFlightControlExample STATIC Runtime/flight_example.cpp)
    target_include_directories(PPOFlightControlExample PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Runtime" "${CMAKE_SOURCE_DIR}/Examples/data")
    target_link_libraries(PPOFlightControlExample PUBLIC GenericEnv FlightControlRuntime)
    if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
        add_executable(FlightControl_ecs_tests tests/ecs_tests.cpp)
        target_link_libraries(FlightControl_ecs_tests PRIVATE PPOFlightControlExample)
        target_compile_definitions(FlightControl_ecs_tests PRIVATE
            PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/resources/settings/Scenarios/FlightControl3D_T6_Joint.json")
        add_test(NAME FlightControl_ecs_tests COMMAND FlightControl_ecs_tests)
    endif()
endif()
