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

# JSBSim 1.3.3, immutable release commit. Native C++ library and bundled c310 data.
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
file(COPY "${jsbsim_SOURCE_DIR}/aircraft/c310" DESTINATION "${PPO_JSBSIM_DATA_ROOT}/aircraft")
file(COPY "${jsbsim_SOURCE_DIR}/engine/engIO470D.xml" "${jsbsim_SOURCE_DIR}/engine/propC10v.xml"
    DESTINATION "${PPO_JSBSIM_DATA_ROOT}/engine")
file(COPY "${jsbsim_SOURCE_DIR}/systems/GNCUtilities.xml" DESTINATION "${PPO_JSBSIM_DATA_ROOT}/systems")
file(COPY "${jsbsim_SOURCE_DIR}/COPYING" DESTINATION "${PPO_JSBSIM_DATA_ROOT}")

add_library(FlightControlCore STATIC Core/flight_config.cpp Core/flight_mission.cpp Core/runway_motion.cpp)
target_compile_features(FlightControlCore PUBLIC cxx_std_17)
target_include_directories(FlightControlCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Core" "${CMAKE_CURRENT_SOURCE_DIR}/Training")
target_link_libraries(FlightControlCore PUBLIC nlohmann_json::nlohmann_json)

add_library(FlightControlSimulator STATIC Simulator/jsbsim_aircraft.cpp Simulator/flight_environment.cpp)
target_include_directories(FlightControlSimulator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Simulator")
target_compile_definitions(FlightControlSimulator PRIVATE PPO_JSBSIM_DATA_ROOT="${PPO_JSBSIM_DATA_ROOT}")
target_link_libraries(FlightControlSimulator PUBLIC FlightControlCore PRIVATE libJSBSim)

add_library(FlightControlLearner STATIC Training/flight_learner.cpp Training/flight_model_store.cpp)
target_link_libraries(FlightControlLearner PUBLIC FlightControlCore PRIVATE RLtools::Core)

add_library(FlightControlRuntime STATIC Runtime/flight_session.cpp)
target_include_directories(FlightControlRuntime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")
target_link_libraries(FlightControlRuntime PUBLIC FlightControlSimulator FlightControlLearner)
add_executable(FlightControlHeadless Runtime/flight_headless.cpp)
target_link_libraries(FlightControlHeadless PRIVATE FlightControlRuntime)
target_compile_definitions(FlightControlHeadless PRIVATE
    PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/resources/settings/Scenarios/FlightControl3D.json")

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    enable_testing()
    foreach(test_name IN ITEMS core simulator learner session)
        add_executable(FlightControl_${test_name}_tests tests/${test_name}_tests.cpp)
        target_link_libraries(FlightControl_${test_name}_tests PRIVATE FlightControlRuntime)
        target_compile_definitions(FlightControl_${test_name}_tests PRIVATE
            PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/resources/settings/Scenarios/FlightControl3D.json")
        add_test(NAME FlightControl_${test_name}_tests COMMAND FlightControl_${test_name}_tests)
    endforeach()
    target_sources(FlightControl_learner_tests PRIVATE tests/other_rltools_backend.cpp)
    target_link_libraries(FlightControl_learner_tests PRIVATE RLtools::Core)
endif()

if (PPO_BUILD_DESKTOP)
    add_library(FlightControlRenderer STATIC Renderer/flight_view.cpp)
    target_include_directories(FlightControlRenderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Renderer")
    target_link_libraries(FlightControlRenderer PUBLIC FlightControlRuntime raylib imgui)
    add_library(PPOFlightControlExample STATIC Runtime/flight_example.cpp)
    target_include_directories(PPOFlightControlExample PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Runtime" "${CMAKE_SOURCE_DIR}/Examples/data")
    target_link_libraries(PPOFlightControlExample PUBLIC GenericEnv FlightControlRuntime)
endif()
