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

add_library(FlightControlCurriculum STATIC Training/flight_curriculum.cpp Runtime/flight_reward_tuning.cpp)
target_include_directories(FlightControlCurriculum PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Training" "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")
target_link_libraries(FlightControlCurriculum PUBLIC FlightControlCore PPOKit::Core PRIVATE PPOKit::Artifacts)

# Historical learner retained for converters and numerical qualification only.
add_library(FlightControlLearner STATIC EXCLUDE_FROM_ALL Training/flight_learner.cpp Training/flight_model_store.cpp)
target_link_libraries(FlightControlLearner PUBLIC FlightControlCore PPOKit::Core PRIVATE RLtools::Core PPOKit::Artifacts)
# Actor and critic types come from graph.json (PPOKit code generation).
ppokit_generate_networks(TARGET FlightControlLearner TASK task.json GRAPH graph.json HEADER flight_networks.hpp)

add_library(FlightControlRuntime STATIC EXCLUDE_FROM_ALL Runtime/flight_session.cpp)
target_include_directories(FlightControlRuntime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Runtime")
target_link_libraries(FlightControlRuntime PUBLIC FlightControlSimulator FlightControlCurriculum FlightControlLearner PPOKit::Training)
# Reward channel labels of the run journal come from the task file.
ppokit_generate_task_layout(TARGET FlightControlRuntime TASK task.json HEADER flight_task_layout.hpp)
add_executable(FlightControlHeadless EXCLUDE_FROM_ALL Runtime/flight_headless.cpp)
target_link_libraries(FlightControlHeadless PRIVATE FlightControlRuntime PPOKit::Live)

# The example provides the simulation; ppokit_graph_network reads the project topology.
add_library(flight_environment SHARED Environment/flight_environment_library.cpp)
target_link_libraries(flight_environment PRIVATE FlightControlSimulator FlightControlCurriculum PPOKit::Training)
set_target_properties(flight_environment PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "")
ppokit_bind_task_contract(TARGET flight_environment TASK task.json)
ppokit_register_program(flight_environment)
# The FlightControl3D Visualiseur: an observer block of network.json and a 3D view for ppokit-viewer.
add_library(flight_visualiser MODULE Visualiser/flight_visualiser.cpp)
# It decodes the world row with the example's own types and geometry (FlightControlCore).
target_link_libraries(flight_visualiser PRIVATE FlightControlCore PPOKit::Core PPOKit::ComponentSdk PPOKit::Live PPOLayout)
set_target_properties(flight_visualiser PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "")
ppokit_register_program(flight_visualiser)
# Offline V3 migration. Both aircraft use the same tensor mapping and distinct scenario contracts.
add_executable(flight_checkpoint_import EXCLUDE_FROM_ALL Tools/flight_checkpoint_import.cpp)
target_link_libraries(flight_checkpoint_import PRIVATE FlightControlRuntime PPOKit::Schema)
target_compile_definitions(flight_checkpoint_import PRIVATE
    PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
add_dependencies(flight_checkpoint_import flight_visualiser)
if (TARGET PPOKitRepositoryProjectsTests)
    # T6's network.json names the Visualiseur: the repository check opens it.
    add_dependencies(PPOKitRepositoryProjectsTests flight_visualiser)
    target_compile_definitions(PPOKitRepositoryProjectsTests PRIVATE PPOKIT_FLIGHT_LIBRARIES=1)
endif()
if (TARGET PPOKitPublishedModelsTests)
    add_dependencies(PPOKitPublishedModelsTests flight_environment flight_visualiser)
    add_test(NAME PPOKitPublishedModels_T6 COMMAND PPOKitPublishedModelsTests
        "${CMAKE_CURRENT_SOURCE_DIR}/projects/T6" "${CMAKE_CURRENT_BINARY_DIR}/published_models_T6" resume)
    # PC7 has a published historical actor but no committed resumable LATEST.
    add_test(NAME PPOKitPublishedModels_PC7 COMMAND PPOKitPublishedModelsTests
        "${CMAKE_CURRENT_SOURCE_DIR}/projects/PC7" "${CMAKE_CURRENT_BINARY_DIR}/published_models_PC7" scratch)
endif()
target_compile_definitions(FlightControlHeadless PRIVATE
    PPO_FLIGHT_BUILD_CONFIGURATION="$<CONFIG>"
    PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/T6/scenario.json")

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    enable_testing()
    foreach(test_name IN ITEMS core reward simulator learner session golden_trace)
        add_executable(FlightControl_${test_name}_tests tests/${test_name}_tests.cpp)
        target_link_libraries(FlightControl_${test_name}_tests PRIVATE FlightControlRuntime)
        set(test_scenario "${CMAKE_CURRENT_SOURCE_DIR}/projects/T6/scenario.json")
        if(test_name STREQUAL "golden_trace")
            set(test_scenario "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/t6_340_scenario.json")
        endif()
        target_compile_definitions(FlightControl_${test_name}_tests PRIVATE
            PPO_FLIGHT_CONFIG="${test_scenario}"
            PPO_FLIGHT_PC7_CONFIG="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/PC7/scenario.json")
        add_test(NAME FlightControl_${test_name}_tests COMMAND FlightControl_${test_name}_tests)
    endforeach()
    # The Linux reference CI must fail if the executable cannot check its frozen fingerprint.
    option(PPO_FLIGHT_REQUIRE_GOLDEN_REFERENCE "Fail if the frozen Flight trace is skipped" OFF)
    if (NOT PPO_FLIGHT_REQUIRE_GOLDEN_REFERENCE)
        set_tests_properties(FlightControl_golden_trace_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # task.json (PPOKit) must describe the same observation, actions and rewards as the code.
    add_executable(FlightControl_task_layout_tests tests/task_layout_tests.cpp)
    target_link_libraries(FlightControl_task_layout_tests PRIVATE FlightControlRuntime)
    ppokit_generate_task_layout(TARGET FlightControl_task_layout_tests TASK task.json HEADER flight_task_layout.hpp)
    add_test(NAME FlightControl_task_layout_tests COMMAND FlightControl_task_layout_tests)
    # graph.json (PPOKit) generates the same rl_tools networks as FlightLearner.
    add_executable(FlightControl_network_parity_tests tests/network_parity_tests.cpp)
    target_link_libraries(FlightControl_network_parity_tests PRIVATE FlightControlRuntime RLtools::Core)
    target_compile_definitions(FlightControl_network_parity_tests PRIVATE
        PPO_FLIGHT_CONFIG="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/T6/scenario.json")
    ppokit_generate_networks(TARGET FlightControl_network_parity_tests
        TASK task.json GRAPH graph.json HEADER flight_networks.hpp)
    add_test(NAME FlightControl_network_parity_tests COMMAND FlightControl_network_parity_tests)
    # projects/T6 (PPOKit graph engine) reproduces master's trained T-6 (FlightLearner, update 870).
    add_executable(FlightControl_master_checkpoint_parity_tests tests/master_checkpoint_parity_tests.cpp)
    target_link_libraries(FlightControl_master_checkpoint_parity_tests PRIVATE FlightControlRuntime RLtools::Core
        PPOKit::Schema)
    target_compile_definitions(FlightControl_master_checkpoint_parity_tests PRIVATE
        PPO_FLIGHT_T6_PROJECT="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/T6"
        PPO_FLIGHT_MASTER_CHECKPOINT="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/master_t6texan2_joint_v5_flight_policy.txt")
    ppokit_generate_networks(TARGET FlightControl_master_checkpoint_parity_tests
        TASK task.json GRAPH graph.json HEADER flight_networks.hpp)
    add_test(NAME FlightControl_master_checkpoint_parity_tests COMMAND FlightControl_master_checkpoint_parity_tests)
    # Capture the published actor through its project modules for a replay against a separate master checkout.
    add_executable(FlightControl_reward_capture EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tools/ppokit/flight_reward_capture.cpp")
    target_link_libraries(FlightControl_reward_capture PRIVATE PPOKit::Network PPOKit::Schema PPOKit::Artifacts)
    add_dependencies(FlightControl_reward_capture flight_environment ppokit_graph_network)
    target_compile_definitions(FlightControl_reward_capture PRIVATE
        PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME FlightControl_campaign_comparison
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/ppokit/test_flight_campaign.py)
    add_test(NAME FlightControl_v3_migration COMMAND ${CMAKE_COMMAND}
        -DIMPORT=$<TARGET_FILE:flight_checkpoint_import> -DFROZEN=$<TARGET_FILE:FlightControl_master_checkpoint_parity_tests>
        -DPYTHON=${Python3_EXECUTABLE}
        -DAUDIT=${CMAKE_SOURCE_DIR}/tools/ppokit/compare_flight_checkpoint.py
        -DFLIGHT=${CMAKE_CURRENT_SOURCE_DIR} -DWORK=${CMAKE_CURRENT_BINARY_DIR}/migration_test
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/v3_migration.cmake)
    # ppokit-run evaluates projects/T6 as master's BEST record of the same checkpoint; the PC-7
    # project refuses the T-6 policy.
    add_test(NAME FlightControl_t6_evaluation_smoke
        COMMAND ${CMAKE_COMMAND} -DRUN=$<TARGET_FILE:ppokit-run>
            -DPROJECT=${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/T6
            -DPC7=${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/PC7
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/t6_evaluation_smoke
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/t6_evaluation_smoke.cmake)
    set_tests_properties(FlightControl_t6_evaluation_smoke PROPERTIES TIMEOUT 600)
    # The Visualiseur end to end: ppokit-run plays T6, FlightView draws its frames; and only
    # training moves the curriculum of the environment DLL.
    add_executable(FlightControl_visualiser_tests tests/visualiser_tests.cpp)
    target_link_libraries(FlightControl_visualiser_tests PRIVATE FlightControlCore PPOKit::View PPOKit::Network
        nlohmann_json::nlohmann_json)
    add_dependencies(FlightControl_visualiser_tests ppokit-run flight_environment ppokit_graph_network flight_visualiser)
    target_compile_definitions(FlightControl_visualiser_tests PRIVATE
        PPOKIT_RUN="$<TARGET_FILE:ppokit-run>"
        PPO_FLIGHT_T6_PROJECT="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/T6"
        FLIGHT_ENVIRONMENT_LIBRARY="$<TARGET_FILE:flight_environment>"
        FLIGHT_NETWORK_LIBRARY="$<TARGET_FILE:ppokit_graph_network>"
        FLIGHT_VISUALISER_LIBRARY="$<TARGET_FILE:flight_visualiser>")
    add_test(NAME FlightControl_visualiser_tests COMMAND FlightControl_visualiser_tests)
    target_sources(FlightControl_learner_tests PRIVATE tests/other_rltools_backend.cpp)
    target_link_libraries(FlightControl_learner_tests PRIVATE RLtools::Core)
endif()
