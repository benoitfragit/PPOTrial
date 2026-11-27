# PPOKit: framework shared by the PPOTrial examples.
# Design and migration plan: docs/architecture/ppokit-framework-design.md
# Progress journal for resuming the work: PPOKit/PROGRESS.md
#
# The public headers live under include/ppokit and never expose a backend
# (rl_tools), a simulator or a GUI library.

# PPOKit's libraries also go into DLLs (network libraries): position-independent code.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PPOKIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "PPOKit sources, for its CMake helpers")

# --- ppokit::core and ppokit::heads : algorithmic building blocks (header-only)
add_library(ppokit_core INTERFACE)
add_library(PPOKit::Core ALIAS ppokit_core)
target_include_directories(ppokit_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_core INTERFACE cxx_std_17)

# --- ppokit::schema and ppokit::eval : task, graph and evaluation files ------
include(FetchContent)
if (NOT TARGET nlohmann_json::nlohmann_json)
    # Same declaration as dependencies.cmake and FlightControl3D: the first one wins.
    FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0)
    FetchContent_MakeAvailable(json)
endif()

# --- ppokit::component : black-box component DLLs (C ABI, loader, graph node) ---
add_library(ppokit_component STATIC src/component/library.cpp src/component/native_module.cpp)
add_library(PPOKit::Component ALIAS ppokit_component)
target_include_directories(ppokit_component PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_component PUBLIC cxx_std_17)
target_link_libraries(ppokit_component PUBLIC ${CMAKE_DL_LIBS} PRIVATE nlohmann_json::nlohmann_json)
# Component DLLs only need the C ABI and the header-only SDK.
add_library(ppokit_component_sdk INTERFACE)
add_library(PPOKit::ComponentSdk ALIAS ppokit_component_sdk)
target_include_directories(ppokit_component_sdk INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_component_sdk INTERFACE cxx_std_17)

add_library(ppokit_schema STATIC src/schema/json_validation.cpp src/schema/task_schema.cpp
    src/schema/task_codegen.cpp src/schema/network_schema.cpp src/schema/train_schema.cpp
    src/schema/block_graph_schema.cpp src/schema/project.cpp src/schema/policy_reference.cpp
    src/eval/evaluation.cpp src/eval/best_store.cpp)
add_library(PPOKit::Schema ALIAS ppokit_schema)
# Evaluation suites share the JSON validation of the schema files.
add_library(PPOKit::Evaluation ALIAS ppokit_schema)
target_include_directories(ppokit_schema PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_schema PUBLIC cxx_std_17)
target_link_libraries(ppokit_schema PUBLIC PPOKit::Component PPOKit::Artifacts PRIVATE nlohmann_json::nlohmann_json)

# --- ppokit::editor : editable block graphs behind the network editor (no GUI) ---
add_library(ppokit_editor STATIC src/editor/graph_document.cpp src/editor/ppo_help.cpp src/editor/run_series.cpp
    src/editor/train_settings_document.cpp src/editor/live_weights.cpp src/editor/probe_series.cpp
    src/editor/curriculum_editing.cpp src/editor/run_health.cpp src/editor/recent_projects.cpp
    src/editor/process.cpp src/editor/evaluation_document.cpp)
add_library(PPOKit::Editor ALIAS ppokit_editor)
find_package(Threads REQUIRED)
target_link_libraries(ppokit_editor PUBLIC PPOKit::Schema PPOKit::Training PPOKit::Artifacts nlohmann_json::nlohmann_json
    Threads::Threads)

add_executable(ppokit-artifact tools/ppokit_artifact.cpp)
target_link_libraries(ppokit-artifact PRIVATE PPOKit::Schema)
target_compile_definitions(ppokit-artifact PRIVATE PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")

add_executable(ppokit-codegen codegen/main.cpp)
target_link_libraries(ppokit-codegen PRIVATE PPOKit::Schema PPOKit::Artifacts)

# ppokit-new <name> [--into DIR] and the studio's "New project": a project that already trains
# (examples/reach is its output).
add_library(ppokit_scaffold STATIC src/tools/scaffold.cpp)
add_library(PPOKit::Scaffold ALIAS ppokit_scaffold)
target_include_directories(ppokit_scaffold PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_scaffold PUBLIC cxx_std_17)
add_executable(ppokit-new tools/ppokit_new.cpp)
target_link_libraries(ppokit-new PRIVATE PPOKit::Scaffold)

# ppokit_generate_task_layout(TARGET ... TASK ... HEADER ...)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PPOKitTask.cmake")

# --- ppokit::runtime : curriculum (stages, promotion, persistence) -----------
add_library(ppokit_training STATIC src/runtime/curriculum.cpp src/runtime/run_log.cpp src/runtime/run_state.cpp src/runtime/writer_lock.cpp
    src/runtime/weight_request.cpp src/runtime/observers.cpp src/runtime/publication.cpp)
add_library(PPOKit::Training ALIAS ppokit_training)
target_link_libraries(ppokit_training PUBLIC PPOKit::Schema PPOKit::Core
    PRIVATE PPOKit::Artifacts nlohmann_json::nlohmann_json)

# --- ppokit::network : clients of the network and environment DLLs (stable C APIs) ---
add_library(ppokit_network STATIC src/network/network_library.cpp src/environment/environment_library.cpp)
add_library(PPOKit::Network ALIAS ppokit_network)
target_include_directories(ppokit_network PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_network PUBLIC cxx_std_17)
target_link_libraries(ppokit_network PUBLIC PPOKit::Component PRIVATE PPOKit::Schema)

# One graph backend for all declarative single-agent projects.
add_library(ppokit_graph_network SHARED src/network/graph_network_library.cpp)
target_compile_definitions(ppokit_graph_network PRIVATE PPOKIT_DYNAMIC_GRAPH=1
    PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
target_link_libraries(ppokit_graph_network PRIVATE PPOKit::Core PPOKit::Schema PPOKit::Training Threads::Threads)
set_target_properties(ppokit_graph_network PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "")
ppokit_register_program(ppokit_graph_network)

# --- ppokit::live : the studio watches and commands a run (loopback socket) ---
add_library(ppokit_live STATIC src/ipc/control.cpp src/live/connection.cpp src/live/datagram.cpp src/live/live_channel.cpp
    src/live/live_control.cpp)
add_library(PPOKit::Live ALIAS ppokit_live)
target_include_directories(ppokit_live PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_live PUBLIC cxx_std_17)
target_link_libraries(ppokit_live PUBLIC PPOKit::Training PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
if (WIN32)
    target_link_libraries(ppokit_live PRIVATE ws2_32)
endif()

# --- ppokit::view : Visualiseur views fed in UDP, for ppokit-viewer (no graphics library) ---
add_library(ppokit_view STATIC src/view/view_library.cpp)
add_library(PPOKit::View ALIAS ppokit_view)
target_include_directories(ppokit_view PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_view PUBLIC cxx_std_17)
target_link_libraries(ppokit_view PUBLIC PPOKit::Live PPOKit::Component)

# ppokit-run: trains, evaluates and plays a project from its environment and network DLLs.
add_executable(ppokit-run tools/ppokit_run.cpp)
target_link_libraries(ppokit-run PRIVATE nlohmann_json::nlohmann_json PPOKit::Network PPOKit::Live PPOKit::Training PPOKit::Schema)
target_compile_definitions(ppokit-run PRIVATE PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
ppokit_register_program(ppokit-run)
add_dependencies(ppokit-run ppokit_graph_network)

# --- ppokit::backend::rltools : learner on generated networks ---------------
# PPOKit owns the rl_tools dependency; the examples and dependencies.cmake reuse it.
if (NOT TARGET RLtools::Core)
    set(RL_TOOLS_ENABLE_TARGETS OFF CACHE BOOL "Build upstream RLtools examples" FORCE)
    set(RL_TOOLS_ENABLE_TESTS OFF CACHE BOOL "Build upstream RLtools tests" FORCE)
    set(RL_TOOLS_ENABLE_HDF5 OFF CACHE BOOL "Use RLtools HDF5" FORCE)
    set(RL_TOOLS_ENABLE_TENSORBOARD OFF CACHE BOOL "Use RLtools TensorBoard" FORCE)
    set(RL_TOOLS_BACKEND_ENABLE_CUDA OFF CACHE BOOL "Use RLtools CUDA" FORCE)
    # Same declaration as dependencies.cmake and FlightControl3D: header-only, no submodules.
    FetchContent_Declare(rltools
        GIT_REPOSITORY https://github.com/rl-tools/rl-tools.git
        GIT_TAG 63a7520b2d0d967920ff14c7336ce4f7d682c21c
        GIT_SUBMODULES "" GIT_SUBMODULES_RECURSE FALSE)
    FetchContent_MakeAvailable(rltools)
endif()
add_library(ppokit_backend_rltools INTERFACE)
add_library(PPOKit::BackendRltools ALIAS ppokit_backend_rltools)
target_link_libraries(ppokit_backend_rltools INTERFACE PPOKit::Core RLtools::Core)

# --- ppokit::artifacts : transactional artifact files ------------------------
add_library(ppokit_artifacts STATIC src/artifacts/atomic_file.cpp src/artifacts/artifact.cpp src/artifacts/sha256.cpp)
add_library(PPOKit::Artifacts ALIAS ppokit_artifacts)
target_include_directories(ppokit_artifacts PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_artifacts PUBLIC cxx_std_17)
target_link_libraries(ppokit_artifacts PRIVATE nlohmann_json::nlohmann_json)

if (PPO_BUILD_CORE_TESTS)
    add_executable(PPOKitAtomicFileTests tests/atomic_file_tests.cpp)
    target_link_libraries(PPOKitAtomicFileTests PRIVATE PPOKit::Artifacts)
    add_test(NAME PPOKitAtomicFileTests COMMAND PPOKitAtomicFileTests)

    add_executable(PPOKitGaeTests tests/gae_tests.cpp)
    target_link_libraries(PPOKitGaeTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitGaeTests COMMAND PPOKitGaeTests)
    add_executable(PPOKitUpdateEngineTests tests/ppo_update_engine_tests.cpp)
    target_link_libraries(PPOKitUpdateEngineTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitUpdateEngineTests COMMAND PPOKitUpdateEngineTests)

    add_executable(PPOKitGraphBlocksTests tests/graph_blocks_tests.cpp)
    target_link_libraries(PPOKitGraphBlocksTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitGraphBlocksTests COMMAND PPOKitGraphBlocksTests)

    add_executable(PPOKitGraphEngineTests tests/graph_engine_tests.cpp)
    target_link_libraries(PPOKitGraphEngineTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitGraphEngineTests COMMAND PPOKitGraphEngineTests)

    add_executable(PPOKitTeamRolloutTests tests/team_rollout_tests.cpp)
    target_link_libraries(PPOKitTeamRolloutTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitTeamRolloutTests COMMAND PPOKitTeamRolloutTests)

    add_executable(PPOKitHostedTrainerTests tests/hosted_trainer_tests.cpp)
    target_link_libraries(PPOKitHostedTrainerTests PRIVATE PPOKit::Schema PPOKit::Core)
    target_compile_definitions(PPOKitHostedTrainerTests PRIVATE
        PPOKIT_TEAM_REACH_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/team_reach/task.json"
        PPOKIT_TEAM_REACH_GRAPH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/team_reach/graph.json")
    add_test(NAME PPOKitHostedTrainerTests COMMAND PPOKitHostedTrainerTests)

    add_executable(PPOKitRuntimeTests tests/runtime_tests.cpp)
    target_link_libraries(PPOKitRuntimeTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitRuntimeTests COMMAND PPOKitRuntimeTests)

    add_executable(PPOKitHeadsTests tests/heads_tests.cpp)
    target_link_libraries(PPOKitHeadsTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitHeadsTests COMMAND PPOKitHeadsTests)

    add_executable(PPOKitTaskSchemaTests tests/task_schema_tests.cpp)
    target_link_libraries(PPOKitTaskSchemaTests PRIVATE PPOKit::Schema)
    target_compile_definitions(PPOKitTaskSchemaTests PRIVATE
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json")
    ppokit_generate_task_layout(TARGET PPOKitTaskSchemaTests
        TASK tests/fixtures/sample_task.json HEADER sample_task_layout.hpp)
    add_test(NAME PPOKitTaskSchemaTests COMMAND PPOKitTaskSchemaTests)

    add_executable(PPOKitEvaluationTests tests/evaluation_tests.cpp)
    target_link_libraries(PPOKitEvaluationTests PRIVATE PPOKit::Evaluation PPOKit::Core)
    add_test(NAME PPOKitEvaluationTests COMMAND PPOKitEvaluationTests)

    add_executable(PPOKitBestStoreTests tests/best_store_tests.cpp)
    target_link_libraries(PPOKitBestStoreTests PRIVATE PPOKit::Evaluation PPOKit::Artifacts)
    add_test(NAME PPOKitBestStoreTests COMMAND PPOKitBestStoreTests)

    add_executable(PPOKitBackendLearnerTests tests/backend_learner_tests.cpp)
    target_link_libraries(PPOKitBackendLearnerTests PRIVATE PPOKit::BackendRltools PPOKit::Evaluation)
    target_compile_definitions(PPOKitBackendLearnerTests PRIVATE
        PPOKIT_CORRIDOR_SMOKE_SUITE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/eval/smoke.json")
    ppokit_generate_networks(TARGET PPOKitBackendLearnerTests TASK tests/fixtures/corridor/task.json
        GRAPH tests/fixtures/corridor/graph.json HEADER corridor_networks.hpp)
    add_test(NAME PPOKitBackendLearnerTests COMMAND PPOKitBackendLearnerTests)

    add_executable(PPOKitAdamRlToolsParityTests tests/adam_rltools_parity_tests.cpp)
    target_link_libraries(PPOKitAdamRlToolsParityTests PRIVATE PPOKit::BackendRltools PPOKit::Training)
    target_compile_definitions(PPOKitAdamRlToolsParityTests PRIVATE
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json")
    add_test(NAME PPOKitAdamRlToolsParityTests COMMAND PPOKitAdamRlToolsParityTests)

    add_executable(PPOKitCurriculumTests tests/curriculum_tests.cpp)
    target_link_libraries(PPOKitCurriculumTests PRIVATE PPOKit::Training)
    target_compile_definitions(PPOKitCurriculumTests PRIVATE
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json"
        PPOKIT_SAMPLE_TRAIN="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_train.json")
    ppokit_generate_task_layout(TARGET PPOKitCurriculumTests
        TASK tests/fixtures/sample_task.json HEADER sample_task_layout.hpp)
    add_test(NAME PPOKitCurriculumTests COMMAND PPOKitCurriculumTests)

    add_executable(PPOKitPublicationTests tests/publication_tests.cpp)
    target_link_libraries(PPOKitPublicationTests PRIVATE PPOKit::Training)
    add_test(NAME PPOKitPublicationTests COMMAND PPOKitPublicationTests)

    add_executable(PPOKitRunLogTests tests/run_log_tests.cpp)
    target_link_libraries(PPOKitRunLogTests PRIVATE PPOKit::Training)
    add_test(NAME PPOKitRunLogTests COMMAND PPOKitRunLogTests)

    add_executable(PPOKitRunSeriesTests tests/run_series_tests.cpp)
    target_link_libraries(PPOKitRunSeriesTests PRIVATE PPOKit::Editor)
    add_test(NAME PPOKitRunSeriesTests COMMAND PPOKitRunSeriesTests)
    # A program the process tests start: echoes its arguments, waits or exits with a code.
    add_executable(ppokit_echo_program tests/support/echo_program.cpp)
    add_executable(PPOKitProcessTests tests/process_tests.cpp)
    target_link_libraries(PPOKitProcessTests PRIVATE PPOKit::Editor)
    target_compile_definitions(PPOKitProcessTests PRIVATE PPOKIT_ECHO_PROGRAM="$<TARGET_FILE:ppokit_echo_program>")
    add_dependencies(PPOKitProcessTests ppokit_echo_program)
    add_test(NAME PPOKitProcessTests COMMAND PPOKitProcessTests)
    add_executable(PPOKitProjectTests tests/project_tests.cpp)
    target_link_libraries(PPOKitProjectTests PRIVATE PPOKit::Schema)
    add_test(NAME PPOKitProjectTests COMMAND PPOKitProjectTests)
    # The repository's projects (FlightControl3D, BluePatrol) when PPOKit is built inside PPOTrial.
    if (EXISTS "${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects")
        add_executable(PPOKitRepositoryProjectsTests tests/repository_projects_tests.cpp)
        target_link_libraries(PPOKitRepositoryProjectsTests PRIVATE PPOKit::Schema nlohmann_json::nlohmann_json)
        target_compile_definitions(PPOKitRepositoryProjectsTests PRIVATE PPOTRIAL_REPOSITORY="${CMAKE_SOURCE_DIR}"
            PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
        add_test(NAME PPOKitRepositoryProjectsTests COMMAND PPOKitRepositoryProjectsTests)
        add_executable(PPOKitPublishedModelsTests tests/published_models_tests.cpp)
        target_link_libraries(PPOKitPublishedModelsTests PRIVATE PPOKit::Network PPOKit::Schema PPOKit::Artifacts)
        target_compile_definitions(PPOKitPublishedModelsTests PRIVATE
            PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
        add_dependencies(PPOKitPublishedModelsTests ppokit_graph_network)
    endif()
    add_executable(PPOKitRunHealthTests tests/run_health_tests.cpp)
    target_link_libraries(PPOKitRunHealthTests PRIVATE PPOKit::Editor)
    add_test(NAME PPOKitRunHealthTests COMMAND PPOKitRunHealthTests)
    add_executable(PPOKitRecentProjectsTests tests/recent_projects_tests.cpp)
    target_link_libraries(PPOKitRecentProjectsTests PRIVATE PPOKit::Editor)
    add_test(NAME PPOKitRecentProjectsTests COMMAND PPOKitRecentProjectsTests)
    add_executable(PPOKitWeightRequestTests tests/weight_request_tests.cpp)
    target_link_libraries(PPOKitWeightRequestTests PRIVATE PPOKit::Editor)
    add_test(NAME PPOKitWeightRequestTests COMMAND PPOKitWeightRequestTests)
    add_executable(PPOKitTrainSettingsTests tests/train_settings_tests.cpp)
    target_link_libraries(PPOKitTrainSettingsTests PRIVATE PPOKit::Editor)
    add_test(NAME PPOKitTrainSettingsTests COMMAND PPOKitTrainSettingsTests)

    add_executable(PPOKitNetworkSchemaTests tests/network_schema_tests.cpp)
    target_link_libraries(PPOKitNetworkSchemaTests PRIVATE PPOKit::Schema)
    add_test(NAME PPOKitNetworkSchemaTests COMMAND PPOKitNetworkSchemaTests)

    # Block graphs are checked against the real example tasks they must cover.
    add_executable(PPOKitBlockGraphSchemaTests tests/block_graph_schema_tests.cpp)
    target_link_libraries(PPOKitBlockGraphSchemaTests PRIVATE PPOKit::Schema)
    target_compile_definitions(PPOKitBlockGraphSchemaTests PRIVATE
        PPOKIT_BLUE_TASK="${PROJECT_SOURCE_DIR}/Examples/BlueDefense/Agents/BluePatrol/task.json"
        PPOKIT_FLIGHT_TASK="${PROJECT_SOURCE_DIR}/Examples/FlightControl3D/task.json"
        PPOKIT_BLUE_ATTENTION_GRAPH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/blue_attention/graph.json")
    add_test(NAME PPOKitBlockGraphSchemaTests COMMAND PPOKitBlockGraphSchemaTests)

    add_executable(PPOKitGraphLearnerTests tests/graph_learner_tests.cpp)
    target_link_libraries(PPOKitGraphLearnerTests PRIVATE PPOKit::Schema PPOKit::Evaluation PPOKit::Core)
    target_compile_definitions(PPOKitGraphLearnerTests PRIVATE
        PPOKIT_CORRIDOR_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/task.json"
        PPOKIT_CORRIDOR_BLOCKS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/graph_blocks.json"
        PPOKIT_CORRIDOR_SMOKE_SUITE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/eval/smoke.json"
        PPOKIT_SWITCH_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/switch/task.json"
        PPOKIT_SWITCH_GRAPH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/switch/graph.json"
        PPOKIT_SWITCH_SHARED_GRAPH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/switch/shared_graph.json"
        PPOKIT_BLUE_TASK="${PROJECT_SOURCE_DIR}/Examples/BlueDefense/Agents/BluePatrol/task.json"
        PPOKIT_BLUE_ATTENTION_GRAPH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/blue_attention/graph.json")
    add_test(NAME PPOKitGraphLearnerTests COMMAND PPOKitGraphLearnerTests)

    # Component DLLs built with the SDK, loaded by the component tests.
    add_library(ppokit_test_components MODULE tests/components/test_components.cpp)
    target_link_libraries(ppokit_test_components PRIVATE PPOKit::ComponentSdk)
    add_library(ppokit_not_a_component MODULE tests/components/not_a_component.cpp)
    target_link_libraries(ppokit_not_a_component PRIVATE PPOKit::ComponentSdk)
    add_executable(PPOKitComponentTests tests/component_tests.cpp)
    target_link_libraries(PPOKitComponentTests PRIVATE PPOKit::Schema PPOKit::Core PPOKit::Training)
    add_dependencies(PPOKitComponentTests ppokit_test_components ppokit_not_a_component)
    target_compile_definitions(PPOKitComponentTests PRIVATE
        PPOKIT_TEST_COMPONENTS="$<TARGET_FILE:ppokit_test_components>"
        PPOKIT_TEST_COMPONENTS_NAME="ppokit_test_components"
        PPOKIT_NOT_A_COMPONENT="$<TARGET_FILE:ppokit_not_a_component>"
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json"
        PPOKIT_OBSERVER_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/observer_task.json"
        PPOKIT_CORRIDOR_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/task.json")
    add_test(NAME PPOKitComponentTests COMMAND PPOKitComponentTests)

    # A Visualiseur DLL (observer + view) and the viewer session, without a window.
    add_library(ppokit_test_view MODULE tests/components/test_view.cpp)
    target_link_libraries(ppokit_test_view PRIVATE PPOKit::ComponentSdk PPOKit::Live)
    add_executable(PPOKitViewTests tests/view_tests.cpp)
    target_link_libraries(PPOKitViewTests PRIVATE PPOKit::View)
    add_dependencies(PPOKitViewTests ppokit_test_view)
    target_compile_definitions(PPOKitViewTests PRIVATE PPOKIT_TEST_VIEW="$<TARGET_FILE:ppokit_test_view>")
    add_test(NAME PPOKitViewTests COMMAND PPOKitViewTests)

    add_executable(PPOKitLiveChannelTests tests/live_channel_tests.cpp)
    target_link_libraries(PPOKitLiveChannelTests PRIVATE PPOKit::Live)
    add_test(NAME PPOKitLiveChannelTests COMMAND PPOKitLiveChannelTests)

    add_executable(PPOKitArtifactTests tests/artifact_tests.cpp)
    target_link_libraries(PPOKitArtifactTests PRIVATE PPOKit::Schema PPOKit::Core)
    set(_ppokit_corridor "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor")
    target_compile_definitions(PPOKitArtifactTests PRIVATE
        PPOKIT_CORRIDOR_TASK="${_ppokit_corridor}/task.json"
        PPOKIT_CORRIDOR_BLOCKS="${_ppokit_corridor}/graph_blocks.json"
        PPOKIT_CORRIDOR_CHECKPOINT="${_ppokit_corridor}/training.ppokit")
    add_test(NAME PPOKitArtifactTests COMMAND PPOKitArtifactTests)
    add_executable(PPOKitRandomStateTests tests/random_state_tests.cpp)
    target_include_directories(PPOKitRandomStateTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(NAME PPOKitRandomStateTests COMMAND PPOKitRandomStateTests)
    # The command line tool on the committed checkpoint.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME PPOKitLegacyContract COMMAND ${Python3_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/tools/ppokit/test_legacy_contract.py" $<TARGET_FILE:ppokit-artifact> "${_ppokit_corridor}")
    add_test(NAME PPOKitArtifactInspect COMMAND ppokit-artifact inspect "${_ppokit_corridor}/training.ppokit")
    set_tests_properties(PPOKitArtifactInspect PROPERTIES PASS_REGULAR_EXPRESSION "training checkpoint")
    add_test(NAME PPOKitArtifactCheck
        COMMAND ppokit-artifact check "${_ppokit_corridor}/training.ppokit" "${_ppokit_corridor}/task.json"
            "${_ppokit_corridor}/graph_blocks.json")
    set_tests_properties(PPOKitArtifactCheck PROPERTIES PASS_REGULAR_EXPRESSION "fits task.json and graph_blocks.json")

    add_executable(PPOKitCurriculumEditingTests tests/curriculum_editing_tests.cpp)
    target_link_libraries(PPOKitCurriculumEditingTests PRIVATE PPOKit::Editor)
    target_compile_definitions(PPOKitCurriculumEditingTests PRIVATE
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json"
        PPOKIT_SAMPLE_TRAIN="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_train.json")
    add_test(NAME PPOKitCurriculumEditingTests COMMAND PPOKitCurriculumEditingTests)

    add_executable(PPOKitProbesTests tests/probes_tests.cpp)
    target_link_libraries(PPOKitProbesTests PRIVATE PPOKit::Editor)
    target_compile_definitions(PPOKitProbesTests PRIVATE
        PPOKIT_CORRIDOR_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/task.json"
        PPOKIT_CORRIDOR_BLOCKS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/graph_blocks.json")
    add_test(NAME PPOKitProbesTests COMMAND PPOKitProbesTests)

    add_executable(PPOKitGraphDocumentTests tests/graph_document_tests.cpp)
    target_link_libraries(PPOKitGraphDocumentTests PRIVATE PPOKit::Editor)
    target_compile_definitions(PPOKitGraphDocumentTests PRIVATE
        PPOKIT_BLUE_TASK="${PROJECT_SOURCE_DIR}/Examples/BlueDefense/Agents/BluePatrol/task.json"
        PPOKIT_BLUE_GRAPH_BLOCKS="${PROJECT_SOURCE_DIR}/Examples/BlueDefense/Agents/BluePatrol/graph_blocks.json"
        PPOKIT_CORRIDOR_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/corridor/task.json"
        PPOKIT_OBSERVER_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/observer_task.json"
        PPOKIT_TEAM_GRAPH_PROJECT="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/team_graph"
        PPOKIT_TEST_COMPONENTS="$<TARGET_FILE:ppokit_test_components>")
    add_dependencies(PPOKitGraphDocumentTests ppokit_test_components)
    add_test(NAME PPOKitGraphDocumentTests COMMAND PPOKitGraphDocumentTests)
endif()

# --- Examples: what ppokit-new generates, built and tested like any example -------
add_subdirectory(examples/reach)

if (PPO_BUILD_CORE_TESTS)
    set(_ppokit_reach "${CMAKE_CURRENT_SOURCE_DIR}/examples/reach")
    # The committed example is exactly what ppokit-new writes today.
    add_test(NAME PPOKitScaffold
        COMMAND "${CMAKE_COMMAND}" -DGENERATOR=$<TARGET_FILE:ppokit-new>
            -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/scaffold -DEXPECTED=${_ppokit_reach}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompareScaffold.cmake")
    add_test(NAME PPOKitReachCheck COMMAND ppokit_reach check)
    set_tests_properties(PPOKitReachCheck PROPERTIES PASS_REGULAR_EXPRESSION "All files are valid")

    # check, train, resume, live weights, curriculum stage and eval, in process.
    add_executable(PPOKitCommandLineTests tests/command_line_tests.cpp)
    target_link_libraries(PPOKitCommandLineTests PRIVATE PPOKit::Core PPOKit::Schema PPOKit::Training PPOKit::Live)
    target_include_directories(PPOKitCommandLineTests PRIVATE "${_ppokit_reach}")
    target_compile_definitions(PPOKitCommandLineTests PRIVATE PPOKIT_REACH_DIR="${_ppokit_reach}")
    ppokit_generate_task_layout(TARGET PPOKitCommandLineTests TASK "${_ppokit_reach}/task.json"
        HEADER reach_layout.hpp)
    add_test(NAME PPOKitCommandLineTests COMMAND PPOKitCommandLineTests)

    # The reach project's network DLL against the in-process learner.
    ppokit_add_network_library(TARGET reach_network TASK "${_ppokit_reach}/task.json")
    add_executable(PPOKitNetworkLibraryTests tests/network_library_tests.cpp)
    target_link_libraries(PPOKitNetworkLibraryTests PRIVATE PPOKit::Network PPOKit::Core PPOKit::Schema
        PPOKit::Training PPOKit::Live)
    target_include_directories(PPOKitNetworkLibraryTests PRIVATE "${_ppokit_reach}")
    ppokit_generate_task_layout(TARGET PPOKitNetworkLibraryTests TASK "${_ppokit_reach}/task.json"
        HEADER reach_layout.hpp)
    # A team's network DLL (three agents) for the team API and ppokit-run's team loop.
    add_library(team_line_network SHARED tests/networks/team_line_network.cpp)
    target_link_libraries(team_line_network PRIVATE PPOKit::Core)
    ppokit_bind_task_contract(TARGET team_line_network TASK tests/fixtures/team_line_task.json)
    set_target_properties(team_line_network PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "")
    add_dependencies(PPOKitNetworkLibraryTests reach_network ppokit_graph_network ppokit_not_a_component team_line_network)
    target_compile_definitions(PPOKitNetworkLibraryTests PRIVATE PPOKIT_SHARED_GRAPH_NETWORK="$<TARGET_FILE:ppokit_graph_network>"
        PPOKIT_REACH_DIR="${_ppokit_reach}"
        PPOKIT_TEAM_LINE_NETWORK="$<TARGET_FILE:team_line_network>"
        PPOKIT_REACH_NETWORK="$<TARGET_FILE:reach_network>"
        PPOKIT_NOT_A_NETWORK="$<TARGET_FILE:ppokit_not_a_component>"
        PPOKIT_SAMPLE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_task.json")
    add_test(NAME PPOKitNetworkLibraryTests COMMAND PPOKitNetworkLibraryTests)

    # The team learner of the shared graph backend on a declarative team project.
    add_executable(PPOKitTeamGraphLearnerTests tests/team_graph_learner_tests.cpp)
    target_link_libraries(PPOKitTeamGraphLearnerTests PRIVATE PPOKit::Network PPOKit::Core PPOKit::Schema
        PPOKit::Training nlohmann_json::nlohmann_json)
    add_dependencies(PPOKitTeamGraphLearnerTests ppokit_graph_network)
    target_compile_definitions(PPOKitTeamGraphLearnerTests PRIVATE
        PPOKIT_SHARED_GRAPH_NETWORK="$<TARGET_FILE:ppokit_graph_network>"
        PPOKIT_TEAM_GRAPH_PROJECT="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/team_graph")
    add_test(NAME PPOKitTeamGraphLearnerTests COMMAND PPOKitTeamGraphLearnerTests)

    # ppokit-run on the reach environment DLL and network DLL, with an observer.
    add_library(reach_environment SHARED tests/environments/reach_environment.cpp)
    target_include_directories(reach_environment PRIVATE "${_ppokit_reach}")
    target_link_libraries(reach_environment PRIVATE PPOKit::Core)
    ppokit_bind_task_contract(TARGET reach_environment TASK "${_ppokit_reach}/task.json" HEADER reach_layout.hpp)
    set_target_properties(reach_environment PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "")
    # A team environment (three agent rows, two agents present) for ppokit-run's team loop.
    add_library(team_line_environment SHARED tests/environments/team_line_environment.cpp)
    target_link_libraries(team_line_environment PRIVATE PPOKit::Core)
    ppokit_bind_task_contract(TARGET team_line_environment TASK tests/fixtures/team_line_task.json)
    set_target_properties(team_line_environment PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "")
    add_executable(PPOKitSessionControlTests tests/session_control_tests.cpp)
    target_link_libraries(PPOKitSessionControlTests PRIVATE PPOKit::Live)
    add_test(NAME PPOKitSessionControlTests COMMAND PPOKitSessionControlTests)
    add_executable(PPOKitRunTests tests/ppokit_run_tests.cpp)
    target_link_libraries(PPOKitRunTests PRIVATE PPOKit::Live nlohmann_json::nlohmann_json)
    add_dependencies(PPOKitRunTests ppokit-run reach_environment ppokit_graph_network ppokit_test_components
        team_line_environment team_line_network)
    target_compile_definitions(PPOKitRunTests PRIVATE
        PPOKIT_RUN="$<TARGET_FILE:ppokit-run>"
        PPOKIT_TEAM_LINE_TASK="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/team_line_task.json"
        PPOKIT_REACH_DIR="${_ppokit_reach}"
        PPOKIT_REACH_ENVIRONMENT="$<TARGET_FILE:reach_environment>"
        PPOKIT_REACH_NETWORK="$<TARGET_FILE:ppokit_graph_network>"
        PPOKIT_TEAM_LINE_ENVIRONMENT="$<TARGET_FILE:team_line_environment>"
        PPOKIT_TEAM_LINE_NETWORK="$<TARGET_FILE:team_line_network>"
        PPOKIT_TEST_COMPONENTS="$<TARGET_FILE:ppokit_test_components>")
    add_test(NAME PPOKitRunTests COMMAND PPOKitRunTests)
endif()
