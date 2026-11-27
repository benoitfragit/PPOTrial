add_subdirectory(Scenario/Entities Entities)
add_subdirectory(Scenario/Initialization Initializer)
add_subdirectory(Evaluation)
add_subdirectory(Agents/Interception)
add_subdirectory(Agents/Mission Agents/mission)

add_library(PPOAirspaceExample STATIC
    Scenario/airspaceexample.hpp
    Scenario/airspaceexample.cpp
    Scenario/airspaceworldrules.hpp
    Scenario/airspaceworldrules.cpp
    Scenario/airspacemeta.hpp
    Scenario/airspacemeta.cpp)

target_compile_features(PPOAirspaceExample PUBLIC cxx_std_17)
target_include_directories(PPOAirspaceExample PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Scenario")
target_link_libraries(PPOAirspaceExample PUBLIC
    GenericEntities
    AirspaceEntities
    GenericEnv
    EntityInitializer
    MetaReflection
    AirspaceEvaluation
    Interception
    Mission)

# Validate that historical outcomes and published components remain unchanged.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(AirspaceWorldRulesTests
        tests/airspace_world_rules_tests.cpp)
    target_compile_features(AirspaceWorldRulesTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceWorldRulesTests PRIVATE PPOAirspaceExample)
    add_test(NAME AirspaceWorldRulesTests COMMAND AirspaceWorldRulesTests)

    add_executable(MissionTrackSelectionTests
        tests/mission_track_selection_tests.cpp)
    target_compile_features(MissionTrackSelectionTests PRIVATE cxx_std_17)
    target_include_directories(MissionTrackSelectionTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Agents/Mission")
    target_link_libraries(MissionTrackSelectionTests PRIVATE GenericEntities)
    add_test(NAME MissionTrackSelectionTests COMMAND MissionTrackSelectionTests)
endif()

# Registration must preserve the same concrete factory and reflection names.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(AirspaceMetaTests tests/airspace_meta_tests.cpp)
    target_compile_features(AirspaceMetaTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceMetaTests PRIVATE PPOAirspaceExample)
    add_test(NAME AirspaceMetaTests COMMAND AirspaceMetaTests)
endif()

# ppokit-run: the Airspace world with its RLMission training agent driven from outside
# (environment DLL), and the shared declarative graph network backend.
add_library(rlmission_environment SHARED Environment/rlmission_environment.cpp)
target_link_libraries(rlmission_environment PRIVATE PPOAirspaceExample PPOKit::Core PPOKit::Schema)
set_target_properties(rlmission_environment PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    PREFIX "")
ppokit_bind_task_contract(TARGET rlmission_environment TASK Agents/Mission/task.json)
ppokit_register_program(rlmission_environment)
# RLSNA trains on its scenario (Scenarios/RLSNATraining.json), plays and evaluates on the evaluator's cases.
add_library(rlsna_environment SHARED Environment/rlsna_environment.cpp)
target_link_libraries(rlsna_environment PRIVATE PPOAirspaceExample PPOKit::Core PPOKit::Schema)
set_target_properties(rlsna_environment PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    PREFIX "")
ppokit_bind_task_contract(TARGET rlsna_environment TASK Agents/Interception/task.json)
ppokit_register_program(rlsna_environment)
# The Airspace Visualiseur: an observer block of network.json (both projects) and a 2D tactical view.
add_library(airspace_visualiser MODULE Visualiser/airspace_visualiser.cpp)
target_link_libraries(airspace_visualiser PRIVATE PPOKit::ComponentSdk PPOKit::Live)
set_target_properties(airspace_visualiser PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    PREFIX "")
ppokit_register_program(airspace_visualiser)
if (TARGET PPOKitRepositoryProjectsTests)
    # RLMission's and RLSNA's network.json name the Visualiseur: the repository check opens them.
    add_dependencies(PPOKitRepositoryProjectsTests airspace_visualiser)
    target_compile_definitions(PPOKitRepositoryProjectsTests PRIVATE PPOKIT_AIRSPACE_LIBRARIES=1)
endif()

if (TARGET PPOKitPublishedModelsTests)
    add_dependencies(PPOKitPublishedModelsTests rlmission_environment rlsna_environment airspace_visualiser)
    foreach(agent IN ITEMS RLMission RLSNA)
        add_test(NAME PPOKitPublishedModels_${agent} COMMAND PPOKitPublishedModelsTests
            "${CMAKE_CURRENT_SOURCE_DIR}/projects/${agent}" "${CMAKE_CURRENT_BINARY_DIR}/published_models_${agent}" resume)
    endforeach()
endif()

# Legacy LibTorch checkpoints (exported by Tools/export_legacy_checkpoint.py) -> PPOKit artifacts.
add_executable(airspace_checkpoint_import Tools/airspace_checkpoint_import.cpp)
target_link_libraries(airspace_checkpoint_import PRIVATE Interception Mission GenericRLBindings
    nlohmann_json::nlohmann_json)
target_compile_features(airspace_checkpoint_import PRIVATE cxx_std_17)

# The converted PPOKit policies act as the legacy networks of policy.txt.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(AirspaceConvertedPolicyTests tests/converted_policy_tests.cpp)
    target_link_libraries(AirspaceConvertedPolicyTests PRIVATE Interception Mission GenericRLBindings AirspacePolicies)
    target_compile_features(AirspaceConvertedPolicyTests PRIVATE cxx_std_17)
    target_compile_definitions(AirspaceConvertedPolicyTests PRIVATE
        PPO_REPOSITORY="${CMAKE_SOURCE_DIR}"
        PPO_LEGACY_POLICY_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/legacy_policy_reference.txt")
    add_test(NAME AirspaceConvertedPolicyTests COMMAND AirspaceConvertedPolicyTests)
    # ppokit-run plays RLSNA and RLMission with the Airspace Visualiseur, the view draws them.
    add_executable(AirspaceVisualiserTests tests/visualiser_tests.cpp)
    target_compile_features(AirspaceVisualiserTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceVisualiserTests PRIVATE PPOKit::View nlohmann_json::nlohmann_json)
    add_dependencies(AirspaceVisualiserTests ppokit-run rlmission_environment rlsna_environment
        ppokit_graph_network airspace_visualiser)
    target_compile_definitions(AirspaceVisualiserTests PRIVATE
        PPOKIT_RUN="$<TARGET_FILE:ppokit-run>"
        PPO_AIRSPACE_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}"
        PPO_SHARED_SETTINGS="${CMAKE_SOURCE_DIR}/Examples/Shared/settings"
        RLMISSION_ENVIRONMENT_LIBRARY="$<TARGET_FILE:rlmission_environment>"
        RLMISSION_NETWORK_LIBRARY="$<TARGET_FILE:ppokit_graph_network>"
        RLSNA_ENVIRONMENT_LIBRARY="$<TARGET_FILE:rlsna_environment>"
        RLSNA_NETWORK_LIBRARY="$<TARGET_FILE:ppokit_graph_network>"
        AIRSPACE_VISUALISER_LIBRARY="$<TARGET_FILE:airspace_visualiser>")
    add_test(NAME AirspaceVisualiserTests COMMAND AirspaceVisualiserTests)

    # The reward RLMission and RLSNA train on under ppokit-run (no last-step mission failure).
    add_executable(AirspaceTrainingRewardTests tests/training_reward_tests.cpp)
    target_compile_features(AirspaceTrainingRewardTests PRIVATE cxx_std_17)
    target_link_libraries(AirspaceTrainingRewardTests PRIVATE PPOKit::Network nlohmann_json::nlohmann_json)
    add_dependencies(AirspaceTrainingRewardTests rlmission_environment rlsna_environment)
    add_test(NAME AirspaceTrainingRewardTests COMMAND AirspaceTrainingRewardTests
        "$<TARGET_FILE:rlmission_environment>" "${CMAKE_CURRENT_SOURCE_DIR}/projects/RLMission"
        "${CMAKE_CURRENT_SOURCE_DIR}/settings/Training/RLMission.json")
    add_test(NAME AirspaceTrainingRewardTests_RLSNA COMMAND AirspaceTrainingRewardTests
        "$<TARGET_FILE:rlsna_environment>" "${CMAKE_CURRENT_SOURCE_DIR}/projects/RLSNA"
        "${CMAKE_CURRENT_SOURCE_DIR}/settings/Training/RLSNA.json")

    # ppokit-run evaluates RLMission and RLSNA, then plays and trains RLMission.
    add_test(NAME Airspace_ppokit_run_smoke
        COMMAND ${CMAKE_COMMAND} -DRUN=$<TARGET_FILE:ppokit-run> -DSOURCE=${CMAKE_SOURCE_DIR}
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/ppokit_run_smoke
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/ppokit_run_smoke.cmake)
    set_tests_properties(Airspace_ppokit_run_smoke PROPERTIES TIMEOUT 900)
endif()

# The red policies (policy.ppokit / policy.txt) load and answer without training.
if (PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(PolicyCheckpointTests tests/policy_checkpoint_tests.cpp)
    target_link_libraries(PolicyCheckpointTests PRIVATE Interception Mission GenericRLBindings)
    target_compile_features(PolicyCheckpointTests PRIVATE cxx_std_17)
    add_test(NAME PolicyCheckpointTests COMMAND PolicyCheckpointTests)
endif()
