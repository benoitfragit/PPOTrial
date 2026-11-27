# PPOKit studio: block graph editor, training dashboard and PPO settings (ImGui views, raylib window).
# Built with the desktop application, which provides imgui, imnodes, raylib and rlImGui.

add_library(ppokit_studio_view STATIC curriculum_view.cpp file_dialogs.cpp graph_editor_view.cpp help_tooltip.cpp
    evaluation_view.cpp scenario_view.cpp launch_settings.cpp overview_view.cpp project_window.cpp start_view.cpp studio_app.cpp studio_icons.cpp studio_theme.cpp
    training_dashboard_view.cpp train_settings_view.cpp)
target_include_directories(ppokit_studio_view PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ppokit_studio_view PUBLIC PPOKit::Editor PPOKit::Live PPOKit::Artifacts PPOKit::Scaffold PPOKit::Network imgui imnodes implot)
if (WIN32)
    # Native folder and file pickers (IFileOpenDialog).
    target_link_libraries(ppokit_studio_view PRIVATE ole32 shell32)
endif()
target_compile_features(ppokit_studio_view PUBLIC cxx_std_17)

add_executable(ppokit-studio main.cpp)
target_link_libraries(ppokit-studio PRIVATE ppokit_studio_view raylib rlImGui)
# Where the studio finds the programs of the projects (ppokit_register_program).
target_compile_definitions(ppokit-studio PRIVATE
    PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    set(_ppokit_blue "${CMAKE_SOURCE_DIR}/Examples/BlueDefense/Agents/BluePatrol")
    # The view runs headless: ImGui and imnodes frames without a renderer.
    add_executable(PPOKitStudioViewTests tests/graph_editor_view_tests.cpp)
    target_link_libraries(PPOKitStudioViewTests PRIVATE ppokit_studio_view)
    target_compile_definitions(PPOKitStudioViewTests PRIVATE
        PPOKIT_BLUE_TASK="${_ppokit_blue}/task.json"
        PPOKIT_BLUE_GRAPH_BLOCKS="${_ppokit_blue}/graph_blocks.json"
        PPOKIT_CORRIDOR_TASK="${CMAKE_SOURCE_DIR}/PPOKit/tests/fixtures/corridor/task.json"
        PPOKIT_TEST_COMPONENTS="$<TARGET_FILE:ppokit_test_components>")
    add_dependencies(PPOKitStudioViewTests ppokit_test_components)
    add_test(NAME PPOKitStudioViewTests COMMAND PPOKitStudioViewTests)
    # The dashboard runs headless too, on a journal written like a live run.
    add_executable(PPOKitStudioDashboardTests tests/training_dashboard_view_tests.cpp)
    target_link_libraries(PPOKitStudioDashboardTests PRIVATE ppokit_studio_view)
    add_test(NAME PPOKitStudioDashboardTests COMMAND PPOKitStudioDashboardTests)
    # The PPO settings editor runs headless as well.
    add_executable(PPOKitStudioSettingsTests tests/train_settings_view_tests.cpp)
    target_link_libraries(PPOKitStudioSettingsTests PRIVATE ppokit_studio_view)
    add_test(NAME PPOKitStudioSettingsTests COMMAND PPOKitStudioSettingsTests)
    # The curriculum editor, headless, on the sample task and curriculum.
    # The project window and its overview run headless on a copy of the reach project.
    add_executable(PPOKitStudioProjectTests tests/project_window_tests.cpp)
    target_link_libraries(PPOKitStudioProjectTests PRIVATE ppokit_studio_view nlohmann_json::nlohmann_json)
    target_compile_definitions(PPOKitStudioProjectTests PRIVATE
        PPOKIT_REPOSITORY_ROOT="${CMAKE_SOURCE_DIR}"
        PPOKIT_STUDIO_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt"
        PPOKIT_REACH_PROJECT="${CMAKE_SOURCE_DIR}/PPOKit/examples/reach"
        PPOKIT_FLIGHT_PC7_PROJECT="${CMAKE_SOURCE_DIR}/Examples/FlightControl3D/projects/PC7"
        PPOKIT_REACH_PROGRAM="$<TARGET_FILE:ppokit_reach>"
        PPOKIT_REACH_ENVIRONMENT="$<TARGET_FILE:reach_environment>"
        PPOKIT_REACH_NETWORK="$<TARGET_FILE:reach_network>"
        PPOKIT_RUN_PROGRAM="$<TARGET_FILE:ppokit-run>")
    add_dependencies(PPOKitStudioProjectTests ppokit_reach ppokit-run reach_environment reach_network)
    if (TARGET flight_environment)
        target_compile_definitions(PPOKitStudioProjectTests PRIVATE PPOKIT_FLIGHT_ENVIRONMENT="$<TARGET_FILE:flight_environment>")
        add_dependencies(PPOKitStudioProjectTests flight_environment)
    endif()
    if (TARGET blue_environment)
        target_compile_definitions(PPOKitStudioProjectTests PRIVATE
            PPOKIT_BLUE_ENVIRONMENT="$<TARGET_FILE:blue_environment>"
            PPOKIT_BLUE_VISUALISER="$<TARGET_FILE:blue_visualiser>")
        add_dependencies(PPOKitStudioProjectTests blue_environment blue_visualiser)
    endif()
    add_test(NAME PPOKitStudioProjectTests COMMAND PPOKitStudioProjectTests)

    # The evaluation suites editor, headless, on a scratch eval/ folder.
    add_executable(PPOKitStudioEvaluationTests tests/evaluation_view_tests.cpp)
    target_link_libraries(PPOKitStudioEvaluationTests PRIVATE ppokit_studio_view)
    add_test(NAME PPOKitStudioEvaluationTests COMMAND PPOKitStudioEvaluationTests)

    add_executable(PPOKitStudioCurriculumTests tests/curriculum_view_tests.cpp)
    target_link_libraries(PPOKitStudioCurriculumTests PRIVATE ppokit_studio_view)
    target_compile_definitions(PPOKitStudioCurriculumTests PRIVATE
        PPOKIT_SAMPLE_TASK="${CMAKE_SOURCE_DIR}/PPOKit/tests/fixtures/sample_task.json"
        PPOKIT_SAMPLE_TRAIN="${CMAKE_SOURCE_DIR}/PPOKit/tests/fixtures/sample_train.json")
    add_test(NAME PPOKitStudioCurriculumTests COMMAND PPOKitStudioCurriculumTests)
    # The command line check validates a graph without a window.
    add_test(NAME PPOKitStudioCheckBlueGraph
        COMMAND ppokit-studio --check "${_ppokit_blue}/task.json" "${_ppokit_blue}/graph_blocks.json")
    set_tests_properties(PPOKitStudioCheckBlueGraph PROPERTIES PASS_REGULAR_EXPRESSION "^valid: ")
endif()
