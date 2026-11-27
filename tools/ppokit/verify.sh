#!/usr/bin/env bash
# Verification of every PPOKit migration step on Linux (no desktop).
#
#   tools/ppokit/verify.sh            build Debug and Release, run all guards
#   tools/ppokit/verify.sh release    one configuration only
#
# Guards:
#   1. CTest in each configuration (includes the Flight golden trace).
#   2. BlueDefense tests that only the Windows desktop build compiles
#      (checkpoints, attention, MAPPO learner), built by hand in build/release
#      as AGENTS.md allows only build/debug and build/release.
#   3. Bit-exact MAPPO learner trace compared with the reference recorded on master.
#
# The first configuration builds JSBSim (about 15 minutes on two cores).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
if [[ $# -eq 0 ]]; then
    configurations=(debug release)
else
    configurations=("$@")
fi

configure_and_test() {
    local name="$1" type
    type="$(tr '[:lower:]' '[:upper:]' <<< "${name:0:1}")${name:1}"
    if [[ ! -f "build/$name/CMakeCache.txt" ]]; then
        cmake -S . -B "build/$name" -G Ninja -DCMAKE_BUILD_TYPE="$type" \
            -DPPO_BUILD_DESKTOP=OFF -DPPO_BUILD_CORE_TESTS=ON -DPPO_BUILD_FLIGHT_CONTROL=ON
    fi
    cmake --build "build/$name" -j"$(nproc)"
    (cd "build/$name" && ctest --output-on-failure)
}

blue_manual_checks() {
    local backend=Examples/BlueDefense/Agents/BluePatrol deps=build/release/_deps out=build/release/manual_checks
    mkdir -p "$out"
    # The actor embeds task.json and graph_blocks.json, as the desktop build does.
    mkdir -p "$out/generated"
    build/release/PPOKit/ppokit-codegen embed ppo::examples::bluedefense::defenders::embedded \
        "BlueTaskJson=$backend/task.json" "BlueGraphBlocksJson=$backend/graph_blocks.json" \
        "$out/generated/blue_actor_graph_files.hpp" || { echo "FAILED: Blue actor graph embedding"; exit 1; }
    local include=(-IPPOKit/include -I"$backend/Core" -I"$backend/Runtime" -I"$backend/Training"
        -IExamples/BlueDefense/Evaluation
        -I"$deps/rltools-src/include" -I"$deps/json-src/include" -I"$out/generated"
        -Ibuild/release/BlueDefenseCoreTests/ppokit_generated/BlueDefenseNetworkTypeTests)
    for test in generalized_actor_tests attention_tests generalized_learner_tests graph_actor_parity_tests; do
        # rl_tools prints informational #pragma messages: keep them in a log.
        g++ -std=c++17 -O2 -w "${include[@]}" "Examples/BlueDefense/tests/$test.cpp" \
            PPOKit/src/artifacts/atomic_file.cpp build/release/PPOKit/libppokit_schema.a \
            build/release/PPOKit/libppokit_component.a -ldl \
            -o "$out/$test" 2> "$out/$test.build.log" \
            || { cat "$out/$test.build.log"; echo "FAILED: Blue $test does not build"; exit 1; }
        (cd /tmp && "$root/$out/$test") > "$out/$test.log" 2>&1 \
            || { echo "FAILED: Blue $test (see $out/$test.log)"; exit 1; }
        echo "Blue $test passed"
    done
    # Wall-clock timings are the only non-deterministic fields of the trace.
    sed -E 's/(_ms|elapsed_ms)=[0-9]+/\1=X/g' "$out/generalized_learner_tests.log" > "$out/blue_learner_trace.txt"
    if diff -u PPOKit/tests/reference/blue_learner_trace_linux_gcc13.txt "$out/blue_learner_trace.txt"; then
        echo "Blue MAPPO learner trace identical to the recorded reference"
    else
        echo "FAILED: Blue MAPPO learner trace changed"; exit 1
    fi
}

# The studio views need Dear ImGui 1.92.1, imnodes and ImPlot, fetched only by the desktop
# build. PPOKIT_STUDIO_DEPS may point to a folder with imgui/, imnodes/ and implot/
# checkouts to run their headless tests here: git clone --branch v1.92.1-docking
# ocornut/imgui, Nelarius/imnodes at 2d0c280, --branch v0.17 epezent/implot.
studio_manual_checks() {
    local deps="$PPOKIT_STUDIO_DEPS" out=build/release/manual_checks/studio
    local blue="$root/Examples/BlueDefense/Agents/BluePatrol"
    mkdir -p "$out"
    # Third-party sources without warning flags; ours with -Wall -Wextra and no warning allowed.
    local objects=() source object
    for source in "$deps"/imgui/imgui.cpp "$deps"/imgui/imgui_draw.cpp "$deps"/imgui/imgui_tables.cpp \
        "$deps"/imgui/imgui_widgets.cpp "$deps"/imnodes/imnodes.cpp "$deps"/implot/implot.cpp \
        "$deps"/implot/implot_items.cpp; do
        object="$out/$(basename "$source" .cpp).o"
        [[ -f "$object" ]] || g++ -std=c++17 -O1 -c -I"$deps/imgui" "$source" -o "$object" \
            || { echo "FAILED: $source does not build"; exit 1; }
        objects+=("$object")
    done
    local libraries=(build/release/PPOKit/libppokit_editor.a build/release/PPOKit/libppokit_live.a build/release/PPOKit/libppokit_training.a
        build/release/PPOKit/libppokit_schema.a build/release/PPOKit/libppokit_component.a
        build/release/PPOKit/libppokit_artifacts.a build/release/PPOKit/libppokit_scaffold.a -ldl -lpthread)
    local views=(PPOKit/studio/curriculum_view.cpp PPOKit/studio/evaluation_view.cpp PPOKit/studio/file_dialogs.cpp
        PPOKit/studio/graph_editor_view.cpp
        PPOKit/studio/help_tooltip.cpp PPOKit/studio/scenario_view.cpp PPOKit/studio/launch_settings.cpp PPOKit/studio/overview_view.cpp PPOKit/studio/project_window.cpp
        PPOKit/studio/start_view.cpp PPOKit/studio/studio_app.cpp PPOKit/studio/studio_icons.cpp PPOKit/studio/studio_theme.cpp
        PPOKit/studio/training_dashboard_view.cpp PPOKit/studio/train_settings_view.cpp)
    local view
    for view in graph_editor_view training_dashboard_view train_settings_view curriculum_view evaluation_view \
        project_window; do
        g++ -std=c++17 -O1 -Wall -Wextra -IPPOKit/include -IPPOKit/studio -I"$deps/imgui" -I"$deps/imnodes" \
            -I"$deps/implot" -Ibuild/release/_deps/json-src/include \
            -DPPOKIT_BLUE_TASK="\"$blue/task.json\"" -DPPOKIT_BLUE_GRAPH_BLOCKS="\"$blue/graph_blocks.json\"" \
            -DPPOKIT_SAMPLE_TASK="\"$root/PPOKit/tests/fixtures/sample_task.json\"" \
            -DPPOKIT_SAMPLE_TRAIN="\"$root/PPOKit/tests/fixtures/sample_train.json\"" \
            -DPPOKIT_CORRIDOR_TASK="\"$root/PPOKit/tests/fixtures/corridor/task.json\"" \
            -DPPOKIT_TEST_COMPONENTS="\"$root/build/release/PPOKit/libppokit_test_components.so\"" \
            -DPPOKIT_REACH_PROJECT="\"$root/PPOKit/examples/reach\"" \
            -DPPOKIT_FLIGHT_PC7_PROJECT="\"$root/Examples/FlightControl3D/projects/PC7\"" \
            -DPPOKIT_REACH_ENVIRONMENT="\"$root/build/release/PPOKit/reach_environment.so\"" \
            -DPPOKIT_REACH_NETWORK="\"$root/build/release/PPOKit/reach_network.so\"" \
            -DPPOKIT_RUN_PROGRAM="\"$root/build/release/PPOKit/ppokit-run\"" \
            -DPPOKIT_REACH_PROGRAM="\"$root/build/release/PPOKit/examples/reach/ppokit_reach\"" \
            "${views[@]}" "PPOKit/studio/tests/${view}_tests.cpp" "${objects[@]}" "${libraries[@]}" \
            -o "$out/${view}_tests" 2> "$out/$view.build.log" \
            || { cat "$out/$view.build.log"; echo "FAILED: studio $view tests do not build"; exit 1; }
        if grep -q "warning" "$out/$view.build.log"; then
            cat "$out/$view.build.log"; echo "FAILED: studio $view sources raise warnings"; exit 1
        fi
        "$out/${view}_tests" || { echo "FAILED: studio $view tests"; exit 1; }
    done
}

for configuration in "${configurations[@]}"; do
    configure_and_test "$configuration"
done
if [[ -d build/release/_deps/rltools-src ]]; then
    blue_manual_checks
else
    echo "Blue manual checks skipped: configure build/release first"
fi
if [[ -n "${PPOKIT_STUDIO_DEPS:-}" ]]; then
    studio_manual_checks
else
    echo "Studio view checks skipped: set PPOKIT_STUDIO_DEPS (desktop CI runs them)"
fi
echo "All PPOKit guards passed"
