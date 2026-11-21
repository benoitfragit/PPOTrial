#!/usr/bin/env bash
# Verification of every PPOKit migration step on Linux (no desktop, no libtorch).
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
    local include=(-IPPOKit/include -I"$backend/Core" -I"$backend/Runtime" -I"$backend/Training"
        -IExamples/BlueDefense/Evaluation -IFramework/RL/Core/include
        -I"$deps/rltools-src/include" -I"$deps/json-src/include")
    for test in generalized_actor_tests attention_tests generalized_learner_tests; do
        # rl_tools prints informational #pragma messages: keep them in a log.
        g++ -std=c++17 -O2 -w "${include[@]}" "Examples/BlueDefense/tests/$test.cpp" \
            PPOKit/src/artifacts/atomic_file.cpp -o "$out/$test" 2> "$out/$test.build.log" \
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

for configuration in "${configurations[@]}"; do
    configure_and_test "$configuration"
done
if [[ -d build/release/_deps/rltools-src ]]; then
    blue_manual_checks
else
    echo "Blue manual checks skipped: configure build/release first"
fi
echo "All PPOKit guards passed"
