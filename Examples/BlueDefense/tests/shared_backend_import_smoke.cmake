# The committed BluePatrol checkpoints convert to the shared graph backend: the tool reloads
# each result and checks it (same state, same means for 1..9 blues) before keeping it.
# Arguments: IMPORT, FIXTURE (shared backend), SHIPPED, CHECKPOINTS, WORK (scratch folder).
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}/prod")
file(WRITE "${WORK}/prod/policy.ppokit.check" "existing temporary file")
# Keep graph declaration order: it determines the checkpoint's parameter order.
# A named visualiser component must resolve through the build registry.
file(MAKE_DIRECTORY "${WORK}/project")
file(COPY "${SHIPPED}/task.json" "${FIXTURE}/training.json" DESTINATION "${WORK}/project")
file(READ "${FIXTURE}/network.json" shared_network)
string(REPLACE "\"features\": \"critic_features\"," "\"features\": \"critic_features\", \"pad_features\": true,"
    shared_network "${shared_network}")
string(REGEX REPLACE "}[ \t\r\n]*$" "" shared_network "${shared_network}")
file(WRITE "${WORK}/project/network.json" "${shared_network},\n  \"components\": [\"blue_visualiser\"]\n}\n")
execute_process(COMMAND "${IMPORT}" --project "${WORK}/project"
        --training "${CHECKPOINTS}/latest/latest.ppokit" --to-training "${WORK}/latest/latest.ppokit"
        --policy "${CHECKPOINTS}/latest/policy.ppokit" --to-policy "${WORK}/latest/policy.ppokit"
        --policy "${CHECKPOINTS}/prod/policy.ppokit" --to-policy "${WORK}/prod/policy.ppokit"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
if (NOT result EQUAL 0 OR NOT output MATCHES "update 5250, same state on reload")
    message(FATAL_ERROR "blue_shared_backend_import failed (${result}):\n${output}\n${errors}")
endif()
file(READ "${WORK}/prod/policy.ppokit.check" old_temporary)
if (NOT old_temporary STREQUAL "existing temporary file")
    message(FATAL_ERROR "conversion changed an existing .check file")
endif()
if (EXISTS "${WORK}/latest/latest.ppokit.conversion" OR EXISTS "${WORK}/prod/policy.ppokit.conversion")
    message(FATAL_ERROR "conversion left its staging directory behind")
endif()
# Nothing is overwritten: a second conversion to the same files is refused.
execute_process(COMMAND "${IMPORT}" --project "${FIXTURE}"
        --policy "${CHECKPOINTS}/prod/policy.ppokit" --to-policy "${WORK}/prod/policy.ppokit"
    RESULT_VARIABLE again ERROR_VARIABLE refused)
if (again EQUAL 0 OR NOT refused MATCHES "nothing is overwritten")
    message(FATAL_ERROR "an existing destination was not refused: ${refused}")
endif()
file(MAKE_DIRECTORY "${WORK}/blocked.ppokit.conversion")
file(WRITE "${WORK}/blocked.ppokit.conversion/sentinel" "keep")
execute_process(COMMAND "${IMPORT}" --project "${FIXTURE}"
        --policy "${CHECKPOINTS}/prod/policy.ppokit" --to-policy "${WORK}/blocked.ppokit"
    RESULT_VARIABLE blocked ERROR_VARIABLE blocked_error)
file(READ "${WORK}/blocked.ppokit.conversion/sentinel" sentinel)
if (blocked EQUAL 0 OR NOT blocked_error MATCHES "nothing is overwritten"
        OR NOT sentinel STREQUAL "keep" OR EXISTS "${WORK}/blocked.ppokit")
    message(FATAL_ERROR "an existing staging directory was not preserved: ${blocked_error}")
endif()
execute_process(COMMAND "${IMPORT}" --project "${FIXTURE}"
        --policy "${CHECKPOINTS}/prod/policy.ppokit" --policy "${CHECKPOINTS}/latest/policy.ppokit"
        --to-policy "${WORK}/ambiguous.ppokit"
    RESULT_VARIABLE ambiguous ERROR_VARIABLE ambiguous_error)
if (ambiguous EQUAL 0 OR NOT ambiguous_error MATCHES "matching --to-policy"
        OR EXISTS "${WORK}/ambiguous.ppokit")
    message(FATAL_ERROR "ambiguous policy arguments were not refused: ${ambiguous_error}")
endif()
file(REMOVE_RECURSE "${WORK}")
