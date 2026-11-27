# Exercise the offline importer and compare the committed T-6 independently against V3.
set(t6 "${FLIGHT}/projects/T6")
set(source "${FLIGHT}/tests/fixtures/master_t6texan2_joint_v5_flight_policy.txt")
set(level "${t6}/checkpoints/ppokit_curriculum_level.txt")
file(REMOVE_RECURSE "${WORK}")
execute_process(COMMAND "${IMPORT}" "${source}" "${level}" "${t6}" "${WORK}" centreline1200 COMMAND_ERROR_IS_FATAL ANY)
foreach(directory IN ITEMS "${WORK}" "${t6}/checkpoints")
    foreach(file IN ITEMS latest policy)
        execute_process(COMMAND "${PYTHON}" "${AUDIT}" --profile centreline1200 "${source}" "${directory}/${file}.ppokit"
            "${level}" "${directory}/ppokit_curriculum_level.txt" COMMAND_ERROR_IS_FATAL ANY)
    endforeach()
endforeach()
execute_process(COMMAND "${IMPORT}" "${source}" "${level}" "${FLIGHT}/projects/PC7" "${WORK}-pc7"
    RESULT_VARIABLE result ERROR_VARIABLE message)
if(result EQUAL 0 OR NOT message MATCHES "signature differs")
    message(FATAL_ERROR "cross-aircraft conversion was not refused by its contract: ${message}")
endif()

# Independent C++ implementations, one update from the same archived state and fixed transitions.
execute_process(COMMAND "${FROZEN}" --frozen "${t6}" "${source}" "${WORK}" "${WORK}/next"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${PYTHON}" "${AUDIT}" --profile centreline1200 --frozen-update "${WORK}/next/flight.txt"
    "${WORK}/next/latest.ppokit" "${level}" "${WORK}/ppokit_curriculum_level.txt" COMMAND_ERROR_IS_FATAL ANY)

execute_process(COMMAND "${IMPORT}" "${source}" "${level}" "${t6}" "${WORK}/wrong-guidance"
    RESULT_VARIABLE mismatch ERROR_VARIABLE explanation)
if(mismatch EQUAL 0 OR NOT explanation MATCHES "guidance profile differs")
    message(FATAL_ERROR "an undeclared current-guidance archive was accepted: ${explanation}")
endif()

# Keep the original 2 km / level 11 behavior executable after master changes its defaults.
set(archived "${WORK}/archived-project")
file(COPY "${t6}/" DESTINATION "${archived}" PATTERN "checkpoints" EXCLUDE)
file(COPY_FILE "${FLIGHT}/tests/fixtures/t6_340_scenario.json" "${archived}/scenario.json")
file(READ "${archived}/task.json" old_task)
string(REPLACE "@8-centreline1200" "@8" old_task "${old_task}")
file(WRITE "${archived}/task.json" "${old_task}")
set(old_v3 "${FLIGHT}/tests/fixtures/t6_340_flight_policy.txt")
set(old_level "${FLIGHT}/tests/fixtures/t6_340_level.txt")
execute_process(COMMAND "${IMPORT}" "${old_v3}" "${old_level}" "${archived}" "${archived}/checkpoints"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${PYTHON}" "${AUDIT}" "${old_v3}" "${archived}/checkpoints/latest.ppokit"
    "${old_level}" "${archived}/checkpoints/ppokit_curriculum_level.txt" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${FROZEN}" "${archived}" "${old_v3}" "${archived}/checkpoints"
    "${FLIGHT}/tests/fixtures/t6_340_best.json" COMMAND_ERROR_IS_FATAL ANY)
