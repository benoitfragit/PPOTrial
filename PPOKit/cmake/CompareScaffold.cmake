# cmake -DGENERATOR=<ppokit-new> -DOUTPUT=<dir> -DEXPECTED=<examples/reach> -P CompareScaffold.cmake
# Regenerates the "reach" example and fails when a file differs from the committed one
# (line endings aside: a Windows checkout may have converted them).
file(REMOVE_RECURSE "${OUTPUT}")
execute_process(COMMAND "${GENERATOR}" reach --into "${OUTPUT}" RESULT_VARIABLE result)
if (NOT result EQUAL 0)
    message(FATAL_ERROR "ppokit-new failed (${result})")
endif()
file(GLOB_RECURSE generated RELATIVE "${OUTPUT}/reach" "${OUTPUT}/reach/*" "${OUTPUT}/reach/.*")
list(REMOVE_DUPLICATES generated)
foreach(name IN LISTS generated)
    file(READ "${OUTPUT}/reach/${name}" made)
    if (NOT EXISTS "${EXPECTED}/${name}")
        message(FATAL_ERROR "examples/reach/${name} is missing: run ppokit-new reach --into PPOKit/examples")
    endif()
    file(READ "${EXPECTED}/${name}" committed)
    string(REPLACE "\r\n" "\n" committed "${committed}")
    if (NOT made STREQUAL committed)
        message(FATAL_ERROR "examples/reach/${name} differs from ppokit-new's output: "
            "regenerate it (ppokit-new reach --into PPOKit/examples) or update the template")
    endif()
endforeach()
message(STATUS "examples/reach matches ppokit-new (${generated})")
