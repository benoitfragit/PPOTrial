# RLMission and RLSNA under ppokit-run (evaluation, play, training), on a copy of the Airspace folders (the projects keep
# their checkpoints in ../../models: the repository's are never written).
# Arguments: RUN (ppokit-run), SOURCE (repository root), WORK (scratch folder).
if(DEFINED ENV{PPOKIT_RUN_TEST_EXECUTABLE})
    set(RUN "$ENV{PPOKIT_RUN_TEST_EXECUTABLE}")
endif()
file(REMOVE_RECURSE "${WORK}")
foreach(folder Examples/Airspace/projects Examples/Airspace/models Examples/Airspace/settings Examples/Shared/settings)
    get_filename_component(parent "${WORK}/${folder}" DIRECTORY)
    file(COPY "${SOURCE}/${folder}" DESTINATION "${parent}")
endforeach()
set(projects "${WORK}/Examples/Airspace/projects")
set(models "${WORK}/Examples/Airspace/models")

# The evaluation suite of a project plays the evaluator's cases: one episode per case, each
# with its score. Its mean is pinned: the score of the eval policy given by the former
# in-world Evaluator (RLMission 465.708, RLSNA 5485.94, checked equal to 1e-4 before it was
# removed), to the digits of `expected`.
function(check_evaluation family expected)
    execute_process(COMMAND "${RUN}" eval --project "${projects}/${family}"
        --policy "${models}/${family}/eval/policy.ppokit"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
    if (NOT result EQUAL 0 OR NOT output MATCHES "100 episodes")
        message(FATAL_ERROR "${family} ppokit-run eval failed (${result}):\n${output}\n${errors}")
    endif()
    file(READ "${projects}/${family}/checkpoints/evaluation_campaign.json" report)
    string(JSON mean GET "${report}" aggregates score.mean)
    string(FIND "${mean}" "${expected}" at)
    if (NOT at EQUAL 0)
        message(FATAL_ERROR "${family}: score.mean ${mean}, the former Evaluator's notation was ${expected}...")
    endif()
    message(STATUS "${family}: score.mean ${mean}")
endfunction()

check_evaluation(RLMission "465.70")
check_evaluation(RLSNA "5485.93")

set(project "${projects}/RLMission")
set(training "${WORK}/Examples/Airspace/models/RLMission/training.ppokit")

# The converted eval policy plays two episodes. The first starts from the world's
# re-initialization like the next ones: on the scenario's raw state it ended at its first step.
execute_process(COMMAND "${RUN}" play --project "${project}" --episodes 2
    --policy "${WORK}/Examples/Airspace/models/RLMission/eval/policy.ppokit"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
if (NOT result EQUAL 0 OR NOT output MATCHES "episode 2: return")
    message(FATAL_ERROR "ppokit-run play failed (${result}):\n${output}\n${errors}")
endif()
if (output MATCHES "episode 1: return [^,]*, 1 steps")
    message(FATAL_ERROR "the first RLMission episode ended at its first step:\n${output}")
endif()

# Training resumes the converted checkpoint (model version 127) and runs one update.
execute_process(COMMAND "${RUN}" train --project "${project}" --updates 1
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
if (NOT result EQUAL 0 OR NOT output MATCHES "Resuming from [^\n]*training.ppokit at update 127")
    message(FATAL_ERROR "ppokit-run train failed (${result}):\n${output}\n${errors}")
endif()
file(READ "${training}" header LIMIT 4096)
if (NOT header MATCHES "\"updates\":128")
    message(FATAL_ERROR "the training checkpoint was not saved at update 128")
endif()
# RLSNA trains on its scenario: it resumes the converted checkpoint (model version 888).
set(training "${WORK}/Examples/Airspace/models/RLSNA/training.ppokit")
execute_process(COMMAND "${RUN}" train --project "${projects}/RLSNA" --updates 1
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
if (NOT result EQUAL 0 OR NOT output MATCHES "Resuming from [^\n]*training.ppokit at update 888")
    message(FATAL_ERROR "ppokit-run train RLSNA failed (${result}):\n${output}\n${errors}")
endif()
file(READ "${training}" header LIMIT 4096)
if (NOT header MATCHES "\"updates\":889")
    message(FATAL_ERROR "the RLSNA training checkpoint was not saved at update 889")
endif()
file(REMOVE_RECURSE "${WORK}")
