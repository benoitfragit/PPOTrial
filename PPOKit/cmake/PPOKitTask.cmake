# ppokit_generate_task_layout(TARGET <target> TASK <task.json> HEADER <file.hpp>)
#
# Generates <file.hpp> from a PPOKit task file and makes it includable by <target>.
# The header is regenerated whenever the task file or the generator changes:
#
#   ppokit_generate_task_layout(TARGET MyExampleTests TASK task.json HEADER my_task_layout.hpp)
#   // in C++: #include "my_task_layout.hpp"
function(ppokit_generate_task_layout)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;TASK;HEADER" "")
    if (NOT ARG_TARGET OR NOT ARG_TASK OR NOT ARG_HEADER)
        message(FATAL_ERROR "ppokit_generate_task_layout: TARGET, TASK and HEADER are required")
    endif()
    get_filename_component(task "${ARG_TASK}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(output_directory "${CMAKE_CURRENT_BINARY_DIR}/ppokit_generated/${ARG_TARGET}")
    set(header "${output_directory}/${ARG_HEADER}")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND ppokit-codegen "${task}" "${header}"
        DEPENDS ppokit-codegen "${task}"
        COMMENT "Generating ${ARG_HEADER} from ${ARG_TASK}"
        VERBATIM)
    # Listing the header as a source orders the generation before compilation.
    target_sources(${ARG_TARGET} PRIVATE "${header}")
    target_include_directories(${ARG_TARGET} PRIVATE "${output_directory}")
endfunction()
