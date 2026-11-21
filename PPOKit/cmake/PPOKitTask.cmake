# CMake helpers that generate PPOKit headers at build time.
#
#   ppokit_generate_task_layout(TARGET <target> TASK <task.json> HEADER <layout.hpp>)
#       Constants of the task: offsets, sizes, names, shape hash.
#
#   ppokit_generate_networks(TARGET <target> TASK <task.json> GRAPH <graph.json> HEADER <networks.hpp>)
#       rl_tools types of the networks described in graph.json. Include the rl_tools
#       operation headers before the generated header.
#
# Each header is regenerated whenever one of its input files or the generator changes,
# and becomes includable by <target>:
#
#   ppokit_generate_task_layout(TARGET MyExampleTests TASK task.json HEADER my_task_layout.hpp)
#   // in C++: #include "my_task_layout.hpp"

# _ppokit_add_generated_header(TARGET <t> HEADER <h> INPUTS <files...> ARGUMENTS <codegen arguments...>)
# Adds the generation command and makes the output a source of the target, which
# orders the generation before compilation.
function(_ppokit_add_generated_header)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;HEADER" "INPUTS;ARGUMENTS")
    set(output_directory "${CMAKE_CURRENT_BINARY_DIR}/ppokit_generated/${ARG_TARGET}")
    set(output "${output_directory}/${ARG_HEADER}")
    add_custom_command(
        OUTPUT "${output}"
        COMMAND ppokit-codegen ${ARG_ARGUMENTS} "${output}"
        DEPENDS ppokit-codegen ${ARG_INPUTS}
        COMMENT "Generating ${ARG_HEADER}"
        VERBATIM)
    target_sources(${ARG_TARGET} PRIVATE "${output}")
    target_include_directories(${ARG_TARGET} PRIVATE "${output_directory}")
endfunction()

function(ppokit_generate_task_layout)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;TASK;HEADER" "")
    if (NOT ARG_TARGET OR NOT ARG_TASK OR NOT ARG_HEADER)
        message(FATAL_ERROR "ppokit_generate_task_layout: TARGET, TASK and HEADER are required")
    endif()
    get_filename_component(task "${ARG_TASK}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    _ppokit_add_generated_header(TARGET ${ARG_TARGET} HEADER ${ARG_HEADER}
        INPUTS "${task}" ARGUMENTS layout "${task}")
endfunction()

function(ppokit_generate_networks)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;TASK;GRAPH;HEADER" "")
    if (NOT ARG_TARGET OR NOT ARG_TASK OR NOT ARG_GRAPH OR NOT ARG_HEADER)
        message(FATAL_ERROR "ppokit_generate_networks: TARGET, TASK, GRAPH and HEADER are required")
    endif()
    get_filename_component(task "${ARG_TASK}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    get_filename_component(graph "${ARG_GRAPH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    _ppokit_add_generated_header(TARGET ${ARG_TARGET} HEADER ${ARG_HEADER}
        INPUTS "${task}" "${graph}" ARGUMENTS networks "${task}" "${graph}")
endfunction()
