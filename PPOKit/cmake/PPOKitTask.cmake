# CMake helpers that generate PPOKit headers at build time.
#
#   ppokit_generate_task_layout(TARGET <target> TASK <task.json> HEADER <layout.hpp>)
#       Constants of the task: offsets, sizes, names, shape hash.
#
#   ppokit_generate_networks(TARGET <target> TASK <task.json> GRAPH <graph.json> HEADER <networks.hpp>)
#       rl_tools types of the networks described in graph.json. Include the rl_tools
#       operation headers before the generated header.
#
#   ppokit_embed_files(TARGET <target> HEADER <files.hpp> NAMESPACE <ns> FILES <Name> <file> ...)
#       String constants holding the files (e.g. task.json and graph.json), so that a
#       program finds them without depending on its working directory.
#
#   ppokit_add_network_library(TARGET <name> TASK <task.json>)
#       The network DLL of a project (ppokit/network/network_abi.h): stable C API for
#       inference and training, the PPOKit graph engine inside. It reads network.json and
#       training.json of the project folder it is opened on, so it fits every graph of this
#       task. Registered like a program, so that ppokit-run and the studio find it by name.
#
# Each header is regenerated whenever one of its input files or the generator changes,
# and becomes includable by <target> (and by the consumers of an INTERFACE target):
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
    get_target_property(target_type ${ARG_TARGET} TYPE)
    if (target_type STREQUAL "INTERFACE_LIBRARY")
        # A header-only target compiles nothing itself: generate through a dedicated target
        # that every consumer depends on (dependencies of INTERFACE libraries need CMake 3.19).
        add_custom_target(${ARG_TARGET}_${ARG_HEADER}_generation DEPENDS "${output}")
        add_dependencies(${ARG_TARGET} ${ARG_TARGET}_${ARG_HEADER}_generation)
        target_include_directories(${ARG_TARGET} INTERFACE "${output_directory}")
    else()
        target_sources(${ARG_TARGET} PRIVATE "${output}")
        target_include_directories(${ARG_TARGET} PRIVATE "${output_directory}")
    endif()
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

function(ppokit_embed_files)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;HEADER;NAMESPACE" "FILES")
    list(LENGTH ARG_FILES count)
    math(EXPR odd "${count} % 2")
    if (NOT ARG_TARGET OR NOT ARG_HEADER OR NOT ARG_NAMESPACE OR count EQUAL 0 OR odd)
        message(FATAL_ERROR "ppokit_embed_files: TARGET, HEADER, NAMESPACE and FILES <Name> <file>... are required")
    endif()
    set(entries)
    set(inputs)
    math(EXPR last "${count} - 1")
    foreach(index RANGE 0 ${last} 2)
        math(EXPR next "${index} + 1")
        list(GET ARG_FILES ${index} name)
        list(GET ARG_FILES ${next} file)
        get_filename_component(file "${file}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        list(APPEND entries "${name}=${file}")
        list(APPEND inputs "${file}")
    endforeach()
    _ppokit_add_generated_header(TARGET ${ARG_TARGET} HEADER ${ARG_HEADER}
        INPUTS ${inputs} ARGUMENTS embed "${ARG_NAMESPACE}" ${entries})
endfunction()

# ppokit_register_program(<target>)
#     ppokit-studio launches the program of a project by name (project.json "executable").
#     A registered program is found wherever the build put it: its path is written to
#     <build>/ppokit_programs_<config>.txt by ppokit_write_program_registry(), called once
#     at the end of the configuration.
function(ppokit_register_program target)
    set_property(GLOBAL APPEND PROPERTY PPOKIT_PROGRAMS ${target})
endfunction()

function(ppokit_write_program_registry)
    get_property(programs GLOBAL PROPERTY PPOKIT_PROGRAMS)
    set(content "")
    foreach(program IN LISTS programs)
        string(APPEND content "$<TARGET_FILE:${program}>\n")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt" CONTENT "${content}")
endfunction()

# Attach the compiled task identity to an environment or native network DLL.
# HEADER lets a module reuse the layout it already includes.
function(ppokit_bind_task_contract)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;TASK;HEADER" "")
    if (NOT ARG_TARGET OR NOT ARG_TASK)
        message(FATAL_ERROR "ppokit_bind_task_contract: TARGET and TASK are required")
    endif()
    if (NOT ARG_HEADER)
        set(ARG_HEADER "${ARG_TARGET}_layout.hpp")
    endif()
    get_filename_component(task "${ARG_TASK}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    # The generated layout is in the task's namespace ("namespace", else ppokit::tasks::<name>).
    file(READ "${task}" task_text)
    string(JSON task_namespace ERROR_VARIABLE no_namespace GET "${task_text}" namespace)
    if (no_namespace)
        string(JSON task_name GET "${task_text}" name)
        set(task_namespace "ppokit::tasks::${task_name}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${task}")
    ppokit_generate_task_layout(TARGET ${ARG_TARGET} TASK "${task}" HEADER "${ARG_HEADER}")
    target_sources(${ARG_TARGET} PRIVATE "${PPOKIT_SOURCE_DIR}/src/component/module_task_contract.cpp")
    target_compile_definitions(${ARG_TARGET} PRIVATE
        PPOKIT_TASK_LAYOUT="${ARG_HEADER}" PPOKIT_TASK_NAMESPACE=${task_namespace})
endfunction()

function(ppokit_add_network_library)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "TARGET;TASK" "")
    if (NOT ARG_TARGET OR NOT ARG_TASK)
        message(FATAL_ERROR "ppokit_add_network_library: TARGET and TASK are required")
    endif()
    add_library(${ARG_TARGET} SHARED "${PPOKIT_SOURCE_DIR}/src/network/graph_network_library.cpp")
    ppokit_bind_task_contract(TARGET ${ARG_TARGET} TASK "${ARG_TASK}")
    target_compile_definitions(${ARG_TARGET} PRIVATE PPOKIT_PROGRAM_REGISTRY="${CMAKE_BINARY_DIR}/ppokit_programs_$<CONFIG>.txt")
    target_link_libraries(${ARG_TARGET} PRIVATE PPOKit::Core PPOKit::Schema PPOKit::Training Threads::Threads)
    # Only the C API is exported; "lib" is not prepended, so the file is named like the target.
    set_target_properties(${ARG_TARGET} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "")
    ppokit_register_program(${ARG_TARGET})
endfunction()
