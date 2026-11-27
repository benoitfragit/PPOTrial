# The reach project: add this folder with add_subdirectory() in a build that has PPOKit.
add_executable(ppokit_reach main.cpp)
target_link_libraries(ppokit_reach PRIVATE PPOKit::Core PPOKit::Schema PPOKit::Training PPOKit::Live)
target_compile_definitions(ppokit_reach PRIVATE PPOKIT_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# Sizes, channel and scenario names of task.json, regenerated whenever task.json changes.
ppokit_generate_task_layout(TARGET ppokit_reach TASK task.json HEADER reach_layout.hpp)
# ppokit-studio finds the program here to train, evaluate and play the project.
ppokit_register_program(ppokit_reach)
