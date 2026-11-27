# The reach project: its DLL contains the simulation only; ppokit-run owns PPO.
add_library(reach_environment SHARED environment.cpp)
target_link_libraries(reach_environment PRIVATE PPOKit::Core nlohmann_json::nlohmann_json)
set_target_properties(reach_environment PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "")
# Sizes, channel and scenario names of task.json, regenerated whenever task.json changes.
ppokit_bind_task_contract(TARGET reach_environment TASK task.json HEADER reach_layout.hpp)
ppokit_register_program(reach_environment)
