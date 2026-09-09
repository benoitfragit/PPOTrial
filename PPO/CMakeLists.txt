# Historical simulator compatibility wrapper, not the generic RL target.
add_library(GenericPPO INTERFACE)
target_include_directories(GenericPPO INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/data)
target_link_libraries(GenericPPO INTERFACE ppo_torch GenericEntities)
