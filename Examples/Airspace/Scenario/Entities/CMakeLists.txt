# Legacy Airspace target kept as a compatibility boundary.
# Shared model implementations are built only once in SharedWorld.
add_library(AirspaceEntities INTERFACE)
target_include_directories(AirspaceEntities INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(AirspaceEntities INTERFACE SharedWorld)
