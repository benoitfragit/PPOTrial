# Shared rendering interface only; concrete Airspace drawing lives in its example.
add_library(GenericRenderer INTERFACE)
target_compile_features(GenericRenderer INTERFACE cxx_std_17)
target_include_directories(GenericRenderer INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(GenericRenderer INTERFACE EnTT::EnTT)
