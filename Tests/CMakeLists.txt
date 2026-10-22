add_executable(PolicyCheckpointTests model_checkpoint_tests.cpp)
target_link_libraries(PolicyCheckpointTests PRIVATE Interception Mission GenericPPO)
target_compile_features(PolicyCheckpointTests PRIVATE cxx_std_17)
target_compile_definitions(PolicyCheckpointTests PRIVATE
    PPO_MODEL_RESOURCE_DIR="${CMAKE_SOURCE_DIR}/resources/model")

# The existing LibTorch distribution contains runtime DLLs outside the target dir.
add_custom_command(TARGET PolicyCheckpointTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/libtorch/lib"
        "$<TARGET_FILE_DIR:PolicyCheckpointTests>")

add_test(NAME PolicyCheckpointTests COMMAND PolicyCheckpointTests)
