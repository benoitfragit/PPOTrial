add_library(GenericRLCore INTERFACE)
target_include_directories(GenericRLCore INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(GenericRLCore INTERFACE cxx_std_17)

if (PPO_BUILD_CORE_TESTS)
    add_executable(GenericRLCoreTests tests/core_tests.cpp)
    target_link_libraries(GenericRLCoreTests PRIVATE GenericRLCore)
    add_test(NAME GenericRLCoreTests COMMAND GenericRLCoreTests)
endif()
