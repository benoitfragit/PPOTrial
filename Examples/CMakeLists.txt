add_subdirectory(Airspace)
add_subdirectory(BlueDefense)

# Explicit composition of compiled examples, separate from Main and the core.
add_library(PPOExampleCatalog STATIC
    data/example.hpp
    data/examplecatalog.hpp
    src/examplecatalog.cpp)
target_compile_features(PPOExampleCatalog PUBLIC cxx_std_17)
target_include_directories(PPOExampleCatalog PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOExampleCatalog PUBLIC PPOAirspaceExample PPOBlueDefenseExample)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    # Contract-only test: no LibTorch, renderer or concrete example needed.
    add_executable(ExampleCatalogTests tests/examplecatalog_tests.cpp)
    target_compile_features(ExampleCatalogTests PRIVATE cxx_std_17)
    target_include_directories(ExampleCatalogTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME ExampleCatalogTests COMMAND ExampleCatalogTests)
endif()
