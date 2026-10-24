add_library(PPOExistingExample STATIC
    data/existingexample.hpp
    src/existingexample.cpp)

target_compile_features(PPOExistingExample PUBLIC cxx_std_17)
target_include_directories(PPOExistingExample PUBLIC
    "${CMAKE_SOURCE_DIR}/Examples/data"
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(PPOExistingExample PUBLIC
    GenericEntities
    GenericEnv
    MetaReflection)
