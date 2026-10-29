cmake_minimum_required(VERSION 3.15)
project(MetaReflection LANGUAGES CXX)

add_library(${PROJECT_NAME}
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data/metaregistry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metaregistry.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GenericEntities
        GenericEnv
        LegacyAgentAPI
        EnTT::EnTT)

add_dependencies(${PROJECT_NAME}
    GenericEntities
    GenericEnv
    LegacyAgentAPI)

# Guard the abstraction boundary: generic reflection must compile and work
# without linking any Airspace agent, platform or trained model.
if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(GenericMetaContractTests tests/meta_contract_tests.cpp)
    target_compile_features(GenericMetaContractTests PRIVATE cxx_std_17)
    target_link_libraries(GenericMetaContractTests PRIVATE MetaReflection)
    add_test(NAME GenericMetaContractTests COMMAND GenericMetaContractTests)
endif()
