# Red production policies of the Airspace example, without LibTorch: the RLMission track
# attention and the RLSNA MLP, read from the policy.txt files converted from model.pt.
add_library(AirspacePolicies STATIC
    legacy_policy_file.cpp
    mission_policy.cpp
    interception_policy.cpp)
target_compile_features(AirspacePolicies PUBLIC cxx_std_17)
target_include_directories(AirspacePolicies PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../Mission")
target_link_libraries(AirspacePolicies PUBLIC GenericRLCore)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(LegacyPolicyTests "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/legacy_policy_tests.cpp")
    target_link_libraries(LegacyPolicyTests PRIVATE AirspacePolicies)
    target_compile_definitions(LegacyPolicyTests PRIVATE
        PPO_REPOSITORY="${CMAKE_SOURCE_DIR}"
        PPO_LEGACY_POLICY_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/legacy_policy_reference.txt")
    add_test(NAME LegacyPolicyTests COMMAND LegacyPolicyTests)
endif()
