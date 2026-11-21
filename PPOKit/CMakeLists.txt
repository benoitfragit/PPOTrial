# PPOKit: framework shared by the PPOTrial examples.
# Design and migration plan: docs/architecture/ppokit-framework-design.md
# Progress journal for resuming the work: PPOKit/PROGRESS.md
#
# The public headers live under include/ppokit and never expose a backend
# (rl_tools, libtorch), a simulator or a GUI library.

# --- ppokit::artifacts : transactional artifact files ------------------------
add_library(ppokit_artifacts STATIC src/artifacts/atomic_file.cpp)
add_library(PPOKit::Artifacts ALIAS ppokit_artifacts)
target_include_directories(ppokit_artifacts PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_artifacts PUBLIC cxx_std_17)

if (PPO_BUILD_CORE_TESTS)
    add_executable(PPOKitAtomicFileTests tests/atomic_file_tests.cpp)
    target_link_libraries(PPOKitAtomicFileTests PRIVATE PPOKit::Artifacts)
    add_test(NAME PPOKitAtomicFileTests COMMAND PPOKitAtomicFileTests)
endif()
