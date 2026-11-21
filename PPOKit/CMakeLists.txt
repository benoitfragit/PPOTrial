# PPOKit: framework shared by the PPOTrial examples.
# Design and migration plan: docs/architecture/ppokit-framework-design.md
# Progress journal for resuming the work: PPOKit/PROGRESS.md
#
# The public headers live under include/ppokit and never expose a backend
# (rl_tools, libtorch), a simulator or a GUI library.

# --- ppokit::core and ppokit::heads : algorithmic building blocks (header-only)
add_library(ppokit_core INTERFACE)
add_library(PPOKit::Core ALIAS ppokit_core)
target_include_directories(ppokit_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_core INTERFACE cxx_std_17)

# --- ppokit::artifacts : transactional artifact files ------------------------
add_library(ppokit_artifacts STATIC src/artifacts/atomic_file.cpp)
add_library(PPOKit::Artifacts ALIAS ppokit_artifacts)
target_include_directories(ppokit_artifacts PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ppokit_artifacts PUBLIC cxx_std_17)

if (PPO_BUILD_CORE_TESTS)
    add_executable(PPOKitAtomicFileTests tests/atomic_file_tests.cpp)
    target_link_libraries(PPOKitAtomicFileTests PRIVATE PPOKit::Artifacts)
    add_test(NAME PPOKitAtomicFileTests COMMAND PPOKitAtomicFileTests)

    add_executable(PPOKitGaeTests tests/gae_tests.cpp)
    target_link_libraries(PPOKitGaeTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitGaeTests COMMAND PPOKitGaeTests)

    add_executable(PPOKitHeadsTests tests/heads_tests.cpp)
    target_link_libraries(PPOKitHeadsTests PRIVATE PPOKit::Core)
    add_test(NAME PPOKitHeadsTests COMMAND PPOKitHeadsTests)
endif()
