include(FetchContent)
# Restore the original project's EnTT version only in the example dependency graph.
if(NOT TARGET EnTT::EnTT)
    FetchContent_Declare(entt
        GIT_REPOSITORY https://github.com/skypjack/entt.git
        GIT_TAG v3.16.0)
    FetchContent_MakeAvailable(entt)
endif()
add_library(navigation_environment STATIC src/navigation.cpp)
target_include_directories(navigation_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(navigation_environment PUBLIC ppo_environment EnTT::EnTT)
set_target_properties(navigation_environment PROPERTIES CXX_EXTENSIONS OFF)
if(BUILD_TESTING)
    add_executable(navigation_tests tests.cpp)
    target_link_libraries(navigation_tests PRIVATE navigation_environment)
    add_test(NAME navigation_runtime COMMAND navigation_tests)
    add_test(NAME navigation_scenarios COMMAND navigation_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/scenarios/default.nav"
        "${CMAKE_CURRENT_SOURCE_DIR}/scenarios/multiple.nav")
endif()
