# Reusable two-dimensional entities and their original motion/zone semantics.
# Airspace and new examples depend on this library, never the other way round.
add_library(SharedWorld STATIC
    data/mobile.hpp
    data/restrictedzone.hpp
    data/objective.hpp
    data/sharedworldcomponents.hpp
    data/restrictedzonegeometry.hpp
    data/restrictedzonelidar.hpp
    data/snasystem.hpp
    data/snacontrolsystem.hpp
    data/flightmodelsystem.hpp
    src/mobile.cpp
    src/restrictedzone.cpp
    src/snasystem.cpp
    src/restrictedzonegeometry.cpp
    src/restrictedzonelidar.cpp
    src/snacontrolsystem.cpp
    src/flightmodelsystem.cpp)
target_compile_features(SharedWorld PUBLIC cxx_std_17)
target_include_directories(SharedWorld PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(SharedWorld PUBLIC GenericEntities GenericSerializer EnTT::EnTT)

if (PPO_BUILD_CORE_TESTS OR PPO_BUILD_CHECKPOINT_TESTS)
    add_executable(SharedWorldLidarTests tests/restrictedzone_lidar_tests.cpp)
    target_compile_features(SharedWorldLidarTests PRIVATE cxx_std_17)
    target_link_libraries(SharedWorldLidarTests PRIVATE SharedWorld)
    add_test(NAME SharedWorldLidarTests COMMAND SharedWorldLidarTests)
endif()
