# Reusable two-dimensional entities and their historical motion/zone semantics.
# Both Airspace and future examples may link SharedWorld directly.
add_library(SharedWorld STATIC
    data/mobile.hpp
    data/restrictedzone.hpp
    data/objective.hpp
    data/sharedworldcomponents.hpp
    data/snasystem.hpp
    data/snacontrolsystem.hpp
    data/flightmodelsystem.hpp
    src/mobile.cpp
    src/restrictedzone.cpp
    src/snasystem.cpp
    src/snacontrolsystem.cpp
    src/flightmodelsystem.cpp)
target_compile_features(SharedWorld PUBLIC cxx_std_17)
target_include_directories(SharedWorld PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(SharedWorld PUBLIC GenericEntities GenericSerializer EnTT::EnTT)
