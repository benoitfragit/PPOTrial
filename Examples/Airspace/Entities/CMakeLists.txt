# Domain-specific ECS entities, dynamics and rules for Airspace only.
# Keep GenericEntities free of these concrete types.
add_library(AirspaceEntities STATIC
    data/airspacecomponents.hpp
    data/aircraft.hpp
    data/sam.hpp
    data/objective.hpp
    data/snasystem.hpp
    data/snacontrolsystem.hpp
    data/flightmodelsystem.hpp
    src/aircraft.cpp
    src/sam.cpp
    src/snasystem.cpp
    src/snacontrolsystem.cpp
    src/flightmodelsystem.cpp)

target_compile_features(AirspaceEntities PUBLIC cxx_std_17)
target_include_directories(AirspaceEntities PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(AirspaceEntities PUBLIC
    GenericEntities
    GenericSerializer
    EnTT::EnTT)
