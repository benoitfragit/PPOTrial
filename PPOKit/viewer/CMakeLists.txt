# ppokit-viewer: a window on the world of a run (Visualiseur views fed in UDP), with raylib.
# Built with the desktop application, which provides raylib.
add_executable(ppokit-viewer main.cpp raylib_draw.cpp raylib_draw.hpp)
target_link_libraries(ppokit-viewer PRIVATE PPOKit::View raylib)
target_compile_features(ppokit-viewer PRIVATE cxx_std_17)
# The studio launches it from an observer's "View" button.
ppokit_register_program(ppokit-viewer)
