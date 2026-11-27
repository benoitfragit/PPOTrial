/**
 * @file main.cpp
 * @brief Program of the reach project: check, train, eval, play (run with "help").
 */

#include "task.hpp"

#include "ppokit/tools/command_line.hpp"

int main(int argc, char** argv)
{
    using namespace reach_example;
    // PPOKIT_EXAMPLE_DIR (set by CMakeLists.txt) is this project folder; the studio passes
    // --project to work on another copy.
    return ppokit::tools::RunCommandLine<ObservationSize, ActionSize>(argc, argv, MakeTask, PPOKIT_EXAMPLE_DIR);
}
