/**
 * @file PPOKit/tools/ppokit_new.cpp
 * @brief ppokit-new: creates a PPOKit project that already trains, to start from.
 * @details
 * @verbatim
 *   ppokit-new <name> [--into DIR]
 * @endverbatim
 * writes DIR/<name>/ (DIR defaults to the current folder); see ppokit/tools/scaffold.hpp.
 */

#include "ppokit/tools/scaffold.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void PrintUsage(std::ostream& stream)
{
    stream << "Usage: ppokit-new <name> [--into DIR]\n"
              "Creates DIR/<name>/ (DIR defaults to the current folder): a PPOKit project that\n"
              "already trains, to be changed into your own task. <name> uses lowercase letters,\n"
              "digits and '_', and starts with a letter (e.g. lander, patrol_2d).\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help")
    {
        PrintUsage(argc < 2 ? std::cerr : std::cout);
        return argc < 2 ? 2 : 0;
    }
    std::string const name = argv[1];
    std::filesystem::path into = ".";
    for (int index = 2; index < argc; ++index)
    {
        std::string const argument = argv[index];
        if (argument == "--into" && index + 1 < argc)
        {
            into = argv[++index];
            continue;
        }
        std::cerr << "ppokit-new: unknown argument '" << argument << "'\n";
        PrintUsage(std::cerr);
        return 2;
    }
    try
    {
        std::filesystem::path const folder = ppokit::tools::WriteNewProject(name, into);
        std::cout << "Created " << folder.string() << ".\n"
                  << "Next:\n"
                  << "  1. add it to your build: add_subdirectory(" << folder.generic_string() << ") and build ppokit_"
                  << name << "\n"
                  << "  2. open the folder in ppokit-studio: Train, Evaluate, Inference\n"
                  << "See " << (folder / "README.md").string() << " to make it your own task.\n";
    }
    catch (std::invalid_argument const& error)
    {
        std::cerr << "ppokit-new: " << error.what() << '\n';
        PrintUsage(std::cerr);
        return 2;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ppokit-new: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
