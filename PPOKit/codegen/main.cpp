/**
 * @file PPOKit/codegen/main.cpp
 * @brief Command-line generator of PPOKit headers.
 *
 *     ppokit-codegen layout   <task.json>              <output.hpp>
 *     ppokit-codegen networks <task.json> <graph.json> <output.hpp>
 *
 * `layout` writes the constants of the task (offsets, names, hash); `networks` writes
 * the rl_tools types of the networks described in graph.json. The build system runs
 * the generator only when an input changed, and the header is then always rewritten
 * so that its timestamp records the generation. Errors are printed as
 * "<file>: <key path>: <message>".
 */
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/schema/network_schema.hpp"
#include "ppokit/schema/task_codegen.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

/** @brief Writes the generated header text as is. */
struct HeaderWriter final
{
    std::string const& text;
    void operator()(std::ostream& stream) const { stream << text; }
};

void WriteHeader(std::filesystem::path const& output, std::string const& header)
{
    ppokit::artifacts::AtomicWriteOptions options;
    options.mode = std::ios::binary;
    ppokit::artifacts::WriteFileAtomically(output, HeaderWriter{header}, options, "ppokit-codegen");
}

int PrintUsage()
{
    std::cerr << "usage: ppokit-codegen layout <task.json> <output.hpp>\n"
                 "       ppokit-codegen networks <task.json> <graph.json> <output.hpp>\n";
    return 2;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        return PrintUsage();
    }
    std::string const command = argv[1];
    try
    {
        if (command == "layout" && argc == 4)
        {
            std::filesystem::path const taskFile = argv[2];
            auto const task = ppokit::schema::LoadTaskSchema(taskFile);
            WriteHeader(argv[3], ppokit::schema::GenerateTaskLayoutHeader(task, taskFile.filename().string()));
            return 0;
        }
        if (command == "networks" && argc == 5)
        {
            std::filesystem::path const graphFile = argv[3];
            auto const task = ppokit::schema::LoadTaskSchema(argv[2]);
            auto const networks = ppokit::schema::LoadNetworkSchema(graphFile, task);
            WriteHeader(argv[4],
                ppokit::schema::GenerateNetworksHeader(networks, task, graphFile.filename().string()));
            return 0;
        }
        return PrintUsage();
    }
    catch (std::exception const& error)
    {
        std::cerr << "ppokit-codegen: " << error.what() << '\n';
        return 1;
    }
}
