/**
 * @file PPOKit/tools/ppokit_artifact.cpp
 * @brief ppokit-artifact: what a checkpoint holds, and whether it fits a task and a graph.
 *
 * Usage: ppokit-artifact inspect <checkpoint>
 *        ppokit-artifact check <checkpoint> <task.json> <graph.json>
 *
 * `check` exits with 0 when the checkpoint loads with these files, 1 otherwise, and says
 * why in plain words (the same message the learner gives).
 */
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/component/native_module.hpp"
#include <fstream>

#include <iomanip>
#include <iostream>
#include <string>

namespace {

void PrintList(char const* title, std::vector<std::string> const& values)
{
    std::cout << "  " << std::left << std::setw(14) << title;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        std::cout << (index == 0 ? "" : ", ") << values[index];
    }
    std::cout << '\n';
}

int Inspect(std::filesystem::path const& path)
{
    auto const header = ppokit::artifacts::ReadArtifactHeader(path);
    auto const& identity = header.identity;
    std::cout << path.filename().string() << ": " << ppokit::artifacts::ArtifactKindName(header.kind) << " checkpoint\n"
              << "  task          " << identity.task << " (observation " << identity.observationSize << ", actions "
              << identity.actionSize << ")\n"
              << "  networks      " << identity.backend << " / " << identity.scalar << '\n'
              << "  updates       " << header.updates << "\n"
              << "  seed          " << header.seed << "\n"
              << "  written       " << header.created << '\n';
    PrintList("actor", identity.actorBlocks);
    PrintList("critic", identity.criticBlocks);
    for (auto const& [component, hash] : identity.components)
    {
        std::cout << "  component     " << component << " (" << std::hex << hash << std::dec << ")\n";
    }
    return 0;
}

int Check(std::filesystem::path const& path, std::filesystem::path const& taskPath, std::filesystem::path const& graphPath)
{
    auto const task = ppokit::schema::LoadTaskSchema(taskPath);
    auto const graphs = ppokit::schema::LoadBlockGraph(graphPath, task);
    auto const header = ppokit::artifacts::ReadArtifactHeader(path);
    try
    {
        ppokit::artifacts::RequireCompatible(
            header.identity, ppokit::schema::ArtifactIdentityOf(task, graphs, header.identity.backend),
            path.filename().string());
    }
    catch (ppokit::artifacts::ArtifactMismatch const& mismatch)
    {
        std::cout << mismatch.what() << '\n';
        return 1;
    }
    std::cout << path.filename().string() << ": fits " << taskPath.filename().string() << " and "
              << graphPath.filename().string() << " (" << ppokit::artifacts::ArtifactKindName(header.kind)
              << ", " << header.updates << " updates)\n";
    return 0;
}

/** @brief Bind legacy weights to their original contract, without changing the payload. */
int Migrate(std::filesystem::path const& input, std::filesystem::path const& taskPath,
    std::filesystem::path const& graphPath, std::filesystem::path const& output)
{
    auto header = ppokit::artifacts::ReadArtifactHeader(input);
    if (header.identity.semanticHash || std::filesystem::exists(output))
    {
        throw std::runtime_error("migration requires a legacy source and a new output file");
    }
    auto const task = ppokit::schema::LoadTaskSchema(taskPath);
    auto const graphs = ppokit::schema::LoadBlockGraph(graphPath, task);
    auto const identity = ppokit::schema::ArtifactIdentityOf(task, graphs, header.identity.backend);
    // The caller supplies the archived training definitions. Dimensions alone cannot recover them.
    header.identity.semanticHash = identity.semanticHash;
    ppokit::artifacts::RequireCompatible(header.identity, identity, input.string());
    if (header.identity.contract.empty()) header.identity.contract = identity.contract;
    std::ifstream source(input, std::ios::binary);
    std::string oldHeader;
    if (!std::getline(source, oldHeader)) throw std::runtime_error("missing source header");
    ppokit::artifacts::WriteFileAtomically(output, [&](std::ostream& out) {
        out << ppokit::artifacts::HeaderLine(header) << source.rdbuf();
        if (source.bad() || !out) throw std::runtime_error("could not copy the checkpoint payload");
    }, ppokit::artifacts::AtomicWriteOptions::Binary(), "legacy contract migration");
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        ppokit::component::SetKnownModules(ppokit::schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
        std::string const command = argc > 1 ? argv[1] : "";
        if (command == "inspect" && argc == 3)
        {
            return Inspect(argv[2]);
        }
        if (command == "check" && argc == 5)
        {
            return Check(argv[2], argv[3], argv[4]);
        }
        if (command == "migrate" && argc == 6)
        {
            return Migrate(argv[2], argv[3], argv[4], argv[5]);
        }
        std::cerr << "usage: ppokit-artifact inspect <checkpoint>\n"
                     "       ppokit-artifact check <checkpoint> <task.json> <graph.json>\n"
                     "       ppokit-artifact migrate <legacy> <original-task.json> <original-graph.json> <new-file>\n";
        return 2;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ppokit-artifact: " << error.what() << '\n';
        return 1;
    }
}
