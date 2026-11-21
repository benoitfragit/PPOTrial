/**
 * @file PPOKit/codegen/main.cpp
 * @brief Command-line generator of task layout headers.
 *
 *     ppokit-codegen <task.json> <output.hpp>
 *
 * The build system runs it only when the task file or the generator changed; the
 * header is then always rewritten, so that its timestamp tells the build tool the
 * generation is up to date. Errors are printed as "<file>: <key path>: <message>".
 */
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/schema/task_codegen.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {

/** @brief Writes the generated header text as is. */
struct HeaderWriter final
{
    std::string const& text;
    void operator()(std::ostream& stream) const { stream << text; }
};

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: ppokit-codegen <task.json> <output.hpp>\n";
        return 2;
    }
    std::filesystem::path const taskFile = argv[1];
    std::filesystem::path const output = argv[2];
    try
    {
        auto const schema = ppokit::schema::LoadTaskSchema(taskFile);
        std::string const header = ppokit::schema::GenerateTaskLayoutHeader(schema, taskFile.filename().string());
        ppokit::artifacts::AtomicWriteOptions options;
        options.mode = std::ios::binary;
        ppokit::artifacts::WriteFileAtomically(output, HeaderWriter{header}, options, "ppokit-codegen");
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "ppokit-codegen: " << error.what() << '\n';
        return 1;
    }
}
