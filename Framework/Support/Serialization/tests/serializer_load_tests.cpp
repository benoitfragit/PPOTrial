/**
 * @file serializer_load_tests.cpp
 * @brief The settings loader: a file that fails to load leaves its callbacks uncalled,
 *        search folders resolve in order, late registrations and Unregister.
 */
#include "serializer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using ppo::serialization::Serializer;
using ppo::serialization::json;

namespace {

void Write(fs::path const& path, std::string const& text)
{
    std::ofstream out(path, std::ios::binary);
    out << text;
}

bool Report(bool condition, char const* description)
{
    if (!condition)
        std::cerr << "FAILED: " << description << '\n';
    return condition;
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path()
        / ("ppo_settings_load_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    bool valid = true;
    try
    {
        fs::create_directories(root / "example");
        fs::create_directories(root / "shared");
        auto& serializer = Serializer::GetInstance();

        // Damaged JSON, a wrong format version, a missing file: no callback runs.
        Write(root / "example" / "broken.json", "{ damaged user configuration }");
        Write(root / "example" / "old.json", R"({"formatVersion":1,"value":9})");
        Write(root / "example" / "good.json", R"({"formatVersion":2,"value":5})");
        int calls = 0;
        int good = 0;
        auto const count = [&](json const&) { ++calls; };
        auto const broken = serializer.Register("broken.json", count);
        auto const old = serializer.Register("old.json", count);
        auto const missing = serializer.Register("missing.json", count);
        auto const read = serializer.Register("good.json", [&](json const& content) { good = content.at("value").get<int>(); });
        serializer.SetSearchPaths({root / "example", root / "shared"});
        serializer.Load();
        valid &= Report(calls == 0, "a file that fails to load calls no callback");
        valid &= Report(good == 5, "a valid file is read");

        // An example's own folder is searched before the shared one.
        Write(root / "example" / "both.json", R"({"formatVersion":2,"value":1})");
        Write(root / "shared" / "both.json", R"({"formatVersion":2,"value":2})");
        Write(root / "shared" / "shared.json", R"({"formatVersion":2,"value":3})");
        int both = 0;
        int shared = 0;
        auto const first = serializer.Register("both.json", [&](json const& content) { both = content.at("value").get<int>(); });
        auto const second = serializer.Register("shared.json", [&](json const& content) { shared = content.at("value").get<int>(); });
        serializer.Load();
        valid &= Report(both == 1 && shared == 3, "the search folders resolve in order");

        // A removed registration is not called.
        Write(root / "example" / "removed.json", R"({"formatVersion":2,"value":7})");
        int removed = 0;
        auto const gone = serializer.Register("removed.json", [&](json const& content) { removed = content.at("value").get<int>(); });
        serializer.Unregister(gone);
        serializer.Load();
        valid &= Report(removed == 0, "an unregistered callback is not called");
        valid &= Report(serializer.Register("none.json", nullptr) == 0U, "a null callback is not registered");

        for (auto const registration : {broken, old, missing, read, first, second})
            serializer.Unregister(registration);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        valid = false;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    return valid ? 0 : 1;
}
