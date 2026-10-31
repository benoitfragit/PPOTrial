/**
 * @file serializer_safety_tests.cpp
 * @brief Guard user JSON against accidental replacement after load errors.
 */
#include "serializer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;
using ppo::serialization::Serializer;
using ppo::serialization::json;

namespace {

/**
 * @brief Sérialise ou écrit les données associées à Write selon les garanties implémentées.
 * @details Utilise des fichiers ou chemins.
 */
void Write(fs::path const& path, std::string const& text)
{
    std::ofstream out(path, std::ios::binary);
    out << text;
}

std::string Read(fs::path const& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

} // namespace

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main()
{
    fs::path const root = fs::temp_directory_path()
        / ("ppo_settings_save_safety_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    bool valid = false;
    try
    {
        fs::create_directories(root);
        std::string const damaged = "{ damaged user configuration: do not overwrite }";
        Write(root / "broken.json", damaged);
        Write(root / "good.json", R"({"formatVersion":2,"value":5})");

        int value = 0;
        auto& serializer = Serializer::GetInstance();
        serializer.SetPath(root);
        serializer.Register("broken.json",
            [](json const& content) { (void)content.at("value"); },
            [](json& content) { content["value"] = -1; });
        serializer.Register("good.json",
            [&](json const& content) { value = content.at("value").get<int>(); },
            [&](json& content) { content["value"] = value; });

        serializer.Load();
        serializer.Save();

        json const good = json::parse(Read(root / "good.json"));
        valid = Read(root / "broken.json") == damaged
             && good.at("value").get<int>() == 5
             && good.at("formatVersion").get<int>() == 2;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Serializer safety exception: " << e.what() << '\n';
    }

    std::error_code ignored;
    fs::remove_all(root, ignored);

    if (!valid)
        std::cerr << "Serializer incorrectly rewrote failed input or changed valid settings.\n";
    return valid ? 0 : 1;
}
