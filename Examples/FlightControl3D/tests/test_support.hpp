#pragma once
#include "flight_environment.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

inline void Check(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
inline bool Near(double a, double b, double epsilon=1e-6) noexcept { return std::abs(a-b)<=epsilon; }
template<typename Function> void Rejects(Function&& operation, char const* message)
{
    bool rejected=false;
    try { operation(); } catch (std::exception const&) { rejected=true; }
    Check(rejected,message);
}
struct TemporaryDirectory final
{
    std::filesystem::path path;
    explicit TemporaryDirectory(char const* name)
        : path(std::filesystem::temp_directory_path()/(std::string{name}+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path,error); }
};
