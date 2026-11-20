#pragma once
#include "flight_environment.hpp"
#include "flight_rng_checkpoint.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

inline void Check(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
inline bool Near(double a, double b, double epsilon=1e-6) noexcept { return std::abs(a-b)<=epsilon; }
/** @brief Compare every model/Adam byte and future RNG draws across native stream representations. */
inline bool SameLearnerCheckpoint(std::filesystem::path const& first,std::filesystem::path const& second)
{
    std::ifstream a(first),b(second);
    std::string left,right;
    for(int line=0;line<3;++line)
        if(!std::getline(a,left) || !std::getline(b,right) || left!=right) return false;
    auto rngA=ppo::examples::flight::detail::ReadCheckpointRng(a);
    auto rngB=ppo::examples::flight::detail::ReadCheckpointRng(b);
    for(int draw=0;draw<4096;++draw) if(rngA()!=rngB()) return false;
    std::string const remainingA((std::istreambuf_iterator<char>(a)),{});
    std::string const remainingB((std::istreambuf_iterator<char>(b)),{});
    return remainingA==remainingB;
}
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
