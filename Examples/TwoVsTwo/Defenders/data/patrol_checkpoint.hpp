#pragma once
// Canonical checkpoint serialization shared by GUI learning and the
// optional headless utility. Preserves BLUE_PATROL_PPO_V2 file contract.
#include "patrol_rltools.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace ppo::examples::twovstwo::defenders {
template<typename MATRIX> void WritePatrolMatrix(std::ostream& os, MATRIX const& m)
{
    for (typename MATRIX::SPEC::TI i=0;i<MATRIX::SPEC::ROWS;++i)
        for (typename MATRIX::SPEC::TI j=0;j<MATRIX::SPEC::COLS;++j)
            os << rlt::get(m,i,j) << ' ';
    os << '\n';
}
template<typename LAYER> void WritePatrolLayer(std::ostream& os,LAYER const& layer)
{
    WritePatrolMatrix(os,layer.weights.parameters);
    WritePatrolMatrix(os,layer.biases.parameters);
}
template<typename BLOCK> void WritePatrolAttention(std::ostream& os,BLOCK const& block)
{
    WritePatrolMatrix(os,block.encoderW.parameters);
    WritePatrolMatrix(os,block.encoderB.parameters);
    WritePatrolMatrix(os,block.queryW.parameters);
    WritePatrolMatrix(os,block.queryB.parameters);
}
template<typename ACTOR> void WritePatrolPolicy(std::ostream& os,ACTOR const& actor)
{
    os << "BLUE_PATROL_PPO_V2 " << ObservationLayout::Version << ' '
       << ObservationLayout::Size << ' ' << ObservationLayout::ActionSize << ' '
       << TrainingParameters::ACTOR_HIDDEN_DIM << ' '
       << TrainingParameters::ACTOR_NUM_LAYERS << '\n'
       << std::setprecision(std::numeric_limits<float>::max_digits10);
    WritePatrolAttention(os,actor.red);
    WritePatrolAttention(os,actor.friendlies);
    WritePatrolLayer(os,actor.input_layer);
    for (auto const& layer:actor.hidden_layers) WritePatrolLayer(os,layer);
    WritePatrolLayer(os,actor.output_layer);
    WritePatrolMatrix(os,actor.log_std.parameters);
    if (!os) throw std::runtime_error("Failed to serialize patrol actor");
}
template<typename ACTOR>
void SavePatrolPolicy(std::filesystem::path const& path,ACTOR const& actor)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    auto const temp=path.string()+".tmp";
    auto const backup=path.string()+".bak";
    {
        std::ofstream file{temp,std::ios::trunc};
        if (!file) throw std::runtime_error("Cannot create patrol checkpoint");
        WritePatrolPolicy(file,actor);
        file.flush();
        if (!file) throw std::runtime_error("Cannot flush patrol checkpoint");
    }
    std::filesystem::remove(backup);
    bool const existed=std::filesystem::exists(path);
    if (existed) std::filesystem::rename(path,backup);
    try { std::filesystem::rename(temp,path); }
    catch (...) {
        if (existed) std::filesystem::rename(backup,path);
        throw;
    }
    if (existed) std::filesystem::remove(backup);
}
} // namespace ppo::examples::twovstwo::defenders
