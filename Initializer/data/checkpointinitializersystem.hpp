#pragma once

#include "entity.hpp"

namespace ppo::model {

static constexpr int const CHECKPOINT_DIMENSION = 6;
using CheckPointBuffer = std::array<float, CHECKPOINT_DIMENSION>;

struct sEntityCheckPoint {
    std::string                                 m_name{ "" };
    int                                         m_index{ 0 };
    std::vector<ppo::model::CheckPointBuffer>   m_buffers{};
};

struct sEvaluatorName {
    std::string  m_filename{ "" };
};

class CheckpointInitializerSystem 
{
public:
    static void Load(Entity& entity);
    static void Init(Entity& entity);
};

} /* ppo::model */