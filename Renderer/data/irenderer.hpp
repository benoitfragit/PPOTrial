#pragma once

namespace ppo::renderer {

class IRenderer
{
public:
    virtual ~IRenderer(void) = default;

    virtual void Run(void) noexcept = 0;
};

} /* ppo::renderer */