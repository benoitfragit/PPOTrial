#pragma once

namespace ppo::environment {

class IRenderer
{
public:
    virtual ~IRenderer(void) = default;

    virtual void Run(void) noexcept = 0;
};

} /* ppo::environment */