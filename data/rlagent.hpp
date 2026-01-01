#pragma once

#include <memory>
#include <cstdint>

namespace ppo::model {

class RLAgent {
public:
                RLAgent         (long long state_dim, 
                                 long long action_dim, 
                                 long long buffer_dim,
                                 float* const stateVector,
                                 float* const actionVector);

                ~RLAgent        (void);

                RLAgent         (RLAgent const&) = delete;
    RLAgent&    operator=       (RLAgent const&) = delete;
                RLAgent         (RLAgent&&)      = delete;
    RLAgent&    operator=       (RLAgent&&)      = delete;

    void        Init            (void);
    void        ChooseAction    (void);
    void        SetNewState     (float reward, bool done);
    bool        IsTraining      (void) const noexcept;
    long long   GetCurrentStep  (void) const noexcept;
    long long   GetMaxStep      (void) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */