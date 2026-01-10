#pragma once
#include <array>
#include <memory>
#include <filesystem>

#include "trainingnet.hpp"
#include <torch/torch.h>

namespace ppo::model {

template<typename Policy, typename Model>
class ProdAgent:
    public IAgent
{
public:
    using StateVector  = std::array<float, Model::STATES.size()>;
    using ActionVector = std::array<float, Model::ACTIONS.size()>;

                             ProdAgent      (Model* const model, 
                                             std::string const& name, 
                                             std::filesystem::path const& modelpath);
                             
    void                     Init           (void) override;

    void                     ChooseAction   (void) override;
    bool                     IsTrained      (void)                  const noexcept;
    bool                     SetNewState    (void) override;
    std::string const&       GetName        (void)                  const noexcept;

    bool                     Update         (void) noexcept override { return false;} ;

private:
    Model*                              p_model         {nullptr};
    StateVector                         m_stateVector   {};
    ActionVector                        m_actionVector  {};
    std::string                         m_name          { "" };
    std::filesystem::path               m_modelpath     { "" };
    std::shared_ptr<Policy>             p_policy         { nullptr };
    torch::Tensor                       m_states_t      { torch::from_blob(m_stateVector.data(), {1, Model::STATES.size() }, torch::kFloat) };
};

} /* ppo::model */

#include "prodagent_def.hpp"