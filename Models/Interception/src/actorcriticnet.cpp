#include "actorcriticnet.hpp"

namespace ppo::interception {

ActorCriticNet::ActorCriticNet(int64_t state_dim, int64_t action_dim, int64_t hidden_dim)
{
    std::string fc1_name{ "fc1" };
    std::string fc2_name{ "fc2" };
    std::string  mu_name{ "mu_head" };
    std::string log_name{ "logstd_head" };
    std::string critic_name{ "critic" };

    fc1 = register_module(fc1_name, torch::nn::Linear(state_dim, hidden_dim));
    fc2 = register_module(fc2_name, torch::nn::Linear(hidden_dim, hidden_dim));

    mu_head     = register_module(mu_name,     torch::nn::Linear(hidden_dim, action_dim));
    logstd_head = register_module(log_name, torch::nn::Linear(hidden_dim, action_dim));

    critic = register_module(critic_name, torch::nn::Linear(hidden_dim, 1));

    // ===== ORTHOGONAL INIT =====
    torch::nn::init::orthogonal_(fc1->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(fc2->weight, std::sqrt(2.0));

    torch::nn::init::orthogonal_(mu_head->weight, 0.01);
    torch::nn::init::orthogonal_(logstd_head->weight, 0.01);
    torch::nn::init::orthogonal_(critic->weight, 1.0);

    // Bias à zéro
    torch::nn::init::constant_(fc1->bias, 0.0);
    torch::nn::init::constant_(fc2->bias, 0.0);
    torch::nn::init::constant_(mu_head->bias, 0.0);
    torch::nn::init::constant_(logstd_head->bias, 0.0);
    torch::nn::init::constant_(critic->bias, 0.0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticNet::forward(torch::Tensor state)
{
    if (!fc1
        || !fc2
        || !mu_head
        || !logstd_head
        || !critic)
        return { {}, {}, {} };


    torch::Tensor x = torch::relu(fc1->forward(state));
    x = torch::relu(fc2->forward(x));

    auto mu = mu_head->forward(x);
    auto log_std = torch::clamp(logstd_head->forward(x), -20.0, 2.0);
    auto value = critic->forward(x);

    return {mu, log_std, value};
}

torch::Tensor
ActorCriticNet::forward_mu(torch::Tensor state)
{
    torch::Tensor x = torch::relu(fc1->forward(state));
    x = torch::relu(fc2->forward(x));

    auto mu = mu_head->forward(x);
    return mu;
}

ActorCriticNetProd::ActorCriticNetProd(int64_t state_dim, int64_t action_dim, int64_t hidden_dim ) :
    p_policy{ std::make_shared<ActorCriticNet>(state_dim, action_dim, hidden_dim)}
{

}

void 
ActorCriticNetProd::load(std::filesystem::path const& modelpath)
{
    if (p_policy == nullptr)
        return;

    torch::serialize::InputArchive archive{};

    archive.load_from(modelpath.string());

    p_policy->load(archive);
    p_policy->eval();
}


torch::Tensor 
ActorCriticNetProd::forward(torch::Tensor state)
{
    if (p_policy == nullptr)
        return {};

    torch::NoGradGuard no_grad;
    return p_policy->forward_mu(state);
}

} /* ppo::interception */