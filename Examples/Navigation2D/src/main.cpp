#include "navigation.hpp"
#include "ppo/rl/runtime.hpp"
#include <iostream>
namespace {
class ScriptedPolicy final : public ppo::rl::IPolicy
{
public:
    // Smoke example only: coast straight ahead; not a trained navigation policy.
    ppo::rl::Value act(const ppo::rl::Value&) override {return std::vector<float>{0.F,0.F};}
};
}
int main(int argc,char** argv)
{
    try
    {
        if(argc!=2) {std::cerr<<"Usage: navigation2d scenario.nav\n";return 2;}
        navigation::Environment environment(navigation::load_scenario(argv[1]));
        ScriptedPolicy policy;
        const auto episode=ppo::rl::run_episode(environment,policy,{42,{}},environment.scenario().max_steps);
        std::cout<<"reward="<<episode.reward<<" steps="<<episode.length
                 <<" terminated="<<episode.terminated<<" truncated="<<episode.truncated<<'\n';
        return 0;
    }
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
