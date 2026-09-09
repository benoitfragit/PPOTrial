#include "navigation.hpp"
#include "torch_policy.hpp"
#include <iostream>

int main(int argc,char** argv)
{
    try
    {
        if(argc!=4) {std::cerr<<"Usage: navigation2d_policy scenario.nav model.pt episode-count\n";return 2;}
        std::size_t parsed=0;
        const std::string count_text=argv[3];
        const auto count=std::stoul(count_text,&parsed);
        if(parsed!=count_text.size() || count==0 || count>10000) throw std::invalid_argument("invalid episode count");
        navigation::Environment environment(navigation::load_scenario(argv[1]));
        const auto dimension=std::get<ppo::rl::Box>(environment.observation_space()).low().size();
        ppo::rl::FrozenTorchPolicy policy(dimension,2,argv[2]);
        std::vector<ppo::rl::Episode> episodes;
        episodes.reserve(count);
        for(std::size_t index=0;index<count;++index)
            episodes.push_back(ppo::rl::run_episode(environment,policy,{index,{}},environment.scenario().max_steps));
        std::cout<<"success_rate="<<ppo::rl::aggregate(episodes,{"success",ppo::rl::Reduction::Rate,1.0})<<'\n';
        return 0;
    }
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
