#include "navigation.hpp"
#include "torch_policy.hpp"
#include <fstream>
#include <iostream>

// This executable is opt-in and is never invoked by the test suite.
int main(int argc,char** argv)
{
    try
    {
        if(argc!=5) {std::cerr<<"Usage: navigation2d_train scenario.nav score.metrics new-output-directory episodes\n";return 2;}
        std::size_t parsed=0;
        const std::string text=argv[4];
        const auto count=std::stoul(text,&parsed);
        if(parsed!=text.size() || count==0 || count>1000000) throw std::invalid_argument("invalid episode count");
        const auto scenario=navigation::load_scenario(argv[1]);
        const auto criteria=ppo::rl::load_metrics(argv[2]);
        const std::filesystem::path output=argv[3];
        if(std::filesystem::exists(output)) throw std::invalid_argument("use a new output directory per training run");
        std::filesystem::create_directories(output);
        std::filesystem::copy_file(argv[1],output/"scenario.nav");
        std::filesystem::copy_file(argv[2],output/"score.metrics");
        navigation::Environment environment(scenario);
        const auto dimension=std::get<ppo::rl::Box>(environment.observation_space()).low().size();
        ppo::rl::TorchLearner learner(dimension,2,512,output);
        for(std::size_t episode=0;episode<count;++episode)
        {
            ppo::rl::train_episode(environment,learner,{episode,{}},scenario.max_steps);
            const auto version="navigation2d-v1-"+std::to_string(episode);
            const auto candidate=output/"candidates"/version;
            learner.save_candidate(candidate);
            std::filesystem::copy_file(output/"scenario.nav",candidate/"scenario.nav");
            ppo::rl::FrozenTorchPolicy frozen(dimension,2,candidate/"eval"/"model.pt");
            navigation::Environment evaluation_environment(scenario);
            const std::vector<ppo::rl::Episode> evaluation{
                ppo::rl::run_episode(evaluation_environment,frozen,{42,{}},scenario.max_steps)};
            const double candidate_score=ppo::rl::score(evaluation,criteria);
            const bool promoted=ppo::rl::promote_checkpoint(candidate,output/"best",version,candidate_score);
            std::cout<<version<<" score="<<candidate_score<<" promoted="<<promoted<<'\n';
        }
        return 0;
    }
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
