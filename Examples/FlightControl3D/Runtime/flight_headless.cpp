#include "flight_session.hpp"
#include <charconv>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>

namespace {
std::size_t Positive(char const* text)
{
    std::string_view value(text); std::size_t result{};
    auto const parsed=std::from_chars(value.data(),value.data()+value.size(),result);
    if (parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || result==0 || result>100000000)
        throw std::invalid_argument("Expected an integer in [1,100000000]");
    return result;
}
}
int main(int argc, char** argv)
{
    using namespace ppo::examples::flight;
    try
    {
        std::filesystem::path configPath(PPO_FLIGHT_CONFIG),output;
        bool training=true;
        std::optional<std::size_t> updates,episodes;
        for (int i=1;i<argc;++i)
        {
            std::string_view option(argv[i]);
            if (option=="--evaluate") { training=false; continue; }
            if (option=="--help")
            {
                std::cout<<"FlightControlHeadless [--config FILE] [--output DIRECTORY] [--updates N] [--evaluate --episodes N]\n";
                return 0;
            }
            if (option!="--config" && option!="--output" && option!="--updates" && option!="--episodes")
                throw std::invalid_argument("Unknown option");
            if (++i==argc) throw std::invalid_argument("Missing option value");
            if (option=="--config") configPath=argv[i];
            else if (option=="--output") output=argv[i];
            else if (option=="--updates") updates=Positive(argv[i]);
            else episodes=Positive(argv[i]);
        }
        auto config=LoadConfig(configPath);
        if (updates) config.training.maxUpdates=*updates;
        if (episodes) config.evaluation.episodes=*episodes;
        if ((training && episodes) || (!training && updates)) throw std::invalid_argument("--updates is for training; --episodes is for evaluation");
        if (output.empty()) output=std::filesystem::absolute(configPath).parent_path().parent_path().parent_path()/config.training.checkpointDirectory;
        FlightSession session(config,training,output/"flight_hybrid_v1.txt");
        std::filesystem::create_directories(output);
        std::ofstream telemetry(output/(training ? "training.csv" : "evaluation.csv"));
        if (!telemetry) throw std::runtime_error("Cannot write FlightControl3D telemetry");
        telemetry<<"episode,updates,outcome,touchdown_m,stop_m,ground_roll_m,touchdowns,stops,crashes,timeouts\n";
        std::uint64_t previousEpisodes=0,previousUpdates=session.Snapshot().learning.updates;
        while (!session.Finished())
        {
            session.Tick(); auto const state=session.Snapshot();
            if (state.episodes!=previousEpisodes)
            {
                auto const& mission=state.lastEpisode;
                telemetry<<state.episodes<<','<<state.learning.updates<<','<<static_cast<int>(mission.outcome)<<',';
                if (mission.touchedDown) telemetry<<mission.touchdownFromThreshold;
                telemetry<<',';
                if (mission.outcome==Outcome::Stopped) telemetry<<mission.stopFromThreshold;
                telemetry<<','<<mission.groundRoll<<','<<state.touchdowns<<','<<state.stops<<','<<state.crashes<<','<<state.timeouts<<'\n';
                telemetry.flush(); previousEpisodes=state.episodes;
            }
            if (state.learning.updates!=previousUpdates)
            {
                std::cout<<"[FLIGHT] updates="<<state.learning.updates<<" episodes="<<state.episodes
                    <<" touchdowns="<<state.touchdowns<<" stops="<<state.stops<<" loss="<<state.learning.policyLoss<<std::endl;
                previousUpdates=state.learning.updates;
            }
        }
        auto const state=session.Snapshot();
        std::cout<<"[FLIGHT] completed episodes="<<state.episodes<<" touchdowns="<<state.touchdowns<<" stops="<<state.stops
            <<" crashes="<<state.crashes<<" timeouts="<<state.timeouts<<'\n';
    }
    catch(std::exception const& error) { std::cerr<<"[FLIGHT_ERROR] "<<error.what()<<'\n'; return 1; }
}
