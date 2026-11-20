#include "flight_session.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
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
template<typename Values>
double Maximum(Values const& values) noexcept
{
    return *std::max_element(values.begin(),values.end());
}
double ActorMeanMaximum(ppo::examples::flight::FlightDiagnostics const& diagnostics) noexcept
{
    return *std::max_element(diagnostics.policyMaximumAbsolute.begin(),
        diagnostics.policyMaximumAbsolute.begin()+ppo::examples::flight::ContinuousActions);
}
template<typename Values>
double MeanAbsolute(Values const& sums,double samples) noexcept
{
    return std::accumulate(sums.begin(),sums.end(),0.0)/(samples*static_cast<double>(sums.size()));
}
template<typename Counts>
std::uint64_t Sum(Counts const& counts) noexcept
{
    return std::accumulate(counts.begin(),counts.end(),std::uint64_t{});
}
}
int main(int argc, char** argv)
{
    using namespace ppo::examples::flight;
    try
    {
        std::filesystem::path configPath(PPO_FLIGHT_CONFIG),output,checkpointToCheck;
        bool training=true,inspectContract=false;
        std::optional<std::size_t> updates,episodes;
        for (int i=1;i<argc;++i)
        {
            std::string_view option(argv[i]);
            if (option=="--evaluate") { training=false; continue; }
            if (option=="--inspect-contract") { inspectContract=true; continue; }
            if (option=="--help")
            {
                std::cout<<"FlightControlHeadless [--config FILE] [--output DIRECTORY] [--updates N] "
                    <<"[--evaluate --episodes N] [--inspect-contract | --check-checkpoint FILE]\n";
                return 0;
            }
            if (option!="--config" && option!="--output" && option!="--updates" && option!="--episodes"
                && option!="--check-checkpoint")
                throw std::invalid_argument("Unknown option");
            if (++i==argc) throw std::invalid_argument("Missing option value");
            if (option=="--config") configPath=argv[i];
            else if (option=="--check-checkpoint") checkpointToCheck=argv[i];
            else if (option=="--output") output=argv[i];
            else if (option=="--updates") updates=Positive(argv[i]);
            else episodes=Positive(argv[i]);
        }
        auto config=LoadConfig(configPath);
        if (updates) config.training.maxUpdates=*updates;
        Validate(config);
        if ((training && episodes) || (!training && updates)) throw std::invalid_argument("--updates is for training; --episodes is for evaluation");
        if(inspectContract || !checkpointToCheck.empty())
        {
            if((inspectContract && !checkpointToCheck.empty()) || updates || episodes || !output.empty() || !training)
                throw std::invalid_argument("Checkpoint/contract inspection is read-only and cannot be combined with execution options");
            if(inspectContract)
                std::cout<<nlohmann::json{{"observations",ObservationSize},{"actions",ContinuousActions},
                    {"outputs",PolicyOutputs},{"width",config.training.hiddenWidth},
                    {"build",PPO_FLIGHT_BUILD_CONFIGURATION},
                    {"signature",nlohmann::json::parse(config.checkpointSignature)}}.dump()<<'\n';
            else
            {
                FlightLearner candidate(config);candidate.Load(checkpointToCheck);
                std::cout<<"[FLIGHT_CHECKPOINT] compatible=true updates="<<candidate.Stats().updates<<'\n';
            }
            return 0;
        }
        std::cout<<"[FLIGHT_RUNTIME] build="<<PPO_FLIGHT_BUILD_CONFIGURATION<<" aircraft="<<config.aircraft
            <<" mode=headless autorun=true unthrottled=true config="<<configPath.string()<<std::endl;
        if (output.empty()) output=std::filesystem::absolute(configPath).parent_path().parent_path().parent_path()/config.training.checkpointDirectory;
        FlightSession session(config,training,output,episodes,TrainingBudget::TotalUpdates);
        std::filesystem::create_directories(output);
        std::ofstream telemetry(output/(training ? "training.csv" : "evaluation.csv"));
        if (!telemetry) throw std::runtime_error("Cannot write FlightControl3D telemetry");
        telemetry<<"episode,updates,outcome,simulated_seconds,touchdown_m,stop_m,ground_roll_m,touchdowns,stops,"
            "crashes,timeouts,first_wheel_contact,valid_touchdown,touchdown_cross_m,touchdown_sink_mps,"
            "touchdown_speed_mps,gear_position,touchdown_quality,curriculum_level\n";
        std::ofstream diagnostics;
        if(config.diagnostics.enabled)
        {
            diagnostics.open(output/"diagnostics.csv");
            if(!diagnostics) throw std::runtime_error("Cannot write FlightControl3D diagnostics");
            diagnostics<<"updates,samples,observation_saturation_ratio,observation_max_abs,actor_mean_max_abs,raw_action_mean_abs,applied_action_mean_abs,raw_action_saturation_ratio,applied_action_saturation_ratio,reward_mean,reward_min,reward_max,actor_steps,maximum_minibatch_approx_kl,maximum_minibatch_sampled_kl,critic_steps,actor_kl_skips\n";
        }
        std::uint64_t previousEpisodes=0,previousUpdates=session.Snapshot().learning.updates,previousValidations=0;
        while (!session.Finished())
        {
            session.Tick(); auto const state=session.Snapshot();
            if (state.episodes!=previousEpisodes)
            {
                auto const& mission=state.lastEpisode;
                telemetry<<state.episodes<<','<<state.learning.updates<<','
                    <<static_cast<int>(mission.outcome)<<','<<state.lastEpisodeSeconds<<',';
                if (mission.contactAttempted) telemetry<<mission.touchdownFromThreshold;
                telemetry<<',';
                if (mission.outcome==Outcome::Stopped) telemetry<<mission.stopFromThreshold;
                telemetry<<','<<mission.groundRoll<<','<<state.touchdowns<<','<<state.stops<<','<<state.crashes<<','<<state.timeouts
                    <<','<<mission.firstWheelContact<<','<<mission.touchedDown<<','<<mission.touchdownCrossMeters<<','<<mission.touchdownSinkMps
                    <<','<<mission.touchdownSpeedMps<<','<<mission.touchdownGearPosition
                    <<','<<(mission.touchedDown ? mission.touchdownQuality : 0.0)<<','<<state.curriculumLevel<<'\n';
                telemetry.flush(); previousEpisodes=state.episodes;
            }
            if (state.learning.updates!=previousUpdates)
            {
                std::cout<<"[FLIGHT] updates="<<state.learning.updates<<" episodes="<<state.episodes
                    <<" touchdowns="<<state.touchdowns<<" stops="<<state.stops<<" loss="<<state.learning.policyLoss<<std::endl;
                if(diagnostics && state.learning.updates%config.diagnostics.logEveryUpdates==0)
                {
                    auto const& d=state.lastDiagnostics;
                    double const samples=static_cast<double>(std::max<std::uint64_t>(d.samples,1));
                    diagnostics<<state.learning.updates<<','<<d.samples<<','
                        <<static_cast<double>(Sum(d.observationSaturations))/(samples*ObservationSize)<<','
                        <<Maximum(d.observationMaximumAbsolute)<<','<<ActorMeanMaximum(d)<<','
                        <<MeanAbsolute(d.rawAbsoluteSums,samples)<<','
                        <<MeanAbsolute(d.appliedAbsoluteSums,samples)<<','
                        <<static_cast<double>(Sum(d.rawSaturations))/(samples*ContinuousActions)<<','
                        <<static_cast<double>(Sum(d.appliedSaturations))/(samples*ContinuousActions)<<','
                        <<d.rewardSum/samples<<','<<d.rewardMinimum<<','<<d.rewardMaximum<<','
                        <<state.learning.actorSteps<<','
                        <<state.learning.maximumApproximateKl<<','
                        <<state.learning.maximumSampledKl<<','<<state.learning.criticSteps<<','
                        <<state.learning.actorKlSkips<<'\n';
                    diagnostics.flush();
                }
                previousUpdates=state.learning.updates;
            }
            if(state.validationCampaigns!=previousValidations)
            {
                double const success=state.gateEvaluated>0
                    ? static_cast<double>(state.gateSuccesses)/static_cast<double>(state.gateEvaluated) : 0;
                std::cout<<"[FLIGHT_VALIDATION] update="<<state.lastValidation.update<<" episodes="<<state.lastValidation.episodes
                    <<" difficulty="<<state.lastValidation.difficulty<<" success="<<success<<" best="<<(state.hasBest ? state.bestScore : -1)<<std::endl;
                previousValidations=state.validationCampaigns;
            }
        }
        auto const state=session.Snapshot();
        if(training && state.gateEvaluated>0)
        {
            // Validation follows the curriculum; only a campaign on the full
            // task can pass the gate, however easy levels score.
            constexpr double FullTaskDifficulty=1.0;
            double const difficulty=state.lastValidation.difficulty;
            bool const passed=difficulty>=FullTaskDifficulty && state.gateSuccesses>=static_cast<std::size_t>(
                std::ceil(config.evaluation.minimumSuccessRate*static_cast<double>(state.gateEvaluated)));
            std::cout<<"[FLIGHT_GATE] successes="<<state.gateSuccesses<<" episodes="<<state.gateEvaluated
                <<" difficulty="<<difficulty<<" passed="<<(passed ? "true" : "false")<<std::endl;
        }
        std::cout<<"[FLIGHT] completed episodes="<<state.episodes<<" touchdowns="<<state.touchdowns<<" stops="<<state.stops
            <<" crashes="<<state.crashes<<" timeouts="<<state.timeouts<<'\n';
    }
    catch(std::exception const& error) { std::cerr<<"[FLIGHT_ERROR] "<<error.what()<<'\n'; return 1; }
}
