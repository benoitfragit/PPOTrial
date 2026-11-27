/**
 * @file flight_reward_capture.cpp
 * @brief Capture two Flight validation campaigns through the public PPOKit environment and policy ABIs.
 * @details Usage: FlightControl_reward_capture PROJECT MODULE_DIRECTORY OUTPUT. Replay with flight_reward_replay compiled
 * against the historical simulator. The stream contains the actual decisions, rewards and boundaries.
 */
#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/component/native_module.hpp"
#include "ppokit/environment/environment_library.hpp"
#include "ppokit/network/network_library.hpp"
#include "ppokit/schema/policy_reference.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr std::uint32_t CampaignSize = 20;
constexpr std::uint32_t MaxSteps = 12000;
constexpr std::array<std::uint32_t, 2> FirstSeeds{4342, 90001};
constexpr std::size_t ObservationSize = 35;
constexpr std::size_t ActionSize = 6;
constexpr std::size_t ChannelCount = 7;

/** @brief Write unpadded x64 little-endian fields. Each seed block resets the simulator instance. */
template<class T>
void Write(std::ostream& stream, T const& value)
{
    static_assert(std::is_trivially_copyable_v<T>, "trace fields must have a fixed representation");
    stream.write(reinterpret_cast<char const*>(&value), sizeof(value));
}

/** @brief Select the archived actor only after checking the digest from its catalog. */
ppokit::schema::PolicyReference HistoricalPolicy(std::filesystem::path const& project)
{
    for (auto const& ref : ppokit::schema::LoadPolicyReferences(project))
    {
        if (ref.role != "HISTORICAL_BEST") continue;
        if (!ref.environmentState || ppokit::artifacts::Sha256OfFile(ref.file) != ref.sha256)
            throw std::runtime_error("historical policy state or digest differs");
        return ref;
    }
    throw std::runtime_error("missing HISTORICAL_BEST reference");
}

/** @brief Capture one episode, then fill in its length once the environment ends it. */
void CaptureEpisode(ppokit::environment::Environment& environment, ppokit::network::Policy& policy, std::uint32_t seed,
    std::uint32_t index, std::vector<float>& world, std::ostream& output)
{
    std::array<float, ObservationSize> observation{};
    std::array<float, ActionSize> action{};
    std::array<double, ChannelCount> channels{};
    std::uint8_t available{};
    double reward{};
    environment.ResetEvaluation(seed, index, CampaignSize, observation.data(), &available, world.data());
    Write(output, seed);
    Write(output, index);
    Write(output, CampaignSize);
    auto const lengthPosition = output.tellp();
    Write(output, std::uint32_t{});
    for (std::uint32_t step = 1; step <= MaxSteps; ++step)
    {
        if (available != 1) throw std::runtime_error("Flight agent missing at a decision");
        policy.Act(observation.data(), action.data(), true);
        Write(output, observation);
        Write(output, action);
        auto const end = environment.Step(action.data(), observation.data(), &available, world.data(),
            &reward, channels.data());
        Write(output, channels);
        Write(output, reward);
        Write(output, static_cast<std::uint32_t>(end));
        if (ppokit::runtime::EndsEpisode(end))
        {
            auto const next = output.tellp();
            output.seekp(lengthPosition);
            Write(output, step);
            output.seekp(next);
            return;
        }
        environment.BeginStep(observation.data(), &available, world.data());
    }
    throw std::runtime_error("Flight episode exceeded its step limit");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4) return 2;
    try
    {
        auto const project = ppokit::schema::LoadProject(argv[1]);
        auto const model = HistoricalPolicy(project.root);
        auto const simulator = ppokit::environment::EnvironmentLibrary::Open(project.environmentLibrary, argv[2]);
        ppokit::component::SetKnownModules(ppokit::schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
        auto const task = ppokit::schema::LoadTaskSchema(project.Task());
        auto const network = ppokit::network::NetworkLibrary::Open(project.networkLibrary, argv[2], &task);
        auto const contract = ppokit::schema::SemanticHash(task);
        simulator->VerifyTaskContract(contract);
        network->VerifyTaskContract(contract);
        auto const& info = simulator->Info();
        auto const& networkInfo = network->Info();
        if (info.agentCapacity != 1 || info.observationSize != ObservationSize || info.actionSize != ActionSize
            || info.rewardChannels != ChannelCount || networkInfo.agentCapacity != 1
            || networkInfo.observationSize != ObservationSize || networkInfo.actionSize != ActionSize)
            throw std::runtime_error("unsupported Flight trace layout");
        ppokit::network::Policy policy(network, project.root, model.file);
        std::ofstream output(argv[3], std::ios::binary);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        Write(output, static_cast<std::uint32_t>(FirstSeeds.size() * CampaignSize));
        std::vector<float> world(info.worldSize);
        for (auto const first : FirstSeeds)
        {
            ppokit::environment::Environment environment(simulator, project.root, first);
            environment.SetLearning(false);
            environment.RestoreTrainingState(*model.environmentState);
            for (std::uint32_t index = 0; index < CampaignSize; ++index)
                CaptureEpisode(environment, policy, first + index, index, world, output);
        }
        output.close();
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
