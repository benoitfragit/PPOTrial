#include "patrol_rltools.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace blue = ppo::examples::twovstwo::defenders;
namespace rlt = rl_tools;

namespace {
template<typename Matrix>
void WriteMatrix(std::ostream& out, Matrix const& matrix)
{
    for (typename Matrix::SPEC::TI i = 0; i < Matrix::SPEC::ROWS; ++i)
        for (typename Matrix::SPEC::TI j = 0; j < Matrix::SPEC::COLS; ++j)
            out << rlt::get(matrix, i, j) << ' ';
    out << '\n';
}

template<typename Layer>
void WriteLayer(std::ostream& out, Layer const& layer)
{
    WriteMatrix(out, layer.weights.parameters);
    WriteMatrix(out, layer.biases.parameters);
}

template<typename Block>
void WriteAttention(std::ostream& out,Block const& b)
{
    WriteMatrix(out,b.encoderW.parameters);
    WriteMatrix(out,b.encoderB.parameters);
    WriteMatrix(out,b.queryW.parameters);
    WriteMatrix(out,b.queryB.parameters);
}

template<typename Actor>
void SavePolicy(std::filesystem::path const& path, Actor const& actor)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    auto const temp = path.string() + ".tmp";
    std::ofstream out{temp, std::ios::trunc};
    if (!out)
        throw std::runtime_error("Cannot open temporary policy file");
    out << "BLUE_PATROL_PPO_V2 " << blue::ObservationLayout::Version << ' '
        << blue::ObservationLayout::Size << ' '
        << blue::ObservationLayout::ActionSize << ' '
        << blue::TrainingParameters::ACTOR_HIDDEN_DIM << ' '
        << blue::TrainingParameters::ACTOR_NUM_LAYERS << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10);
    WriteAttention(out,actor.red);
    WriteAttention(out,actor.friendlies);
    WriteLayer(out, actor.input_layer);
    for (auto const& layer : actor.hidden_layers)
        WriteLayer(out, layer);
    WriteLayer(out, actor.output_layer);
    WriteMatrix(out, actor.log_std.parameters);
    out.flush();
    if (!out)
        throw std::runtime_error("Failed to serialize blue PPO policy");
    out.close();
    // Replace safely on Windows as well: retain the prior checkpoint until the
    // new file is in place, restoring it if the final rename fails.
    auto const backup = path.string() + ".bak";
    std::filesystem::remove(backup);
    bool const existing = std::filesystem::exists(path);
    if (existing)
        std::filesystem::rename(path, backup);
    try
    {
        std::filesystem::rename(temp, path);
    }
    catch (...)
    {
        if (existing)
            std::filesystem::rename(backup, path);
        throw;
    }
    if (existing)
        std::filesystem::remove(backup);
}


blue::Parameters ReadParameters()
{
    namespace fs = std::filesystem;
    fs::path const settings{PPO_BLUE_SETTINGS_DIR};
    auto read = [](fs::path const& path) {
        std::ifstream file{path};
        if (!file)
            throw std::runtime_error("Cannot open configuration: " + path.string());
        nlohmann::json data;
        file >> data;
        if (data.at("formatVersion").get<int>() != 2)
            throw std::runtime_error("Unsupported configuration version: " + path.string());
        return data;
    };
    auto const world = read(settings / "Environment.json").at("environment");
    auto const flight = read(settings / "Platforms/Mobile/defender_car.json")
        .at("parameters").at("FlightModel");
    blue::Parameters p{};
    p.halfNorth = world.at("maxNorth").get<float>() * 0.5F;
    p.halfEast = world.at("maxEast").get<float>() * 0.5F;
    p.dt = world.at("timeStep").get<float>();
    p.maxSpeed = flight.at("MaxSpeed").get<float>();
    p.maxAcceleration = flight.at("MaxAcceleration").get<float>();
    p.maxRotationRate = flight.at("MaxRotAcceleration").get<float>();
    if (!blue::Valid(p))
        throw std::invalid_argument("Invalid Blue Patrol world/vehicle configuration");
    return p;
}

unsigned ParseUnsigned(char const* input)
{
    std::string const text{input};
    std::size_t consumed = 0;
    unsigned long const number = std::stoul(text, &consumed, 10);
    if (consumed != text.size() || number > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Expected unsigned integer argument");
    return static_cast<unsigned>(number);
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        // Optional seed and output path. No dependency on LibTorch or GPU.
        if (argc > 4)
        {
            std::cerr << "Usage: PPOBlueTrain [seed] [checkpoint.txt] [updates]\n";
            return 2;
        }
        unsigned const seed = argc >= 2 ? ParseUnsigned(argv[1]) : 1U;
        std::filesystem::path const output = argc >= 3
            ? std::filesystem::path{argv[2]}
            : std::filesystem::path{PPO_BLUE_MODEL_PATH};

        unsigned const updates = argc >= 4 ? ParseUnsigned(argv[3]) :
            static_cast<unsigned>(blue::TrainingParameters::STEP_LIMIT);
        if (updates == 0U)
            throw std::invalid_argument("PPO update count must be positive");

        blue::Parameters const params = ReadParameters();
        if (!blue::Valid(params))
            throw std::invalid_argument("Invalid patrol environment parameters");

        blue::Device device;
        blue::TrainingState training{};
        for (auto& env : training.envs)
            env.parameters = params;
        rlt::malloc(device, training);
        rlt::init(device, training, seed);
        training.actor_optimizer.parameters.alpha = 3.e-4F;
        training.critic_optimizer.parameters.alpha = 3.e-4F;
        while (training.step < updates)
        {
            if (rlt::step(device, training))
                break;
            if (training.step % 10U == 0U)
                std::cout << "PPO update " << training.step
                          << " / " << updates << '\n';
        }
        SavePolicy(output, training.ppo.actor);
        std::cout << "Saved actor: " << output.string() << '\n';
        // RLtools' pinned core release has a typo in its aggregate free(State)
        // overload. Free exactly the allocations made by malloc(State).
        rlt::free(device, training.ppo);
        rlt::free(device, training.ppo_buffers);
        rlt::free(device, training.on_policy_runner_dataset);
        rlt::free(device, training.on_policy_runner);
        rlt::free(device, training.actor_eval_buffers);
        rlt::free(device, training.actor_buffers);
        rlt::free(device, training.critic_buffers);
        rlt::free(device, training.critic_buffers_gae);
        rlt::free(device, training.observation_normalizer);
        rlt::free(device, training.observations_mean);
        rlt::free(device, training.observations_std);
        // BluePatrol has no dynamically allocated resources.
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Blue PPO training failed: " << e.what() << '\n';
        return 1;
    }
}
