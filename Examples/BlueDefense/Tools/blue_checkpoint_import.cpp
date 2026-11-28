/**
 * @file Examples/BlueDefense/Tools/blue_checkpoint_import.cpp
 * @brief Converts Blue MAPPO checkpoints of the former formats (V7 training state, V10 actor)
 *        into PPOKit artifacts (`.ppokit`), and proves that nothing was lost.
 *
 * Usage:
 * @code
 * blue_checkpoint_import --seed <training seed>
 *     [--training <V7 file> --to-training <latest.ppokit>]
 *     [--actor <V10 file> --to-policy <policy.ppokit>]
 * @endcode
 *
 * Each source is read by the generalized learner, written as an artifact, read back by a
 * fresh learner and written again in its former format: the result must equal the source
 * byte for byte (parameters, both Adams with settings, ages and moments, RNG, updates, P/N).
 * Only then is the artifact kept. An existing destination is never overwritten. The seed is
 * recorded in the artifact header (the training seed of blue_patrol_training.json); a V10
 * actor records no update count, so its policy artifact says update 0.
 */
#include "generalized_patrol_learner.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

namespace defenders = ppo::examples::bluedefense::defenders;

/** @brief Header tokens before the P/N metadata: V7 training (magic and 10 values), V10 actor (magic and 7). */
constexpr int TrainingHeaderTokens = 11;
constexpr int ActorHeaderTokens = 8;

struct Options final
{
    std::optional<std::uint32_t> seed;
    std::filesystem::path training;
    std::filesystem::path toTraining;
    std::filesystem::path actor;
    std::filesystem::path toPolicy;
};

/** @brief Value of the option at `index`, which must exist. */
std::string ValueAfter(int argc, char** argv, int index)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string(argv[index]) + " needs a value");
    }
    return argv[index + 1];
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; index += 2)
    {
        std::string const name = argv[index];
        std::string const value = ValueAfter(argc, argv, index);
        if (name == "--seed")
        {
            options.seed = static_cast<std::uint32_t>(std::stoul(value));
        }
        else if (name == "--training")
        {
            options.training = value;
        }
        else if (name == "--to-training")
        {
            options.toTraining = value;
        }
        else if (name == "--actor")
        {
            options.actor = value;
        }
        else if (name == "--to-policy")
        {
            options.toPolicy = value;
        }
        else
        {
            throw std::invalid_argument("unknown option " + name);
        }
    }
    bool const trainingPair = options.training.empty() == options.toTraining.empty();
    bool const actorPair = options.actor.empty() == options.toPolicy.empty();
    bool const anything = !options.training.empty() || !options.actor.empty();
    if (!options.seed || !trainingPair || !actorPair || !anything)
    {
        throw std::invalid_argument("usage: blue_checkpoint_import --seed <seed> [--training <V7> --to-training "
                                    "<latest.ppokit>] [--actor <V10> --to-policy <policy.ppokit>]");
    }
    return options;
}

std::string FileText(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

/** @brief P/N recorded in a former file, after its first `skippedTokens` header tokens. */
defenders::ActiveTeamCounts LegacyTeamCounts(std::filesystem::path const& path, int skippedTokens)
{
    std::ifstream stream(path);
    std::string token;
    for (int index = 0; index < skippedTokens; ++index)
    {
        if (!(stream >> token))
        {
            throw std::runtime_error(path.string() + ": header too short");
        }
    }
    defenders::GeneralizedCheckpointMetadata const metadata =
        defenders::generalized_checkpoint_detail::ReadMetadata(stream);
    return {metadata.activeBlueMobiles, metadata.activeRedMobiles};
}

void RequireAbsent(std::filesystem::path const& path)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(path.string() + " exists: remove it first (nothing is overwritten)");
    }
}

/** @brief Removes a rejected artifact, then reports the source that did not survive the round trip. */
[[noreturn]] void Reject(std::filesystem::path const& artifact, std::filesystem::path const& source)
{
    std::error_code ignored;
    std::filesystem::remove(artifact, ignored);
    throw std::runtime_error(source.string() + ": the converted artifact does not give back the same file");
}

/** @brief V7 -> training artifact, checked by writing the V7 file again from the artifact. */
void ConvertTraining(std::filesystem::path const& source, std::filesystem::path const& destination,
    std::uint32_t seed)
{
    RequireAbsent(destination);
    defenders::ActiveTeamCounts const counts = LegacyTeamCounts(source, TrainingHeaderTokens);
    defenders::GeneralizedPatrolLearner reader{counts, seed, 1U};
    reader.LoadTrainingState(source);
    reader.SaveTrainingState(destination);

    defenders::GeneralizedPatrolLearner restored{counts, seed, 1U};
    restored.LoadTrainingState(destination);
    std::filesystem::path const again = destination.string() + ".check";
    restored.SaveV7TrainingState(again);
    bool const same = FileText(again) == FileText(source) && restored.StepCount() == reader.StepCount();
    std::filesystem::remove(again);
    if (!same)
    {
        Reject(destination, source);
    }
    std::cout << source.string() << " -> " << destination.string() << ": update " << restored.StepCount()
              << ", P=" << counts.blueMobiles << " N=" << counts.redMobiles << ", identical round trip\n";
}

/** @brief V10 -> policy artifact, checked by writing the V10 file again from the artifact. */
void ConvertActor(std::filesystem::path const& source, std::filesystem::path const& destination, std::uint32_t seed)
{
    RequireAbsent(destination);
    defenders::ActiveTeamCounts const counts = LegacyTeamCounts(source, ActorHeaderTokens);
    defenders::GeneralizedPatrolLearner reader{counts, seed, 1U};
    reader.LoadActor(source);
    reader.SaveActor(destination);

    defenders::GeneralizedPatrolLearner restored{counts, seed, 1U};
    restored.LoadActor(destination);
    std::filesystem::path const again = destination.string() + ".check";
    restored.SaveV10Actor(again);
    bool const same = FileText(again) == FileText(source);
    std::filesystem::remove(again);
    if (!same)
    {
        Reject(destination, source);
    }
    std::cout << source.string() << " -> " << destination.string() << ": P=" << counts.blueMobiles
              << " N=" << counts.redMobiles << ", identical round trip\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Options const options = ParseOptions(argc, argv);
        if (!options.training.empty())
        {
            ConvertTraining(options.training, options.toTraining, *options.seed);
        }
        if (!options.actor.empty())
        {
            ConvertActor(options.actor, options.toPolicy, *options.seed);
        }
        return EXIT_SUCCESS;
    }
    catch (std::exception const& error)
    {
        std::cerr << "blue_checkpoint_import: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
