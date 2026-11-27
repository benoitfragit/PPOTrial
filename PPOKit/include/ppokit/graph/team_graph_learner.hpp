/**
 * @file PPOKit/include/ppokit/graph/team_graph_learner.hpp
 * @brief MAPPO team learner of declarative projects: task.json and network.json, no client code.
 * @details One rollout of agentCapacity rows (task.json "agents" max): the shared actor acts
 * for every agent of the episode, the centralized critic reads what network.json's
 * "team_critic" section builds from one actor block of the whole team (ppokit/schema:
 * TeamCriticSpec). HostedTrainer runs the PPO update; TeamGraphModels holds the graphs and
 * their Adam optimizers.
 *
 * The members of an episode are the agents present at its first step: an agent that is not
 * available afterwards was destroyed (it still shares the return and trains the critic), a
 * row that was never present is absent (left out of everything). The team may change size
 * from one episode to the next (a curriculum stage), not within one.
 *
 * @code
 * TeamGraphLearner learner(task, graphs, plan.ppo.Settings(), plan.ppo.RolloutSteps());
 * learner.Act(observations, available, actions, false);        // agentCapacity rows
 * // ... environment step ...
 * bool const updated = learner.Observe(observations, available, rewards, end);
 * @endcode
 *
 * Every buffer is allocated at construction: a team step never allocates. Not thread-safe.
 */
#pragma once

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/core/ppo_settings.hpp"
#include "ppokit/graph/checkpoint_io.hpp"
#include "ppokit/graph/team_graph_models.hpp"
#include "ppokit/runtime/hosted_trainer.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::graph {

/** @brief The PPO settings of a hosted MAPPO update, from a learner's settings. */
inline runtime::TeamPpoSettings TeamSettingsOf(core::PpoSettings const& settings) noexcept
{
    runtime::TeamPpoSettings team;
    team.gamma = static_cast<float>(settings.gamma);
    team.lambda = static_cast<float>(settings.lambda);
    team.clip = static_cast<float>(settings.clip);
    team.entropyCoefficient = static_cast<float>(settings.entropyCoefficient);
    team.targetKl = static_cast<float>(settings.targetKl);
    team.epochs = settings.epochs;
    team.minibatch = settings.minibatch;
    return team;
}

/**
 * @brief Builds the critic input of every member of the team from one actor block of each.
 * @details Own row (zero once destroyed), mean and maximum over the living members (0 when
 * none lives), living members and members over the agent capacity, as TeamCriticSpec says.
 */
class TeamCriticBuilder final
{
public:
    TeamCriticBuilder(schema::TeamCriticSpec const& spec, schema::InputLayout const& critic, std::size_t capacity)
        : width_(spec.width), capacity_(capacity), criticSize_(critic.size),
          ownOffset_(critic.groups.at(spec.own).offset), meanOffset_(critic.groups.at(spec.mean).offset),
          maxOffset_(critic.groups.at(spec.max).offset),
          livingOffset_(critic.groups.at(spec.livingFraction).offset),
          teamOffset_(critic.groups.at(spec.teamFraction).offset), mean_(spec.width), maximum_(spec.width)
    {
    }

    /**
     * @param features `capacity` rows of `width` values (the actor block of each row).
     * @param members, alive Of each row; alive implies member.
     * @param critics Receives `capacity` rows of the critic input; rows of non-members are zero.
     */
    void Build(float const* features, std::vector<bool> const& members, std::vector<bool> const& alive,
        float* critics)
    {
        std::fill(mean_.begin(), mean_.end(), 0.F);
        std::fill(maximum_.begin(), maximum_.end(), -std::numeric_limits<float>::infinity());
        std::size_t living = 0;
        std::size_t memberCount = 0;
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            memberCount += members[agent] ? 1 : 0;
            if (!alive[agent])
            {
                continue;
            }
            ++living;
            float const* const row = features + agent * width_;
            for (std::size_t feature = 0; feature < width_; ++feature)
            {
                mean_[feature] += row[feature];
                maximum_[feature] = std::max(maximum_[feature], row[feature]);
            }
        }
        float const inverseLiving = InverseOf(living);
        float const livingFraction = static_cast<float>(living) / static_cast<float>(capacity_);
        float const teamFraction = static_cast<float>(memberCount) / static_cast<float>(capacity_);
        std::fill(critics, critics + capacity_ * criticSize_, 0.F);
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            if (!members[agent])
            {
                continue;
            }
            float* const critic = critics + agent * criticSize_;
            float const* const row = features + agent * width_;
            for (std::size_t feature = 0; feature < width_; ++feature)
            {
                // A destroyed member's own block stays zero: that is how the critic sees its death.
                if (alive[agent])
                {
                    critic[ownOffset_ + feature] = row[feature];
                }
                critic[meanOffset_ + feature] = mean_[feature] * inverseLiving;
                critic[maxOffset_ + feature] = living == 0 ? 0.F : maximum_[feature];
            }
            critic[livingOffset_] = livingFraction;
            critic[teamOffset_] = teamFraction;
        }
    }

private:
    static float InverseOf(std::size_t living) noexcept
    {
        if (living == 0)
        {
            return 0.F;
        }
        return 1.F / static_cast<float>(living);
    }

    std::size_t width_;
    std::size_t capacity_;
    std::size_t criticSize_;
    std::size_t ownOffset_;
    std::size_t meanOffset_;
    std::size_t maxOffset_;
    std::size_t livingOffset_;
    std::size_t teamOffset_;
    std::vector<float> mean_;
    std::vector<float> maximum_;
};

/** @brief Shared-actor MAPPO of a team task built from its schemas; see the file comment. */
class TeamGraphLearner final
{
public:
    /**
     * @param settings Learner settings (train.json); its seed seeds the weights and sampling.
     * @param teamSteps Team steps collected per update.
     * @throws std::invalid_argument for a task that is not a team task, or graphs without
     *         a "team_critic" section.
     */
    TeamGraphLearner(schema::TaskSchema const& task, schema::BlockGraphSchema const& graphs,
        core::PpoSettings const& settings, std::size_t teamSteps)
        : capacity_(RequireTeam(task, graphs)), observationSize_(task.observation.size),
          criticSize_(task.critic->size), actionSize_(task.actionSize), seed_(settings.seed),
          models_(schema::BuildGraph(graphs.actor), schema::BuildGraph(graphs.critic),
              schema::MakeInputSlices(graphs.actor, "actor", task), schema::MakeCriticInputSlices(graphs.critic, task),
              settings, observationSize_, criticSize_, actionSize_),
          random_(settings.seed), trainer_(models_, random_, capacity_, teamSteps, TeamSettingsOf(settings)),
          critics_(*graphs.teamCritic, *task.critic, capacity_),
          featureNode_(models_.Actor().Find(graphs.teamCritic->features)), featureWidth_(graphs.teamCritic->width),
          members_(capacity_, false), alive_(capacity_, false), presence_(capacity_),
          observationRows_(capacity_, nullptr), criticRows_(capacity_, nullptr), means_(capacity_ * actionSize_),
          features_(capacity_ * featureWidth_), criticValues_(capacity_ * criticSize_),
          rewards_(capacity_), ends_(capacity_)
    {
    }

    /** @brief The trainer refers to the models and the generator: the learner stays in place. */
    TeamGraphLearner(TeamGraphLearner const&) = delete;
    TeamGraphLearner& operator=(TeamGraphLearner const&) = delete;
    TeamGraphLearner(TeamGraphLearner&&) = delete;
    TeamGraphLearner& operator=(TeamGraphLearner&&) = delete;

    std::uint64_t Updates() const noexcept { return updates_; }
    runtime::TeamUpdateStatistics const& LastUpdate() const noexcept { return trainer_.LastUpdate(); }
    runtime::TeamPpoSettings const& Settings() const noexcept { return trainer_.Settings(); }
    TeamGraphModels& Models() noexcept { return models_; }
    /** @brief True between two rollouts: nothing collected, nothing pending (save, evaluate). */
    bool AtUpdateBoundary() const noexcept { return trainer_.AtUpdateBoundary(); }

    /**
     * @brief Actions of the agents of the episode, sampled (and recorded) or their means.
     * @param observations agentCapacity rows; rows of absent agents are ignored.
     * @param available Of each row: an agent that acts now.
     * @param actions Receives agentCapacity rows; rows of agents that do not act are zero.
     * @param deterministic True for the means, nothing recorded (between two rollouts only).
     */
    void Act(float const* observations, std::uint8_t const* available, float* actions, bool deterministic)
    {
        if (deterministic)
        {
            ActMeans(observations, available, actions);
            return;
        }
        if (episodeStarts_)
        {
            BeginEpisode(available);
        }
        ReadTeam(observations, available);
        models_.ActorMeansAndFeatures(observationRows_.data(), capacity_, means_.data(), featureNode_, featureWidth_,
            features_.data());
        critics_.Build(features_.data(), members_, alive_, criticValues_.data());
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            presence_[agent] = PresenceOf(agent);
        }
        trainer_.ChooseActionsFromMeans(observationRows_.data(), means_.data(), criticRows_.data(), presence_.data(),
            actions);
    }

    /**
     * @brief Records the step after the sampled actions; updates when the rollout is full.
     * @param observations, available The next decision observations and the agents present.
     * @param rewards Each row's reward (agentCapacity values; ignored for absent rows).
     * @param end The team's episode end.
     * @return True when an update was made.
     */
    bool Observe(float const* observations, std::uint8_t const* available, double const* rewards,
        runtime::StepEnd end)
    {
        ReadTeam(observations, available);
        models_.ActorMeansAndFeatures(observationRows_.data(), capacity_, means_.data(), featureNode_, featureWidth_,
            features_.data());
        critics_.Build(features_.data(), members_, alive_, criticValues_.data());
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            float const reward = static_cast<float>(rewards[agent]);
            if (members_[agent] && !std::isfinite(reward))
            {
                throw std::invalid_argument("PPOKit team learner: non-finite reward for agent " + std::to_string(agent));
            }
            rewards_[agent] = members_[agent] ? reward : 0.F;
            ends_[agent] = end;
        }
        bool const updated = trainer_.Observe(criticRows_.data(), rewards_.data(), ends_.data());
        episodeStarts_ = runtime::EndsEpisode(end);
        updates_ += updated ? 1 : 0;
        return updated;
    }

    /** @brief Writes the training state (resumable): networks, both Adams, generator, updates. */
    void SaveTraining(std::filesystem::path const& path, artifacts::ArtifactIdentity const& identity)
    {
        RequireBoundary("save a training state");
        Write(path, identity, artifacts::ArtifactKind::Training);
    }

    /** @brief Writes the actor alone, for evaluation and inference. */
    void SavePolicy(std::filesystem::path const& path, artifacts::ArtifactIdentity const& identity)
    {
        Write(path, identity, artifacts::ArtifactKind::Policy);
    }

    /**
     * @brief Resumes from a training state of the same networks; unchanged on failure.
     * @throws artifacts::ArtifactMismatch for other networks; std::runtime_error for a
     *         policy or a damaged file.
     */
    void LoadTraining(std::filesystem::path const& path, artifacts::ArtifactIdentity const& expected)
    {
        RequireBoundary("load a training state");
        Read(path, expected, true);
    }

    /** @brief Loads the actor of a policy or training checkpoint. */
    void LoadPolicy(std::filesystem::path const& path, artifacts::ArtifactIdentity const& expected)
    {
        Read(path, expected, false);
    }

private:
    /** @brief The agent capacity of a team task whose graphs build the critic input. */
    static std::size_t RequireTeam(schema::TaskSchema const& task, schema::BlockGraphSchema const& graphs)
    {
        if (task.maximumAgents < 2 || !task.critic || !graphs.teamCritic)
        {
            throw std::invalid_argument("PPOKit team learner: a team task (agents.max above 1, a \"critic\" "
                                        "input) and a \"team_critic\" section in network.json are required");
        }
        return task.maximumAgents;
    }

    /** @brief The members of the episode are the agents present at its first step. */
    void BeginEpisode(std::uint8_t const* available)
    {
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            members_[agent] = available[agent] != 0;
        }
        if (std::find(members_.begin(), members_.end(), true) == members_.end())
        {
            throw std::invalid_argument("PPOKit team learner: an episode starts without any agent");
        }
        episodeStarts_ = false;
    }

    /** @brief Row pointers and the living members; an agent cannot join a running episode. */
    void ReadTeam(float const* observations, std::uint8_t const* available)
    {
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            bool const present = available[agent] != 0;
            if (present && !members_[agent])
            {
                throw std::invalid_argument("PPOKit team learner: agent " + std::to_string(agent)
                    + " appeared during an episode (the team is fixed at the episode's first step)");
            }
            alive_[agent] = present;
            observationRows_[agent] = members_[agent] ? observations + agent * observationSize_ : nullptr;
            criticRows_[agent] = criticValues_.data() + agent * criticSize_;
        }
    }

    runtime::AgentPresence PresenceOf(std::size_t agent) const noexcept
    {
        if (!members_[agent])
        {
            return runtime::AgentPresence::Absent;
        }
        return alive_[agent] ? runtime::AgentPresence::Active : runtime::AgentPresence::Destroyed;
    }

    /** @brief Means of the available agents, zero elsewhere; nothing recorded or drawn. */
    void ActMeans(float const* observations, std::uint8_t const* available, float* actions)
    {
        if (!trainer_.AtUpdateBoundary())
        {
            throw std::logic_error("PPOKit team learner: deterministic actions only between two rollouts");
        }
        for (std::size_t agent = 0; agent < capacity_; ++agent)
        {
            observationRows_[agent] = available[agent] != 0 ? observations + agent * observationSize_ : nullptr;
        }
        models_.ActorMeans(observationRows_.data(), capacity_, actions);
    }

    void RequireBoundary(char const* operation) const
    {
        if (!trainer_.AtUpdateBoundary())
        {
            throw std::logic_error(std::string("PPOKit team learner: ") + operation
                + " requires an update boundary (an empty rollout)");
        }
    }

    void Write(std::filesystem::path const& path, artifacts::ArtifactIdentity const& identity,
        artifacts::ArtifactKind kind)
    {
        artifacts::ArtifactHeader header;
        header.kind = kind;
        header.identity = identity;
        header.updates = updates_;
        header.seed = seed_;
        header.created = artifacts::UtcNow();
        bool const training = kind == artifacts::ArtifactKind::Training;
        artifacts::WriteFileAtomically(
            path,
            [&](std::ostream& out) {
                out << artifacts::HeaderLine(header);
                checkpoint::WriteTensors(out, "actor", models_.Actor().Parameters());
                if (!training)
                {
                    return;
                }
                checkpoint::WriteTensors(out, "critic", models_.Critic().Parameters());
                checkpoint::WriteAdam(out, "actor_adam", models_.ActorOptimizer().Save());
                checkpoint::WriteAdam(out, "critic_adam", models_.CriticOptimizer().Save());
                checkpoint::WriteRandomState(out, random_.Engine());
            },
            artifacts::AtomicWriteOptions{}, "PPOKit team learner checkpoint");
    }

    void Read(std::filesystem::path const& path, artifacts::ArtifactIdentity const& expected, bool training)
    {
        std::string const source = path.filename().string();
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error(source + ": cannot open the checkpoint");
        }
        artifacts::ArtifactHeader const header = artifacts::ReadCompatibleHeader(in, expected, source, training);
        // Everything is read before anything is applied: a damaged file leaves the learner as it was.
        std::vector<Parameter*> const actorParameters = models_.Actor().Parameters();
        checkpoint::Tensors const actor =
            checkpoint::ReadTensors(in, "actor", checkpoint::SizesOf(actorParameters), source);
        if (!training)
        {
            if (header.kind == artifacts::ArtifactKind::Policy)
            {
                artifacts::RequirePayloadEnd(in, source);
            }
            checkpoint::Apply(actor, actorParameters);
            return;
        }
        std::vector<Parameter*> const criticParameters = models_.Critic().Parameters();
        checkpoint::Tensors const critic =
            checkpoint::ReadTensors(in, "critic", checkpoint::SizesOf(criticParameters), source);
        Adam::State actorState = checkpoint::ReadAdam(in, "actor_adam", actorParameters, source);
        Adam::State criticState = checkpoint::ReadAdam(in, "critic_adam", criticParameters, source);
        std::mt19937 const engine = checkpoint::ReadRandomState(in, source);
        artifacts::RequirePayloadEnd(in, source);
        models_.ActorOptimizer().ValidateState(actorState);
        models_.CriticOptimizer().ValidateState(criticState);
        checkpoint::Apply(actor, actorParameters);
        checkpoint::Apply(critic, criticParameters);
        models_.ActorOptimizer().Restore(std::move(actorState));
        models_.CriticOptimizer().Restore(std::move(criticState));
        random_.Engine() = engine;
        seed_ = header.seed;
        updates_ = header.updates;
    }

    std::size_t capacity_;
    std::size_t observationSize_;
    std::size_t criticSize_;
    std::size_t actionSize_;
    std::uint64_t seed_;
    TeamGraphModels models_;
    StandardTeamRandom random_;
    runtime::HostedTrainer<TeamGraphModels, StandardTeamRandom> trainer_;
    TeamCriticBuilder critics_;
    NodeId featureNode_;
    std::size_t featureWidth_;
    std::vector<bool> members_;
    std::vector<bool> alive_;
    std::vector<runtime::AgentPresence> presence_;
    std::vector<float const*> observationRows_;
    std::vector<float const*> criticRows_;
    std::vector<float> means_;
    std::vector<float> features_;
    std::vector<float> criticValues_;
    std::vector<float> rewards_;
    std::vector<runtime::StepEnd> ends_;
    std::uint64_t updates_ = 0;
    bool episodeStarts_ = true;
};

} // namespace ppokit::graph
