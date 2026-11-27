/**
 * @file PPOKit/tests/hosted_trainer_tests.cpp
 * @brief Hosted MAPPO: the test's own loop drives a team of three agents and calls the
 *        trainer each step, as a client simulation would.
 *
 * Team reach: each agent moves on a line towards its own target; the team shares one
 * reward, minus the mean distance to the targets. The third agent is destroyed halfway
 * through every episode: it stops moving, still counts in the team reward, and must not
 * train the actor. The centralized critic sees the whole team.
 *
 * Plus the mechanics with instrumented models: call order, padding rows, masked rows.
 * Smoke tests of the trainer, not convergence claims.
 */
#include "ppokit/graph/team_graph_models.hpp"
#include "ppokit/runtime/hosted_trainer.hpp"
#include "ppokit/schema/block_graph_schema.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using ppokit::runtime::AgentPresence;
using ppokit::runtime::HostedTrainer;
using ppokit::runtime::StepEnd;
using ppokit::runtime::TeamPpoSettings;

constexpr std::size_t Agents = 3;
using Models = ppokit::graph::TeamGraphModels;
using Trainer = HostedTrainer<Models, ppokit::graph::StandardTeamRandom>;
/** @brief One agent's rows in the team reach task. */
using Observation = std::array<float, 2>;
using CriticInput = std::array<float, 2 * Agents>;
using Actions = std::array<float, Agents>;

int failures = 0;

/** @brief One pointer per agent row, as the trainer takes them. */
template<typename Row, std::size_t Count>
std::array<float const*, Count> RowsOf(std::array<Row, Count> const& rows) noexcept
{
    std::array<float const*, Count> pointers{};
    for (std::size_t agent = 0; agent < Count; ++agent)
    {
        pointers[agent] = rows[agent].data();
    }
    return pointers;
}

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief The client's simulation: it owns its loop and its episode logic. */
class TeamReach final
{
public:
    static constexpr int EpisodeSteps = 12;
    static constexpr int DestroyedAt = 6;
    static constexpr float Speed = 0.25F;

    explicit TeamReach(std::uint32_t seed) : rng_(seed) { Reset(); }

    void Reset()
    {
        std::uniform_real_distribution<float> uniform(-1.F, 1.F);
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            position_[agent] = uniform(rng_);
            target_[agent] = uniform(rng_);
        }
        step_ = 0;
    }

    bool Available(std::size_t agent) const noexcept { return agent != Agents - 1 || step_ < DestroyedAt; }

    std::array<Observation, Agents> Observations() const
    {
        std::array<Observation, Agents> observations{};
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            observations[agent] = {position_[agent], target_[agent]};
        }
        return observations;
    }

    /** @brief Every agent's critic input is the whole team. */
    std::array<CriticInput, Agents> Critics() const
    {
        CriticInput team{};
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            team[2 * agent] = position_[agent];
            team[2 * agent + 1] = target_[agent];
        }
        std::array<CriticInput, Agents> critics{};
        critics.fill(team);
        return critics;
    }

    /** @brief Applies the actions of the available agents; returns the shared reward. */
    float Step(Actions const& actions)
    {
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            if (Available(agent))
            {
                float const velocity = std::clamp(actions[agent], -1.F, 1.F);
                position_[agent] = std::clamp(position_[agent] + Speed * velocity, -1.5F, 1.5F);
            }
        }
        ++step_;
        return -MeanDistance();
    }

    bool TimeUp() const noexcept { return step_ >= EpisodeSteps; }

    float MeanDistance() const noexcept
    {
        float distance = 0.F;
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            distance += std::abs(position_[agent] - target_[agent]);
        }
        return distance / static_cast<float>(Agents);
    }

private:
    std::mt19937 rng_;
    std::array<float, Agents> position_{};
    std::array<float, Agents> target_{};
    int step_ = 0;
};

/** @brief Mean final distance of the deterministic policy on fresh episodes. */
float EvaluatePolicy(Models& models, std::uint32_t seed)
{
    constexpr int Episodes = 100;
    TeamReach simulation(seed);
    float total = 0.F;
    for (int episode = 0; episode < Episodes; ++episode)
    {
        simulation.Reset();
        while (!simulation.TimeUp())
        {
            auto const observations = simulation.Observations();
            auto const rows = RowsOf(observations);
            Actions means{};
            models.ActorMeans(rows.data(), Agents, means.data());
            simulation.Step(means);
        }
        total += simulation.MeanDistance();
    }
    return total / Episodes;
}

/** @brief Graph models of the team reach task, the same for the same seed. */
Models MakeModels(std::uint32_t seed, double maximumGradientNorm = 0.0)
{
    auto const task = ppokit::schema::LoadTaskSchema(PPOKIT_TEAM_REACH_TASK);
    auto const graphs = ppokit::schema::LoadBlockGraph(PPOKIT_TEAM_REACH_GRAPH, task);
    ppokit::core::PpoSettings modelSettings;
    modelSettings.actorLearningRate = 3e-3;
    modelSettings.criticLearningRate = 3e-3;
    modelSettings.maxGradientNorm = maximumGradientNorm;
    modelSettings.seed = seed;
    return Models(ppokit::schema::BuildGraph(graphs.actor), ppokit::schema::BuildGraph(graphs.critic),
        ppokit::schema::MakeInputSlices(graphs.actor, "actor", task),
        ppokit::schema::MakeCriticInputSlices(graphs.critic, task), modelSettings, task.observation.size,
        task.critic->size, task.actionSize);
}

/** @brief The configured norm limit applies independently to actor and critic gradients. */
void TeamGradientClipping()
{
    constexpr double limit = 0.25;
    Models models = MakeModels(5, limit);
    std::array<float, 2> const observation{1.F, -1.F};
    std::array<float, 2 * Agents> const critic{1.F, -1.F, 2.F, -2.F, 3.F, -3.F};
    float const* observationRow[] = {observation.data()};
    float const* criticRow[] = {critic.data()};
    float output = 0.F;
    float const gradient = 100.F;
    models.ActorForward(observationRow, 1, &output);
    models.ActorBackwardAndStep(&gradient, &gradient);
    models.CriticForward(criticRow, 1, &output);
    models.CriticBackwardAndStep(&gradient);
    for (auto* graph : {&models.Actor(), &models.Critic()})
    {
        double squares = 0.0;
        for (auto const* parameter : graph->Parameters())
            for (float const value : parameter->gradient) squares += static_cast<double>(value) * value;
        double const norm = std::sqrt(squares);
        Check(norm > limit * 0.99 && norm <= limit * 1.00001,
            "team models: each optimizer clips its global gradient norm");
    }
}

/** @brief Whether two graphs hold the same parameter values, bit for bit. */
bool SameParameters(ppokit::graph::Graph& left, ppokit::graph::Graph& right)
{
    auto const leftParameters = left.Parameters();
    auto const rightParameters = right.Parameters();
    if (leftParameters.size() != rightParameters.size())
    {
        return false;
    }
    for (std::size_t tensor = 0; tensor < leftParameters.size(); ++tensor)
    {
        auto const& a = leftParameters[tensor]->value;
        auto const& b = rightParameters[tensor]->value;
        if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0)
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief A team of 3 in rows 0, 2 and 4 of 5 (rows 1 and 3 absent) trains exactly as the
 *        same team in 3 rows: same actions, statistics and weights, bit for bit.
 */
void AbsentRowsTrainAsASmallerTeam()
{
    constexpr std::size_t Rows = 5;
    constexpr std::array<std::size_t, Agents> RowOf{0, 2, 4};
    Models compact = MakeModels(5);
    Models spread = MakeModels(5);
    ppokit::graph::StandardTeamRandom compactRandom(11);
    ppokit::graph::StandardTeamRandom spreadRandom(11);
    TeamPpoSettings settings;
    settings.minibatch = 16;
    settings.epochs = 3;
    Trainer compactTrainer(compact, compactRandom, Agents, 24, settings);
    Trainer spreadTrainer(spread, spreadRandom, Rows, 24, settings);
    TeamReach simulation(3);
    bool same = true;
    int updates = 0;
    while (updates < 3)
    {
        auto const observations = simulation.Observations();
        auto const critics = simulation.Critics();
        auto const observationRows = RowsOf(observations);
        auto const criticRows = RowsOf(critics);
        std::array<AgentPresence, Agents> compactPresence{};
        std::array<float const*, Rows> spreadObservations{};
        std::array<float const*, Rows> spreadCritics{};
        std::array<AgentPresence, Rows> spreadPresence{};
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            AgentPresence const presence =
                simulation.Available(agent) ? AgentPresence::Active : AgentPresence::Destroyed;
            compactPresence[agent] = presence;
            spreadPresence[RowOf[agent]] = presence;
            spreadObservations[RowOf[agent]] = observationRows[agent];
            spreadCritics[RowOf[agent]] = criticRows[agent];
        }
        Actions compactActions{};
        std::array<float, Rows> spreadActions{};
        spreadActions.fill(7.F);
        compactTrainer.ChooseActions(observationRows.data(), criticRows.data(), compactPresence.data(),
            compactActions.data());
        spreadTrainer.ChooseActions(spreadObservations.data(), spreadCritics.data(), spreadPresence.data(),
            spreadActions.data());
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            same = same && compactActions[agent] == spreadActions[RowOf[agent]];
        }
        same = same && spreadActions[1] == 0.F && spreadActions[3] == 0.F;
        float const reward = simulation.Step(compactActions);
        StepEnd const end = simulation.TimeUp() ? StepEnd::Truncated : StepEnd::Running;
        auto const nextCriticValues = simulation.Critics();
        auto const nextCritics = RowsOf(nextCriticValues);
        std::array<float const*, Rows> spreadNextCritics{};
        std::array<float, Agents> compactRewards{};
        compactRewards.fill(reward);
        // Rewards of absent rows are ignored: give them noise.
        std::array<float, Rows> spreadRewards{reward, 99.F, reward, -99.F, reward};
        std::array<StepEnd, Agents> compactEnds{};
        compactEnds.fill(end);
        std::array<StepEnd, Rows> spreadEnds{};
        spreadEnds.fill(end);
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            spreadNextCritics[RowOf[agent]] = nextCritics[agent];
        }
        bool const compactUpdated =
            compactTrainer.Observe(nextCritics.data(), compactRewards.data(), compactEnds.data());
        bool const spreadUpdated =
            spreadTrainer.Observe(spreadNextCritics.data(), spreadRewards.data(), spreadEnds.data());
        same = same && compactUpdated == spreadUpdated;
        if (compactUpdated)
        {
            ++updates;
            auto const& a = compactTrainer.LastUpdate();
            auto const& b = spreadTrainer.LastUpdate();
            same = same && a.policyLoss == b.policyLoss && a.valueLoss == b.valueLoss
                && a.approximateKl == b.approximateKl && a.explainedVariance == b.explainedVariance
                && a.learnableRows == b.learnableRows && a.criticRows == b.criticRows
                && b.criticRows == Agents * 24;
        }
        if (simulation.TimeUp())
        {
            simulation.Reset();
        }
    }
    Check(same && SameParameters(compact.Actor(), spread.Actor()) && SameParameters(compact.Critic(), spread.Critic()),
        "absent rows: a team spread over larger rows trains bit for bit as the compact team");
}

void TeamLearnsToReach()
{
    Models models = MakeModels(5);
    ppokit::graph::StandardTeamRandom random(11);
    TeamPpoSettings settings;
    settings.minibatch = 64;
    settings.epochs = 4;
    Trainer trainer(models, random, Agents, 96, settings);
    // Probes stay attached the whole run: they only read, so the team must learn as without them.
    ppokit::graph::GraphProbes actorProbes(models.Actor());
    ppokit::graph::GraphProbes criticProbes(models.Critic());
    models.SetProbes(&actorProbes, &criticProbes);

    float const before = EvaluatePolicy(models, 777);
    TeamReach simulation(3);
    int updates = 0;
    bool availableRowsLearnable = true;
    while (updates < 60)
    {
        // The client's loop: observe, ask the trainer, advance, report.
        auto const observations = simulation.Observations();
        auto const critics = simulation.Critics();
        std::array<AgentPresence, Agents> presence{};
        for (std::size_t agent = 0; agent < Agents; ++agent)
        {
            presence[agent] = simulation.Available(agent) ? AgentPresence::Active : AgentPresence::Destroyed;
        }
        Actions actions{};
        auto const observationRows = RowsOf(observations);
        auto const criticRows = RowsOf(critics);
        trainer.ChooseActions(observationRows.data(), criticRows.data(), presence.data(), actions.data());
        float const reward = simulation.Step(actions);
        StepEnd const end = simulation.TimeUp() ? StepEnd::Truncated : StepEnd::Running;
        auto const nextCriticValues = simulation.Critics();
        auto const nextCritics = RowsOf(nextCriticValues);
        std::array<float, Agents> rewards{};
        rewards.fill(reward);
        std::array<StepEnd, Agents> ends{};
        ends.fill(end);
        if (trainer.CollectedTeamSteps() + 1 == trainer.TeamSteps())
        {
            // Last step before the update: the always available agent 0 is learnable on every row.
            auto const& rows = trainer.RolloutRows();
            for (std::size_t step = 0; step + 1 < trainer.TeamSteps(); ++step)
            {
                availableRowsLearnable = availableRowsLearnable && rows[step * Agents].learnable;
            }
        }
        bool const lastUpdate = updates + 1 == 60;
        if (lastUpdate && trainer.CollectedTeamSteps() + 1 == trainer.TeamSteps())
        {
            // The update starts inside the next Observe: probe only that one.
            actorProbes.Reset();
            criticProbes.Reset();
        }
        if (trainer.Observe(nextCritics.data(), rewards.data(), ends.data()))
        {
            ++updates;
            auto const& statistics = trainer.LastUpdate();
            Check(std::isfinite(statistics.policyLoss) && std::isfinite(statistics.valueLoss),
                "team reach: finite losses");
            Check(statistics.learnableRows < statistics.criticRows,
                "team reach: destroyed rows train the critic only");
        }
        if (simulation.TimeUp())
        {
            simulation.Reset();
        }
    }
    float const after = EvaluatePolicy(models, 777);
    std::cout << "hosted team reach: mean final distance " << before << " -> " << after << ", explained variance "
              << trainer.LastUpdate().explainedVariance << '\n';
    Check(availableRowsLearnable, "team reach: available agents are learnable");
    Check(after < 0.6F * before && after < 0.25F, "team reach: the team learns to reach its targets");
    Check(trainer.AtUpdateBoundary() && trainer.CompletedUpdates() == 60, "an update leaves the trainer at a boundary");
    auto const probed = actorProbes.Summary();
    bool measured = !probed.empty() && !criticProbes.Summary().empty();
    for (auto const& block : probed)
    {
        measured = measured && block.samples > 0 && (!block.hasGradient || block.gradientNorm > 0.0);
    }
    Check(measured, "team reach: the probes measure every block of the last update");
}

/** @brief Models that record what the trainer sends, for the mechanics. */
struct RecordingModels final
{
    std::size_t paddingRows = 0;
    std::size_t actorSteps = 0;
    std::size_t criticSteps = 0;
    float maskedGradient = 0.F;
    float trainingMeanShift = 0.F;
    std::vector<float const*> lastActorRows;

    std::size_t ObservationSize() const noexcept { return 1; }
    std::size_t CriticSize() const noexcept { return 1; }
    std::size_t ActionSize() const noexcept { return 1; }
    float LogStd(std::size_t) const { return -1.F; }
    void ActorMeans(float const* const* rows, std::size_t count, float* means)
    {
        for (std::size_t row = 0; row < count; ++row)
        {
            means[row] = rows[row] == nullptr ? 0.F : 0.1F * rows[row][0];
        }
    }
    void CriticValues(float const* const* rows, std::size_t count, float* values)
    {
        for (std::size_t row = 0; row < count; ++row)
        {
            paddingRows += rows[row] == nullptr ? 1 : 0;
            values[row] = rows[row] == nullptr ? 0.F : rows[row][0];
        }
    }
    void ActorForward(float const* const* rows, std::size_t count, float* means)
    {
        lastActorRows.assign(rows, rows + count);
        ActorMeans(rows, count, means);
        for (std::size_t row = 0; row < count; ++row)
        {
            if (rows[row] != nullptr) means[row] += trainingMeanShift;
        }
    }
    void ActorBackwardAndStep(float const* gradients, float const*)
    {
        // Observations of destroyed agents are negative in this test.
        for (std::size_t row = 0; row < lastActorRows.size(); ++row)
        {
            if (lastActorRows[row] != nullptr && lastActorRows[row][0] < 0.F)
            {
                maskedGradient += std::abs(gradients[row]);
            }
        }
        ++actorSteps;
    }
    /** @brief Training forward: same values, padding not counted (only the baselines are). */
    void CriticForward(float const* const* rows, std::size_t count, float* values)
    {
        for (std::size_t row = 0; row < count; ++row)
        {
            values[row] = rows[row] == nullptr ? 0.F : rows[row][0];
        }
    }
    void CriticBackwardAndStep(float const*) { ++criticSteps; }
};

/** @brief A KL violation skips the actor step while the critic still learns. */
void TargetKlGuard()
{
    for (float const target : {0.F, 0.01F})
    {
        RecordingModels models;
        ppokit::graph::StandardTeamRandom random(7);
        TeamPpoSettings settings;
        settings.minibatch = 2;
        settings.epochs = 2;
        settings.targetKl = target;
        HostedTrainer<RecordingModels, ppokit::graph::StandardTeamRandom> trainer(models, random, 2, 2, settings);
        using Row = std::array<float, 1>;
        std::array<Row, 2> const observations{{{1.F}, {-1.F}}};
        std::array<Row, 2> const critics{{{0.5F}, {0.5F}}};
        auto const observationRows = RowsOf(observations);
        auto const criticRows = RowsOf(critics);
        std::array<AgentPresence, 2> const presence{AgentPresence::Active, AgentPresence::Destroyed};
        std::array<float, 2> actions{};
        std::array<float, 2> const rewards{1.F, 1.F};
        std::array<StepEnd, 2> const ends{StepEnd::Running, StepEnd::Running};
        for (int step = 0; step < 2; ++step)
        {
            trainer.ChooseActions(observationRows.data(), criticRows.data(), presence.data(), actions.data());
            if (step == 1) models.trainingMeanShift = 100.F;
            trainer.Observe(criticRows.data(), rewards.data(), ends.data());
        }
        auto const& statistics = trainer.LastUpdate();
        Check(models.criticSteps == 4, "target KL: critic updates every minibatch");
        if (target == 0.F)
            Check(models.actorSteps > 0 && statistics.actorKlSkips == 0, "target KL: zero disables the guard");
        else
            Check(models.actorSteps == 0 && statistics.actorKlSkips > 0, "target KL: violation skips actor steps");
    }
}

void Mechanics()
{
    RecordingModels models;
    ppokit::graph::StandardTeamRandom random(1);
    TeamPpoSettings settings;
    settings.minibatch = 4;
    settings.epochs = 2;
    HostedTrainer<RecordingModels, ppokit::graph::StandardTeamRandom> trainer(models, random, 2, 5, settings);
    using Row = std::array<float, 1>;

    bool refused = false;
    try
    {
        std::array<Row, 2> const critics{};
        std::array<float, 2> rewards{};
        std::array<StepEnd, 2> ends{};
        trainer.Observe(RowsOf(critics).data(), rewards.data(), ends.data());
    }
    catch (std::logic_error const&)
    {
        refused = true;
    }
    Check(refused, "Observe before ChooseActions is refused");

    bool updated = false;
    for (int step = 0; step < 5; ++step)
    {
        // Agent 1 is destroyed: its observation is negative so the models can spot its rows.
        std::array<Row, 2> const observations{{{1.F + static_cast<float>(step)}, {-1.F}}};
        auto const observationRows = RowsOf(observations);
        std::array<Row, 2> const criticValues{{{0.5F}, {0.5F}}};
        auto const critics = RowsOf(criticValues);
        std::array<AgentPresence, 2> presence{AgentPresence::Active, AgentPresence::Destroyed};
        std::array<float, 2> actions{};
        trainer.ChooseActions(observationRows.data(), critics.data(), presence.data(), actions.data());
        if (step == 0)
        {
            refused = false;
            try
            {
                trainer.ChooseActions(observationRows.data(), critics.data(), presence.data(), actions.data());
            }
            catch (std::logic_error const&)
            {
                refused = true;
            }
            Check(refused, "a second ChooseActions without feedback is refused");
        }
        std::array<float, 2> rewards{1.F, 1.F};
        std::array<StepEnd, 2> ends{StepEnd::Running, StepEnd::Running};
        updated = trainer.Observe(critics.data(), rewards.data(), ends.data());
    }
    Check(updated, "the fifth team step fills the rollout and updates");
    // 10 rows, minibatches of 4: 3 batches per epoch, the last one with 2 padding rows.
    Check(models.criticSteps == 3 * 2 && models.actorSteps <= 3 * 2, "one optimizer step per minibatch and network");
    Check(models.paddingRows == 2 * 2, "baselines pad the last minibatch (value and next value)");
    Check(models.maskedGradient == 0.F, "destroyed agents send no gradient to the actor");
    Check(trainer.LastUpdate().learnableRows == 5 && trainer.LastUpdate().criticRows == 10,
        "statistics count available and critic rows");
}

} // namespace

int main()
{
    try
    {
        Mechanics();
        TargetKlGuard();
        TeamGradientClipping();
        AbsentRowsTrainAsASmallerTeam();
        TeamLearnsToReach();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " hosted trainer check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit hosted trainer tests passed\n";
    return 0;
}
