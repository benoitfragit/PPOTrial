/**
 * @file PPOKit/tests/component_tests.cpp
 * @brief Component DLLs: loading and its messages, manifests and configurations, the
 *        wiring rules of graph.json, values and gradients through a graph, a learner
 *        training with a trainable component, and observers (Visualiseur blocks).
 */
#include "ppokit/component/component_node.hpp"
#include "ppokit/component/library.hpp"
#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/graph/probes.hpp"
#include "ppokit/runtime/observers.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "support/corridor_task.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

using namespace ppokit;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool Contains(std::string const& text, std::string const& part)
{
    return text.find(part) != std::string::npos;
}

/** @brief The message of what `body` throws, or empty. */
template<typename Body>
std::string ErrorOf(Body&& body)
{
    try
    {
        body();
    }
    catch (std::exception const& error)
    {
        return error.what();
    }
    return {};
}

std::filesystem::path const Plugin(PPOKIT_TEST_COMPONENTS);

void Loading()
{
    auto const library = component::Library::Open(Plugin.string(), {});
    Check(library->Count() == 7 && library->At(0).Id() == "scale@1" && library->At(1).Id() == "count_present@1",
        "the DLL lists its components");
    auto const byName = component::Library::Open(PPOKIT_TEST_COMPONENTS_NAME, Plugin.parent_path());
    Check(byName == library, "a bare name finds the same, already loaded, library");
    auto const scale = library->Find("scale", 1);
    Check(scale.manifest->capabilities.trainable && scale.manifest->config.size() == 1, "the manifest is read");
    Check(Contains(ErrorOf([&] { library->Find("scale", 2); }), "it provides: scale@1, count_present@1"),
        "an unknown version lists what the library provides");
    Check(Contains(ErrorOf([] { component::Library::Open("no_such_component", "."); }), "looked for:"),
        "a missing library says where it was looked for");
    Check(Contains(ErrorOf([] { component::Library::Open(PPOKIT_NOT_A_COMPONENT, {}); }),
              "not a PPOKit component library"),
        "a DLL without the entry point is refused with the reason");
}

void Configurations()
{
    auto const scale = component::Library::Open(Plugin.string(), {})->Find("scale", 1);
    Check(component::ValidateConfig(*scale.manifest, "{}") == R"({"initial":1.0})", "defaults are filled");
    Check(Contains(ErrorOf([&] { component::ValidateConfig(*scale.manifest, R"({"inital":2})"); }),
              "config.inital: unknown setting (settings: initial)"),
        "a misspelt setting lists the settings");
    Check(Contains(ErrorOf([&] { component::ValidateConfig(*scale.manifest, R"({"initial":20})"); }),
              "is outside [-10.0, 10.0]"),
        "an out-of-range setting gives the range");
    Check(component::ComponentHash(scale, R"({"initial":1.0})") != component::ComponentHash(scale, R"({"initial":2.0})"),
        "the hash changes with the configuration");
}

/** @brief A sample-task actor using the components, with `extra` nodes and edges spliced in. */
std::string Graph(std::string const& actorNodes, std::string const& actorEdges)
{
    return R"({ "schema": "ppokit.graph/2", "components": [")" + std::string(PPOKIT_TEST_COMPONENTS_NAME) + R"("],
      "actor": { "nodes": { "own": { "type": "input", "from": "observation.own" },
                            "red": { "type": "input", "from": "observation.red_tracks" },
                            )" + actorNodes + R"(,
                            "policy": { "type": "gaussian_head" } },
                 "edges": [ )" + actorEdges + R"( ] },
      "critic": { "nodes": { "own": { "type": "input", "from": "observation.own" },
                             "trunk": { "type": "mlp", "layers": 1, "width": 4, "activation": "tanh" },
                             "value": { "type": "value" } },
                  "edges": [ ["own", "trunk"], ["trunk", "value"] ] } })";
}

std::string const ValidNodes = R"("counted": { "type": "component:count_present@1" },
    "gated": { "type": "component:set_gate@1", "config": { "gate": 0.25 } },
    "pooled": { "type": "masked_pool", "mode": "mean" },
    "scaled": { "type": "component:scale@1", "config": { "initial": 2 } },
    "joined": { "type": "concat" },
    "trunk": { "type": "mlp", "layers": 1, "width": 8, "activation": "tanh" })";
std::string const ValidEdges = R"(["red", "counted"], ["red", "gated"], ["gated", "pooled"], ["own", "scaled"],
    ["scaled", "joined.in1"], ["counted", "joined.in2"], ["pooled", "joined.in3"], ["joined", "trunk"],
    ["trunk", "policy"])";

/** @brief Loss sum(policy * weights) and its gradient, for the gradient check. */
double Loss(graph::Graph& actor, std::vector<graph::Signal> const& inputs)
{
    graph::Signal const output = actor.Forward(inputs);
    double loss = 0.0;
    for (std::size_t index = 0; index < actor.OutputShape().width; ++index)
    {
        loss += static_cast<double>(output.values[index]) * static_cast<double>(index + 1);
    }
    return loss;
}

void ThroughAGraph(schema::TaskSchema const& task)
{
    auto const schema = schema::ParseBlockGraph(Graph(ValidNodes, ValidEdges), "graph.json", task, Plugin.parent_path());
    auto actor = schema::BuildGraph(schema.actor);
    std::mt19937 rng(5);
    actor.Initialize(rng);
    float const own[2] = {0.5F, -1.F};
    float const red[9] = {1.F, 2.F, 1.F, 7.F, 7.F, 0.F, 3.F, 4.F, 1.F};
    std::uint8_t const mask[3] = {1, 0, 1};
    std::vector<graph::Signal> const inputs{{own, nullptr}, {red, mask}};
    actor.Forward(inputs);
    Check(actor.OutputOf(actor.Find("counted")).values[0] == 2.F, "the counter sees the two present entities");
    Check(actor.OutputOf(actor.Find("scaled")).values[1] == -2.F, "scale multiplies by its configured weight");
    graph::Signal const gated = actor.OutputOf(actor.Find("gated"));
    Check(gated.values[0] == 0.25F && gated.mask == mask, "a set output follows its input's mask");

    // The trainable component's parameter gradient agrees with a finite difference.
    actor.ZeroGradients();
    Loss(actor, inputs);
    std::vector<float> outputGradient(actor.OutputShape().width);
    for (std::size_t index = 0; index < outputGradient.size(); ++index)
    {
        outputGradient[index] = static_cast<float>(index + 1);
    }
    actor.Backward(outputGradient.data());
    graph::Parameter* weights = nullptr;
    auto& node = dynamic_cast<component::ComponentNode&>(actor.At(actor.Find("scaled")));
    std::vector<graph::Parameter*> parameters;
    node.CollectParameters(parameters);
    weights = parameters.at(0);
    double const analytic = weights->gradient[0];
    float const step = 1e-3F;
    weights->value[0] += step;
    double const up = Loss(actor, inputs);
    weights->value[0] -= 2.F * step;
    double const down = Loss(actor, inputs);
    weights->value[0] += step;
    double const numeric = (up - down) / (2.0 * step);
    Check(std::abs(analytic - numeric) < 1e-3 * std::max(1.0, std::abs(numeric)),
        "the component's parameter gradient matches a finite difference (" + std::to_string(analytic) + " vs "
            + std::to_string(numeric) + ")");

    graph::GraphProbes probes(actor);
    actor.Forward(inputs);
    probes.ObserveForward();
    bool probed = false;
    for (auto const& block : probes.Summary())
    {
        probed = probed || (block.block == "scaled" && block.type == "component" && block.hasGradient);
    }
    Check(probed, "probes measure component blocks like the others");
}

void WiringRules(schema::TaskSchema const& task)
{
    auto const error = [&](std::string const& nodes, std::string const& edges) {
        return ErrorOf([&] { schema::ParseBlockGraph(Graph(nodes, edges), "graph.json", task, Plugin.parent_path()); });
    };
    std::string const afterTrainable = error(R"("encoded": { "type": "entity_encoder", "width": 4, "activation": "tanh" },
        "counted": { "type": "component:count_present@1" },
        "joined": { "type": "concat" },
        "trunk": { "type": "mlp", "layers": 1, "width": 8, "activation": "tanh" })",
        R"(["red", "encoded"], ["encoded", "counted"], ["own", "joined.in1"], ["counted", "joined.in2"],
        ["joined", "trunk"], ["trunk", "policy"])");
    Check(Contains(afterTrainable, "actor.nodes.counted: component count_present@1 has no gradient, so the trainable "
                                   "block 'encoded' before it could not learn"),
        "a component without gradient after a trainable block is refused: " + afterTrainable);
    // Attention has no parameters: a component without gradient after attention over raw inputs is fine.
    std::string const afterAttention = error(R"("query": { "type": "select", "indices": [0, 1, 0] },
        "attended": { "type": "masked_attention" },
        "after": { "type": "component:failing@1" },
        "trunk": { "type": "mlp", "layers": 1, "width": 8, "activation": "tanh" })",
        R"(["own", "query"], ["query", "attended.query"], ["red", "attended.set"], ["attended", "after"],
        ["after", "trunk"], ["trunk", "policy"])");
    Check(afterAttention.empty(), "attention alone is not a trainable block: " + afterAttention);
    std::string const stateful = error(R"("remembered": { "type": "component:memory@1" },
        "trunk": { "type": "mlp", "layers": 1, "width": 8, "activation": "tanh" },
        "pooled": { "type": "masked_pool", "mode": "max" }, "joined": { "type": "concat" })",
        R"(["own", "remembered"], ["remembered", "joined.in1"], ["red", "pooled"], ["pooled", "joined.in2"],
        ["joined", "trunk"], ["trunk", "policy"])");
    Check(Contains(stateful, "is stateful"), "stateful components are refused in graphs for now");
    std::string const unknown = error(R"("x": { "type": "component:radar@1" })", R"(["own", "x"], ["x", "policy"])");
    Check(Contains(unknown, "no component radar@1") && Contains(unknown, "scale@1"),
        "an unknown component lists those available: " + unknown);
    std::string const port = error(R"("gated": { "type": "component:set_gate@1" },
        "pooled": { "type": "masked_pool", "mode": "max" }, "joined": { "type": "concat" })",
        R"(["red", "gated.trakcs"], ["gated", "pooled"], ["pooled", "joined.in1"], ["own", "joined.in2"],
        ["joined", "policy"])");
    Check(Contains(port, "has no port 'trakcs'") && Contains(port, "tracks"), "a misspelt port lists the ports: " + port);
    std::string const kind = error(R"("gated": { "type": "component:set_gate@1" })",
        R"(["own", "gated"], ["gated", "policy"])");
    Check(Contains(kind, "port 'tracks' expects a set"), "a vector wired to a set port is named: " + kind);
    std::string const config = error(R"("scaled": { "type": "component:scale@1", "config": { "initial": 99 } })",
        R"(["own", "scaled"], ["scaled", "policy"])");
    Check(Contains(config, "actor.nodes.scaled.config") && Contains(config, "outside"), "a bad setting names the node");

    auto const failing = schema::ParseBlockGraph(Graph(R"("broken": { "type": "component:failing@1" },
        "pooled": { "type": "masked_pool", "mode": "max" }, "joined": { "type": "concat" })",
        R"(["own", "broken"], ["broken", "joined.in1"], ["red", "pooled"], ["pooled", "joined.in2"],
        ["joined", "policy"])"), "graph.json", task, Plugin.parent_path());
    auto actor = schema::BuildGraph(failing.actor);
    float const own[2] = {0.F, 0.F};
    float const red[9] = {};
    std::uint8_t const mask[3] = {};
    Check(Contains(ErrorOf([&] { actor.Forward({{own, nullptr}, {red, mask}}); }), "forward failed: sensor offline"),
        "a component's exception reaches PPOKit as its message");
}

/** @brief The corridor learns with a trainable component in front of its network. */
void Training()
{
    auto const task = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
    std::string const text = R"({ "schema": "ppokit.graph/2", "components": [")"
        + std::string(PPOKIT_TEST_COMPONENTS_NAME) + R"("],
      "actor": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                            "scaled": { "type": "component:scale@1" },
                            "trunk": { "type": "mlp", "layers": 2, "width": 16, "activation": "tanh" },
                            "policy": { "type": "gaussian_head" } },
                 "edges": [ ["state", "scaled"], ["scaled", "trunk"], ["trunk", "policy"] ] },
      "critic": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                             "trunk": { "type": "mlp", "layers": 2, "width": 16, "activation": "tanh" },
                             "value": { "type": "value" } },
                  "edges": [ ["state", "trunk"], ["trunk", "value"] ] } })";
    auto const graphs = schema::ParseBlockGraph(text, "graph.json", task, Plugin.parent_path());
    core::PpoSettings settings;
    settings.actorLearningRate = 3e-3;
    settings.criticLearningRate = 3e-3;
    settings.seed = 7;
    graph::GraphLearner<2, 1> learner(schema::BuildGraph(graphs.actor), schema::BuildGraph(graphs.critic),
        schema::MakePolicyLayout(graphs, task), settings);
    tests::CorridorTask corridor;
    graph::GraphLearner<2, 1>::Buffer buffer(256);
    runtime::RolloutCollector<tests::CorridorTask, graph::GraphLearner<2, 1>, graph::GraphLearner<2, 1>::Buffer>
        collector(corridor, learner);
    auto& scale = dynamic_cast<component::ComponentNode&>(learner.Actor().At(learner.Actor().Find("scaled")));
    std::vector<graph::Parameter*> parameters;
    scale.CollectParameters(parameters);
    float const before = parameters.at(0)->value[0];
    bool finite = true;
    for (int update = 0; update < 5; ++update)
    {
        collector.Collect(buffer);
        auto const statistics = learner.Update(buffer);
        finite = finite && std::isfinite(statistics.policyLoss) && std::isfinite(statistics.valueLoss);
    }
    Check(finite, "training with a component keeps finite losses");
    Check(parameters.at(0)->value[0] != before, "PPOKit's optimizer updates the component's parameters");
}

} // namespace

/** @brief A graph of the observer task: ValidNodes in the actor, and `observers` as given. */
std::string GraphWithObservers(std::string const& observers)
{
    std::string graph = Graph(ValidNodes, ValidEdges);
    graph.pop_back(); // the closing brace of the root object
    return graph + ", \"observers\": " + observers + " }";
}

/** @brief The checksum observer wired to a world, an observation and every action and reward. */
std::string CheckedObservers(double expectedSum)
{
    return R"({ "nodes": { "pose": { "type": "input", "from": "world.pose" },
                           "tracks": { "type": "input", "from": "observation.red_tracks" },
                           "controls": { "type": "input", "from": "action" },
                           "rewards": { "type": "input", "from": "reward" },
                           "check": { "type": "component:checksum@1",
                                      "config": { "expected_sum": )" + std::to_string(expectedSum) + R"(,
                                                  "expected_agents": 2 } } },
                "edges": [ ["pose", "check.pose"], ["tracks", "check.tracks"],
                           ["controls", "check.controls"], ["rewards", "check.rewards"] ] })";
}

void ObserverManifests()
{
    auto const manifestError = [](std::string const& ports, std::string const& capabilities) {
        std::string const json = R"({"schema":"ppokit.component/1","name":"watch","version":1,"ports":[)" + ports
            + R"(],"capabilities":{)" + capabilities + "}}";
        return ErrorOf([&] { component::ParseManifest(json, "watch"); });
    };
    std::string const in = R"({"name":"x","direction":"in","dtype":"f32","shape":[3]})";
    std::string const out = R"({"name":"y","direction":"out","dtype":"f32","shape":[3]})";
    Check(manifestError(in, R"("observer":true,"batch":true)").empty(), "an observer with inputs is accepted");
    Check(Contains(manifestError(in + "," + out, R"("observer":true,"batch":true)"), "has no output port ('y')"),
        "an observer with an output is refused");
    Check(Contains(manifestError(in, R"("observer":true,"batch":true,"differentiable":true)"), "only watches"),
        "a differentiable observer is refused");
    Check(Contains(manifestError(in, R"("observer":true)"), "must declare batch"),
        "an observer must take every agent at once");
}

void ObserverWiring()
{
    auto const task = schema::LoadTaskSchema(PPOKIT_OBSERVER_TASK);
    auto const parse = [&](std::string const& observers) {
        return schema::ParseBlockGraph(GraphWithObservers(observers), "graph.json", task, Plugin.parent_path());
    };
    auto const error = [&](std::string const& observers) { return ErrorOf([&] { parse(observers); }); };

    schema::BlockGraphSchema const graphs = parse(CheckedObservers(0.0));
    Check(graphs.observers.size() == 1 && graphs.observers[0].inputs.size() == 4, "the observer and its four ports");
    schema::ObservedInput const& tracks = graphs.observers[0].inputs[1];
    Check(tracks.signal == schema::ObservedSignal::Observation && tracks.offset == 7 && tracks.size == 9
            && tracks.shape == std::vector<std::size_t>{3, 3},
        "a set group is read where it is, with its per-agent shape");
    Check(graphs.observers[0].shapes == R"({"N":3})", "symbols are fixed by the wiring");
    auto const withoutObservers = schema::ParseBlockGraph(Graph(ValidNodes, ValidEdges), "graph.json", task,
        Plugin.parent_path());
    Check(schema::ArtifactIdentityOf(task, graphs).graphHash
            == schema::ArtifactIdentityOf(task, withoutObservers).graphHash,
        "observers do not change the identity of the networks");

    std::string const inNetwork = ErrorOf([&] {
        schema::ParseBlockGraph(Graph(ValidNodes + R"(, "seen": { "type": "component:checksum@1" })", ValidEdges),
            "graph.json", task, Plugin.parent_path());
    });
    Check(Contains(inNetwork, "is an observer: it goes in the \"observers\" section"),
        "an observer is refused in a network: " + inNetwork);
    std::string const notObserver = error(R"({ "nodes": { "view": { "type": "component:scale@1" } }, "edges": [] })");
    Check(Contains(notObserver, "is not an observer"), "a computing component is refused as observer: " + notObserver);
    std::string const unwired = error(R"({ "nodes": { "pose": { "type": "input", "from": "world.pose" },
        "check": { "type": "component:checksum@1" } }, "edges": [ ["pose", "check.pose"] ] })");
    Check(Contains(unwired, "port 'tracks' is not wired"), "every port of an observer is wired: " + unwired);
    std::string const unused = error(R"({ "nodes": { "pose": { "type": "input", "from": "world.pose" } },
        "edges": [] })");
    Check(Contains(unused, "input 'pose' (world.pose) feeds no observer"), "an unused input is named: " + unused);
    std::string const shape = error(R"({ "nodes": { "own": { "type": "input", "from": "observation.own" },
        "check": { "type": "component:checksum@1" } }, "edges": [ ["own", "check.pose"] ] })");
    Check(Contains(shape, "port 'pose' expects 3 on axis 0, 'observation.own' gives [2]"),
        "a wrong shape names port and source: " + shape);
    std::string const unknown = error(R"({ "nodes": { "wind": { "type": "input", "from": "world.wind" } },
        "edges": [] })");
    Check(Contains(unknown, "no world group 'wind' (groups: pose, zones)"), "an unknown group lists them: " + unknown);
    auto const sample = schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
    std::string const noWorld = ErrorOf([&] {
        schema::ParseBlockGraph(GraphWithObservers(CheckedObservers(0.0)), "graph.json", sample, Plugin.parent_path());
    });
    Check(Contains(noWorld, "task.json has no \"world\" section"), "world needs a world section: " + noWorld);
}

/** @brief Fills `values` with `value`. */
void Fill(std::vector<float>& values, float value)
{
    std::fill(values.begin(), values.end(), value);
}

void ObserverCalls()
{
    auto const task = schema::LoadTaskSchema(PPOKIT_OBSERVER_TASK);
    // Two agents: 9 track values of 1, 3 pose values of 2, 3 actions of 0.5 and 2 rewards of -1 each.
    double const expected = 2.0 * (9 * 1.0 + 3 * 2.0 + 3 * 0.5 - 2 * 1.0);
    auto const graphs = schema::ParseBlockGraph(GraphWithObservers(CheckedObservers(expected)), "graph.json", task,
        Plugin.parent_path());
    runtime::ObserverSet observers(task, graphs.observers, 4, 7);
    std::vector<float> observation(2 * task.observation.size);
    std::vector<float> world(2 * task.world->size);
    std::vector<float> actions(2 * task.actionSize);
    std::vector<float> rewards(2 * task.rewards.size());
    Fill(observation, 1.F);
    Fill(world, 2.F);
    Fill(actions, 0.5F);
    Fill(rewards, -1.F);
    runtime::ObservedStep step;
    step.agents = 2;
    step.observation = observation.data();
    step.world = world.data();
    step.actions = actions.data();
    step.rewards = rewards.data();
    observers.Notify(step);
    Check(observers.Active(0) && observers.Error(0).empty(), "the observer sees what graph.json wires to it: "
            + observers.Error(0));
    Fill(world, 3.F);
    observers.Notify(step);
    Check(!observers.Active(0) && Contains(observers.Error(0), "instead of"),
        "a failing observer is switched off with its reason, without stopping the run");
}

int main()
{
    try
    {
        Loading();
        Configurations();
        auto const task = schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
        ThroughAGraph(task);
        WiringRules(task);
        Training();
        ObserverManifests();
        ObserverWiring();
        ObserverCalls();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " component check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit component tests passed\n";
    return EXIT_SUCCESS;
}
