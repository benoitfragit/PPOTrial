/**
 * @file PPOKit/tests/block_graph_schema_tests.cpp
 * @brief `ppokit.graph/2`: shape inference on the real Blue and Flight tasks, execution of
 *        the built graphs, and one error message per kind of mistake.
 */
#include "ppokit/schema/block_graph_schema.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ppokit;
using schema::BlockGraphSchema;
using schema::SchemaError;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string ReadFile(char const* path)
{
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

/** @brief Replaces the first occurrence of `from`; the test fails loudly if it is absent. */
std::string Replace(std::string text, std::string const& from, std::string const& to)
{
    std::size_t const position = text.find(from);
    if (position == std::string::npos)
    {
        std::cerr << "test setup: '" << from << "' not found\n";
        ++failures;
        return text;
    }
    return text.replace(position, from.size(), to);
}

schema::BlockSpec const* FindBlock(schema::BlockNetworkSpec const& network, std::string const& name)
{
    for (schema::BlockSpec const& block : network.blocks)
    {
        if (block.name == name)
        {
            return &block;
        }
    }
    return nullptr;
}

/** @brief The attention actor and the pooled critic of Blue infer their shapes from task.json. */
void BlueShapes(schema::TaskSchema const& task, std::string const& text)
{
    BlockGraphSchema const graphs = schema::ParseBlockGraph(text, "graph.json", task);
    schema::BlockNetworkSpec const& actor = graphs.actor;
    Check((actor.inputs == std::vector<std::string>{"own", "red", "friends", "context"}),
        "blue: inputs keep the file order");
    Check((actor.outputs == std::vector<std::string>{"policy"}), "blue: the Gaussian head is the actor output");
    Check(FindBlock(actor, "red")->output == graph::Shape::Set(10, 13), "blue: red tracks are a set[10 x 13]");
    Check(FindBlock(actor, "red_embed")->output == graph::Shape::Set(10, 8), "blue: the encoder keeps the capacity");
    Check(FindBlock(actor, "features")->output == graph::Shape::Vector(7 + 10 + 8 + 8 + 8 + 6),
        "blue: concat sums its inputs");
    Check(FindBlock(actor, "policy")->output == graph::Shape::Vector(2), "blue: one mean per action component");
    Check(FindBlock(graphs.critic, "all")->output == graph::Shape::Vector(3 * 156), "blue: critic input width");
    Check((graphs.critic.outputs == std::vector<std::string>{"value"}), "blue: the value head is the critic output");

    // Blocks come after their sources.
    bool ordered = true;
    for (std::size_t index = 0; index < actor.blocks.size(); ++index)
    {
        for (std::string const& source : actor.blocks[index].sources)
        {
            bool before = false;
            for (std::size_t earlier = 0; earlier < index; ++earlier)
            {
                before = before || actor.blocks[earlier].name == source;
            }
            ordered = ordered && before;
        }
    }
    Check(ordered, "blue: blocks are in topological order");
}

/** @brief The built actor runs on random observations with partial sets. */
void BlueExecutes(schema::TaskSchema const& task, std::string const& text)
{
    BlockGraphSchema const graphs = schema::ParseBlockGraph(text, "graph.json", task);
    graph::Graph actor = schema::BuildGraph(graphs.actor);
    std::mt19937 rng(17);
    actor.Initialize(rng);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::vector<std::vector<float>> values;
    std::vector<std::vector<std::uint8_t>> masks;
    std::vector<graph::Signal> inputs;
    for (std::string const& name : graphs.actor.inputs)
    {
        graph::Shape const shape = FindBlock(graphs.actor, name)->output;
        values.emplace_back(shape.Size());
        for (float& value : values.back())
        {
            value = uniform(rng);
        }
        masks.emplace_back(shape.capacity, 0);
        for (std::size_t entity = 0; entity < shape.capacity; entity += 2)
        {
            masks.back()[entity] = 1;
        }
    }
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        bool const isSet = FindBlock(graphs.actor, graphs.actor.inputs[index])->output.kind == graph::SignalKind::Set;
        inputs.push_back({values[index].data(), isSet ? masks[index].data() : nullptr});
    }
    graph::Signal const means = actor.Forward(inputs);
    Check(std::isfinite(means.values[0]) && std::isfinite(means.values[1]), "blue: the built actor runs");
    Check(actor.OutputCount() == 1 && actor.InputCount() == 4, "blue: graph inputs and outputs follow the spec");

    graph::Graph critic = schema::BuildGraph(graphs.critic);
    critic.Initialize(rng);
    Check(critic.OutputShape() == graph::Shape::Vector(1), "blue: the critic outputs one value");
}

/** @brief Flight needs two heads: Gaussian controls, then the Bernoulli gear. */
void FlightTwoHeads(schema::TaskSchema const& task)
{
    std::string const text = R"({
      "schema": "ppokit.graph/2",
      "actor": {
        "nodes": {
          "runway": { "type": "input", "from": "observation.runway" },
          "phase": { "type": "input", "from": "observation.phase" },
          "all": { "type": "concat" },
          "trunk": { "type": "mlp", "layers": 2, "width": 16 },
          "controls": { "type": "gaussian_head" },
          "gear": { "type": "bernoulli_head" }
        },
        "edges": [ ["runway", "all.in1"], ["phase", "all.in2"], ["all", "trunk"],
                   ["trunk", "gear"], ["trunk", "controls"] ]
      },
      "critic": {
        "nodes": { "runway": { "type": "input", "from": "observation.runway" },
                   "value": { "type": "value" } },
        "edges": [ ["runway", "value"] ]
      }
    })";
    BlockGraphSchema const graphs = schema::ParseBlockGraph(text, "graph.json", task);
    Check((graphs.actor.outputs == std::vector<std::string>{"controls", "gear"}),
        "flight: Gaussian head first, Bernoulli head second");
    Check(FindBlock(graphs.actor, "controls")->output == graph::Shape::Vector(5), "flight: five controls");
    Check(FindBlock(graphs.actor, "gear")->output == graph::Shape::Vector(1), "flight: one gear logit");
    graph::Graph actor = schema::BuildGraph(graphs.actor);
    Check(actor.OutputCount() == 2, "flight: the built actor has two outputs");

    std::string const withoutGear =
        Replace(Replace(text, R"(["trunk", "gear"],)", ""), R"(,
          "gear": { "type": "bernoulli_head" })", "");
    try
    {
        schema::ParseBlockGraph(withoutGear, "graph.json", task);
        Check(false, "flight: a missing Bernoulli head is refused");
    }
    catch (SchemaError const& error)
    {
        Check(std::string(error.what()).find("actor.nodes: missing a bernoulli_head node") != std::string::npos,
            std::string("flight: missing Bernoulli head reported, got: ") + error.what());
    }
}

/**
 * @brief Separate attention keys, a log-std computed from the state and a critic block
 *        sharing an actor block: parsed, built, and their mistakes named. A network that
 *        uses none of them keeps its identity.
 */
void ExtendedBlocks(schema::TaskSchema const& blue, std::string const& blueText, schema::TaskSchema const& flight)
{
    std::string const keyed = Replace(Replace(blueText, R"("red_attend":     { "type": "masked_attention" },)",
                                          R"("red_attend":     { "type": "masked_attention", "keys": true },
                                             "red_keys":       { "type": "entity_encoder", "width": 8, "activation": "identity" },)"),
        R"(["red_embed", "red_attend.set"],)", R"(["red_embed", "red_attend.set"], ["red_embed", "red_keys"], ["red_keys", "red_attend.keys"],)");
    BlockGraphSchema const withKeys = schema::ParseBlockGraph(keyed, "graph.json", blue);
    schema::BlockSpec const* attend = FindBlock(withKeys.actor, "red_attend");
    Check(attend != nullptr && attend->keys && attend->sources.size() == 3 && attend->sources[2] == "red_keys",
        "attention keys: a third port, wired from the key projection");
    graph::Graph built = schema::BuildGraph(withKeys.actor);
    Check(built.OutputCount() == 1, "attention keys: the actor builds");
    BlockGraphSchema const plain = schema::ParseBlockGraph(blueText, "graph.json", blue);
    Check(schema::ArtifactIdentityOf(blue, plain).graphHash != schema::ArtifactIdentityOf(blue, withKeys).graphHash,
        "attention keys: part of the network identity");
    Check((schema::BlockInputPorts(schema::BlockType::MaskedAttention, 0, true)
              == std::vector<std::string>{"query", "set", "keys"}),
        "attention keys: the editor sees the keys port");

    std::string const shared = R"({
      "schema": "ppokit.graph/2",
      "actor": {
        "nodes": {
          "runway": { "type": "input", "from": "observation.runway" },
          "trunk": { "type": "mlp", "layers": 2, "width": 16, "activation": "relu" },
          "controls": { "type": "gaussian_head", "log_std": "state", "log_std_min": -20, "log_std_max": 2 },
          "gear": { "type": "bernoulli_head" }
        },
        "edges": [ ["runway", "trunk"], ["trunk", "controls"], ["trunk", "gear"] ]
      },
      "critic": {
        "nodes": { "runway": { "type": "input", "from": "observation.runway" },
                   "trunk": { "type": "mlp", "layers": 2, "width": 16, "activation": "relu", "share": "trunk" },
                   "value": { "type": "value" } },
        "edges": [ ["runway", "trunk"], ["trunk", "value"] ]
      }
    })";
    BlockGraphSchema const graphs = schema::ParseBlockGraph(shared, "graph.json", flight);
    schema::BlockSpec const* controls = FindBlock(graphs.actor, "controls");
    Check(controls != nullptr && controls->logStd.stateDependent && controls->logStd.minimum == -20.F
            && controls->logStd.maximum == 2.F,
        "state log-std: mode and bounds read");
    graph::PolicyLayout const layout = schema::MakePolicyLayout(graphs, flight);
    Check(layout.sharedBlocks.size() == 1 && layout.sharedBlocks[0].critic == "trunk"
            && layout.sharedBlocks[0].actor == "trunk",
        "share: the layout names the critic block and the actor block it uses");

    struct Case final
    {
        char const* description;
        std::string text;
        char const* expected;
    };
    Case const cases[] = {
        {"share in the actor", Replace(shared, R"("width": 16, "activation": "relu" },)",
                                   R"("width": 16, "activation": "relu", "share": "trunk" },)"),
            "actor.nodes.trunk.share: only a critic block shares the parameters of an actor block"},
        {"share an unknown block", Replace(shared, R"("share": "trunk")", R"("share": "body")"),
            "critic.nodes.trunk.share: no actor block 'body'"},
        {"share another layer", Replace(shared, R"("layers": 2, "width": 16, "activation": "relu", "share")",
                                    R"("layers": 2, "width": 32, "activation": "relu", "share")"),
            "critic.nodes.trunk.share: actor block 'trunk' is not the same layer"},
        {"log-std without bounds", Replace(shared, R"(, "log_std_min": -20, "log_std_max": 2)", ""),
            "actor.nodes.controls.log_std_min: a log-std computed from the state needs its bounds"},
        {"log-std mode", Replace(shared, R"("log_std": "state")", R"("log_std": "learned")"),
            "actor.nodes.controls.log_std: expected \"parameter\" or \"state\""},
        {"log-std bounds of a learned one", Replace(shared, R"("log_std": "state", )", ""),
            "actor.nodes.controls: log_std_min and log_std_max bound a log-std computed from the state"},
        {"keys flag", Replace(keyed, R"("keys": true)", R"("keys": "yes")"),
            "actor.nodes.red_attend.keys: expected true or false"},
        {"keys port unwired", Replace(keyed, R"(["red_keys", "red_attend.keys"],)", ""),
            "actor.nodes.red_attend.keys: input port is not wired"},
    };
    for (Case const& item : cases)
    {
        schema::TaskSchema const& task = item.text.find("red_attend") != std::string::npos ? blue : flight;
        try
        {
            schema::ParseBlockGraph(item.text, "graph.json", task);
            Check(false, std::string("error not detected: ") + item.description);
        }
        catch (SchemaError const& error)
        {
            Check(std::string(error.what()).find(item.expected) != std::string::npos,
                std::string(item.description) + ": got '" + error.what() + "'");
        }
    }
}

/** @brief LibTorch masking by fill value: read, part of the identity, refused where it means nothing. */
void MaskFill(schema::TaskSchema const& blue, std::string const& blueText)
{
    std::string const filled = Replace(Replace(Replace(blueText,
        R"("red_embed":      { "type": "entity_encoder", "width": 8, "activation": "tanh" },)",
        R"("red_embed":      { "type": "entity_encoder", "width": 8, "activation": "tanh", "encode_absent": true },)"),
        R"("red_attend":     { "type": "masked_attention" },)",
        R"("red_attend":     { "type": "masked_attention", "mask_fill": -1e9 },)"),
        R"("red_max":        { "type": "masked_pool", "mode": "max" },)",
        R"("red_max":        { "type": "masked_pool", "mode": "max", "mask_fill": -1e9 },)");
    BlockGraphSchema const graphs = schema::ParseBlockGraph(filled, "graph.json", blue);
    schema::BlockSpec const* embed = FindBlock(graphs.actor, "red_embed");
    schema::BlockSpec const* attend = FindBlock(graphs.actor, "red_attend");
    schema::BlockSpec const* pooled = FindBlock(graphs.actor, "red_max");
    Check(embed != nullptr && embed->encodeAbsent && attend != nullptr && attend->maskFill == -1e9F
            && pooled != nullptr && pooled->maskFill == -1e9F,
        "mask fill: encode_absent and mask_fill read");
    BlockGraphSchema const plain = schema::ParseBlockGraph(blueText, "graph.json", blue);
    Check(!FindBlock(plain.actor, "red_max")->maskFill && !FindBlock(plain.actor, "red_embed")->encodeAbsent,
        "mask fill: off unless written");
    Check(schema::ArtifactIdentityOf(blue, plain).graphHash != schema::ArtifactIdentityOf(blue, graphs).graphHash,
        "mask fill: part of the network identity");
    graph::Graph built = schema::BuildGraph(graphs.actor);
    Check(built.OutputCount() == 1, "mask fill: the actor builds");

    struct Case final
    {
        char const* description;
        std::string text;
        char const* expected;
    };
    Case const cases[] = {
        {"fill of a mean", Replace(blueText, R"("red_mean":       { "type": "masked_pool", "mode": "mean" },)",
                               R"("red_mean":       { "type": "masked_pool", "mode": "mean", "mask_fill": -1e9 },)"),
            "actor.nodes.red_mean.mask_fill: a fill value applies to a \"max\" pool only"},
        {"fill not a number", Replace(filled, R"("mask_fill": -1e9 },)", R"("mask_fill": "low" },)"),
            "mask_fill: expected a finite number"},
        {"encode_absent flag", Replace(filled, R"("encode_absent": true)", R"("encode_absent": 1)"),
            "actor.nodes.red_embed.encode_absent: expected true or false"},
        {"fill of a dense block", Replace(blueText, R"("red_embed":      { "type": "entity_encoder", "width": 8, "activation": "tanh" },)",
                                      R"("red_embed":      { "type": "entity_encoder", "width": 8, "activation": "tanh", "mask_fill": 0 },)"),
            "red_embed"},
    };
    for (Case const& item : cases)
    {
        try
        {
            schema::ParseBlockGraph(item.text, "graph.json", blue);
            Check(false, std::string("error not detected: ") + item.description);
        }
        catch (SchemaError const& error)
        {
            Check(std::string(error.what()).find(item.expected) != std::string::npos,
                std::string(item.description) + ": got '" + error.what() + "'");
        }
    }
}

/** @brief Each mistake is refused with the network, node, port or edge named. */
void Errors(schema::TaskSchema const& task, std::string const& text)
{
    struct Case final
    {
        char const* description;
        std::string text;
        char const* expected;
    };
    Case const cases[] = {
        {"unknown type", Replace(text, R"("type": "masked_attention" },)", R"("type": "attention" },)"),
            "actor.nodes.red_attend.type: unknown block type 'attention'"},
        {"vector on a set port", Replace(text, R"(["red", "red_embed"])", R"(["own", "red_embed"])"),
            "actor.nodes.red_embed.in: expects a set, node 'own' gives vector[7]"},
        {"query width", Replace(text, R"("friends_query":  { "type": "dense", "width": 6)",
                            R"("friends_query":  { "type": "dense", "width": 5)"),
            "actor.nodes.friends_attend.query: expects vector[6], node 'friends_query' gives vector[5]"},
        {"unwired port", Replace(text, R"(["red_embed", "red_attend.set"],)", ""),
            "actor.nodes.red_attend.set: input port is not wired"},
        {"port wired twice", Replace(text, R"(["red", "red_embed"])", R"(["red", "red_embed"], ["red", "red_embed.in"])"),
            "actor.edges[1]: port 'red_embed.in' is already wired from 'red' (actor.edges[0])"},
        {"bare attention target", Replace(text, R"(["red_query", "red_attend.query"])", R"(["red_query", "red_attend"])"),
            "node 'red_attend' has several ports, name one (query or set)"},
        {"concat gap", Replace(text, R"(["friends_attend", "features.in6"])", R"(["friends_attend", "features.in7"])"),
            "node 'features' has no port 'in7' (concat ports are in1..inN without gaps)"},
        {"cycle", Replace(text, R"(["own", "red_query"])", R"(["trunk", "red_query"])"),
            "actor.edges: cycle between nodes"},
        {"unused output",
            Replace(Replace(text, R"("red_max":)", R"("spare": { "type": "masked_pool", "mode": "mean" }, "red_max":)"),
                R"(["red_embed", "red_max"],)", R"(["red_embed", "red_max"], ["red_embed", "spare"],)"),
            "actor.nodes.spare: output is not used by any node"},
        {"unknown source", Replace(text, R"(["context", "features.in2"])", R"(["ctx", "features.in2"])"),
            "actor.edges[11]: unknown source node 'ctx'"},
        {"unknown group", Replace(text, R"("observation.patrol")", R"("observation.patrols")"),
            "actor.nodes.context.from: no group 'patrols' in observation"},
        {"actor reads critic", Replace(text, R"("observation.patrol")", R"("critic.own")"),
            "actor.nodes.context.from: the actor reads only observation groups"},
        {"head feeds a node",
            Replace(Replace(text, R"("trunk":)", R"("extra": { "type": "dense", "width": 2 }, "trunk":)"),
                R"(["trunk", "policy"])", R"(["trunk", "policy"], ["policy", "extra"])"),
            "actor.nodes.policy: a head is an output and cannot feed other nodes"},
        {"critic with an action head", Replace(text, R"("value": { "type": "value" })", R"("value": { "type": "gaussian_head" })"),
            "critic.nodes.value: a critic has no action head"},
        {"missing actor head", Replace(text, R"("type": "gaussian_head")", R"("type": "dense", "width": 2)"),
            "actor.nodes.policy: output is not used by any node"},
        {"pool mode", Replace(text, R"("mode": "max")", R"("mode": "sum")"),
            "actor.nodes.red_max.mode: expected \"mean\" or \"max\""},
        {"unknown key", Replace(text, R"("type": "mlp", "layers": 2, "width": 64,)", R"("type": "mlp", "depth": 2, "width": 64,)"),
            "actor.nodes.trunk"},
        {"select index", Replace(Replace(text, R"("context":        { "type": "input", "from": "observation.patrol" },)",
                                  R"("context_all":   { "type": "input", "from": "observation.patrol" },
                                     "context":       { "type": "select", "indices": [0, 10] },)"),
                             R"(["context", "features.in2"])", R"(["context_all", "context"], ["context", "features.in2"])"),
            "actor.nodes.context: PPOKit select: index 10 is outside an input of width 10"},
        {"select without indices", Replace(Replace(text, R"("context":        { "type": "input", "from": "observation.patrol" },)",
                                  R"("context_all":   { "type": "input", "from": "observation.patrol" },
                                     "context":       { "type": "select" },)"),
                             R"(["context", "features.in2"])", R"(["context_all", "context"], ["context", "features.in2"])"),
            "actor.nodes.context.indices: missing"},
        {"concat_sets of a vector", Replace(Replace(text, R"("friends":        { "type": "input", "from": "observation.blue_tracks" },)",
                                  R"("tracks":         { "type": "input", "from": "observation.blue_tracks" },
                                     "friends":        { "type": "concat_sets" },)"),
                             R"(["friends", "friends_embed"])", R"(["tracks", "friends.in1"], ["own", "friends.in2"], ["friends", "friends_embed"])"),
            "actor.nodes.friends.in2: expects a set, node 'own' gives vector[7]"},
        {"wrong version", Replace(text, "ppokit.graph/2", "ppokit.graph/1"), "schema: expected \"ppokit.graph/2\""},
    };
    for (Case const& item : cases)
    {
        try
        {
            schema::ParseBlockGraph(item.text, "graph.json", task);
            Check(false, std::string("error not detected: ") + item.description);
        }
        catch (SchemaError const& error)
        {
            Check(std::string(error.what()).find(item.expected) != std::string::npos,
                std::string(item.description) + ": got '" + error.what() + "'");
        }
    }
}

} // namespace

int main()
{
    try
    {
        schema::TaskSchema const blue = schema::LoadTaskSchema(PPOKIT_BLUE_TASK);
        schema::TaskSchema const flight = schema::LoadTaskSchema(PPOKIT_FLIGHT_TASK);
        std::string const blueGraph = ReadFile(PPOKIT_BLUE_ATTENTION_GRAPH);
        BlueShapes(blue, blueGraph);
        BlueExecutes(blue, blueGraph);
        FlightTwoHeads(flight);
        Errors(blue, blueGraph);
        ExtendedBlocks(blue, blueGraph, flight);
        MaskFill(blue, blueGraph);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " block graph check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit block graph schema tests passed\n";
    return 0;
}
