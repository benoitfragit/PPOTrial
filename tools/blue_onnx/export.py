"""Read-only V10/V7 Blue MAPPO checkpoint -> ONNX exporter.

The graph represents inference. It never mutates a checkpoint or training state.
The contract is defined by generalized_checkpoint.hpp and
generalized_training_checkpoint.hpp; only Blue RLTools is supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_SIZE = 295
COMPACT_SIZE = 156
ACTOR_WIDTH = 128
CRITIC_SIZE = 470
CRITIC_WIDTH = 256
SEMANTIC = "UNIFIED_PHYSICAL_GUIDANCE_V1"


class CheckpointError(ValueError):
    """Invalid, truncated or incompatible checkpoint."""


class Reader:
    def __init__(self, source: str):
        self.tokens = source.split()
        self.index = 0

    def read(self) -> str:
        if self.index >= len(self.tokens):
            raise CheckpointError("Truncated checkpoint")
        value = self.tokens[self.index]
        self.index += 1
        return value

    def expect(self, token: str) -> None:
        actual = self.read()
        if actual != token:
            raise CheckpointError(f"Expected {token!r}; got {actual!r}")

    def integer(self) -> int:
        item = self.read()
        try:
            value = int(item)
        except ValueError as exc:
            raise CheckpointError(f"Invalid integer: {item!r}") from exc
        if value < 0:
            raise CheckpointError("Negative checkpoint count")
        return value

    def matrix(self, rows: int, columns: int) -> np.ndarray:
        end = self.index + rows * columns
        if end > len(self.tokens):
            raise CheckpointError("Truncated weight matrix")
        try:
            matrix = np.asarray(
                self.tokens[self.index:end], dtype=np.float32
            ).reshape(rows, columns)
        except ValueError as exc:
            raise CheckpointError("Invalid floating-point weight") from exc
        self.index = end
        if not np.isfinite(matrix).all():
            raise CheckpointError("Nonfinite checkpoint weight")
        return matrix

    def end(self) -> None:
        if self.index != len(self.tokens):
            raise CheckpointError("Unexpected trailing checkpoint data")


def _header(reader: Reader, full: bool) -> dict:
    reader.expect("BLUE_TEAM_TRAINING_V7" if full else "BLUE_TEAM_MAPPO_V10")
    expected = ((7, 10, RAW_SIZE, COMPACT_SIZE, 2, ACTOR_WIDTH, 3,
                 CRITIC_SIZE, CRITIC_WIDTH, 3) if full
                else (10, 10, RAW_SIZE, COMPACT_SIZE, 2, ACTOR_WIDTH, 3))
    actual = tuple(reader.integer() for _ in expected)
    if actual != expected:
        raise CheckpointError(
            f"Unsupported Blue {'V7' if full else 'V10'} header: {actual}"
        )
    counts = tuple(reader.integer() for _ in range(7))
    if (not 1 <= counts[0] <= 9 or not 0 <= counts[1] <= 10
            or counts[2:] != (9, 10, 8, 1, 61)):
        raise CheckpointError(f"Invalid team capacities: {counts}")
    reader.expect(SEMANTIC)
    return {"blue": counts[0], "red": counts[1]}


def _attention(reader: Reader, width: int, features: int) -> dict:
    return {
        "encoder_w": reader.matrix(width, features),
        "encoder_b": reader.matrix(1, width),
        "query_w": reader.matrix(width, 7),
        "query_b": reader.matrix(1, width),
    }


def _layer(reader: Reader, inputs: int, outputs: int) -> dict:
    return {"w": reader.matrix(outputs, inputs),
            "b": reader.matrix(1, outputs)}


def _actor(reader: Reader) -> tuple[dict, dict]:
    metadata = _header(reader, False)
    return {
        "red": _attention(reader, 8, 13),
        "friendly": _attention(reader, 6, 9),
        "layers": [
            _layer(reader, COMPACT_SIZE, ACTOR_WIDTH),
            _layer(reader, ACTOR_WIDTH, ACTOR_WIDTH),
            _layer(reader, ACTOR_WIDTH, 2),
        ],
        "log_std": reader.matrix(1, 2),
    }, metadata


def _skip_moments(reader: Reader, rows: int, cols: int) -> None:
    reader.matrix(rows, cols)
    reader.matrix(rows, cols)


def _skip_actor_moments(reader: Reader) -> None:
    for width, features in ((8, 13), (6, 9)):
        for rows, cols in ((width, features), (1, width),
                           (width, 7), (1, width)):
            _skip_moments(reader, rows, cols)
    for inputs, outputs in ((COMPACT_SIZE, ACTOR_WIDTH),
                            (ACTOR_WIDTH, ACTOR_WIDTH),
                            (ACTOR_WIDTH, 2)):
        _skip_moments(reader, outputs, inputs)
        _skip_moments(reader, 1, outputs)
    _skip_moments(reader, 1, 2)


def _critic(reader: Reader) -> dict:
    layers = []
    for inputs, outputs in ((CRITIC_SIZE, CRITIC_WIDTH),
                            (CRITIC_WIDTH, CRITIC_WIDTH),
                            (CRITIC_WIDTH, 1)):
        layers.append(_layer(reader, inputs, outputs))
        _skip_moments(reader, outputs, inputs)
        _skip_moments(reader, 1, outputs)
    return {"layers": layers}


@dataclass(frozen=True)
class Checkpoint:
    actor: dict
    critic: dict | None
    metadata: dict
    format: str


def parse_checkpoint(data: str) -> Checkpoint:
    reader = Reader(data)
    if not reader.tokens:
        raise CheckpointError("Empty checkpoint")
    if reader.tokens[0] == "BLUE_TEAM_MAPPO_V10":
        actor, metadata = _actor(reader)
        reader.end()
        return Checkpoint(actor, None, metadata, "V10")
    if reader.tokens[0] != "BLUE_TEAM_TRAINING_V7":
        raise CheckpointError("Expected generalized Blue V10 or full V7")
    meta = _header(reader, True)
    reader.expect("ACTOR")
    actor, actor_meta = _actor(reader)
    if meta != actor_meta:
        raise CheckpointError("V7/V10 team metadata mismatch")
    reader.expect("ACTOR_MOMENTS")
    _skip_actor_moments(reader)
    reader.expect("CRITIC")
    critic = _critic(reader)
    reader.expect("OPTIMIZERS")
    for _ in range(2):
        values = reader.matrix(1, 8)
        if values[0, 0] <= 0 or values[0, 3] <= 0:
            raise CheckpointError("Invalid Adam optimizer parameters")
        if reader.integer() < 1:
            raise CheckpointError("Invalid Adam age")
    reader.expect("UPDATES")
    updates = reader.integer()
    reader.expect("RNG")
    if (reader.index >= len(reader.tokens) - 1
            or reader.tokens[-1] != "END_BLUE_TEAM_TRAINING"):
        raise CheckpointError("Missing RNG or V7 terminator")
    # The RNG serializes implementation-specific engine state; not needed by ONNX.
    reader.index = len(reader.tokens)
    reader.end()
    return Checkpoint(actor, critic, {**meta, "updates": updates}, "V7")


class Graph:
    def __init__(self):
        from onnx import helper, numpy_helper
        self.helper = helper
        self.numpy_helper = numpy_helper
        self.nodes = []
        self.initializers = []

    def literal(self, name: str, value, dtype=np.float32) -> str:
        array = np.asarray(value, dtype=dtype)
        self.initializers.append(self.numpy_helper.from_array(array, name))
        return name

    def op(self, operation: str, name: str, *inputs: str, **attributes) -> str:
        self.nodes.append(
            self.helper.make_node(operation, list(inputs), [name],
                                  name=name, **attributes)
        )
        return name

    def cut(self, name: str, input_name: str,
            start: int, stop: int, axis: int = 1) -> str:
        return self.op(
            "Slice", name, input_name,
            self.literal(name + "_start", [start], np.int64),
            self.literal(name + "_stop", [stop], np.int64),
            self.literal(name + "_axis", [axis], np.int64)
        )

    def expand(self, name: str, input_name: str, axis: int) -> str:
        return self.op("Unsqueeze", name, input_name,
                       self.literal(name + "_axis", [axis], np.int64))

    def reduce(self, operation: str, name: str,
               input_name: str, axis: int, keepdims: int = 0) -> str:
        return self.op(operation, name, input_name,
                       self.literal(name + "_axis", [axis], np.int64),
                       keepdims=keepdims)

    def dense(self, name: str, x: str, layer: dict, relu: bool) -> str:
        weights = self.literal(name + "_weights_T", layer["w"].T.copy())
        bias = self.literal(name + "_bias", layer["b"])
        affine = self.op(
            "Add", name + "_affine",
            self.op("MatMul", name + "_matmul", x, weights), bias
        )
        return self.op("Relu", name + "_relu", affine) if relu else affine

    def mlp(self, prefix: str, x: str, layers: list) -> str:
        for index, layer in enumerate(layers):
            x = self.dense(f"{prefix}_layer_{index}", x, layer,
                           index != len(layers) - 1)
        return x

    def model(self, title: str, input_name: str, input_size: int,
              outputs: list[tuple[str, str, list]], properties: dict):
        import onnx
        from onnx import TensorProto
        for public_name, internal_name, _ in outputs:
            if public_name != internal_name:
                self.op("Identity", public_name, internal_name)
        graph = self.helper.make_graph(
            self.nodes, title,
            [self.helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, ["batch", input_size]
            )],
            [self.helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
             for name, _, dims in outputs],
            initializer=self.initializers,
        )
        model = self.helper.make_model(
            graph, producer_name="PPOTrial-blue-onnx",
            opset_imports=[self.helper.make_operatorsetid("", 18)]
        )
        # Portable across ONNX Runtime versions that support opset 18.
        model.ir_version = 10
        self.helper.set_model_props(
            model, {k: str(v) for k, v in properties.items()}
        )
        onnx.checker.check_model(model)
        return model


def _masked_attention(graph: Graph, prefix: str, x: str, own: str,
                      offset: int, number: int, feature_count: int,
                      width: int, weights: dict) -> str:
    flat = graph.cut(prefix + "_flat", x, offset,
                     offset + number * feature_count)
    tracks = graph.op(
        "Reshape", prefix + "_tracks", flat,
        graph.literal(prefix + "_shape", [-1, number, feature_count], np.int64)
    )
    mask = graph.cut(prefix + "_mask", tracks, feature_count - 1,
                     feature_count, 2)
    present = graph.op("Greater", prefix + "_present_3d", mask,
                       graph.literal(prefix + "_mask_threshold", [0.5]))
    present = graph.op(
        "Squeeze", prefix + "_present", present,
        graph.literal(prefix + "_squeeze_axis", [2], np.int64)
    )
    mask_f = graph.op("Cast", prefix + "_mask_float", present, to=1)
    mask_expanded = graph.expand(prefix + "_mask_expanded", mask_f, 2)
    features = graph.op(
        "Tanh", prefix + "_features",
        graph.op(
            "Add", prefix + "_encode_affine",
            graph.op(
                "MatMul", prefix + "_encode_mul", tracks,
                graph.literal(prefix + "_encoder_weights_T",
                              weights["encoder_w"].T.copy())
            ),
            graph.literal(prefix + "_encoder_bias", weights["encoder_b"])
        )
    )
    query = graph.op(
        "Tanh", prefix + "_query",
        graph.op(
            "Add", prefix + "_query_affine",
            graph.op(
                "MatMul", prefix + "_query_mul", own,
                graph.literal(prefix + "_query_weights_T",
                              weights["query_w"].T.copy())
            ),
            graph.literal(prefix + "_query_bias", weights["query_b"])
        )
    )
    dot = graph.op(
        "Mul", prefix + "_dot",
        features, graph.expand(prefix + "_query_expand", query, 1)
    )
    scores = graph.op(
        "Mul", prefix + "_scores",
        graph.reduce("ReduceSum", prefix + "_dot_sum", dot, 2),
        graph.literal(prefix + "_score_scale", [1 / math.sqrt(width)])
    )
    masked = graph.op(
        "Where", prefix + "_masked_scores", present, scores,
        graph.literal(prefix + "_negative_score", [-1e9])
    )
    softmax = graph.op("Softmax", prefix + "_softmax", masked, axis=1)
    unnormalized = graph.op("Mul", prefix + "_masked_probs", softmax, mask_f)
    denom = graph.op(
        "Max", prefix + "_denom_safe",
        graph.reduce("ReduceSum", prefix + "_prob_sum",
                     unnormalized, 1, 1),
        graph.literal(prefix + "_epsilon", [1e-12])
    )
    probs = graph.op("Div", prefix + "_probs", unnormalized, denom)
    context = graph.reduce(
        "ReduceSum", prefix + "_context",
        graph.op(
            "Mul", prefix + "_weighted_features", features,
            graph.expand(prefix + "_prob_expand", probs, 2)
        ), 1
    )
    count = graph.reduce("ReduceSum", prefix + "_count", mask_f, 1, 1)
    mean = graph.op(
        "Div", prefix + "_mean",
        graph.reduce(
            "ReduceSum", prefix + "_features_sum",
            graph.op("Mul", prefix + "_valid_features",
                     features, mask_expanded), 1
        ),
        graph.op("Max", prefix + "_safe_count", count,
                 graph.literal(prefix + "_one", [1]))
    )
    maxima = graph.reduce(
        "ReduceMax", prefix + "_maximum_raw",
        graph.op(
            "Where", prefix + "_max_candidates",
            graph.expand(prefix + "_present_expand", present, 2), features,
            graph.literal(prefix + "_minus_large", [-1e9])
        ), 1
    )
    maximum = graph.op(
        "Where", prefix + "_maximum",
        graph.op("Greater", prefix + "_has_tracks", count,
                 graph.literal(prefix + "_zero", [0])),
        maxima,
        graph.literal(prefix + "_empty_max",
                      np.zeros((1, width), dtype=np.float32))
    )
    return graph.op(
        "Concat", prefix + "_context_mean_max",
        context, mean, maximum, axis=1
    )


def actor_model(actor: dict, properties: dict):
    graph = Graph()
    raw = "raw_observation"
    own = graph.cut("own_state", raw, 0, 7)
    own_lidar = graph.cut("own_lidar", raw, 0, 68)
    red = _masked_attention(graph, "red", raw, own, 70, 10, 13, 8,
                            actor["red"])
    friendly = _masked_attention(graph, "friendly_objective", raw, own,
                                 200, 9, 9, 6, actor["friendly"])
    # Exact generalized_attention_actor.hpp::transform compact layout:
    # 0..67 own+LiDAR, 68..91 red, 92..93 team counts,
    # 94..111 friendly+objective, 112..121 patrol/role,
    # 122 global 0, 123 formation slot, 124..155 unused legacy center.
    zero_tail = graph.op(
        "Mul", "unused_legacy_center_zero",
        graph.cut("legacy_center_size", raw, 0, 32),
        graph.literal("literal_zero", [0])
    )
    compact = graph.op(
        "Concat", "compact_observation_156",
        own_lidar, red, graph.cut("team_counts", raw, 293, 295),
        friendly, graph.cut("patrol_before_role", raw, 281, 289),
        graph.cut("role", raw, 68, 69),
        graph.cut("patrol_after_role", raw, 290, 291),
        graph.cut("global_first", raw, 291, 292),
        graph.cut("formation_slot", raw, 69, 70),
        zero_tail, axis=1
    )
    action_mean = graph.mlp("actor", compact, actor["layers"])
    log_std = graph.literal("learned_log_std", actor["log_std"])
    return graph.model(
        "Blue_MAPPO_shared_actor", raw, RAW_SIZE,
        [("action_mean", action_mean, ["batch", 2]),
         ("log_std", log_std, [1, 2])],
        {**properties, "network": "actor",
         "inference": "deterministic Gaussian mean; no PPO sampling"}
    )


def critic_model(critic: dict, properties: dict):
    graph = Graph()
    values = graph.mlp("critic", "central_observation", critic["layers"])
    return graph.model(
        "Blue_MAPPO_centralized_critic", "central_observation", CRITIC_SIZE,
        [("state_value", values, ["batch", 1])],
        {**properties, "network": "critic",
         "inputs": "own compact + alive team mean/max + count fractions"}
    )


def export(source: Path, output_dir: Path) -> list[Path]:
    import onnx

    data = source.read_bytes()
    try:
        parsed = parse_checkpoint(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CheckpointError("Non-UTF8 Blue text checkpoint") from exc
    metadata = {
        "checkpoint_format": parsed.format,
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
        **parsed.metadata,
    }
    models = [("blue_actor.onnx", actor_model(parsed.actor, metadata))]
    if parsed.critic is not None:
        models.append(("blue_critic.onnx",
                       critic_model(parsed.critic, metadata)))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, model in models:
        path = output_dir / filename
        onnx.save_model(model, str(path))
        onnx.checker.check_model(str(path))
        paths.append(path)
    manifest = {
        "version": 1, "checkpoint_file": source.name,
        "sha256": metadata["checkpoint_sha256"],
        "format": parsed.format, "metadata": parsed.metadata,
        "files": [path.name for path in paths],
        "validation": "ONNX graph checker passed; RLTools parity not yet certified",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    try:
        for path in export(args.checkpoint, args.output_dir):
            print(f"Created: {path}")
    except (OSError, CheckpointError) as exc:
        cli.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
