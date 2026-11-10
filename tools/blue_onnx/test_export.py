"""Deterministic synthetic checkpoint and ONNX graph regression tests."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export import (CheckpointError, actor_model, critic_model, export,
                    parse_checkpoint)


class Fixture:
    def __init__(self):
        self.rng = np.random.default_rng(418)
        self.parts = []

    def literal(self, text: str) -> None:
        self.parts.append(text)

    def matrix(self, rows: int, columns: int, *, zero: bool = False) -> None:
        if zero:
            array = np.zeros((rows, columns), dtype=np.float32)
        else:
            array = self.rng.normal(0, 0.02, (rows, columns)).astype(np.float32)
        self.parts.extend(format(float(v), ".9g") for v in array.flat)

    def attention(self, width: int, features: int) -> None:
        for rows, cols in ((width, features), (1, width),
                           (width, 7), (1, width)):
            self.matrix(rows, cols)

    def layer(self, inputs: int, outputs: int) -> None:
        self.matrix(outputs, inputs)
        self.matrix(1, outputs)

    def moments(self, rows: int, columns: int) -> None:
        self.matrix(rows, columns, zero=True)
        self.matrix(rows, columns, zero=True)

    def actor(self) -> None:
        self.literal("BLUE_TEAM_MAPPO_V10 10 10 295 156 2 128 3 "
                     "2 2 9 10 8 1 61 UNIFIED_PHYSICAL_GUIDANCE_V1")
        self.attention(8, 13)
        self.attention(6, 9)
        for first, second in ((156, 128), (128, 128), (128, 2)):
            self.layer(first, second)
        self.matrix(1, 2)

    def training(self) -> None:
        self.literal("BLUE_TEAM_TRAINING_V7 7 10 295 156 2 128 3 "
                     "470 256 3 2 2 9 10 8 1 61 "
                     "UNIFIED_PHYSICAL_GUIDANCE_V1 ACTOR")
        self.actor()
        self.literal("ACTOR_MOMENTS")
        for width, features in ((8, 13), (6, 9)):
            for rows, cols in ((width, features), (1, width),
                               (width, 7), (1, width)):
                self.moments(rows, cols)
        for first, second in ((156, 128), (128, 128), (128, 2)):
            self.moments(second, first)
            self.moments(1, second)
        self.moments(1, 2)
        self.literal("CRITIC")
        for first, second in ((470, 256), (256, 256), (256, 1)):
            self.layer(first, second)
            self.moments(second, first)
            self.moments(1, second)
        self.literal("OPTIMIZERS")
        self.literal("0.001 0.9 0.999 1e-8 0 0 0 1 1")
        self.literal("0.001 0.9 0.999 1e-8 0 0 0 1 1")
        self.literal("UPDATES 42 RNG 123 456 789 END_BLUE_TEAM_TRAINING")

    def text(self) -> str:
        return "\n".join(self.parts) + "\n"


def reference_attention(raw: np.ndarray, own: np.ndarray,
                        offset: int, number: int, width: int,
                        features: int, weights: dict) -> np.ndarray:
    slots = raw[offset:offset + number * features].reshape(number, features)
    valid = slots[:, -1] > 0.5
    if not valid.any():
        return np.zeros(3 * width, dtype=np.float32)
    slots = slots[valid]
    embeddings = np.tanh(slots @ weights["encoder_w"].T
                         + weights["encoder_b"])
    query = np.tanh(own @ weights["query_w"].T + weights["query_b"][0])
    scores = (embeddings @ query) / math.sqrt(width)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    context = (probs[:, None] * embeddings).sum(axis=0)
    return np.concatenate((context, embeddings.mean(axis=0),
                           embeddings.max(axis=0))).astype(np.float32)


def reference_actor(raw: np.ndarray, actor: dict) -> np.ndarray:
    result = []
    for observation in raw:
        own = observation[:7]
        reds = reference_attention(observation, own, 70, 10, 8, 13,
                                   actor["red"])
        friends = reference_attention(observation, own, 200, 9, 6, 9,
                                      actor["friendly"])
        compact = np.concatenate((
            observation[:68], reds, observation[293:295], friends,
            observation[281:289], observation[68:69],
            observation[290:291], observation[291:292],
            observation[69:70], np.zeros(32, dtype=np.float32)
        )).astype(np.float32)
        assert compact.shape == (156,)
        x = compact
        for index, layer in enumerate(actor["layers"]):
            x = x @ layer["w"].T + layer["b"][0]
            if index != 2:
                x = np.maximum(x, 0)
        result.append(x)
    return np.asarray(result, dtype=np.float32)


def reference_critic(raw: np.ndarray, critic: dict) -> np.ndarray:
    x = raw.copy()
    for index, layer in enumerate(critic["layers"]):
        x = x @ layer["w"].T + layer["b"]
        if index != 2:
            x = np.maximum(x, 0)
    return x


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        fixture = Fixture()
        fixture.actor()
        self.v10 = fixture.text()

    def test_v10_read_and_reject_bad_contract(self):
        parsed = parse_checkpoint(self.v10)
        self.assertEqual(parsed.format, "V10")
        self.assertEqual(parsed.actor["red"]["encoder_w"].shape, (8, 13))
        self.assertEqual(parsed.actor["layers"][0]["w"].shape, (128, 156))
        self.assertIsNone(parsed.critic)
        with self.assertRaises(CheckpointError):
            parse_checkpoint(self.v10.replace("295 156", "294 156", 1))
        with self.assertRaises(CheckpointError):
            parse_checkpoint(self.v10 + "UNEXPECTED")
        with self.assertRaises(CheckpointError):
            parse_checkpoint(self.v10[:-80])

    def test_v7_actor_and_critic(self):
        fixture = Fixture()
        fixture.training()
        parsed = parse_checkpoint(fixture.text())
        self.assertEqual(parsed.format, "V7")
        self.assertEqual(parsed.metadata["updates"], 42)
        self.assertEqual(parsed.critic["layers"][0]["w"].shape, (256, 470))
        self.assertEqual(parsed.critic["layers"][2]["b"].shape, (1, 1))

    def test_v10_real_onnx_graph_and_numerical_equivalence(self):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnx / onnxruntime not installed")
        actor = parse_checkpoint(self.v10).actor
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "blue_v10.txt"
            source.write_text(self.v10, encoding="utf-8")
            results = export(source, Path(directory) / "output")
            self.assertEqual([p.name for p in results], ["blue_actor.onnx"])
            session = ort.InferenceSession(str(results[0]), providers=["CPUExecutionProvider"])
            rng = np.random.default_rng(12)
            raw = rng.normal(0, 0.5, (4, 295)).astype(np.float32)
            raw[:, 82] = 0
            raw[:, 95] = 0
            raw[:, 208] = 0
            raw[:, 280] = 1
            raw[1, 82] = 1
            raw[2, 82] = 1
            raw[2, 95] = 1
            raw[3, 208] = 1
            # Permuting two active reds must preserve the actor prediction.
            raw[3, 70:83] = raw[2, 83:96]
            raw[3, 83:96] = raw[2, 70:83]
            raw[3, 280] = raw[2, 280]
            raw[2, 200:281] = raw[3, 200:281]
            raw[3, :70] = raw[2, :70]
            raw[3, 96:200] = raw[2, 96:200]
            raw[3, 281:295] = raw[2, 281:295]
            actual, std = session.run(None, {"raw_observation": raw})
            expected = reference_actor(raw, actor)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(actual[2], actual[3], atol=2e-5)
            np.testing.assert_allclose(std, actor["log_std"], atol=0)
            onnx.checker.check_model(str(results[0]))

    def test_critic_onnx_matches_numpy(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")
        fixture = Fixture()
        fixture.training()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training_v7.txt"
            source.write_text(fixture.text(), encoding="utf-8")
            files = export(source, Path(directory) / "output")
            self.assertEqual(len(files), 2)
            parsed = parse_checkpoint(fixture.text())
            session = ort.InferenceSession(str(files[1]), providers=["CPUExecutionProvider"])
            raw = np.random.default_rng(23).normal(
                0, 0.2, (3, 470)
            ).astype(np.float32)
            expected = reference_critic(raw, parsed.critic)
            result = session.run(None, {"central_observation": raw})[0]
            np.testing.assert_allclose(result, expected, atol=2e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
