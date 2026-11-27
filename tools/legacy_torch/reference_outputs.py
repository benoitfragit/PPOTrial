#!/usr/bin/env python3
"""Writes reference observations and actions of the converted red policies.

The actions are computed with numpy in float32, following the LibTorch forward passes
the policies replace (ActorCriticTrackAttention::forward_mu, ActorCriticNet::forward_mu).
LegacyPolicyTests checks the C++ policies against them; before LibTorch left the
repository, LegacyPolicyTorchParityTests checked the same observations against LibTorch.

Usage: reference_outputs.py <repository> <output>
"""

import sys

import numpy as np

MISSION_TRACKS, MISSION_FEATURES, MISSION_GLOBALS = 10, 9, 62
SNA_INPUTS = 6


def read_policy(path):
    parameters = {}
    with open(path) as stream:
        lines = stream.read().split("\n")
    index = 2
    while lines[index] != "end":
        _, name, rows, columns = lines[index].split()
        rows, columns = int(rows), int(columns)
        values = [float(v) for line in lines[index + 1:index + 1 + rows] for v in line.split()]
        parameters[name] = np.array(values, dtype=np.float32).reshape(rows, columns)
        index += 1 + rows
    return parameters


def dense(parameters, module, x):
    return (x @ parameters[module + ".weight"].T + parameters[module + ".bias"][0]).astype(np.float32)


def relu(x):
    return np.maximum(x, np.float32(0))


def mission_action(p, observation):
    tracks = observation[:MISSION_TRACKS * MISSION_FEATURES].reshape(MISSION_TRACKS, MISSION_FEATURES)
    mask = np.clip(tracks[:, -1], 0, 1).astype(np.float32)
    feat = relu(dense(p, "enc2", relu(dense(p, "enc1", tracks))))
    m = mask[:, None]
    mean = (feat * m).sum(0) / np.maximum(m.sum(0), np.float32(1e-6))
    maxv = np.where(m > 0.5, feat, np.float32(-1e9)).max(0)
    pooled = np.concatenate([mean, maxv]).astype(np.float32)
    query = dense(p, "attn_q", pooled)
    keys = dense(p, "attn_k", feat)
    scores = (keys * query).sum(-1) / np.float32(8.0) + (np.float32(1) - mask) * np.float32(-1e9)
    scores = scores.astype(np.float32)
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()
    context = (weights[:, None] * feat).sum(0)
    heads = np.concatenate([pooled, context, observation[MISSION_TRACKS * MISSION_FEATURES:]]).astype(np.float32)
    return np.concatenate([dense(p, "mu_head", heads), dense(p, "fire_head", heads)])


def sna_action(p, observation):
    return dense(p, "mu_head", relu(dense(p, "fc2", relu(dense(p, "fc1", observation)))))


def mission_observations(rng):
    """No track, one to ten present tracks, a fractional presence, and large values."""
    cases = []
    for present in range(MISSION_TRACKS + 1):
        tracks = rng.uniform(-1, 1, (MISSION_TRACKS, MISSION_FEATURES)).astype(np.float32)
        tracks[:, -1] = 0
        tracks[:present, -1] = 1
        globals_ = rng.uniform(0, 1, MISSION_GLOBALS).astype(np.float32)
        cases.append(np.concatenate([tracks.ravel(), globals_]))
    fractional = cases[4].copy()
    fractional[MISSION_FEATURES - 1] = 0.3
    fractional[2 * MISSION_FEATURES - 1] = 1.7
    cases.append(fractional)
    large = cases[6].copy() * np.float32(25)
    for track in range(MISSION_TRACKS):
        large[track * MISSION_FEATURES + MISSION_FEATURES - 1] = 1 if track < 6 else 0
    cases.append(large)
    return cases


def write(lines, name, architecture_file, observations, action):
    lines.append("model %s %d" % (name, len(observations)))
    for observation in observations:
        lines.append(" ".join("%.9g" % v for v in observation))
        lines.append(" ".join("%.9g" % v for v in action(architecture_file, observation)))


def main(arguments):
    if len(arguments) != 2:
        sys.stderr.write(__doc__)
        return 2
    repository, output = arguments
    rng = np.random.default_rng(20261123)
    lines = ["ppokit.legacy_policy_reference/1"]
    for version in ("prod", "eval"):
        mission = read_policy("%s/Examples/Airspace/models/RLMission/%s/policy.txt" % (repository, version))
        write(lines, "RLMission/" + version, mission, mission_observations(rng), mission_action)
        sna = read_policy("%s/Examples/Airspace/models/RLSNA/%s/policy.txt" % (repository, version))
        sna_cases = [rng.uniform(-2, 2, SNA_INPUTS).astype(np.float32) for _ in range(8)]
        write(lines, "RLSNA/" + version, sna, sna_cases, sna_action)
    lines.append("end")
    with open(output, "w", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
