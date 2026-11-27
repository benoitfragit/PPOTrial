#!/usr/bin/env python3
"""Exports a legacy LibTorch Airspace checkpoint (model.pt) as tensors in PPOKit order.

The legacy agents (RLMission, RLSNA) were trained by LibTorch PPO; their model.pt held the
network and its Adam state. They left the repository in commit 05bcb4f; this script reads
them back from git history and writes, for airspace_checkpoint_import:

    {"agent": "RLMission", "version": "eval", "source": "<git object>",
     "adam_steps": 8197, "lr": ..., "betas": [...], "eps": ...,
     "actor":  [[...], ...],   actor parameters,  Graph::Parameters() order of graph.json
     "critic": [[...], ...],   critic parameters, same order (shared layers: the actor's values)
     "first":  [[...], ...],   Adam first moments,  GraphLearner::JointParameters() order
     "second": [[...], ...]}   Adam second moments, same order

torch Linear weights are (out, in) row-major, as PPOKit Dense weights: values are copied
flat, without transpose. The joint order (actor parameters, then the critic's own) is the
legacy registration order, which is torch's Adam order. All parameters of a legacy file
share one Adam step.

Usage: python3 -I export_legacy_checkpoint.py <repository> <RLMission|RLSNA> <prod|eval> <out.json>
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legacy_pt_reader as reader  # noqa: E402

LEGACY_COMMIT = "05bcb4f^"

# Legacy module names in PPOKit order: (actor modules, critic modules, critic modules of its own).
LAYOUTS = {
    "RLMission": (["enc1", "enc2", "attn_q", "attn_k", "mu_head", "logstd_head", "fire_head"],
                  ["enc1", "enc2", "attn_q", "attn_k", "critic1", "critic2"],
                  ["critic1", "critic2"]),
    "RLSNA": (["fc1", "fc2", "mu_head", "logstd_head"],
              ["fc1", "fc2", "critic"],
              ["critic"]),
}


def tensor_names(modules):
    """weight then bias of each module, in order."""
    return ["%s.%s" % (module, kind) for module in modules for kind in ("weight", "bias")]


def flat(array):
    return [float(value) for value in array.astype("float32").reshape(-1)]


def main(repository, agent, version, output):
    actor_modules, critic_modules, own_modules = LAYOUTS[agent]
    path = "resources/model/%s/%s/model.pt" % (agent, version)
    blob = subprocess.run(["git", "-C", repository, "show", "%s:%s" % (LEGACY_COMMIT, path)],
                          check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as folder:
        archive = os.path.join(folder, "model.pt")
        with open(archive, "wb") as stream:
            stream.write(blob)
        root = reader.load(archive)
        parameters = dict((name, tensor.array) for name, tensor in reader.parameters(root))
        optimizer = reader.adam(root)
    if optimizer["group_count"] != 1:
        raise SystemExit("expected one Adam parameter group")
    group = optimizer["groups"][0]
    registration = [name for name, _ in reader.parameters(root)]
    joint = tensor_names(actor_modules) + tensor_names(own_modules)
    if joint != registration:
        raise SystemExit("joint order %s differs from the legacy registration order %s" % (joint, registration))
    steps = sorted(set(entry["step"] for entry in group["params"]))
    if len(steps) != 1:
        raise SystemExit("Adam steps differ between parameters: %s" % steps)
    for name, entry in zip(registration, group["params"]):
        if entry["exp_avg"].array.shape != parameters[name].shape or entry["other_fields"]:
            raise SystemExit("unexpected Adam state for %s" % name)
    options = group["options"]
    document = {
        "agent": agent, "version": version, "source": "%s:%s" % (LEGACY_COMMIT, path),
        "adam_steps": steps[0],
        "lr": float(options["lr"]), "betas": [float(b) for b in options["betas"]], "eps": float(options["eps"]),
        "weight_decay": float(options["weight_decay"]),
        "actor": [flat(parameters[name]) for name in tensor_names(actor_modules)],
        "critic": [flat(parameters[name]) for name in tensor_names(critic_modules)],
        "first": [flat(entry["exp_avg"].array) for entry in group["params"]],
        "second": [flat(entry["exp_avg_sq"].array) for entry in group["params"]],
    }
    with open(output, "w") as stream:
        # repr of a float32 widened to double round-trips to the same float32.
        json.dump(document, stream)
    print("%s %s: Adam step %d, %d actor, %d critic, %d joint tensors -> %s"
          % (agent, version, steps[0], len(document["actor"]), len(document["critic"]), len(document["first"]), output))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
