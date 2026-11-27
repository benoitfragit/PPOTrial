"""Compare a historical FlightControl V3 with the serialized PPOKit tensors in graph order."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import struct
import sys


def floats(tokens, count):
    """Round text to the float32 actually loaded by both C++ engines."""
    if len(tokens) != count:
        raise ValueError(f"expected {count} floats, found {len(tokens)}")
    data = struct.unpack(f"={count}f", struct.pack(f"={count}f", *map(float, tokens)))
    if not all(map(math.isfinite, data)):
        raise ValueError("non-finite checkpoint value")
    return data


def random_stream(line):
    words = list(map(int, line.split()))
    if len(words) == 624:
        words.append(624)
    if len(words) != 625 or not 0 <= words[-1] <= 624 or any(not 0 <= x < 2**32 for x in words[:-1]):
        raise ValueError("invalid mt19937 state")
    rng = random.Random()
    rng.setstate((3, tuple(words), None))
    # Two consecutive complete recurrence windows, including the next twist.
    return tuple(rng.getrandbits(32) for _ in range(1248))


def legacy(path):
    lines = iter(path.read_text().splitlines())
    magic, obs, axes, outputs, width, updates = next(lines).split()
    if (magic, obs, axes, outputs, width) != ("PPO_FLIGHT_POLICY_V3", "35", "5", "6", "128"):
        raise ValueError("unsupported FlightControl layout")
    signature = json.loads(json.loads(next(lines)))
    ages = tuple(map(int, next(lines).split()))
    rng = random_stream(next(lines))

    def tensor(count):
        return tuple(floats(next(lines).split(), count) for _ in range(3))

    def trunk():
        return [("input.weight", tensor(35 * 128)), ("input.bias", tensor(128)),
                ("hidden.weight", tensor(128 * 128)), ("hidden.bias", tensor(128))]

    actor = trunk()
    weights, biases, std = tensor(6 * 128), tensor(6), tensor(6)
    for name, source, region in [
        ("gaussian.weight", weights, slice(0, 640)), ("gaussian.bias", biases, slice(0, 5)),
        ("gaussian.log_std", std, slice(0, 5)), ("gear.weight", weights, slice(640, 768)),
        ("gear.bias", biases, slice(5, 6)),
    ]:
        actor.append((name, tuple(values[region] for values in source)))
    critic = trunk() + [("value.weight", tensor(128)), ("value.bias", tensor(1))]
    if list(lines):
        raise ValueError("trailing legacy data")
    return int(updates), signature, ages, rng, {"actor": actor, "critic": critic}


def compare(old_file, new_file, level_file, converted_level, output, frozen_update=False, profile="legacy"):
    update, signature, ages, rng, networks = legacy(old_file)
    lines = iter(new_file.read_text().splitlines())
    header = json.loads(next(lines))
    expected_contract = f"{signature['aircraft']}@{signature['contractVersion']}"
    if profile == "centreline1200":
        expected_contract += "-centreline1200"
    if header["updates"] != update or header["identity"]["contract"] != expected_contract:
        raise ValueError("update or aircraft differs")
    rows = []
    failures = []

    def section(tag, network, component, column):
        tensors = networks[network]
        if next(lines).split() != [tag, str(len(tensors))]:
            raise ValueError(f"incorrect {network} {tag} section")
        for name, fields in tensors:
            expected = fields[column]
            size, *tokens = next(lines).split()
            if int(size) != len(expected):
                raise ValueError(f"{name} dimension differs")
            observed = floats(tokens, len(expected))
            absolute = [abs(a-b) for a, b in zip(expected, observed)]
            relative = [d/abs(a) if a else (math.inf if d else 0.) for d, a in zip(absolute, expected)]
            exact = struct.pack(f"={len(expected)}f", *expected) == struct.pack(f"={len(observed)}f", *observed)
            rows.append([network, name, component, size, max(absolute), max(relative), int(exact)])
            # Only an optimizer replay permits rounding differences between C++ backends.
            bounds = {"parameter": 3e-6, "m": 3e-5, "v": 8e-5}
            if not exact and (not frozen_update or max(absolute) > bounds[component]):
                failures.append(f"{network}.{name}.{component}")

    section("actor", "actor", "parameter", 0)
    if header["kind"] == "training":
        section("critic", "critic", "parameter", 0)
        for network, age in zip(("actor", "critic"), ages):
            if next(lines).split() != [network + "_adam", str(age - 1)]:
                failures.append(network + " Adam step")
            section("first", network, "m", 1)
            section("second", network, "v", 2)
        if next(lines) != "rng" or random_stream(next(lines)) != rng:
            failures.append("mt19937")
    elif header["kind"] != "policy":
        raise ValueError("unknown artifact kind")
    if list(lines) or int(level_file.read_text()) != int(converted_level.read_text()):
        failures.append("end of file or curriculum")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["network", "tensor", "field", "float_count", "max_abs", "max_rel", "bit_exact"])
    writer.writerows(rows)
    if failures:
        raise ValueError("differences: " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical", "converted", "historical_level", "converted_level"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--frozen-update", action="store_true", help="allow backend rounding after one update")
    parser.add_argument("--profile", choices=("legacy", "centreline1200"), default="legacy")
    args = parser.parse_args()
    compare(args.historical, args.converted, args.historical_level, args.converted_level, sys.stdout, args.frozen_update, args.profile)
