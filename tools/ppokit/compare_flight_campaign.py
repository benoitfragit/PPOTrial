"""Compare a historical FlightControl campaign CSV with a PPOKit evaluation report."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

FIELDS = ("steps", "stop", "touchdown", "crash", "return", "stop_m", "touchdown_m")


def reference_rows(source, seed_range):
    """Read the native campaign or the master columns of a previously published comparison."""
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(("seed,steps,", "seed,master_steps,"))), None)
    if start is None:
        raise ValueError(f"{source}: missing historical campaign header")
    reader = csv.DictReader(lines[start:])
    paired = "master_steps" in reader.fieldnames
    columns = ["master_" + name for name in FIELDS] if paired else [
        "steps", "stop", "touchdown", "crash", "reward", "stop_m", "touchdown_m"]
    rows = []
    for row in reader:
        seed = int(row["seed"])
        if seed_range is None or seed_range[0] <= seed <= seed_range[1]:
            values = [int(row[key]) for key in columns[:4]] + [float(row[key]) for key in columns[4:]]
            rows.append((seed, values))
    if seed_range is not None:
        first, last = seed_range
        if last < first or len(rows) != last - first + 1 or any(seed != first + i for i, (seed, _) in enumerate(rows)):
            raise ValueError(f"{source}: expected one ordered reference episode per seed from {first} through {last}")
    return rows


def check_episode(seed, old, new, episode):
    """Compare exact historical values from the same simulator configuration and seed."""
    for field, expected, actual in zip(FIELDS, old, new):
        if actual is None and field in ("stop_m", "touchdown_m"):
            continue  # The caller checked that neither simulator recorded the corresponding event.
        if actual is None or not math.isfinite(expected) or not math.isfinite(actual) or actual != expected:
            raise ValueError(f"seed {seed}, {field}: master={expected}, PPOKit={actual}")
    terminal = bool(old[1] or old[3])  # Stopped and Crash are absorbing. The other Flight outcome is Timeout.
    if episode["terminated"] is not terminal or episode["metrics"][3] != int(not terminal):
        raise ValueError(f"seed {seed}: episode boundary disagrees with the historical outcome")


def compare(source, report, output, *, check=False, seed_range=None):
    historical = reference_rows(source, seed_range)
    episodes = json.loads(report.read_text(encoding="utf-8-sig"))["episodes"]
    if not historical or len(historical) != len(episodes):
        raise ValueError(f"different or empty campaigns: master={len(historical)}, PPOKit={len(episodes)}")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["seed", *[f"{engine}_{name}" for name in FIELDS for engine in ("master", "ppokit")]])
    for (seed, old), episode in zip(historical, episodes):
        if seed != episode["seed"]:
            raise ValueError(f"seed order differs: master={seed}, PPOKit={episode['seed']}")
        metrics = episode["metrics"]
        if check and any(flag not in (0, 1) for flag in metrics[:4]):
            raise ValueError(f"seed {seed}: expected binary Flight outcome flags")
        new = [episode["length"], *map(int, metrics[:3]), episode["return"], metrics[4], metrics[5]]
        for distance, flag in ((5, 1), (6, 2)):
            if new[distance] is None and (old[flag] or new[flag]):
                raise ValueError(f"seed {seed}, {FIELDS[distance]}: missing distance for a recorded event")
        writer.writerow([seed, *[n for pair in zip(old, new) for n in pair]])
        if check:
            check_episode(seed, old, new, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_csv", type=Path)
    parser.add_argument("ppokit_report", type=Path)
    parser.add_argument("--check", action="store_true", help="exit nonzero at the first differing episode field")
    parser.add_argument("--seed-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="select a complete inclusive range from a multi-campaign reference")
    args = parser.parse_args()
    try:
        compare(args.historical_csv, args.ppokit_report, sys.stdout, check=args.check, seed_range=args.seed_range)
    except (IndexError, KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
