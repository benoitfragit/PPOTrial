#!/usr/bin/env python3
"""Headless, checkpoint-preserving MAPPO curriculum; one Release binary per CI run.

The generalized V6 checkpoint supports changing P/N between process launches.
No C++ agent, simulator, optimizer or observation schema is modified here.

Each block runs `ppokit-run train` on the BluePatrol project (blue_environment and
blue_network). Its evaluations are the project's suite eval/validation.json, made
the campaign of the training file for the block (seeds, cases, cadence): they
reach the run journal and the suite's BEST store. After each block the best campaign of the
block is promoted to PROD as PPO.exe's Training mode promoted it (score above the
.score of the same P/N, or no such score).
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import blue_checkpoint
import mappo_process_watchdog


PROJECT = Path("Examples/BlueDefense/projects/BluePatrol")
TRAINING_FILE = PROJECT / "blue_patrol_training.json"
SCENARIO_FILE = PROJECT / "scenario.json"
SUITE_FILE = PROJECT / "eval/validation.json"
MODEL_ROOT = PROJECT / "checkpoints"
LATEST = MODEL_ROOT / "latest" / blue_checkpoint.TRAINING_NAME
PROD = MODEL_ROOT / "prod" / blue_checkpoint.POLICY_NAME
PROD_SCORE = Path(str(PROD) + blue_checkpoint.SCORE_SUFFIX)
PROGRESS = MODEL_ROOT / "curriculum-progress.json"
LATEST_ACTOR = MODEL_ROOT / "latest" / blue_checkpoint.POLICY_NAME
JOURNAL = MODEL_ROOT / "latest/metrics.jsonl"
SUITE = "validation"
# ppokit-run's BEST store of the suite; the gate champions live in best/<gate>/.
BEST_MANIFEST = MODEL_ROOT / "best" / f"{SUITE}.json"
SCORE_MAGIC = "BLUE_PATROL_GENERALIZED_EVALUATION_V1"
# Only the current Blue contract is reset (under its new and former names);
# old candidate files remain available.
FRESH_SEEDS = (LATEST, LATEST_ACTOR, PROD, PROD_SCORE,
               MODEL_ROOT / "latest" / blue_checkpoint.FORMER_TRAINING_NAME,
               MODEL_ROOT / "latest" / blue_checkpoint.FORMER_POLICY_NAME,
               MODEL_ROOT / "prod" / blue_checkpoint.FORMER_POLICY_NAME,
               MODEL_ROOT / "prod" / (blue_checkpoint.FORMER_POLICY_NAME
                                      + blue_checkpoint.SCORE_SUFFIX))
OUTPUT = Path("build/mappo-telemetry")
# ppokit-run's line after each campaign of a suite during training.
EVAL_PATTERN = re.compile(r"evaluation '([A-Za-z0-9_.-]+)' at update (\d+)")
STAGE_NAMES = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def read_updates(path: Path = LATEST) -> int:
    """Read the authoritative PPO update count, not an actor/CI timestamp.

    The training checkpoint's contract is validated too (either format)."""
    if not path.exists():
        return 0
    return blue_checkpoint.read_updates(path)


def validate_curriculum(document: dict) -> list[dict]:
    if document.get("formatVersion") != 1:
        raise ValueError("Curriculum formatVersion must be 1")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Curriculum needs a nonempty stages array")
    seen: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {
            "id", "blue", "red", "hostileZones", "updates"
        }:
            raise ValueError("Each stage needs id, blue, red, hostileZones, updates")
        name, blue, red, zones, updates = (
            stage["id"], stage["blue"], stage["red"],
            stage["hostileZones"], stage["updates"]
        )
        if not isinstance(name, str) or not STAGE_NAMES.fullmatch(name) or name in seen:
            raise ValueError(f"Invalid or duplicate curriculum stage id: {name!r}")
        seen.add(name)
        if (type(blue) is not int or not 1 <= blue <= 9
                or type(red) is not int or not 0 <= red <= 10
                or type(zones) is not bool or type(updates) is not int
                or not 1 <= updates <= 10000):
            raise ValueError(f"Invalid counts, zones or update count for {name}")
    if (stages[-1]["blue"], stages[-1]["red"]) != (4, 4):
        raise ValueError("Final stage must be the reference 4v4 case")
    return stages


def fingerprint(document: dict) -> str:
    serialized = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_progress(config_hash: str, checkpoint_updates: int) -> dict:
    if not PROGRESS.is_file():
        progress = {
            "formatVersion": 1,
            "curriculumHash": config_hash,
            "baselineUpdates": checkpoint_updates,
            "nextStage": 0,
        }
        write_json(PROGRESS, progress)
        return progress
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    if (progress.get("formatVersion") != 1
            or progress.get("curriculumHash") != config_hash
            or type(progress.get("baselineUpdates")) is not int
            or type(progress.get("nextStage")) is not int):
        raise ValueError("Existing curriculum progress is incompatible; start a fresh run")
    if checkpoint_updates < progress["baselineUpdates"]:
        raise ValueError("Checkpoint predates saved curriculum progress")
    return progress


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def apply_stage(stage: dict, target_updates: int,
                original_training: dict, original_scenario: dict) -> None:
    training = json.loads(json.dumps(original_training))
    scenario = json.loads(json.dumps(original_scenario))
    training["training"]["maxUpdates"] = target_updates
    settings = scenario["blueDefense"]
    settings["blueTeam"]["activeCount"] = stage["blue"]
    settings["redTeam"]["activeCount"] = stage["red"]
    settings["restrictedZones"]["enabled"] = stage["hostileZones"]
    objective_layout = stage.get("objectiveLayout", "fixed")
    if objective_layout not in ("fixed", "randomized"):
        raise ValueError("Unknown objective layout")
    # Always apply the selected variant, even when the source JSON was randomized.
    settings.setdefault("objectiveSpawn", {})["mode"] = objective_layout
    if not stage["hostileZones"]:
        settings["restrictedZoneSpawn"]["enabledProbability"] = 0.0
    write_json(TRAINING_FILE, training)
    write_json(SCENARIO_FILE, scenario)


def suite_of_training(training: dict) -> dict:
    """The project's suite evaluated during training, made the training file's campaign.

    PPO.exe's campaign came from the training file (seeds, cases, cadence), which CI
    changes (independent seeds); the suite takes them, and keeps the rest.
    """
    suite = json.loads(SUITE_FILE.read_text(encoding="utf-8"))
    if suite.get("suite") != SUITE:
        raise ValueError(f"{SUITE_FILE} must be the suite '{SUITE}'")
    if suite.get("best") != [{"metric": "score.mean", "prefer": "higher"}]:
        raise ValueError(f"{SUITE_FILE} must keep its BEST at the highest score.mean "
                         "(PPO.exe's promotion rule)")
    evaluation = training["evaluation"]
    suite["seeds"] = {"first": evaluation["seed"], "count": evaluation["episodes"]}
    suite["episodes"] = 1
    suite["during_training"] = {"every_updates": training["training"]["checkpointEvery"]}
    return suite


def clear_suite_best() -> None:
    """Empty the suite's BEST store so that it keeps the best campaign of one block."""
    if BEST_MANIFEST.is_file():
        BEST_MANIFEST.unlink()
    # The store names its policies after the policy file (.ppokit; .txt before), and an
    # interrupted copy leaves a .partial file.
    for pattern in (f"{SUITE}_*.ppokit", f"{SUITE}_*.txt", f"{SUITE}_*.partial"):
        for policy in BEST_MANIFEST.parent.glob(pattern):
            policy.unlink()


def float32(value: float) -> float:
    """The value as the C++ float of PPO.exe's campaign score."""
    return array.array("f", [value])[0]


def promoted_score(blue: int, red: int) -> float | None:
    """The .score of PROD for this P/N, as PPO.exe's LoadPromotedScore reads it."""
    if not PROD.is_file() or not PROD_SCORE.is_file():
        return None
    fields = PROD_SCORE.read_text(encoding="utf-8").split()
    try:
        if (len(fields) >= 4 and fields[0] == SCORE_MAGIC
                and int(fields[1]) == blue and int(fields[2]) == red):
            return float32(float(fields[3]))
    except ValueError:
        pass
    return None


def promote_like_training(stage: dict) -> bool:
    """PPO.exe's PromoteEvaluatedActor over one block: the block's best campaign
    becomes PROD when its score is above the .score of the same P/N (or there is
    none). Returns True when PROD changed."""
    if not BEST_MANIFEST.is_file():
        return False
    best = json.loads(BEST_MANIFEST.read_text(encoding="utf-8"))
    score = float32(float(best["report"]["aggregates"]["score.mean"]))
    incumbent = promoted_score(stage["blue"], stage["red"])
    if incumbent is not None and not score > incumbent:
        return False
    policy = BEST_MANIFEST.parent / best["policy"]
    # A ppokit.best/2 manifest keeps the digest of the weights that were evaluated.
    if "sha256" in best and hashlib.sha256(policy.read_bytes()).hexdigest() != best["sha256"]:
        raise ValueError(f"{policy} does not hold the evaluated weights of {BEST_MANIFEST}")
    PROD.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROD.with_suffix(PROD.suffix + ".tmp")
    shutil.copyfile(policy, temporary)
    temporary.replace(PROD)
    temporary = PROD_SCORE.with_suffix(PROD_SCORE.suffix + ".tmp")
    temporary.write_text(f"{SCORE_MAGIC} {stage['blue']} {stage['red']} {score:.9g}\n",
                         encoding="utf-8")
    temporary.replace(PROD_SCORE)
    print(f"[BLUE MAPPO V10] Promoted evaluated generalized actor of update "
          f"{best['update']} to {PROD} (score {score:.9g})", flush=True)
    return True


def run_stage(binary: Path, stage: dict, seconds: int, log_path: Path) -> str | None:
    """Supervise a full block outside the hot path with bounded lifecycle.

    The block trains up to the training file's maxUpdates (apply_stage), saving
    every saveEveryUpdates (else every checkpointEvery) update.
    """
    training = json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
    suite = suite_of_training(training)
    settings = training["training"]
    save_every = settings.get("saveEveryUpdates", settings["checkpointEvery"])
    command = [str(binary.resolve()), "train", "--project", str((PROJECT / "project.legacy.json").resolve()),
               "--until-update", str(settings["maxUpdates"]),
               "--save-every", str(save_every),
               "--max-seconds", str(seconds)]
    clear_suite_best()
    original_suite = SUITE_FILE.read_bytes()
    write_json(SUITE_FILE, suite)
    try:
        return mappo_process_watchdog.run(
            command, stage["id"], seconds, log_path,
            OUTPUT / "mappo-training.log", EVAL_PATTERN, read_updates,
            metadata=stage)
    finally:
        SUITE_FILE.write_bytes(original_suite)
        # As in PPO.exe, a campaign finished before an interrupted block still promotes.
        promote_like_training(stage)


def save_candidate_and_restore(baseline: dict[Path, bytes | None],
                               stage: dict, stage_dir: Path) -> None:
    """Keep non-4v4 promotion entirely separate from production's 4v4 score."""
    if (stage["blue"], stage["red"]) == (4, 4):
        return
    for path in (PROD, PROD_SCORE):
        if path.is_file():
            shutil.copy2(path, stage_dir / path.name)
        if baseline[path] is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(baseline[path])


def require_clean_fresh_start() -> None:
    """Avoid the silent legacy V4/V7 migration when no run ID was supplied.

    These are Blue checkpoints in this runner's temporary checkout ONLY.
    The Red production RLMission model is never touched.
    """
    if PROGRESS.exists():
        raise ValueError("Fresh-start requested but curriculum progress already exists")
    for path in FRESH_SEEDS:
        if path.is_file():
            print(f"[CURRICULUM] Removing old Blue seed from checkout: {path}")
            path.unlink()



def validate_curriculum_v2(document: dict) -> tuple[list[dict], int, int]:
    """Validate an independently versioned, explicit 1000-update curriculum."""
    if document.get("formatVersion") not in (2, 3):
        raise ValueError("Staged curriculum requires formatVersion 2 or 3")
    stages = validate_curriculum({
        "formatVersion": 1, "stages": document.get("stages")
    })
    rehearsal = document.get("rehearsal")
    if not isinstance(rehearsal, dict) or set(rehearsal) != {
        "primaryUpdates", "priorUpdates"
    }:
        raise ValueError("V2 rehearsal requires primaryUpdates and priorUpdates")
    primary, prior = rehearsal["primaryUpdates"], rehearsal["priorUpdates"]
    if (type(primary) is not int or type(prior) is not int
            or not 1 <= primary <= 1000 or not 1 <= prior <= 1000):
        raise ValueError("V2 rehearsal quotas must be positive integers")
    total = document.get("totalUpdates")
    if type(total) is not int or not 1 <= total <= 10000:
        raise ValueError("V2 totalUpdates must be in [1, 10000]")
    if sum(stage["updates"] for stage in stages) != total:
        raise ValueError("V2 stage quotas must sum to totalUpdates")
    if document["formatVersion"] == 3 and (
            document.get("observationVersion") != 10
            or document.get("initialization") != "scratch"):
        raise ValueError("V3 requires observationVersion 10 and initialization scratch")
    return stages, primary, prior


def choose_rehearsal_block(stages: list[dict], phase_index: int,
                           phase_start: int, phase_end: int, updates: int,
                           primary_updates: int, prior_updates: int,
                           reference_stage: dict | None = None
                           ) -> tuple[dict, int]:
    """Deterministic stage-dependent rehearsal, safe after partial checkpoints.

    With a warm V10 seed, the first phase also rehearses the reference 4v4
    scenario. Later phases alternate that 4v4 reference with earlier tasks.
    Without a warm seed, the historical from-scratch schedule is unchanged.
    Every phase finishes with its primary task.
    """
    if not 0 <= phase_index < len(stages) or not phase_start <= updates < phase_end:
        raise ValueError("Rehearsal request outside curriculum phase")
    primary = stages[phase_index]
    if phase_index == 0 and reference_stage is None:
        return primary, phase_end
    cycle_length = primary_updates + prior_updates
    cycle, offset = divmod(updates - phase_start, cycle_length)
    if offset < primary_updates:
        return primary, min(phase_end, updates + primary_updates - offset)
    # A final short tail is always trained on the current phase.
    if phase_end - updates <= prior_updates:
        return primary, phase_end
    if reference_stage is not None and phase_index < len(stages) - 1:
        # Rehearse the seed's 4v4 behavior from the very first patrol phase.
        previous = (reference_stage if phase_index == 0 or cycle % 2 == 0
                    else stages[(cycle // 2) % phase_index])
    else:
        previous = stages[cycle % phase_index]
    return previous, min(phase_end, updates + cycle_length - offset)


def run_v2(args: argparse.Namespace, config: dict) -> int:
    """Versioned rehearsal; V3 allows explicit V10 warm import or scratch."""
    stages, primary_updates, prior_updates = validate_curriculum_v2(config)
    version = config["formatVersion"]
    binary = Path(args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Missing built ppokit-run executable: {binary}")
    original_training_bytes = TRAINING_FILE.read_bytes()
    original_scenario_bytes = SCENARIO_FILE.read_bytes()
    original_training = json.loads(original_training_bytes)
    original_scenario = json.loads(original_scenario_bytes)
    warm_start = getattr(args, "warm_start", False)
    if warm_start and (version != 3 or args.fresh_start):
        raise ValueError("Warm start requires V3 without --fresh-start")
    if args.fresh_start:
        require_clean_fresh_start()
    # read_updates rejects a checkpoint of another observation schema (V3: 10).
    updates = read_updates()
    config_hash = fingerprint(config)
    if warm_start and (PROGRESS.exists() or not LATEST.is_file()
                       or updates <= 0):
        raise ValueError(
            "V3 warm start requires a nonzero V10 training checkpoint "
            "without existing curriculum progress")
    if version == 3 and not PROGRESS.exists() and not args.fresh_start and not warm_start:
        raise ValueError("New V3 has no compatible progress; use --fresh-start")
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if version == 3:
            source = progress.get("initializationSource", "scratch")
            baseline = progress.get("baselineUpdates")
            if (not LATEST.is_file()
                    or source not in ("scratch", "warm-v10")
                    or (source == "scratch" and baseline != 0)
                    or (source == "warm-v10" and (type(baseline) is not int
                                                or baseline <= 0))):
                raise ValueError("V3 resume needs compatible checkpoint provenance")
        if (progress.get("formatVersion") != version
                or progress.get("curriculumHash") != config_hash
                or type(progress.get("baselineUpdates")) is not int
                or type(progress.get("nextStage")) is not int
                or progress.get("targetUpdates") != config["totalUpdates"]):
            raise ValueError(
                "Incompatible V1 or modified V2/V3 curriculum progress; "
                "resume with its exact version and configuration")
    else:
        if updates != 0 and not warm_start:
            raise ValueError(
                "Existing training checkpoint has no compatible progress; "
                "use a fresh start or restore a complete run artifact")
        progress = {
            "formatVersion": version,
            "curriculumHash": config_hash,
            "baselineUpdates": updates,
            "nextStage": 0,
            "targetUpdates": config["totalUpdates"],
        }
        if version == 3:
            progress["initializationSource"] = "warm-v10" if warm_start else "scratch"
        write_json(PROGRESS, progress)
        if warm_start:
            print(f"[CURRICULUM_V3] Imported complete V10 optimizer at update "
                  f"{updates}; 4v4 reference rehearsal enabled.", flush=True)
    baseline_updates = progress["baselineUpdates"]
    next_stage = progress["nextStage"]
    if (baseline_updates < 0 or not 0 <= next_stage <= len(stages)
            or updates < baseline_updates
            or updates > baseline_updates + config["totalUpdates"]):
        raise ValueError("Invalid V2 checkpoint / curriculum progress")
    completed_target = baseline_updates + sum(
        stage["updates"] for stage in stages[:next_stage])
    if updates < completed_target:
        raise ValueError("Checkpoint behind completed V2 curriculum phase")
    if next_stage == len(stages):
        if updates != baseline_updates + config["totalUpdates"]:
            raise ValueError("Completed V2 progress contradicts checkpoint update")
        print(f"[CURRICULUM_V{version}] All curriculum updates already completed.")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_root = OUTPUT / f"curriculum-v{version}"
    stage_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_seconds
    seed_source = progress.get("initializationSource", "scratch")
    reference_stage = stages[-1] if version == 3 and seed_source == "warm-v10" else None
    summary = [
        f"# Blue MAPPO curriculum V{version} — interleaved rehearsal",
        "",
        f"Initialization: {seed_source}; baseline: {baseline_updates}.",
        f"Start update: {updates}; target: "
        f"{baseline_updates + config['totalUpdates']}.",
        f"Rehearsal: {primary_updates} primary / {prior_updates} previous "
        "updates per cycle (warm V10: include 4v4 reference).",
        "",
        "| Phase | Scenario | P/N | Zones | Update | Last evaluation |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    try:
        for index in range(next_stage, len(stages)):
            phase_start = baseline_updates + sum(
                stage["updates"] for stage in stages[:index])
            phase_end = phase_start + stages[index]["updates"]
            phase = stages[index]
            updates = read_updates()
            if updates < phase_start:
                raise ValueError("V2 checkpoint behind current phase")
            if updates >= phase_end:
                progress["nextStage"] = index + 1
                write_json(PROGRESS, progress)
                continue
            while updates < phase_end:
                remaining_time = deadline - time.monotonic()
                if remaining_time < 20:
                    print(f"[CURRICULUM_V{version}] Time budget reached; resume via run ID.")
                    break
                selected, block_end = choose_rehearsal_block(
                    stages, index, phase_start, phase_end, updates,
                    primary_updates, prior_updates, reference_stage)
                # maxUpdates stops the binary at block_end; the wall-clock
                # limit is only a safeguard. Give each block the entire
                # remaining budget rather than prorating it by update count.
                # A prorated limit can expire mid-block and prematurely end
                # the curriculum while hours remain.
                seconds = int(remaining_time)
                block_dir = (stage_root / f"{index + 1:02d}-{phase['id']}"
                             / f"update-{updates:06d}-{selected['id']}")
                block_dir.mkdir(parents=True, exist_ok=True)
                baseline = {path: path.read_bytes() if path.is_file() else None
                            for path in (PROD, PROD_SCORE)}
                print(
                    f"[CURRICULUM_V{version}] phase={phase['id']} "
                    f"active={selected['id']} P={selected['blue']} "
                    f"N={selected['red']} zones={selected['hostileZones']} "
                    f"updates={updates}->{block_end} budget_s={seconds}",
                    flush=True)
                apply_stage(selected, block_end,
                            original_training, original_scenario)
                if (selected["blue"], selected["red"]) != (4, 4):
                    PROD.unlink(missing_ok=True)
                    PROD_SCORE.unlink(missing_ok=True)
                try:
                    last_eval = run_stage(
                        binary, selected, seconds, block_dir / "training.log")
                    fresh_updates = read_updates()
                    if not updates < fresh_updates <= block_end:
                        raise RuntimeError(
                            f"Invalid V2 optimizer progress for {selected['id']}: "
                            f"{updates}->{fresh_updates}, target={block_end}")
                    if JOURNAL.is_file():
                        shutil.copy2(JOURNAL, block_dir / "metrics.jsonl")
                    summary.append(
                        f"| {phase['id']} | {selected['id']} | "
                        f"{selected['blue']}/{selected['red']} | "
                        f"{selected['hostileZones']} | {fresh_updates} | "
                        f"{last_eval or 'n/a'} |")
                    (OUTPUT / "curriculum-summary.md").write_text(
                        "\n".join(summary) + "\n", encoding="utf-8")
                finally:
                    save_candidate_and_restore(baseline, selected, block_dir)
                updates = fresh_updates
                if updates < block_end:
                    print(
                        f"[CURRICULUM_V{version}] Partial block: {updates}/{block_end}; "
                        "continuing in this run while time remains.",
                        flush=True)
                    # A partial but progressing block must not terminate the
                    # curriculum. Retry with the same cumulative maxUpdates;
                    # the PPO checkpoint holds the optimizer state.
            if updates < phase_end:
                break
            progress["nextStage"] = index + 1
            write_json(PROGRESS, progress)
    finally:
        TRAINING_FILE.write_bytes(original_training_bytes)
        SCENARIO_FILE.write_bytes(original_scenario_bytes)
    print(f"[CURRICULUM_V{version}] Finished at update={read_updates()} "
          f"next_stage={progress['nextStage']}/{len(stages)}.")
    return 0


def run(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.curriculum).read_text(encoding="utf-8"))
    if getattr(args, "warm_start", False) and config.get("formatVersion") != 3:
        raise ValueError("--warm-start is supported by V3 only")
    if config.get("formatVersion") in (2, 3):
        return run_v2(args, config)
    stages = validate_curriculum(config)
    config_hash = fingerprint(config)
    binary = Path(args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Missing built ppokit-run executable: {binary}")
    original_training = json.loads(TRAINING_FILE.read_text(encoding="utf-8"))
    original_scenario = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
    original_training_bytes = TRAINING_FILE.read_bytes()
    original_scenario_bytes = SCENARIO_FILE.read_bytes()
    if args.fresh_start:
        require_clean_fresh_start()
    updates = read_updates()
    progress = load_progress(config_hash, updates)
    if not 0 <= progress["nextStage"] <= len(stages):
        raise ValueError("Invalid curriculum progress stage")
    completed_target = progress["baselineUpdates"] + sum(
        item["updates"] for item in stages[:progress["nextStage"]]
    )
    if updates < completed_target:
        raise ValueError(
            "Checkpoint is behind completed curriculum stages: "
            f"updates={updates}, required={completed_target}"
        )
    if progress["nextStage"] == len(stages):
        print("[CURRICULUM] All stages already completed.")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_root = OUTPUT / "curriculum"
    stage_root.mkdir(parents=True, exist_ok=True)
    baseline = {path: path.read_bytes() if path.is_file() else None
                for path in (PROD, PROD_SCORE)}
    deadline = time.monotonic() + args.max_seconds
    summary: list[str] = [
        "# Blue MAPPO curriculum CI",
        "",
        f"Start PPO update: {updates}; target stages: {len(stages)}.",
        "",
        "| Stage | P/N | Hostile zones | Target update | Current update | Last evaluation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    try:
        for index in range(progress["nextStage"], len(stages)):
            stage = stages[index]
            target = progress["baselineUpdates"] + sum(
                item["updates"] for item in stages[:index + 1]
            )
            updates = read_updates()
            if updates >= target:
                print(f"[CURRICULUM] Skip completed {stage['id']} at update {updates}.")
                progress["nextStage"] = index + 1
                write_json(PROGRESS, progress)
                continue
            remaining = deadline - time.monotonic()
            if remaining < 20:
                print("[CURRICULUM] Time budget reached; checkpoint can resume.")
                break
            # Equal fair-share time; an incomplete stage never gets skipped.
            seconds = max(1, min(int(remaining), max(
                30, int(remaining / (len(stages) - index))
            )))
            stage_dir = stage_root / f"{index + 1:02d}-{stage['id']}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            print(f"[CURRICULUM] Stage={stage['id']} P={stage['blue']} "
                  f"N={stage['red']} zones={stage['hostileZones']} "
                  f"updates={updates}->{target} budget_s={seconds}",
                  flush=True)
            apply_stage(stage, target, original_training, original_scenario)
            if (stage["blue"], stage["red"]) != (4, 4):
                # Remove the 4v4 production score for isolated level evaluation.
                PROD.unlink(missing_ok=True)
                PROD_SCORE.unlink(missing_ok=True)
            try:
                last_eval = run_stage(
                    binary, stage, seconds, stage_dir / "training.log"
                )
                fresh_updates = read_updates()
                if fresh_updates <= updates:
                    raise RuntimeError(
                        f"Stage {stage['id']} made no optimizer progress: {updates}"
                    )
                if JOURNAL.is_file():
                    shutil.copy2(JOURNAL, stage_dir / "metrics.jsonl")
                if fresh_updates >= target:
                    progress["nextStage"] = index + 1
                    write_json(PROGRESS, progress)
                row = (f"| {stage['id']} | {stage['blue']}/{stage['red']} | "
                       f"{stage['hostileZones']} | {target} | "
                       f"{fresh_updates} | {last_eval or 'n/a'} |")
                summary.append(row)
                (OUTPUT / "curriculum-summary.md").write_text(
                    "\n".join(summary) + "\n", encoding="utf-8"
                )
                if fresh_updates < target:
                    print(f"[CURRICULUM] Stage {stage['id']} incomplete at "
                          f"{fresh_updates}/{target}; resume via run ID.")
                    break
            finally:
                save_candidate_and_restore(baseline, stage, stage_dir)
    finally:
        # Restore pristine checkout resources even on failures.
        TRAINING_FILE.write_bytes(original_training_bytes)
        SCENARIO_FILE.write_bytes(original_scenario_bytes)

    print(f"[CURRICULUM] Finished at update={read_updates()} "
          f"next_stage={progress['nextStage']}/{len(stages)}.")
    return 0


def run_single_stage(args: argparse.Namespace) -> int:
    """PPO.exe's Training mode on the scenario as it is: one block up to maxUpdates."""
    binary = Path(args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Missing built ppokit-run executable: {binary}")
    settings = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))["blueDefense"]
    stage = {"id": "single", "blue": settings["blueTeam"]["activeCount"],
             "red": settings["redTeam"]["activeCount"]}
    run_stage(binary, stage, args.max_seconds, OUTPUT / "single" / "training.log")
    print(f"[CURRICULUM] Single stage {stage['blue']}v{stage['red']} ended at update "
          f"{read_updates()}.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/release/PPOKit/Release/ppokit-run.exe")
    parser.add_argument("--curriculum",
                        default="Examples/BlueDefense/curricula/BluePatrolCurriculumV3.json")
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--fresh-start", action="store_true",
                        help="ignore checked-in legacy Blue seeds (no resume run ID)")
    parser.add_argument("--warm-start", action="store_true",
                        help="one-time import of a full V10 checkpoint into V3")
    parser.add_argument("--single-stage", action="store_true",
                        help="no curriculum: train the scenario as it is up to maxUpdates")
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 18000:
        parser.error("--max-seconds must be in [1, 18000]")
    try:
        return run_single_stage(args) if args.single_stage else run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"[CURRICULUM_ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
