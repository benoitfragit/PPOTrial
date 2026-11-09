"""Bounded, low-frequency supervision for headless MAPPO training blocks.

The watcher runs outside the simulation hot path. Silence alone only raises
an observation warning; a kill requires either an explicit total budget
overrun or a genuine absence of optimizer-update progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Pattern


@dataclass(frozen=True)
class WatchdogLimits:
    heartbeat_seconds: float = 30.0
    quiet_warning_seconds: float = 300.0
    stalled_update_seconds: float = 3600.0
    budget_grace_seconds: float = 180.0
    shutdown_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "WatchdogLimits":
        """Optional tuning without introducing GitHub workflow inputs."""
        defaults = cls()
        values = {}
        for field, variable in (
            ("heartbeat_seconds", "MAPPO_HEARTBEAT_SECONDS"),
            ("quiet_warning_seconds", "MAPPO_QUIET_WARNING_SECONDS"),
            ("stalled_update_seconds", "MAPPO_STALLED_UPDATE_SECONDS"),
            ("budget_grace_seconds", "MAPPO_BUDGET_GRACE_SECONDS"),
            ("shutdown_seconds", "MAPPO_SHUTDOWN_SECONDS"),
        ):
            raw = os.environ.get(variable)
            value = float(raw) if raw is not None else getattr(defaults, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{variable} must be finite and positive")
            values[field] = value
        return cls(**values)


def _assign_windows_job(process: subprocess.Popen[str]) -> object | None:
    """Own all Windows descendants, even if PPO exits before its children.

    A kill-on-close Job Object is available on Windows 2022 runners. It is
    deliberately confined to the external watchdog, not the C++ runtime.
    """
    if os.name != "nt":
        return None
    import ctypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("per_process_time", ctypes.c_longlong),
            ("per_job_time", ctypes.c_longlong),
            ("flags", ctypes.c_uint32),
            ("min_working_set", ctypes.c_size_t),
            ("max_working_set", ctypes.c_size_t),
            ("active_process_limit", ctypes.c_uint32),
            ("affinity", ctypes.c_size_t),
            ("priority_class", ctypes.c_uint32),
            ("scheduling_class", ctypes.c_uint32),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "read_operations", "write_operations", "other_operations",
            "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("basic", BasicLimit), ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t),
                    ("peak_job_memory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetInformationJobObject.restype = ctypes.c_int
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.AssignProcessToJobObject.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    try:
        limits = ExtendedLimit()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
                handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        # Popen retains the real process handle on Windows; assignment
        # cannot depend on a PID that may have been recycled.
        if not kernel.AssignProcessToJobObject(handle, int(process._handle)):
            # Preserve the Win32 error before polling, which may change it.
            error = ctypes.get_last_error()
            if process.poll() is not None:
                # A very short-lived leaf finished before assignment.
                kernel.CloseHandle(handle)
                return None
            if error == 5:  # ERROR_ACCESS_DENIED: exit can race assignment.
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    kernel.CloseHandle(handle)
                    return None
            raise OSError(error, "AssignProcessToJobObject failed")
        return handle
    except BaseException:
        kernel.CloseHandle(handle)
        raise


def _close_windows_job(handle: object | None) -> None:
    """Reap descendants on normal exit, timeout, error and cancellation."""
    if handle is None:
        return
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    kernel.CloseHandle(handle)


def _stop_tree(process: subprocess.Popen[str], grace: float) -> None:
    """Stop the whole process group, then escalate within a fixed bound."""
    if os.name == "nt":
        # The Windows process tree must be collected before its parent exits.
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=grace / 2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=grace, check=False)
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=grace)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=grace)


def run(command: list[str], stage_id: str, budget_seconds: float,
        log_path: Path, combined_path: Path, evaluation: Pattern[str],
        updates: Callable[[], int], limits: WatchdogLimits | None = None,
        metadata: dict | None = None) -> str | None:
    """Stream output without blocking heartbeat, preserve failure provenance."""
    config = limits or WatchdogLimits.from_env()
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("Invalid stage budget")
    status_path = log_path.with_name("watchdog.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    data: queue.Queue[str | None] = queue.Queue()
    start = time.monotonic()
    last_log = start
    last_update_at = start
    next_heartbeat = start + config.heartbeat_seconds
    phase = "unknown"
    last_eval: str | None = None
    error_seen = False
    stop_reason: str | None = None
    heartbeat_count = 0
    last_update: int | None = None
    last_checkpoint_update: int | None = None
    try:
        last_checkpoint_update = updates()
        last_update = last_checkpoint_update
    except (OSError, ValueError):
        pass

    popen_args = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                  "text": True, "encoding": "utf-8", "errors": "replace",
                  "bufsize": 1}
    if os.name == "nt":
        popen_args["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_args["start_new_session"] = True

    with log_path.open("w", encoding="utf-8") as log, (
        combined_path.open("a", encoding="utf-8")
    ) as combined:
        process = subprocess.Popen(command, **popen_args)
        try:
            job = _assign_windows_job(process)
        except BaseException:
            if process.poll() is None:
                _stop_tree(process, config.shutdown_seconds)
            raise
        assert process.stdout is not None

        def read_lines() -> None:
            try:
                for line in process.stdout:
                    data.put(line)
            finally:
                data.put(None)

        reader = threading.Thread(target=read_lines, name="mappo-stdout",
                                  daemon=True)
        reader.start()

        def record(reason: str | None = None) -> dict:
            now = time.monotonic()
            details = metadata or {}
            item = {
                "timestampUtc": datetime.now(timezone.utc).isoformat(),
                "stage": stage_id,
                "primaryGate": details.get("primaryGate", stage_id),
                "blockIndex": details.get("blockIndex"),
                "targetUpdate": details.get("targetUpdate"),
                "scenarioVariant": details.get("scenarioVariant", "fixed_no_zones"),
                "pid": process.pid,
                "alive": process.poll() is None,
                "phase": phase,
                "lastUpdate": last_update,
                "lastCheckpointUpdate": last_checkpoint_update,
                "lastEvaluation": last_eval,
                "lastLogAgeSeconds": round(now - last_log, 3),
                "elapsedSeconds": round(now - start, 3),
                "budgetRemainingSeconds": round(
                    max(0.0, budget_seconds + config.budget_grace_seconds
                        - (now - start)), 3),
                "heartbeatCount": heartbeat_count,
                "status": (("failure" if reason in
                           ("process_error", "mappo_error", "monitor_error")
                           else "incomplete") if reason is not None
                           else "running" if process.poll() is None else "completed"),
                "stopReason": reason,
            }
            temporary = status_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
            temporary.replace(status_path)
            return item

        try:
            while True:
                now = time.monotonic()
                wait = min(1.0, max(0.01, next_heartbeat - now))
                try:
                    item = data.get(timeout=wait)
                except queue.Empty:
                    item = ""  # No output is NOT evidence of a hung process.
                if item is None:
                    break
                if item:
                    sys.stdout.write(item)
                    log.write(item)
                    combined.write(item)
                    last_log = time.monotonic()
                    if "Starting deterministic" in item:
                        phase = "evaluation"
                    elif "Saved actor and training checkpoint" in item:
                        phase = "checkpoint"
                    elif "update=" in item or "PPO update" in item:
                        phase = "training"
                    # Optimizer completion is observable before the next
                    # disk checkpoint; both are valid liveness evidence.
                    completed = re.search(
                        r"^\[MAPPO_UPDATE\] update=(\d+)(?:\s|$)", item)
                    if completed:
                        observed = int(completed.group(1))
                        if last_update is None or observed > last_update:
                            last_update = observed
                            last_update_at = time.monotonic()
                    found = evaluation.search(item)
                    if found:
                        last_eval = found.group(0)
                    if "[MAPPO_ERROR]" in item or "[MAPPO_EVAL_ERROR]" in item:
                        error_seen = True
                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat_count += 1
                    try:
                        last_checkpoint_update = updates()
                        if (last_update is None
                                or last_checkpoint_update > last_update):
                            last_update = last_checkpoint_update
                            last_update_at = now
                    except (OSError, ValueError):
                        pass  # Checkpoints can be temporarily in the middle of a save.
                    snapshot = record()
                    log.flush()
                    combined.flush()
                    print(
                        f"[MAPPO_HEARTBEAT] stage={stage_id} phase={phase}"
                        f" pid={process.pid} update={last_update}"
                        f" silent_s={snapshot['lastLogAgeSeconds']}"
                        f" budget_remaining_s={snapshot['budgetRemainingSeconds']}",
                        flush=True)
                    if now - last_log > config.quiet_warning_seconds:
                        print(f"[MAPPO_HEARTBEAT] Quiet process stage={stage_id}; "
                              "no timeout inferred from missing logs", flush=True)
                    next_heartbeat = now + config.heartbeat_seconds
                if process.poll() is None and (
                        now - last_update_at > config.stalled_update_seconds
                        and last_update is not None):
                    stop_reason = "stalled_optimizer_updates"
                    break
                if now - start > budget_seconds + config.budget_grace_seconds:
                    stop_reason = "budget_timeout"
                    break
            if stop_reason is not None:
                _stop_tree(process, config.shutdown_seconds)
            else:
                # Child exited; stdout reader reached EOF. No indefinite wait.
                process.wait(timeout=config.shutdown_seconds)
        except BaseException:
            if process.poll() is None:
                _stop_tree(process, config.shutdown_seconds)
            record(stop_reason or "monitor_error")
            raise
        finally:
            # Close first: descendants inheriting stdout must not hold the
            # reader open after their PPO parent has exited.
            _close_windows_job(job)
            log.flush()
            combined.flush()
            process.stdout.close()
            reader.join(timeout=config.shutdown_seconds)

        return_code = process.poll()
        if stop_reason is None and (return_code != 0 or error_seen):
            stop_reason = "process_error" if return_code != 0 else "mappo_error"
        record(stop_reason)
        if stop_reason is not None:
            raise RuntimeError(
                f"Curriculum stage {stage_id} incomplete: {stop_reason}, "
                f"exit={return_code}; see {log_path} and {status_path}")
    return last_eval
