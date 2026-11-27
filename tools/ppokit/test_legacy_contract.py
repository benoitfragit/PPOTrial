"""Exercise explicit migration of pre-semantic graph headers on the committed corridor artifact."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main(tool, fixture):
    task, graph = fixture / "task.json", fixture / "graph_blocks.json"
    line, payload = (fixture / "training.ppokit").read_bytes().split(b"\n", 1)
    header = json.loads(line)
    with tempfile.TemporaryDirectory() as base:
        old = Path(base) / "old.ppokit"
        del header["identity"]["semantic_v1"]
        canonical = payload.replace(b"\r\n", b"\n")
        for label, newline in (("lf", b"\n"), ("crlf", b"\r\n")):
            original = canonical.replace(b"\n", newline)
            old.write_bytes(json.dumps(header).encode() + newline + original)
            refused = subprocess.run([tool, "check", old, task, graph], capture_output=True)
            if refused.returncode != 1 or b"missing semantic_v1" not in refused.stdout:
                raise ValueError("ambiguous contract loaded silently")
            new = Path(base) / f"{label}.ppokit"
            subprocess.run([tool, "migrate", old, task, graph, new], check=True)
            subprocess.run([tool, "check", new, task, graph], check=True)
            if new.read_bytes().split(b"\n", 1)[1] != original:
                raise ValueError(f"migration changed the bytes of the {label} checkpoint body")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
