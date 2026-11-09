"""Drive the frozen Registration 4 pipeline (S and G) over fifteen unreviewed recordings.

Thread A runs bundle, extract and prepare for each recording in order (video
decode, CPU). Thread B waits for each prepared recording and runs train,
evaluate, then the two policy exports (S, G) into
<data parent>/work/identity_candidates/<rec>/<run id>/{S,G}. Each step logs to
its own file beside this script; the shared state is written to
fifteen_state.json after every step. Failures are recorded and the next
recording proceeds; nothing is retried.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d import load_paths  # noqa: E402

PY = REPO / ".venv" / "Scripts" / "python.exe"
RUNNER = REPO / "analyses/20261031_p11_identity_evaluation/run_appearance_clean.py"
CONFIG = REPO / "analyses/20261031_p11_identity_evaluation/config_appearance_fifteen_v1.toml"
EXPORT = {
    "S": REPO / "analyses/20261108_six_recording_identity/export_identity_s.py",
    "G": REPO / "analyses/20261108_six_recording_identity/export_identity_g.py",
}
HERE = Path(__file__).resolve().parent  # state and step logs are written beside this file
STATE = HERE / "fifteen_state.json"
KEYS = [
    "k21",
    "k22",
    "k23",
    "p21",
    "p22",
    "p23",
    "e11",
    "e12",
    "e13",
    "e21",
    "e22",
    "e23",
    "g11",
    "g12",
    "g13",
]

cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
paths = load_paths()
ROOT = paths.raw_data_dir.parent
STAGING = paths.appearance_staging_dir
EXPORT_ROOT = ROOT / "work" / "identity_candidates"

state: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "recordings": {}}
lock = threading.Lock()
prepared: dict[str, threading.Event] = {k: threading.Event() for k in KEYS}
prepared_ok: dict[str, bool] = {}


def save():
    with lock:
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def record(key, step, **fields):
    with lock:
        state["recordings"].setdefault(key, {})[step] = fields
    save()


def run_dir(key) -> Path:
    return STAGING / cfg[key]["output"] / cfg["run_ids"][key]


def run(key, step, cmd) -> bool:
    log = HERE / f"fifteen_{key}_{step}.log"
    started = time.time()
    record(key, step, status="running", started=time.strftime("%H:%M:%S"), log=str(log))
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write("# " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        rc = subprocess.run(
            [str(c) for c in cmd], cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT
        ).returncode
    record(
        key,
        step,
        status="ok" if rc == 0 else "failed",
        returncode=rc,
        seconds=round(time.time() - started),
        log=str(log),
    )
    return rc == 0


def runner(key, step) -> bool:
    return run(key, step, [PY, RUNNER, step, "--recording", key, "--config", CONFIG])


def export(key, cand) -> bool:
    out = run_dir(key)
    prov = json.loads((out / "appearance_clean_evaluate.provenance.json").read_text())
    entry = prov.get("outputs", {}).get(f"identity_tracks_{cand}")
    if entry is None or not (out / "evaluation" / f"identity_tracks_{cand}.npz").exists():
        record(
            key,
            f"export_{cand}",
            status="absent",
            note="candidate not produced by evaluate (G is undefined when no anchors disagree)",
        )
        return True
    dest = EXPORT_ROOT / cfg[key]["experiment"] / cfg["run_ids"][key] / cand
    return run(
        key,
        f"export_{cand}",
        [PY, EXPORT[cand], "--run", out, "--sha256", entry["sha256"], "--output", dest],
    )


def thread_a():
    for key in KEYS:
        ok = True
        if run_dir(key).exists():
            record(key, "bundle", status="failed", note="run directory already exists")
            ok = False
        for step in ("bundle", "extract", "prepare"):
            if not ok:
                break
            ok = runner(key, step)
        prepared_ok[key] = ok
        prepared[key].set()
    with lock:
        state["thread_a_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()


def thread_b():
    for key in KEYS:
        prepared[key].wait()
        if not prepared_ok[key]:
            record(key, "train", status="skipped", note="preparation failed")
            continue
        if not runner(key, "train"):
            continue
        if not runner(key, "evaluate"):
            continue
        export(key, "S")
        export(key, "G")
    with lock:
        state["thread_b_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()


def main():
    if STAGING is None:
        raise SystemExit("appearance_staging_dir is not configured")
    save()
    a = threading.Thread(target=thread_a, name="prepare", daemon=False)
    b = threading.Thread(target=thread_b, name="model", daemon=False)
    a.start()
    b.start()
    a.join()
    b.join()
    with lock:
        state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
