from __future__ import annotations

import csv
import math
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
if os.name == "nt":
    default_solver = ROOT / "build-cuda-ascii" / "mc_field_solver.exe"
else:
    default_solver = ROOT / "build" / "mc_field_solver"
SOLVER = Path(os.environ.get("MC_FIELD_SOLVER", default_solver))
RUNS = ROOT / "output" / "web"
RUNS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=None)
solver_lock = threading.Lock()


def read_matrix(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [[float(value) for value in row] for row in csv.reader(handle)]


def summarize(values: list[list[float]]) -> dict[str, float]:
    flat = [value for row in values for value in row if math.isfinite(value)]
    return {"min": min(flat), "max": max(flat), "mean": sum(flat) / len(flat)}


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/assets/<path:name>")
def assets(name: str):
    return send_from_directory(STATIC, name)


@app.get("/api/status")
def status():
    return jsonify({
        "ready": SOLVER.exists(),
        "solver": str(SOLVER),
        "backend": "cpu",
        "cudaCompiled": True,
        "cudaRunnable": False,
        "reason": "No NVIDIA GPU detected on this host",
    })


@app.post("/api/solve")
def solve():
    payload = request.get_json(silent=True) or {}
    try:
        nx = int(payload.get("nx", 40))
        ny = int(payload.get("ny", 40))
        walks = int(payload.get("walks", 1000))
        seed = int(payload.get("seed", 123456789))
    except (TypeError, ValueError):
        return jsonify({"error": "Parameters must be integers"}), 400
    if not 8 <= nx <= 96 or not 8 <= ny <= 96:
        return jsonify({"error": "Grid dimensions must be between 8 and 96"}), 400
    if not 50 <= walks <= 10000:
        return jsonify({"error": "Walks must be between 50 and 10,000"}), 400
    if not 0 <= seed <= 2**63 - 1:
        return jsonify({"error": "Seed is outside the supported range"}), 400
    if not SOLVER.exists():
        return jsonify({"error": "Solver executable is missing; build the project first"}), 503

    run_id = uuid.uuid4().hex[:12]
    output = RUNS / run_id
    # Keep the child-process argument ASCII-only; MSVC narrow argv cannot reliably
    # represent the Chinese workspace path on every Windows code page.
    output_argument = Path("output") / "web" / run_id
    command = [str(SOLVER), "--nx", str(nx), "--ny", str(ny), "--walks",
               str(walks), "--seed", str(seed), "--backend", "cpu",
               "--output", str(output_argument)]
    started = time.perf_counter()
    try:
        with solver_lock:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Solver timed out after 300 seconds"}), 504
    if completed.returncode != 0:
        return jsonify({"error": completed.stderr.strip() or "Solver failed"}), 500

    potential = read_matrix(output / "potential.csv")
    stderr = read_matrix(output / "stderr.csv")
    ex = read_matrix(output / "Ex.csv")
    ey = read_matrix(output / "Ey.csv")
    magnitude = read_matrix(output / "E.csv")
    tokens = dict(item.split("=", 1) for item in completed.stdout.strip().split()
                  if "=" in item)
    return jsonify({
        "runId": run_id,
        "parameters": {"nx": nx, "ny": ny, "walks": walks, "seed": seed},
        "runtimeMs": float(tokens.get("runtime_ms", (time.perf_counter() - started) * 1000)),
        "rmse": float(tokens.get("rmse", "nan")),
        "potential": potential,
        "stderr": stderr,
        "ex": ex,
        "ey": ey,
        "magnitude": magnitude,
        "stats": {
            "potential": summarize(potential),
            "stderr": summarize(stderr),
            "magnitude": summarize(magnitude),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
