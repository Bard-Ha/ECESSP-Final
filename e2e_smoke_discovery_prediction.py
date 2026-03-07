#!/usr/bin/env python3
"""
E2E smoke test:
frontend request -> Express proxy -> FastAPI /api/discover and /api/predict.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "ecessp-ml"
FRONTEND_DIR = REPO_ROOT / "ecessp-frontend"

BACKEND_PORT = int(os.getenv("E2E_BACKEND_PORT", "8012"))
FRONTEND_PORT = int(os.getenv("E2E_FRONTEND_PORT", "5012"))
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"


def wait_for(url: str, timeout_sec: int = 90) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}")


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    backend_proc: subprocess.Popen | None = None
    frontend_proc: subprocess.Popen | None = None
    try:
        py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
        if not py.exists():
            raise FileNotFoundError(f"Missing python: {py}")

        backend_proc = subprocess.Popen(
            [str(py), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for(f"{BACKEND_URL}/api/health")

        env = os.environ.copy()
        env["NODE_ENV"] = "development"
        env["PORT"] = str(FRONTEND_PORT)
        env["HOST"] = "127.0.0.1"
        env["ECESSP_ML_URL"] = BACKEND_URL
        frontend_proc = subprocess.Popen(
            ["npm.cmd" if os.name == "nt" else "npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for(f"{FRONTEND_URL}/api/health")

        # Discover (generative mode)
        discover_payload = {
            "system": {
                "battery_id": "e2e_gen",
                "average_voltage": 3.7,
                "capacity_grav": 150.0,
                "capacity_vol": 500.0,
                "energy_grav": 500.0,
                "energy_vol": 1500.0,
                "stability_charge": 0.9,
                "stability_discharge": 0.9,
            },
            "objective": {
                "objectives": {
                    "average_voltage": 4.0,
                    "capacity_grav": 180.0,
                    "energy_grav": 600.0,
                }
            },
            "explain": False,
            "mode": "generative",
        }
        dr = requests.post(f"{FRONTEND_URL}/api/discover", json=discover_payload, timeout=90)
        if dr.status_code != 200:
            raise RuntimeError(f"/api/discover failed: {dr.status_code} {dr.text[:400]}")
        d = dr.json()
        if "system" not in d:
            raise RuntimeError(f"/api/discover missing system: {d}")

        # Materials list for real component ids
        mr = requests.get(f"{FRONTEND_URL}/api/materials?limit=10", timeout=30)
        if mr.status_code != 200:
            raise RuntimeError(f"/api/materials failed: {mr.status_code} {mr.text[:300]}")
        items = mr.json().get("items", [])
        if len(items) < 5:
            raise RuntimeError("Not enough materials returned for prediction test")

        comps1 = {
            "cathode": items[0]["material_id"],
            "anode": items[1]["material_id"],
            "electrolyte": items[2]["material_id"],
            "separator": items[3]["material_id"],
            "additives": items[4]["material_id"],
        }
        comps2 = {
            "cathode": items[5]["material_id"],
            "anode": items[6]["material_id"],
            "electrolyte": items[7]["material_id"],
            "separator": items[8]["material_id"],
            "additives": items[9]["material_id"],
        }

        p1 = requests.post(f"{FRONTEND_URL}/api/predict", json={"components": comps1}, timeout=60)
        p2 = requests.post(f"{FRONTEND_URL}/api/predict", json={"components": comps2}, timeout=60)
        if p1.status_code != 200 or p2.status_code != 200:
            raise RuntimeError(f"/api/predict failed: {p1.status_code} {p2.status_code}")

        j1 = p1.json()
        j2 = p2.json()
        props1 = j1.get("predicted_properties", {})
        props2 = j2.get("predicted_properties", {})
        if not props1 or not props2:
            raise RuntimeError("Prediction payload missing predicted_properties")

        delta = sum(abs(float(props1[k]) - float(props2.get(k, props1[k]))) for k in props1.keys())
        if delta == 0:
            raise RuntimeError("Predictions are identical for two different component sets")

        print("E2E discovery + prediction smoke test passed.")
        return 0
    finally:
        if frontend_proc is not None:
            terminate_process(frontend_proc)
        if backend_proc is not None:
            terminate_process(backend_proc)


if __name__ == "__main__":
    raise SystemExit(main())

