#!/usr/bin/env python3
"""
E2E smoke test:
frontend request -> Express proxy -> FastAPI /api/discover.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "ecessp-ml"
FRONTEND_DIR = REPO_ROOT / "ecessp-frontend"

BACKEND_PORT = int(os.getenv("E2E_BACKEND_PORT", "8010"))
FRONTEND_PORT = int(os.getenv("E2E_FRONTEND_PORT", "5010"))
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"


def wait_for(url: str, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code < 500:
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
        backend_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
        if not backend_py.exists():
            raise FileNotFoundError(f"Missing python: {backend_py}")

        backend_cmd = [
            str(backend_py),
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(BACKEND_PORT),
        ]
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        wait_for(f"{BACKEND_URL}/api/health", timeout_sec=60)

        env = os.environ.copy()
        env["NODE_ENV"] = "development"
        env["PORT"] = str(FRONTEND_PORT)
        env["HOST"] = "127.0.0.1"
        env["ECESSP_ML_URL"] = BACKEND_URL

        npm_exe = "npm.cmd" if os.name == "nt" else "npm"
        frontend_cmd = [npm_exe, "run", "dev"]
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(FRONTEND_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        wait_for(f"{FRONTEND_URL}/api/health", timeout_sec=90)

        payload = {
            "system": {
                "battery_id": "e2e_smoke",
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
                    "average_voltage": 3.8,
                    "capacity_grav": 160.0,
                    "energy_grav": 520.0,
                }
            },
            "explain": False,
            "mode": "predictive",
        }

        resp = None
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                resp = requests.post(f"{FRONTEND_URL}/api/discover", json=payload, timeout=60)
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(1)
        if resp is None:
            raise RuntimeError(f"Discovery request failed after retries: {last_exc}")
        print("POST /api/discover via frontend:", resp.status_code)
        data = resp.json()
        print("Response keys:", list(data.keys()) if isinstance(data, dict) else type(data))

        if resp.status_code != 200:
            print("Body:", data)
            return 1
        if not isinstance(data, dict) or "system" not in data:
            print("Unexpected body:", data)
            return 1

        print("E2E smoke test passed.")
        return 0

    finally:
        if frontend_proc is not None:
            terminate_process(frontend_proc)
        if backend_proc is not None:
            terminate_process(backend_proc)


if __name__ == "__main__":
    raise SystemExit(main())
