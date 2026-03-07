from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests


pytestmark = pytest.mark.integration


def _wait_for(url: str, timeout_sec: int = 90) -> None:
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


def _terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.skipif(
    os.getenv("RUN_PROXY_INTEGRATION", "0") != "1",
    reason="Set RUN_PROXY_INTEGRATION=1 to run frontend->backend proxy integration test.",
)
def test_frontend_discovery_proxy_response_schema() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root / "ecessp-ml"
    frontend_dir = repo_root / "ecessp-frontend"
    py = repo_root / ".venv" / "Scripts" / "python.exe"
    npm = shutil.which("npm.cmd") or shutil.which("npm")

    if not py.exists():
        pytest.skip("python.exe not found in repo .venv")
    if npm is None:
        pytest.skip("npm executable not found")

    backend_port = int(os.getenv("E2E_BACKEND_PORT", "8014"))
    frontend_port = int(os.getenv("E2E_FRONTEND_PORT", "5014"))
    backend_url = f"http://127.0.0.1:{backend_port}"
    frontend_url = f"http://127.0.0.1:{frontend_port}"

    backend_proc: subprocess.Popen[str] | None = None
    frontend_proc: subprocess.Popen[str] | None = None

    try:
        backend_proc = subprocess.Popen(
            [
                str(py),
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(backend_port),
            ],
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for(f"{backend_url}/api/health")

        if not (frontend_dir / "node_modules").exists():
            subprocess.run([npm, "ci"], cwd=str(frontend_dir), check=True)

        env = os.environ.copy()
        env["NODE_ENV"] = "development"
        env["PORT"] = str(frontend_port)
        env["HOST"] = "127.0.0.1"
        env["ECESSP_ML_URL"] = backend_url
        frontend_proc = subprocess.Popen(
            [npm, "run", "dev"],
            cwd=str(frontend_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for(f"{frontend_url}/api/health")

        payload = {
            "system": {
                "battery_id": "integration_proxy_schema",
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
                    "average_voltage": 3.9,
                    "capacity_grav": 170.0,
                    "energy_grav": 560.0,
                }
            },
            "explain": False,
            "mode": "predictive",
        }
        response = requests.post(f"{frontend_url}/api/discover", json=payload, timeout=90)
        assert response.status_code == 200, response.text

        body = response.json()
        assert isinstance(body, dict)
        assert "system" in body
        assert "score" in body
        assert "metadata" in body
        assert "history" in body
        assert isinstance(body["system"], dict)
        assert isinstance(body["score"], dict)

    finally:
        _terminate_process(frontend_proc)
        _terminate_process(backend_proc)
