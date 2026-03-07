#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


REPORT_DIR = PROJECT_ROOT / "reports" / "reliability_baseline"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pctl(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(float(x) for x in samples)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(p * len(ordered)) - 1)))
    return float(ordered[idx])


def _predict_components(material_ids: list[str], index: int) -> dict[str, str]:
    if len(material_ids) >= 5:
        n = len(material_ids)
        return {
            "cathode": material_ids[(index + 0) % n],
            "anode": material_ids[(index + 1) % n],
            "electrolyte": material_ids[(index + 2) % n],
            "separator": material_ids[(index + 3) % n],
            "additives": material_ids[(index + 4) % n],
        }
    fallback = [
        "LiFePO4",
        "Li4Ti5O12",
        "LiPF6",
        "PE",
        "VC",
        "LiMn2O4",
        "Na3V2(PO4)3",
    ]
    n = len(fallback)
    return {
        "cathode": fallback[(index + 0) % n],
        "anode": fallback[(index + 1) % n],
        "electrolyte": fallback[(index + 2) % n],
        "separator": fallback[(index + 3) % n],
        "additives": fallback[(index + 4) % n],
    }


def _discover_payload(index: int) -> dict[str, Any]:
    return {
        "system": {
            "battery_id": f"baseline_{index}",
            "average_voltage": 3.5 + 0.05 * (index % 4),
            "capacity_grav": 140.0 + 5.0 * (index % 5),
            "capacity_vol": 450.0 + 10.0 * (index % 6),
            "energy_grav": 520.0 + 12.0 * (index % 5),
            "energy_vol": 1400.0 + 20.0 * (index % 6),
            "stability_charge": 0.08,
            "stability_discharge": 0.06,
        },
        "objective": {
            "objectives": {
                "average_voltage": 3.9,
                "capacity_grav": 175.0,
                "energy_grav": 610.0,
            }
        },
        "explain": False,
        "mode": "predictive",
    }


def _run() -> dict[str, Any]:
    latencies_ms: dict[str, list[float]] = {
        "health": [],
        "materials": [],
        "discover": [],
        "predict": [],
    }
    request_counts: dict[str, int] = {k: 0 for k in latencies_ms}
    failed_counts: dict[str, int] = {k: 0 for k in latencies_ms}
    rejected_counts: dict[str, int] = {k: 0 for k in latencies_ms}
    discovery_valid_flags: list[int] = []
    prediction_valid_flags: list[int] = []
    calibration_pairs: list[tuple[float, int]] = []

    with TestClient(app) as client:
        for _ in range(5):
            t0 = time.perf_counter()
            resp = client.get("/api/health")
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms["health"].append(dt)
            request_counts["health"] += 1
            if resp.status_code >= 500:
                failed_counts["health"] += 1

        material_ids: list[str] = []
        t0 = time.perf_counter()
        mat_resp = client.get("/api/materials?limit=30")
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms["materials"].append(dt)
        request_counts["materials"] += 1
        if mat_resp.status_code >= 500:
            failed_counts["materials"] += 1
        else:
            body = mat_resp.json()
            items = body.get("items", []) if isinstance(body, dict) else []
            for item in items:
                if isinstance(item, dict):
                    mid = str(item.get("material_id") or "").strip()
                    if mid:
                        material_ids.append(mid)

        for i in range(8):
            t0 = time.perf_counter()
            resp = client.post("/api/discover", json=_discover_payload(i))
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms["discover"].append(dt)
            request_counts["discover"] += 1
            if resp.status_code >= 500:
                failed_counts["discover"] += 1
                continue
            if resp.status_code >= 400:
                rejected_counts["discover"] += 1
                continue
            body = resp.json()
            metadata = body.get("metadata", {}) if isinstance(body, dict) else {}
            discovery_valid_flags.append(1 if bool(metadata.get("valid", False)) else 0)

        for i in range(8):
            payload = {"components": _predict_components(material_ids, i)}
            t0 = time.perf_counter()
            resp = client.post("/api/predict", json=payload)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms["predict"].append(dt)
            request_counts["predict"] += 1
            if resp.status_code == 422:
                rejected_counts["predict"] += 1
                continue
            if resp.status_code >= 500:
                failed_counts["predict"] += 1
                continue
            if resp.status_code >= 400:
                rejected_counts["predict"] += 1
                continue
            body = resp.json()
            diagnostics = body.get("diagnostics", {}) if isinstance(body, dict) else {}
            valid = 1 if bool(diagnostics.get("valid", False)) else 0
            prediction_valid_flags.append(valid)
            try:
                confidence = float(body.get("confidence_score", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            calibration_pairs.append((max(0.0, min(1.0, confidence)), valid))

    endpoint_metrics: dict[str, dict[str, float]] = {}
    total_requests = 0
    total_failures = 0
    for endpoint, samples in latencies_ms.items():
        count = int(request_counts.get(endpoint, 0))
        fail = int(failed_counts.get(endpoint, 0))
        total_requests += count
        total_failures += fail
        endpoint_metrics[endpoint] = {
            "count": float(count),
            "failed": float(fail),
            "rejected": float(rejected_counts.get(endpoint, 0)),
            "error_rate_percent": (100.0 * fail / count) if count > 0 else 0.0,
            "p50_latency_ms": _pctl(samples, 0.50),
            "p95_latency_ms": _pctl(samples, 0.95),
        }

    brier_proxy = mean((conf - obs) ** 2 for conf, obs in calibration_pairs) if calibration_pairs else None

    bin_stats: list[dict[str, float]] = []
    if calibration_pairs:
        bins = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
        for lo, hi in bins:
            group = [(c, o) for (c, o) in calibration_pairs if lo <= c < hi]
            if not group:
                bin_stats.append(
                    {"bin_low": lo, "bin_high": hi, "count": 0.0, "avg_confidence": 0.0, "observed_valid_rate": 0.0}
                )
                continue
            avg_conf = mean(c for c, _ in group)
            obs_rate = mean(o for _, o in group)
            bin_stats.append(
                {
                    "bin_low": lo,
                    "bin_high": hi,
                    "count": float(len(group)),
                    "avg_confidence": float(avg_conf),
                    "observed_valid_rate": float(obs_rate),
                }
            )

    prediction_valid_rate = mean(prediction_valid_flags) if prediction_valid_flags else 0.0
    discovery_valid_rate = mean(discovery_valid_flags) if discovery_valid_flags else 0.0

    return {
        "generated_at_utc": _now_utc(),
        "method": "fastapi_testclient_baseline_probe",
        "samples": {
            "total_requests": int(total_requests),
            "total_failures": int(total_failures),
            "prediction_responses_scored": int(len(prediction_valid_flags)),
            "discovery_responses_scored": int(len(discovery_valid_flags)),
        },
        "api": {
            "overall_error_rate_percent": (100.0 * total_failures / total_requests) if total_requests > 0 else 0.0,
            "endpoint_metrics": endpoint_metrics,
        },
        "model_reliability": {
            "valid_candidate_rate_prediction": float(prediction_valid_rate),
            "valid_candidate_rate_discovery": float(discovery_valid_rate),
            "uncertainty_calibration_brier_proxy": (float(brier_proxy) if brier_proxy is not None else None),
            "calibration_bins": bin_stats,
        },
        "notes": [
            "Calibration metric is a proxy computed from confidence_score vs diagnostics.valid.",
            "This baseline is intended for trend tracking after reliability hardening changes.",
        ],
    }


def main() -> int:
    report = _run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    latest_json = REPORT_DIR / "reliability_baseline_latest.json"
    stamped_json = REPORT_DIR / f"reliability_baseline_{ts}.json"
    latest_md = REPORT_DIR / "reliability_baseline_latest.md"

    payload = json.dumps(report, ensure_ascii=True, indent=2)
    latest_json.write_text(payload + "\n", encoding="utf-8")
    stamped_json.write_text(payload + "\n", encoding="utf-8")

    md = [
        "# Reliability Baseline",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Total requests: `{report['samples']['total_requests']}`",
        f"- Total failures: `{report['samples']['total_failures']}`",
        f"- Overall error rate (%): `{report['api']['overall_error_rate_percent']:.3f}`",
        f"- Prediction valid candidate rate: `{report['model_reliability']['valid_candidate_rate_prediction']:.3f}`",
        f"- Discovery valid candidate rate: `{report['model_reliability']['valid_candidate_rate_discovery']:.3f}`",
        (
            f"- Calibration Brier proxy: "
            f"`{report['model_reliability']['uncertainty_calibration_brier_proxy']:.4f}`"
            if report["model_reliability"]["uncertainty_calibration_brier_proxy"] is not None
            else "- Calibration Brier proxy: `N/A (no accepted prediction responses)`"
        ),
        "",
        "## Endpoint p95 Latency (ms)",
    ]
    for endpoint, metrics in report["api"]["endpoint_metrics"].items():
        md.append(f"- `{endpoint}`: `{metrics['p95_latency_ms']:.2f}`")
    latest_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(str(latest_json))
    print(str(stamped_json))
    print(str(latest_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
