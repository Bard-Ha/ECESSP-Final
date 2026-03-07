from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.api import routes
from backend.api.schemas import PredictionRequest


class _StubDiscoveryService:
    def __init__(
        self,
        *,
        uncertainty_penalty: float,
        guardrail_status: str = "accept",
        valid: bool = True,
    ):
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.guardrail_status = str(guardrail_status)
        self.valid = bool(valid)

    def discover(self, **_: object) -> dict[str, object]:
        return {
            "system": {
                "average_voltage": 3.9,
                "capacity_grav": 182.0,
                "capacity_vol": 520.0,
                "energy_grav": 640.0,
                "energy_vol": 1780.0,
                "max_delta_volume": 0.07,
                "stability_charge": 0.08,
                "stability_discharge": 0.05,
                "uncertainty": {
                    "model_heads": {
                        "uncertainty_penalty": self.uncertainty_penalty,
                        "compatibility_score_aggregate": 0.6,
                        "role_probabilities": {
                            "cathode": 0.8,
                            "anode": 0.1,
                            "electrolyte_candidate": 0.1,
                        },
                    }
                },
            },
            "metadata": {
                "valid": self.valid,
                "prediction_guardrail": {"status": self.guardrail_status},
            },
            "score": {"score": 0.72, "speculative": False},
        }


def _payload() -> PredictionRequest:
    return PredictionRequest(
        components={
            "cathode": "LiFePO4",
            "anode": "Li4Ti5O12",
            "electrolyte": "LiPF6",
            "separator": "PE",
            "additives": "VC",
        }
    )


def _prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "_ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(
        routes,
        "_predictive_chemistry_gate",
        lambda **_: {"valid": True, "reasons": [], "details": {}},
    )


def test_policy_rejects_high_uncertainty(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_MODE", "reject")
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_MEDIUM", "0.20")
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_HIGH", "0.35")
    svc = _StubDiscoveryService(uncertainty_penalty=0.72)

    with pytest.raises(HTTPException) as exc:
        routes.predict_system(_payload(), discovery_service=svc)

    assert exc.value.status_code == 422
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail["policy"]["action"] == "reject"
    assert "high_uncertainty" in detail["policy"]["reasons"]


def test_policy_fallback_uses_feature_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_MODE", "fallback")
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_HIGH", "0.30")
    svc = _StubDiscoveryService(uncertainty_penalty=0.62)
    payload = _payload()

    response = routes.predict_system(payload, discovery_service=svc)
    hint = routes._components_to_feature_hint(payload.components)

    assert response.diagnostics is not None
    assert response.diagnostics["uncertainty_policy"]["action"] == "fallback"
    assert response.predicted_properties["average_voltage"] == pytest.approx(hint["average_voltage"])
    assert response.predicted_properties["capacity_grav"] == pytest.approx(hint["capacity_grav"])


def test_policy_explain_for_medium_uncertainty(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(monkeypatch)
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_MODE", "explain")
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_MEDIUM", "0.20")
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_HIGH", "0.40")
    svc = _StubDiscoveryService(uncertainty_penalty=0.25)

    response = routes.predict_system(_payload(), discovery_service=svc)

    assert response.diagnostics is not None
    assert response.diagnostics["uncertainty_policy"]["action"] == "explain"
    assert response.score is not None and float(response.score) <= 0.70


def test_policy_rejects_on_guardrail_even_when_low_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare(monkeypatch)
    monkeypatch.setenv("ECESSP_PREDICTION_UNCERTAINTY_MODE", "fallback")
    monkeypatch.setenv("ECESSP_REJECT_ON_GUARDRAIL_REJECT", "1")
    svc = _StubDiscoveryService(uncertainty_penalty=0.05, guardrail_status="reject")

    with pytest.raises(HTTPException) as exc:
        routes.predict_system(_payload(), discovery_service=svc)

    assert exc.value.status_code == 422
    assert "guardrail_reject" in exc.value.detail["policy"]["reasons"]
