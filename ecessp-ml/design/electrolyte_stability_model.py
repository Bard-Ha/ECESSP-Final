from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ElectrolyteWindow:
    reduction_limit: float
    oxidation_limit: float
    source: str


@dataclass(frozen=True)
class ElectrolyteCompatibility:
    valid: bool
    reduction_ok: bool
    oxidation_ok: bool
    window: ElectrolyteWindow


_WINDOW_LIBRARY: Dict[str, ElectrolyteWindow] = {
    "lipf6": ElectrolyteWindow(reduction_limit=0.0, oxidation_limit=4.3, source="LiPF6-carbonate"),
    "napf6": ElectrolyteWindow(reduction_limit=0.1, oxidation_limit=4.2, source="NaPF6-carbonate"),
    "kpf6": ElectrolyteWindow(reduction_limit=0.1, oxidation_limit=4.1, source="KPF6-carbonate"),
    "mg(tfsi)2": ElectrolyteWindow(reduction_limit=0.3, oxidation_limit=3.6, source="Mg(TFSI)2-glyme"),
    "ca(tfsi)2": ElectrolyteWindow(reduction_limit=0.3, oxidation_limit=3.7, source="Ca(TFSI)2-solvent"),
    "znso4": ElectrolyteWindow(reduction_limit=0.2, oxidation_limit=2.2, source="ZnSO4-aqueous"),
    "chloroaluminate": ElectrolyteWindow(reduction_limit=0.2, oxidation_limit=2.5, source="chloroaluminate-ionic-liquid"),
}


def estimate_electrolyte_window(electrolyte: str, *, working_ion: str) -> ElectrolyteWindow:
    text = str(electrolyte or "").strip().lower()
    if text:
        for key, window in _WINDOW_LIBRARY.items():
            if key in text:
                return window

    ion = str(working_ion or "Li").strip()
    if ion == "Na":
        return ElectrolyteWindow(reduction_limit=0.1, oxidation_limit=4.2, source="fallback-na-carbonate")
    if ion == "Mg":
        return ElectrolyteWindow(reduction_limit=0.3, oxidation_limit=3.6, source="fallback-mg-glyme")
    if ion == "K":
        return ElectrolyteWindow(reduction_limit=0.1, oxidation_limit=4.1, source="fallback-k-carbonate")
    if ion == "Zn":
        return ElectrolyteWindow(reduction_limit=0.2, oxidation_limit=2.2, source="fallback-zn-aqueous")
    if ion == "Al":
        return ElectrolyteWindow(reduction_limit=0.2, oxidation_limit=2.5, source="fallback-al-ionic-liquid")
    return ElectrolyteWindow(reduction_limit=0.0, oxidation_limit=4.3, source="fallback-li-carbonate")


def evaluate_electrolyte_stability(
    *,
    electrolyte: str,
    working_ion: str,
    cathode_potential: float,
    anode_potential: float,
) -> ElectrolyteCompatibility:
    window = estimate_electrolyte_window(electrolyte, working_ion=working_ion)
    oxidation_ok = float(cathode_potential) <= float(window.oxidation_limit)
    reduction_ok = float(anode_potential) >= float(window.reduction_limit)
    return ElectrolyteCompatibility(
        valid=bool(oxidation_ok and reduction_ok),
        reduction_ok=bool(reduction_ok),
        oxidation_ok=bool(oxidation_ok),
        window=window,
    )


def sei_requirement_flag(*, anode_potential: float, working_ion: str) -> bool:
    ion = str(working_ion or "Li").strip()
    threshold = 0.5 if ion in {"Li", "Na", "K"} else 0.6
    return float(anode_potential) < float(threshold)


def thermal_risk_flag(*, cathode_potential: float, max_oxidation_state: float) -> bool:
    return float(cathode_potential) >= 4.2 and float(max_oxidation_state) >= 4.0
