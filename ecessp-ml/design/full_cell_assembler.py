from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple

from design.compatibility_model import CompatibilityRecord
from design.system_template import BatterySystem
from design.physics_chemistry import parse_formula
from materials.chemistry_engine import (
    StrictOxidationSolver,
    PolyanionLibrary,
    StructureClassifier,
    InsertionFilter,
    AlkaliValidator,
)
from design.electrolyte_stability_model import (
    evaluate_electrolyte_stability,
    sei_requirement_flag,
    thermal_risk_flag,
)


@dataclass
class AssembledBatteryCandidate:
    candidate_id: str
    working_ion: str
    cathode_formula: str
    anode_formula: str
    electrolyte_formula: str
    separator_material: str
    additive_material: str
    theoretical_voltage_window: Tuple[float, float]
    cathode_redox_potential: float = 0.0
    anode_redox_potential: float = 0.0
    full_cell_voltage: float = 0.0
    np_ratio: float = 1.0
    limiting_electrode: str = "unknown"
    theoretical_capacity_cathode: float = 0.0
    theoretical_capacity_anode: float = 0.0
    electrolyte_window: Dict[str, float] = field(default_factory=dict)
    sei_expected: bool = False
    thermal_risk: bool = False
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    valid_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "working_ion": self.working_ion,
            "cathode_formula": self.cathode_formula,
            "anode_formula": self.anode_formula,
            "electrolyte_formula": self.electrolyte_formula,
            "separator_material": self.separator_material,
            "additive_material": self.additive_material,
            "theoretical_voltage_window": list(self.theoretical_voltage_window),
            "cathode_redox_potential": float(self.cathode_redox_potential),
            "anode_redox_potential": float(self.anode_redox_potential),
            "full_cell_voltage": float(self.full_cell_voltage),
            "np_ratio": float(self.np_ratio),
            "limiting_electrode": self.limiting_electrode,
            "theoretical_capacity_cathode": float(self.theoretical_capacity_cathode),
            "theoretical_capacity_anode": float(self.theoretical_capacity_anode),
            "electrolyte_window": dict(self.electrolyte_window),
            "sei_expected": bool(self.sei_expected),
            "thermal_risk": bool(self.thermal_risk),
            "uncertainty": dict(self.uncertainty),
            "provenance": dict(self.provenance),
            "valid": bool(self.valid),
            "valid_reasons": list(self.valid_reasons),
        }


class FullCellAssembler:
    """
    Stage-4 full-cell assembler producing explicit assembled candidates.
    """

    ION_STACK_DEFAULTS: Dict[str, Dict[str, str]] = {
        "Li": {
            "electrolyte": "1M LiPF6 in EC/EMC",
            "separator_material": "PP/PE trilayer separator",
            "additive_material": "FEC + VC blend",
        },
        "Na": {
            "electrolyte": "NaPF6 in EC/DEC",
            "separator_material": "Ceramic-coated PP/PE separator",
            "additive_material": "FEC + NaDFOB blend",
        },
        "K": {
            "electrolyte": "KPF6 in carbonate solvent",
            "separator_material": "Microporous PP/PE separator",
            "additive_material": "FEC + KFSI stabilizer",
        },
        "Mg": {
            "electrolyte": "Mg(TFSI)2 in glyme solvent",
            "separator_material": "Glass-fiber separator",
            "additive_material": "Interphase stabilizer package",
        },
        "Ca": {
            "electrolyte": "Ca(TFSI)2 solvent blend",
            "separator_material": "Ceramic-coated polymer separator",
            "additive_material": "SEI-promoting additive blend",
        },
        "Zn": {
            "electrolyte": "ZnSO4 aqueous electrolyte",
            "separator_material": "Cellulose or glass-fiber separator",
            "additive_material": "Interface-stabilizer additive blend",
        },
        "Al": {
            "electrolyte": "Chloroaluminate ionic liquid",
            "separator_material": "High-stability polymer membrane",
            "additive_material": "Corrosion inhibitor package",
        },
        "Y": {
            "electrolyte": "Exploratory electrolyte screening",
            "separator_material": "Ceramic-coated exploratory separator",
            "additive_material": "Exploratory additive screening",
        },
    }
    ION_EFFECTIVE_RADIUS_ANG: Dict[str, float] = {
        "Li": 0.76,
        "Na": 1.02,
        "K": 1.38,
        "Mg": 0.72,
        "Ca": 1.00,
        "Zn": 0.74,
        "Al": 0.54,
        "Y": 0.90,
    }
    ION_CHARGE_STATE: Dict[str, str] = {
        "Li": "+1",
        "Na": "+1",
        "K": "+1",
        "Mg": "+2",
        "Ca": "+2",
        "Zn": "+2",
        "Al": "+3",
        "Y": "+3",
    }
    BATTERY_ASSEMBLY_RULES: Dict[str, Any] = {
        "working_ion_constraints": {
            "allowed_charge_states": {"+1", "+2", "+3"},
            "ionic_radius_angstrom_range": (0.5, 1.2),
        },
        "electrode_role_rules": {
            "anode_redox_potential_max_v": 1.2,
            "cathode_redox_potential_min_v": 2.0,
            "anode_volume_change_percent_max": 40.0,
        },
        "voltage_constraints": {
            "minimum_cell_voltage_v": 1.5,
            "maximum_cell_voltage_v": 5.0,
            "reject_if_voltage_negative": True,
        },
    }
    ROLE_ASSIGNMENT_ENGINE: Dict[str, Any] = {
        "sort_by_average_redox_potential": "ascending",
        "role_rules": {
            "lowest_redox_material": "assign_anode",
            "highest_redox_material": "assign_cathode",
        },
        "validation_rules": {
            "minimum_voltage_difference_v": 1.0,
            "reject_if_negative_voltage": True,
            "reject_if_overlap_percentage_above": 65.0,
        },
        "override_allowed": False,
    }
    ADDITIVE_INTERPHASE_RULES: Dict[str, Any] = {
        "max_ionic_conductivity_drop_fraction": 0.15,
        "max_gas_generation_risk": 0.50,
        "require_stable_interphase": True,
    }
    ANODE_POTENTIAL_MAX_BY_ION: Dict[str, float] = {
        "Li": 1.2,
        "Na": 1.5,
        "K": 1.6,
        "Mg": 1.8,
        "Ca": 1.8,
        "Zn": 1.6,
        "Al": 1.8,
        "Y": 1.8,
    }
    STACK_PHYSICS_RULES: Dict[str, Any] = {
        "ion_diffusion_barrier_ev_max": 0.60,
        "ion_diffusion_coefficient_cm2_s_min": 1e-14,
        "charge_transfer_resistance_ohm_max": 500.0,
        "delta_g_cell_ev_max": -0.1,  # must be <= -0.1 eV-equivalent proxy
        "max_voltage_deviation_percent": 10.0,
    }
    ELECTROLYTE_HARD_CONSTRAINTS: Dict[str, Any] = {
        "must_conduct_working_ion": True,
        "must_be_electronically_insulating": True,
        "no_redox_activity_within_cell_window": True,
        "band_gap_eV_min": 3.0,
    }
    _TRANSITION_METALS = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
        "Y", "Zr", "Nb", "Mo", "W", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
    FAMILY_MAX_INTERCALATION_RADIUS_ANG: Dict[str, float] = {
        "Layered oxide": 1.12,
        "Spinel": 1.05,
        "Olivine": 1.00,
        "NASICON": 1.32,
        "Prussian Blue": 1.45,
        "Polyanion framework": 1.20,
    }
    SEPARATOR_LIBRARY: tuple[Dict[str, Any], ...] = (
        {
            "profile": "polyolefin",
            "tokens": ("pp/pe", "polyolefin", "high-porosity pe", "microporous pp/pe"),
            "supported_ions": {"Li", "Na", "K"},
            "max_ion_radius_ang": 1.16,
            "max_voltage": 4.5,
            "allow_aqueous": False,
        },
        {
            "profile": "ceramic_coated",
            "tokens": ("ceramic-coated", "ceramic coated", "ceramic-polymer", "porous-ceramic"),
            "supported_ions": {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y"},
            "max_ion_radius_ang": 1.45,
            "max_voltage": 5.0,
            "allow_aqueous": True,
        },
        {
            "profile": "glass_fiber_or_cellulose",
            "tokens": ("glass-fiber", "glass fiber", "cellulose"),
            "supported_ions": {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y"},
            "max_ion_radius_ang": 1.80,
            "max_voltage": 4.2,
            "allow_aqueous": True,
        },
        {
            "profile": "polyimide",
            "tokens": ("polyimide",),
            "supported_ions": {"Li", "Na", "K", "Mg", "Ca"},
            "max_ion_radius_ang": 1.35,
            "max_voltage": 4.8,
            "allow_aqueous": False,
        },
        {
            "profile": "polymer_membrane",
            "tokens": ("polymer membrane", "high-stability polymer membrane"),
            "supported_ions": {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y"},
            "max_ion_radius_ang": 1.35,
            "max_voltage": 4.6,
            "allow_aqueous": False,
        },
    )
    ADDITIVE_LIBRARY: tuple[Dict[str, Any], ...] = (
        {
            "profile": "fec_vc",
            "tokens": ("fec + vc", "vc blend", "vc"),
            "supported_ions": {"Li", "Na", "K"},
            "min_voltage": 1.8,
            "max_voltage": 4.5,
            "allow_aqueous": False,
        },
        {
            "profile": "lidfob",
            "tokens": ("lidfob",),
            "supported_ions": {"Li"},
            "min_voltage": 2.0,
            "max_voltage": 4.6,
            "allow_aqueous": False,
        },
        {
            "profile": "nadfob_nafsi",
            "tokens": ("nadfob", "nafsi", "dtd"),
            "supported_ions": {"Na"},
            "min_voltage": 1.8,
            "max_voltage": 4.4,
            "allow_aqueous": False,
        },
        {
            "profile": "multivalent_interphase",
            "tokens": ("interphase stabilizer", "chelating", "halide-balance"),
            "supported_ions": {"Mg", "Ca", "Zn", "Al", "Y"},
            "min_voltage": 1.5,
            "max_voltage": 4.0,
            "allow_aqueous": True,
        },
        {
            "profile": "corrosion_inhibitor",
            "tokens": ("corrosion inhibitor",),
            "supported_ions": {"Al", "Zn"},
            "min_voltage": 1.2,
            "max_voltage": 3.5,
            "allow_aqueous": True,
        },
        {
            "profile": "generic_hf_scavenger",
            "tokens": ("hf-scavenger", "impedance-control", "cycle-life booster", "interface-stabilizer"),
            "supported_ions": {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y"},
            "min_voltage": 1.2,
            "max_voltage": 4.8,
            "allow_aqueous": False,
        },
    )

    _CATHODE_TM_POTENTIALS = {
        "Mn": 3.8,
        "Fe": 3.4,
        "Co": 3.9,
        "Ni": 4.0,
        "V": 3.3,
        "Cr": 3.2,
        "Ti": 2.2,
        "Cu": 3.6,
    }
    _ANODE_TM_POTENTIALS = {
        "Mn": 1.2,
        "Fe": 0.9,
        "Co": 1.1,
        "Ni": 1.0,
        "V": 1.3,
        "Cr": 1.0,
        "Ti": 0.8,
        "Cu": 1.2,
        "C": 0.2,
        "Si": 0.25,
        "Sn": 0.45,
    }

    def __init__(self):
        self.strict_oxidation_solver = StrictOxidationSolver()
        self.polyanion_library = PolyanionLibrary()
        self.structure_classifier = StructureClassifier()
        self.insertion_filter = InsertionFilter()
        self.alkali_validator = AlkaliValidator()

    def _stack_defaults(self, working_ion: str) -> Dict[str, str]:
        ion = str(working_ion or "Li").strip()
        if ion in self.ION_STACK_DEFAULTS:
            return dict(self.ION_STACK_DEFAULTS[ion])
        return dict(self.ION_STACK_DEFAULTS["Li"])

    @staticmethod
    def _norm_text(value: str | None) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _match_profile(
        cls,
        *,
        text: str,
        library: tuple[Dict[str, Any], ...],
        kind: str,
    ) -> Dict[str, Any]:
        lowered = cls._norm_text(text)
        for profile in library:
            tokens = profile.get("tokens", ())
            if any(token in lowered for token in tokens):
                out = dict(profile)
                out["matched"] = True
                out["resolved_from"] = "library"
                return out

        if "generated" in lowered and kind in lowered:
            return {
                "profile": f"generated_{kind}",
                "matched": True,
                "resolved_from": "generated_fallback",
                "supported_ions": {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y"},
                "max_ion_radius_ang": 1.40 if kind == "separator" else None,
                "max_voltage": 4.6 if kind == "separator" else 4.5,
                "min_voltage": 1.5 if kind == "additive" else None,
                "allow_aqueous": False,
            }

        return {
            "profile": "unresolved",
            "matched": False,
            "resolved_from": "none",
            "supported_ions": set(),
            "max_ion_radius_ang": None,
            "max_voltage": None,
            "min_voltage": None,
            "allow_aqueous": False,
        }

    @classmethod
    def _ion_radius(cls, working_ion: str) -> float:
        ion = str(working_ion or "Li").strip()
        return float(cls.ION_EFFECTIVE_RADIUS_ANG.get(ion, 0.90))

    @classmethod
    def _ion_charge(cls, working_ion: str) -> str:
        ion = str(working_ion or "Li").strip()
        return str(cls.ION_CHARGE_STATE.get(ion, "+1"))

    @classmethod
    def _family_radius_limit(cls, family: str | None) -> float:
        return float(cls.FAMILY_MAX_INTERCALATION_RADIUS_ANG.get(str(family or ""), 1.15))

    def _evaluate_component_compatibility(
        self,
        *,
        working_ion: str,
        cathode_family: str | None,
        anode_family: str | None,
        cathode_potential: float,
        anode_potential: float,
        full_cell_voltage: float,
        electrolyte_formula: str,
        separator_material: str,
        additive_material: str,
        cathode_diagnostics: Dict[str, Any] | None = None,
        anode_diagnostics: Dict[str, Any] | None = None,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        reasons: list[str] = []
        diagnostics: Dict[str, Any] = {}

        ion = str(working_ion or "Li").strip()
        ion_radius = self._ion_radius(ion)
        cath_limit = self._family_radius_limit(cathode_family)
        an_limit = self._family_radius_limit(anode_family)
        diagnostics["ion_transport"] = {
            "working_ion": ion,
            "ion_radius_ang": float(ion_radius),
            "cathode_family": str(cathode_family or ""),
            "anode_family": str(anode_family or ""),
            "cathode_family_radius_limit_ang": float(cath_limit),
            "anode_family_radius_limit_ang": float(an_limit),
        }

        if ion_radius > cath_limit + 1e-8:
            reasons.append("cathode_intercalation_size_mismatch")
        if ion_radius > an_limit + 1e-8:
            reasons.append("anode_intercalation_size_mismatch")

        cath_dim = str(((cathode_diagnostics or {}).get("structure") or {}).get("diffusion_dimensionality") or "").upper()
        an_dim = str(((anode_diagnostics or {}).get("structure") or {}).get("diffusion_dimensionality") or "").upper()
        if cath_dim == "1D" and ion_radius > 1.05:
            reasons.append("cathode_1d_diffusion_radius_risk")
        if an_dim == "1D" and ion_radius > 1.05:
            reasons.append("anode_1d_diffusion_radius_risk")

        separator_profile = self._match_profile(text=separator_material, library=self.SEPARATOR_LIBRARY, kind="separator")
        additive_profile = self._match_profile(text=additive_material, library=self.ADDITIVE_LIBRARY, kind="additive")

        electrolyte_text = self._norm_text(electrolyte_formula)
        aqueous_electrolyte = any(tok in electrolyte_text for tok in ("aqueous", "h2o", "so4", "water"))
        _, electrolyte_reasons, electrolyte_diag = self._evaluate_electrolyte_hard_constraints(
            working_ion=ion,
            electrolyte_formula=str(electrolyte_formula),
        )
        reasons.extend(electrolyte_reasons)

        sep_supported = set(separator_profile.get("supported_ions", set()) or set())
        if sep_supported and ion not in sep_supported:
            reasons.append("separator_ion_not_supported")
        sep_radius_limit = separator_profile.get("max_ion_radius_ang")
        if sep_radius_limit is not None and ion_radius > float(sep_radius_limit) + 1e-8:
            reasons.append("separator_transport_radius_exceeded")
        sep_v_max = separator_profile.get("max_voltage")
        if sep_v_max is not None and float(cathode_potential) > float(sep_v_max) + 1e-8:
            reasons.append("separator_oxidation_window_exceeded")
        if aqueous_electrolyte and not bool(separator_profile.get("allow_aqueous", False)):
            reasons.append("separator_not_aqueous_compatible")
        if ion in {"Mg", "Ca", "Zn", "Al"} and "pp/pe" in self._norm_text(separator_material):
            reasons.append("separator_multivalent_transport_risk")

        add_supported = set(additive_profile.get("supported_ions", set()) or set())
        if add_supported and ion not in add_supported:
            reasons.append("additive_ion_not_supported")
        add_v_min = additive_profile.get("min_voltage")
        if add_v_min is not None and float(anode_potential) < float(add_v_min) - 1e-8:
            reasons.append("additive_reduction_window_exceeded")
        add_v_max = additive_profile.get("max_voltage")
        if add_v_max is not None and float(cathode_potential) > float(add_v_max) + 1e-8:
            reasons.append("additive_oxidation_window_exceeded")
        if aqueous_electrolyte and not bool(additive_profile.get("allow_aqueous", False)):
            reasons.append("additive_not_aqueous_compatible")

        diagnostics["separator_profile"] = {
            "name": str(separator_profile.get("profile", "unresolved")),
            "resolved_from": str(separator_profile.get("resolved_from", "none")),
            "matched": bool(separator_profile.get("matched", False)),
            "supported_ions": sorted(set(separator_profile.get("supported_ions", set()) or set())),
            "max_ion_radius_ang": separator_profile.get("max_ion_radius_ang"),
            "max_voltage": separator_profile.get("max_voltage"),
            "allow_aqueous": bool(separator_profile.get("allow_aqueous", False)),
        }
        diagnostics["additive_profile"] = {
            "name": str(additive_profile.get("profile", "unresolved")),
            "resolved_from": str(additive_profile.get("resolved_from", "none")),
            "matched": bool(additive_profile.get("matched", False)),
            "supported_ions": sorted(set(additive_profile.get("supported_ions", set()) or set())),
            "min_voltage": additive_profile.get("min_voltage"),
            "max_voltage": additive_profile.get("max_voltage"),
            "allow_aqueous": bool(additive_profile.get("allow_aqueous", False)),
        }
        diagnostics["electrolyte_environment"] = {
            "formula": str(electrolyte_formula),
            "aqueous": bool(aqueous_electrolyte),
            "full_cell_voltage": float(full_cell_voltage),
            "cathode_potential": float(cathode_potential),
            "anode_potential": float(anode_potential),
        }
        diagnostics["electrolyte_constraints"] = dict(electrolyte_diag)
        diagnostics["valid"] = len(reasons) == 0
        diagnostics["reasons"] = list(reasons)

        return (len(reasons) == 0), reasons, diagnostics

    def _estimate_electrolyte_band_gap_proxy(self, *, electrolyte_formula: str) -> float:
        text = self._norm_text(electrolyte_formula)
        if any(tok in text for tok in ("ec", "emc", "dec", "glyme", "solvent", "ionic liquid", "aqueous")):
            return 4.2
        try:
            comp = parse_formula(str(electrolyte_formula or "").replace(" ", ""))
        except Exception:
            comp = {}
        elements = set(comp.keys())
        if any(el in self._TRANSITION_METALS for el in elements):
            return 1.8
        if elements and elements.issubset({"Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y", "P", "F", "S", "O", "N", "Cl", "Br", "I", "B", "C", "H"}):
            return 3.6
        return 3.2

    def _evaluate_electrolyte_hard_constraints(
        self,
        *,
        working_ion: str,
        electrolyte_formula: str,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        reasons: list[str] = []
        rules = self.ELECTROLYTE_HARD_CONSTRAINTS
        text = self._norm_text(electrolyte_formula)
        ion = str(working_ion or "Li").strip()
        ion_token = ion.lower()

        try:
            composition = parse_formula(str(electrolyte_formula or "").replace(" ", ""))
        except Exception:
            composition = {}
        elements = set(composition.keys())

        ion_conductive = bool(ion_token in text or ion in elements)
        electronically_insulating = not any(el in self._TRANSITION_METALS for el in elements)
        redox_inactive = electronically_insulating
        band_gap_proxy_ev = float(self._estimate_electrolyte_band_gap_proxy(electrolyte_formula=electrolyte_formula))

        if bool(rules.get("must_conduct_working_ion", True)) and not ion_conductive:
            reasons.append("electrolyte_not_ion_conductive")
        if bool(rules.get("must_be_electronically_insulating", True)) and not electronically_insulating:
            reasons.append("electrolyte_not_electronically_insulating")
        if bool(rules.get("no_redox_activity_within_cell_window", True)) and not redox_inactive:
            reasons.append("electrolyte_redox_active_in_cell_window")
        if band_gap_proxy_ev < float(rules.get("band_gap_eV_min", 3.0)):
            reasons.append("electrolyte_band_gap_below_minimum")

        diagnostics = {
            "rules": dict(rules),
            "working_ion": ion,
            "electrolyte_formula": str(electrolyte_formula),
            "ion_conductive": bool(ion_conductive),
            "electronically_insulating": bool(electronically_insulating),
            "redox_inactive_within_window": bool(redox_inactive),
            "band_gap_proxy_ev": float(band_gap_proxy_ev),
            "hard_valid": len(reasons) == 0,
            "reasons": list(reasons),
        }
        return (len(reasons) == 0), reasons, diagnostics

    def _evaluate_battery_assembly_rules(
        self,
        *,
        working_ion: str,
        cathode_formula: str,
        anode_formula: str,
        electrolyte_formula: str,
        separator_material: str,
        additive_material: str,
        cathode_potential: float,
        anode_potential: float,
        full_cell_voltage: float,
        delta_volume_ratio: float,
        electrolyte_window_valid: bool,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        reasons: list[str] = []
        rules = self.BATTERY_ASSEMBLY_RULES

        required_components = {
            "working_ion": str(working_ion or "").strip(),
            "anode": str(anode_formula or "").strip(),
            "cathode": str(cathode_formula or "").strip(),
            "electrolyte": str(electrolyte_formula or "").strip(),
            "separator": str(separator_material or "").strip(),
            "electrolyte_additives": str(additive_material or "").strip(),
        }
        for key, value in required_components.items():
            if not value:
                reasons.append(f"missing_required_component:{key}")

        charge_state = self._ion_charge(working_ion)
        allowed_charge = set(rules["working_ion_constraints"]["allowed_charge_states"])
        ion_radius = float(self._ion_radius(working_ion))
        ion_radius_lo, ion_radius_hi = rules["working_ion_constraints"]["ionic_radius_angstrom_range"]
        if charge_state not in allowed_charge:
            reasons.append("working_ion_charge_state_not_allowed")
        if not (float(ion_radius_lo) <= ion_radius <= float(ion_radius_hi)):
            reasons.append("working_ion_radius_out_of_range")

        role_rules = rules["electrode_role_rules"]
        ion_key = str(working_ion or "Li").strip() or "Li"
        anode_potential_max = float(
            self.ANODE_POTENTIAL_MAX_BY_ION.get(
                ion_key,
                role_rules["anode_redox_potential_max_v"],
            )
        )
        if float(anode_potential) > float(anode_potential_max):
            reasons.append("anode_potential_too_high")
        if float(cathode_potential) < float(role_rules["cathode_redox_potential_min_v"]):
            reasons.append("cathode_potential_too_low")
        if float(delta_volume_ratio) * 100.0 > float(role_rules["anode_volume_change_percent_max"]):
            reasons.append("volume_change_exceeds_rule_limit")

        v_rules = rules["voltage_constraints"]
        if bool(v_rules.get("reject_if_voltage_negative", True)) and float(full_cell_voltage) < 0.0:
            reasons.append("cell_voltage_negative")
        if float(full_cell_voltage) < float(v_rules["minimum_cell_voltage_v"]):
            reasons.append("cell_voltage_below_minimum")
        if float(full_cell_voltage) > float(v_rules["maximum_cell_voltage_v"]):
            reasons.append("cell_voltage_above_maximum")
        if not bool(electrolyte_window_valid):
            reasons.append("electrolyte_window_not_covering_voltage")

        role_valid, role_reasons, role_diag = self._evaluate_role_assignment_engine(
            anode_potential=float(anode_potential),
            cathode_potential=float(cathode_potential),
        )
        if not role_valid:
            hard_role_reasons = [r for r in role_reasons if str(r) == "negative_voltage"]
            soft_role_warnings = [r for r in role_reasons if str(r) != "negative_voltage"]
            if hard_role_reasons:
                reasons.extend([f"role_assignment:{r}" for r in hard_role_reasons])
            role_diag = dict(role_diag)
            role_diag["soft_warnings"] = list(soft_role_warnings)
            role_diag["soft_warning_count"] = int(len(soft_role_warnings))

        diagnostics = {
            "required_components": {k: bool(v) for k, v in required_components.items()},
            "working_ion_constraints": {
                "working_ion": str(working_ion),
                "charge_state": charge_state,
                "allowed_charge_states": sorted(allowed_charge),
                "ion_radius_ang": ion_radius,
                "ion_radius_range_ang": [float(ion_radius_lo), float(ion_radius_hi)],
            },
            "electrode_role_rules": {
                "anode_potential_v": float(anode_potential),
                "cathode_potential_v": float(cathode_potential),
                "anode_potential_max_v": float(anode_potential_max),
                "cathode_potential_min_v": float(role_rules["cathode_redox_potential_min_v"]),
                "delta_volume_percent": float(delta_volume_ratio) * 100.0,
                "max_volume_change_percent": float(role_rules["anode_volume_change_percent_max"]),
            },
            "voltage_constraints": {
                "full_cell_voltage_v": float(full_cell_voltage),
                "minimum_cell_voltage_v": float(v_rules["minimum_cell_voltage_v"]),
                "maximum_cell_voltage_v": float(v_rules["maximum_cell_voltage_v"]),
                "electrolyte_window_covers_voltage": bool(electrolyte_window_valid),
            },
            "role_assignment_engine": dict(role_diag),
            "hard_rejection_reasons": list(reasons),
            "hard_valid": len(reasons) == 0,
        }
        return (len(reasons) == 0), reasons, diagnostics

    @staticmethod
    def _estimate_window(
        *,
        target_voltage: float,
        compatibility_score: float,
    ) -> tuple[float, float]:
        center = max(1.0, float(target_voltage))
        half_width = max(0.2, 0.9 - 0.6 * compatibility_score)
        return (max(0.5, center - half_width), center + half_width)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    def _estimate_redox_potential(
        self,
        *,
        formula: str,
        role: str,
        family: str | None,
        working_ion: str,
    ) -> float:
        solver = self.strict_oxidation_solver.solve_formula(formula)
        comp = solver.composition
        if not comp:
            return 3.2 if role == "cathode" else 1.0

        if role == "anode" and "C" in comp and "O" not in comp:
            return 0.2

        table = self._CATHODE_TM_POTENTIALS if role == "cathode" else self._ANODE_TM_POTENTIALS
        num = 0.0
        den = 0.0
        for el, coeff in comp.items():
            if el in table:
                num += float(coeff) * float(table[el])
                den += float(coeff)
        if den > 0.0:
            base = num / den
        else:
            base = 3.3 if role == "cathode" else 1.0

        ion = str(working_ion or "Li").strip()
        if ion == "Na":
            base -= 0.2
        elif ion == "Mg":
            base -= 0.35
        elif ion == "K":
            base -= 0.25

        fam = str(family or "").lower()
        if "polyanion" in fam or "nasicon" in fam:
            base += 0.15
        if "spinel" in fam and role == "cathode":
            base += 0.10
        if "layered" in fam and role == "anode":
            base += 0.10

        if role == "cathode":
            return self._clamp(base, 2.0, 4.5)
        return self._clamp(base, 0.0, 2.8)

    def _estimate_average_redox_potential(
        self,
        *,
        formula: str,
        family: str | None,
        working_ion: str,
    ) -> float:
        solver = self.strict_oxidation_solver.solve_formula(formula)
        comp = solver.composition
        if not comp:
            return 2.2

        if "C" in comp and "O" not in comp:
            return 0.2

        num = 0.0
        den = 0.0
        for el, coeff in comp.items():
            if el in self._CATHODE_TM_POTENTIALS and el in self._ANODE_TM_POTENTIALS:
                value = 0.5 * (
                    float(self._CATHODE_TM_POTENTIALS[el]) + float(self._ANODE_TM_POTENTIALS[el])
                )
            elif el in self._CATHODE_TM_POTENTIALS:
                value = float(self._CATHODE_TM_POTENTIALS[el])
            elif el in self._ANODE_TM_POTENTIALS:
                value = float(self._ANODE_TM_POTENTIALS[el])
            else:
                continue
            num += float(coeff) * value
            den += float(coeff)

        base = (num / den) if den > 0.0 else 2.2
        ion = str(working_ion or "Li").strip()
        if ion == "Na":
            base -= 0.2
        elif ion == "Mg":
            base -= 0.35
        elif ion == "K":
            base -= 0.25

        fam = str(family or "").lower()
        if "polyanion" in fam or "nasicon" in fam:
            base += 0.10
        if "spinel" in fam:
            base += 0.05
        return self._clamp(base, 0.0, 4.6)

    def _evaluate_role_assignment_engine(
        self,
        *,
        anode_potential: float,
        cathode_potential: float,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        rules = self.ROLE_ASSIGNMENT_ENGINE.get("validation_rules", {})
        min_voltage = float(rules.get("minimum_voltage_difference_v", 1.5))
        reject_negative = bool(rules.get("reject_if_negative_voltage", True))
        overlap_threshold = float(rules.get("reject_if_overlap_percentage_above", 50.0))

        voltage_gap = float(cathode_potential - anode_potential)
        reasons: list[str] = []

        if reject_negative and voltage_gap < 0.0:
            reasons.append("negative_voltage")
        if voltage_gap < min_voltage:
            reasons.append("minimum_voltage_difference_not_met")

        # Map voltage separation onto a 0-100 "redox overlap" proxy.
        # 3.0V separation => 0% overlap; 0V separation => 100% overlap.
        span_v = 3.0
        overlap_pct = 100.0 * (1.0 - max(0.0, min(span_v, voltage_gap)) / span_v)
        overlap_pct = float(max(0.0, min(100.0, overlap_pct)))
        if overlap_pct > overlap_threshold:
            reasons.append("overlap_percentage_above_threshold")

        diagnostics = {
            "anode_potential_v": float(anode_potential),
            "cathode_potential_v": float(cathode_potential),
            "voltage_gap_v": float(voltage_gap),
            "minimum_voltage_difference_v": float(min_voltage),
            "reject_if_negative_voltage": bool(reject_negative),
            "redox_overlap_percentage": float(overlap_pct),
            "reject_if_overlap_percentage_above": float(overlap_threshold),
            "override_allowed": bool(self.ROLE_ASSIGNMENT_ENGINE.get("override_allowed", False)),
            "hard_valid": len(reasons) == 0,
            "hard_rejection_reasons": list(reasons),
        }
        return (len(reasons) == 0), reasons, diagnostics

    @staticmethod
    def _between(value: float, a: float, b: float) -> bool:
        lo = min(float(a), float(b))
        hi = max(float(a), float(b))
        return (value > lo + 1e-8) and (value < hi - 1e-8)

    def _evaluate_additive_interphase_validation(
        self,
        *,
        additive_material: str,
        anode_potential: float,
        cathode_potential: float,
        electrolyte_reduction_limit: float,
        electrolyte_oxidation_limit: float,
        component_diag: Dict[str, Any],
        sei_expected: bool,
        thermal_risk: bool,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        reasons: list[str] = []
        additive_profile = (component_diag or {}).get("additive_profile", {}) if isinstance(component_diag, dict) else {}
        add_min = additive_profile.get("min_voltage")
        add_max = additive_profile.get("max_voltage")

        # Place additive proxies between neighboring interfaces by construction.
        additive_reduction_proxy = float(0.5 * (anode_potential + electrolyte_reduction_limit))
        additive_oxidation_proxy = float(0.5 * (cathode_potential + electrolyte_oxidation_limit))

        if not self._between(additive_reduction_proxy, anode_potential, electrolyte_reduction_limit):
            reasons.append("additive_reduction_potential_out_of_band")
        if not self._between(additive_oxidation_proxy, electrolyte_oxidation_limit, cathode_potential):
            reasons.append("additive_oxidation_potential_out_of_band")

        profile_reduction_window_consistent = (
            True
            if not isinstance(add_min, (int, float))
            else self._between(float(add_min), anode_potential, electrolyte_reduction_limit)
        )
        profile_oxidation_window_consistent = (
            True
            if not isinstance(add_max, (int, float))
            else self._between(float(add_max), electrolyte_oxidation_limit, cathode_potential)
        )

        # Simple deterministic transport-risk proxy: unresolved additives and broad windows
        # penalize ionic conductivity retention.
        resolved = bool(additive_profile.get("matched", False))
        ionic_drop = 0.05
        if not resolved:
            ionic_drop += 0.10
        if isinstance(add_min, (int, float)) and isinstance(add_max, (int, float)):
            ionic_drop += max(0.0, (float(add_max) - float(add_min) - 2.8) * 0.03)
        ionic_drop = float(max(0.0, min(0.50, ionic_drop)))
        if ionic_drop > float(self.ADDITIVE_INTERPHASE_RULES["max_ionic_conductivity_drop_fraction"]):
            reasons.append("additive_ionic_conductivity_drop_too_high")

        gas_generation_risk = 0.10
        text = self._norm_text(additive_material)
        if "halide" in text:
            gas_generation_risk += 0.18
        if "generated" in text:
            gas_generation_risk += 0.10
        if thermal_risk:
            gas_generation_risk += 0.18
        gas_generation_risk = float(max(0.0, min(1.0, gas_generation_risk)))
        if gas_generation_risk > float(self.ADDITIVE_INTERPHASE_RULES["max_gas_generation_risk"]):
            reasons.append("additive_gas_generation_risk_high")

        interphase_stable = bool(sei_expected and not thermal_risk)
        if bool(self.ADDITIVE_INTERPHASE_RULES["require_stable_interphase"]) and not interphase_stable:
            reasons.append("unstable_interphase_predicted")

        diagnostics = {
            "additive_material": str(additive_material),
            "additive_reduction_proxy_v": float(additive_reduction_proxy),
            "additive_oxidation_proxy_v": float(additive_oxidation_proxy),
            "electrolyte_reduction_limit_v": float(electrolyte_reduction_limit),
            "electrolyte_oxidation_limit_v": float(electrolyte_oxidation_limit),
            "ionic_conductivity_drop_fraction": float(ionic_drop),
            "max_ionic_conductivity_drop_fraction": float(
                self.ADDITIVE_INTERPHASE_RULES["max_ionic_conductivity_drop_fraction"]
            ),
            "gas_generation_risk": float(gas_generation_risk),
            "max_gas_generation_risk": float(self.ADDITIVE_INTERPHASE_RULES["max_gas_generation_risk"]),
            "profile_reduction_window_consistent": bool(profile_reduction_window_consistent),
            "profile_oxidation_window_consistent": bool(profile_oxidation_window_consistent),
            "interphase_stable": bool(interphase_stable),
            "hard_valid": len(reasons) == 0,
            "hard_rejection_reasons": list(reasons),
        }
        return (len(reasons) == 0), reasons, diagnostics

    def _estimate_transport_proxies(
        self,
        *,
        working_ion: str,
        cathode_diagnostics: Dict[str, Any],
        anode_diagnostics: Dict[str, Any],
        mechanical_strain_risk: float,
    ) -> Dict[str, float]:
        ion_radius = float(self._ion_radius(working_ion))
        cath_dim = str(((cathode_diagnostics or {}).get("structure") or {}).get("diffusion_dimensionality") or "").upper()
        an_dim = str(((anode_diagnostics or {}).get("structure") or {}).get("diffusion_dimensionality") or "").upper()
        one_d_count = int(cath_dim == "1D") + int(an_dim == "1D")

        barrier = 0.28
        barrier += 0.08 * float(one_d_count)
        barrier += 0.10 * max(0.0, ion_radius - 0.90)
        barrier += 0.12 * max(0.0, min(1.0, float(mechanical_strain_risk)))
        barrier = float(max(0.10, min(1.10, barrier)))

        # Arrhenius-like proxy at room temperature scale.
        diffusion_coeff = float(1e-12 * math.exp(-barrier / 0.12))
        charge_transfer_resistance = float(
            120.0 + 420.0 * barrier + 80.0 * max(0.0, min(1.0, float(mechanical_strain_risk)))
        )
        return {
            "ion_diffusion_barrier_ev": float(barrier),
            "ion_diffusion_coefficient_cm2_s": float(diffusion_coeff),
            "charge_transfer_resistance_ohm": float(charge_transfer_resistance),
        }

    def _evaluate_stack_physics_gates(
        self,
        *,
        working_ion: str,
        full_cell_voltage: float,
        delta_volume_ratio: float,
        cathode_diagnostics: Dict[str, Any],
        anode_diagnostics: Dict[str, Any],
        mechanical_strain_risk: float,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        reasons: list[str] = []
        proxies = self._estimate_transport_proxies(
            working_ion=working_ion,
            cathode_diagnostics=cathode_diagnostics,
            anode_diagnostics=anode_diagnostics,
            mechanical_strain_risk=mechanical_strain_risk,
        )
        barrier = float(proxies["ion_diffusion_barrier_ev"])
        diff = float(proxies["ion_diffusion_coefficient_cm2_s"])
        r_ct = float(proxies["charge_transfer_resistance_ohm"])
        delta_g_cell = float(-1.0 * float(full_cell_voltage))
        volume_change_percent = float(delta_volume_ratio) * 100.0

        if delta_g_cell > float(self.STACK_PHYSICS_RULES["delta_g_cell_ev_max"]):
            reasons.append("delta_g_cell_not_negative_enough")
        if volume_change_percent > 40.0:
            reasons.append("volume_change_percent_above_limit")
        if barrier > float(self.STACK_PHYSICS_RULES["ion_diffusion_barrier_ev_max"]):
            reasons.append("ion_diffusion_barrier_above_limit")
        if diff < float(self.STACK_PHYSICS_RULES["ion_diffusion_coefficient_cm2_s_min"]):
            reasons.append("ion_diffusion_coefficient_below_limit")
        if r_ct > float(self.STACK_PHYSICS_RULES["charge_transfer_resistance_ohm_max"]):
            reasons.append("charge_transfer_resistance_above_limit")

        diagnostics = {
            "delta_g_cell_proxy_ev": float(delta_g_cell),
            "delta_g_cell_ev_max": float(self.STACK_PHYSICS_RULES["delta_g_cell_ev_max"]),
            "volume_change_percent": float(volume_change_percent),
            "volume_change_percent_max": 40.0,
            "ion_diffusion_barrier_ev": float(barrier),
            "ion_diffusion_barrier_ev_max": float(self.STACK_PHYSICS_RULES["ion_diffusion_barrier_ev_max"]),
            "ion_diffusion_coefficient_cm2_s": float(diff),
            "ion_diffusion_coefficient_cm2_s_min": float(self.STACK_PHYSICS_RULES["ion_diffusion_coefficient_cm2_s_min"]),
            "charge_transfer_resistance_ohm": float(r_ct),
            "charge_transfer_resistance_ohm_max": float(self.STACK_PHYSICS_RULES["charge_transfer_resistance_ohm_max"]),
            "hard_valid": len(reasons) == 0,
            "hard_rejection_reasons": list(reasons),
        }
        return (len(reasons) == 0), reasons, diagnostics

    def _evaluate_voltage_integrity(
        self,
        *,
        recomputed_voltage: float,
        model_predicted_voltage: float,
    ) -> tuple[bool, list[str], Dict[str, Any]]:
        reasons: list[str] = []
        denom = max(1e-6, abs(float(recomputed_voltage)))
        deviation_pct = float(100.0 * abs(float(model_predicted_voltage) - float(recomputed_voltage)) / denom)
        max_dev = float(self.STACK_PHYSICS_RULES["max_voltage_deviation_percent"])
        if deviation_pct > max_dev + 1e-8:
            reasons.append("voltage_deviation_above_limit")
        diagnostics = {
            "recomputed_voltage_v": float(recomputed_voltage),
            "model_predicted_voltage_v": float(model_predicted_voltage),
            "voltage_deviation_percent": float(deviation_pct),
            "max_voltage_deviation_percent": float(max_dev),
            "hard_valid": len(reasons) == 0,
            "hard_rejection_reasons": list(reasons),
        }
        return (len(reasons) == 0), reasons, diagnostics

    def _validate_insertion_material(self, *, formula: str, working_ion: str) -> tuple[bool, list[str], dict[str, Any]]:
        reasons: list[str] = []
        diagnostics: dict[str, Any] = {}

        strict = self.strict_oxidation_solver.solve_formula(formula)
        diagnostics["strict_oxidation"] = {
            "valid": strict.valid,
            "unique_solution": strict.unique_solution,
            "oxidation_states": dict(strict.oxidation_states),
            "n_electrons_max": strict.n_electrons_max,
            "c_theoretical_mAh_g": strict.c_theoretical_mAh_g,
            "reasons": list(strict.reasons),
        }
        if not strict.valid:
            reasons.append("strict_oxidation_unsolved")

        poly = self.polyanion_library.validate_formula(formula)
        diagnostics["polyanion"] = {
            "valid": poly.valid,
            "recognized_units": list(poly.recognized_units),
            "reasons": list(poly.reasons),
        }
        if not poly.valid:
            reasons.extend(list(poly.reasons))

        struct = self.structure_classifier.classify_formula(
            formula,
            working_ion=working_ion,
        )
        diagnostics["structure"] = {
            "valid": struct.valid,
            "family": struct.family,
            "prototype_name": struct.prototype_name,
            "supported_working_ions": list(struct.supported_working_ions),
            "diffusion_dimensionality": struct.diffusion_dimensionality,
            "typical_voltage_range": list(struct.typical_voltage_range)
            if struct.typical_voltage_range is not None
            else None,
            "reasons": list(struct.reasons),
        }
        if not struct.valid:
            reasons.extend(list(struct.reasons))

        ins = self.insertion_filter.evaluate_formula(formula, structure_family=struct.family)
        diagnostics["insertion_filter"] = {
            "valid": ins.valid,
            "insertion_probability": float(ins.insertion_probability),
            "reasons": list(ins.reasons),
        }
        if not ins.valid:
            reasons.extend(list(ins.reasons))

        alk = self.alkali_validator.validate_formula(formula, working_ion)
        diagnostics["alkali"] = {
            "valid": alk.valid,
            "alkali_elements_present": list(alk.alkali_elements_present),
            "reasons": list(alk.reasons),
        }
        if not alk.valid:
            reasons.extend(list(alk.reasons))

        diagnostics["structure_family"] = struct.family
        diagnostics["strict_capacity_theoretical"] = strict.c_theoretical_mAh_g
        diagnostics["strict_n_electrons_max"] = strict.n_electrons_max
        return (len(reasons) == 0), reasons, diagnostics

    def assemble(
        self,
        *,
        index: int,
        compatibility: CompatibilityRecord,
        target_property_vector: Dict[str, float],
        base_system: BatterySystem,
        component_selection: Optional[Dict[str, Any]] = None,
    ) -> tuple[AssembledBatteryCandidate, BatterySystem]:
        working_ion = compatibility.cathode.working_ion or base_system.working_ion or "Li"
        stack_defaults = self._stack_defaults(working_ion)
        target_voltage = float(target_property_vector.get("average_voltage", 3.7) or 3.7)
        target_capacity = float(target_property_vector.get("capacity_grav", 180.0) or 180.0)
        target_delta_volume = float(target_property_vector.get("max_delta_volume", 0.14) or 0.14)

        compatibility_score = compatibility.aggregate_score()
        v_low, v_high = self._estimate_window(
            target_voltage=target_voltage,
            compatibility_score=compatibility_score,
        )

        material_a_formula = str(compatibility.cathode.framework_formula or "")
        material_b_formula = str(compatibility.anode.framework_formula or "")
        material_a_valid, material_a_reasons, material_a_diag = self._validate_insertion_material(
            formula=material_a_formula,
            working_ion=working_ion,
        )
        material_b_valid, material_b_reasons, material_b_diag = self._validate_insertion_material(
            formula=material_b_formula,
            working_ion=working_ion,
        )
        material_a_family = material_a_diag.get("structure_family")
        material_b_family = material_b_diag.get("structure_family")
        material_a_potential = self._estimate_average_redox_potential(
            formula=material_a_formula,
            family=material_a_family if isinstance(material_a_family, str) else None,
            working_ion=working_ion,
        )
        material_b_potential = self._estimate_average_redox_potential(
            formula=material_b_formula,
            family=material_b_family if isinstance(material_b_family, str) else None,
            working_ion=working_ion,
        )

        # Role lock is always derived from redox ordering; labels never override electrochemistry.
        role_swapped = bool(material_a_potential <= material_b_potential)
        if role_swapped:
            anode_formula = material_a_formula
            cathode_formula = material_b_formula
            an_valid, an_reasons, an_diag = material_a_valid, material_a_reasons, material_a_diag
            cath_valid, cath_reasons, cath_diag = material_b_valid, material_b_reasons, material_b_diag
            anode_avg_potential = float(material_a_potential)
            cathode_avg_potential = float(material_b_potential)
            anode_source_mode = str(compatibility.cathode.source_mode or "")
            cathode_source_mode = str(compatibility.anode.source_mode or "")
        else:
            anode_formula = material_b_formula
            cathode_formula = material_a_formula
            an_valid, an_reasons, an_diag = material_b_valid, material_b_reasons, material_b_diag
            cath_valid, cath_reasons, cath_diag = material_a_valid, material_a_reasons, material_a_diag
            anode_avg_potential = float(material_b_potential)
            cathode_avg_potential = float(material_a_potential)
            anode_source_mode = str(compatibility.anode.source_mode or "")
            cathode_source_mode = str(compatibility.cathode.source_mode or "")
        an_family = an_diag.get("structure_family")
        cath_family = cath_diag.get("structure_family")
        anode_v = self._estimate_redox_potential(
            formula=anode_formula,
            role="anode",
            family=an_family if isinstance(an_family, str) else None,
            working_ion=working_ion,
        )
        cathode_v = self._estimate_redox_potential(
            formula=cathode_formula,
            role="cathode",
            family=cath_family if isinstance(cath_family, str) else None,
            working_ion=working_ion,
        )
        full_cell_v = float(cathode_v - anode_v)

        role_assignment_valid, role_assignment_reasons, role_assignment_diag = self._evaluate_role_assignment_engine(
            anode_potential=float(anode_avg_potential),
            cathode_potential=float(cathode_avg_potential),
        )
        role_assignment_hard_reasons = [r for r in role_assignment_reasons if str(r) == "negative_voltage"]
        role_assignment_soft_warnings = [r for r in role_assignment_reasons if str(r) != "negative_voltage"]
        role_assignment_hard_valid = len(role_assignment_hard_reasons) == 0
        role_assignment_diag = dict(role_assignment_diag)
        role_assignment_diag["soft_warnings"] = list(role_assignment_soft_warnings)
        role_assignment_diag["soft_warning_count"] = int(len(role_assignment_soft_warnings))
        role_assignment_diag["hard_rejection_reasons"] = list(role_assignment_hard_reasons)
        role_assignment_diag["hard_valid"] = bool(role_assignment_hard_valid)

        cap_cath_theory = float(cath_diag.get("strict_capacity_theoretical") or 0.0)
        cap_an_theory = float(an_diag.get("strict_capacity_theoretical") or 0.0)
        if cap_cath_theory <= 0.0:
            cap_cath_theory = max(40.0, target_capacity * 1.1)
        if cap_an_theory <= 0.0:
            cap_an_theory = max(40.0, target_capacity * 1.3)

        np_ratio_raw = cap_an_theory / max(cap_cath_theory, 1e-6)
        np_ratio = self._clamp(np_ratio_raw, 1.05, 1.2)
        limiting = "cathode" if cap_cath_theory <= (cap_an_theory / max(np_ratio, 1e-6)) else "anode"
        limiting_capacity = min(cap_cath_theory, cap_an_theory / max(np_ratio, 1e-6))

        avg_v = max(0.6, min(v_high, max(v_low, full_cell_v)))
        cap_g = max(10.0, min(limiting_capacity, target_capacity * (0.85 + 0.20 * compatibility.chemical_stability_score)))
        cap_v = max(30.0, cap_g * 3.1)
        delta_v = max(0.0, target_delta_volume * (0.90 + 0.45 * compatibility.mechanical_strain_risk))

        validation_reasons: list[str] = []
        validation_reasons.extend(cath_reasons)
        validation_reasons.extend(an_reasons)
        validation_reasons.extend([f"role_assignment:{r}" for r in role_assignment_hard_reasons])
        if full_cell_v <= 1.0:
            validation_reasons.append("voltage_inverted_or_too_low")
        if not (1.05 <= np_ratio <= 1.2):
            validation_reasons.append("np_ratio_out_of_bounds")
        if delta_v > 0.20:
            validation_reasons.append("volume_expansion_exceeds_threshold")

        selection = dict(component_selection or {})
        battery_id = f"assembled_{index:03d}_{base_system.battery_id}"
        battery_formula = f"{working_ion}-{cathode_formula}|{anode_formula}"
        electrolyte_formula = str(
            selection.get("electrolyte_material")
            or compatibility.electrolyte.framework_formula
            or stack_defaults["electrolyte"]
        )
        separator_material = str(
            selection.get("separator_material")
            or stack_defaults["separator_material"]
        )
        additive_material = str(
            selection.get("additive_material")
            or stack_defaults["additive_material"]
        )
        electrolyte_alkali = self.alkali_validator.validate_formula(
            electrolyte_formula,
            working_ion,
        )
        if not bool(electrolyte_alkali.valid):
            validation_reasons.extend([f"electrolyte:{r}" for r in electrolyte_alkali.reasons])
        electrolyte_eval = evaluate_electrolyte_stability(
            electrolyte=electrolyte_formula,
            working_ion=working_ion,
            cathode_potential=float(cathode_v),
            anode_potential=float(anode_v),
        )
        if not bool(electrolyte_eval.valid):
            validation_reasons.append("electrolyte_stability_window_violation")
        cath_ox_states = cath_diag.get("strict_oxidation", {}).get("oxidation_states", {})
        max_cath_ox = max((float(v) for v in cath_ox_states.values()), default=0.0)
        sei_expected = sei_requirement_flag(
            anode_potential=float(anode_v),
            working_ion=working_ion,
        )
        thermal_risk = thermal_risk_flag(
            cathode_potential=float(cathode_v),
            max_oxidation_state=max_cath_ox,
        )
        component_valid, component_reasons, component_diag = self._evaluate_component_compatibility(
            working_ion=working_ion,
            cathode_family=cath_family if isinstance(cath_family, str) else None,
            anode_family=an_family if isinstance(an_family, str) else None,
            cathode_potential=float(cathode_v),
            anode_potential=float(anode_v),
            full_cell_voltage=float(full_cell_v),
            electrolyte_formula=str(electrolyte_formula),
            separator_material=str(separator_material),
            additive_material=str(additive_material),
            cathode_diagnostics=cath_diag,
            anode_diagnostics=an_diag,
        )
        validation_reasons.extend(component_reasons)
        assembly_rule_valid, assembly_rule_reasons, assembly_rule_diag = self._evaluate_battery_assembly_rules(
            working_ion=working_ion,
            cathode_formula=str(cathode_formula),
            anode_formula=str(anode_formula),
            electrolyte_formula=str(electrolyte_formula),
            separator_material=str(separator_material),
            additive_material=str(additive_material),
            cathode_potential=float(cathode_v),
            anode_potential=float(anode_v),
            full_cell_voltage=float(full_cell_v),
            delta_volume_ratio=float(delta_v),
            electrolyte_window_valid=bool(electrolyte_eval.valid),
        )
        validation_reasons.extend([f"assembly_rule:{r}" for r in assembly_rule_reasons])
        stage6_valid, stage6_reasons, stage6_diag = self._evaluate_additive_interphase_validation(
            additive_material=str(additive_material),
            anode_potential=float(anode_v),
            cathode_potential=float(cathode_v),
            electrolyte_reduction_limit=float(electrolyte_eval.window.reduction_limit),
            electrolyte_oxidation_limit=float(electrolyte_eval.window.oxidation_limit),
            component_diag=component_diag,
            sei_expected=bool(sei_expected),
            thermal_risk=bool(thermal_risk),
        )
        validation_reasons.extend([f"stage6:{r}" for r in stage6_reasons])
        stage7_valid, stage7_reasons, stage7_diag = self._evaluate_stack_physics_gates(
            working_ion=working_ion,
            full_cell_voltage=float(full_cell_v),
            delta_volume_ratio=float(delta_v),
            cathode_diagnostics=cath_diag,
            anode_diagnostics=an_diag,
            mechanical_strain_risk=float(compatibility.mechanical_strain_risk),
        )
        validation_reasons.extend([f"stage7:{r}" for r in stage7_reasons])
        stage8_valid, stage8_reasons, stage8_diag = self._evaluate_voltage_integrity(
            recomputed_voltage=float(full_cell_v),
            model_predicted_voltage=float(avg_v),
        )
        validation_reasons.extend([f"stage8:{r}" for r in stage8_reasons])

        system = BatterySystem(
            battery_id=battery_id,
            provenance="generated",
            parent_battery_id=base_system.battery_id,
            battery_type="insertion",
            working_ion=working_ion,
            framework_formula=cathode_formula,
            battery_formula=battery_formula,
            cathode_material=cathode_formula,
            anode_material=anode_formula,
            electrolyte=electrolyte_formula,
            separator_material=separator_material,
            additive_material=additive_material,
            average_voltage=avg_v,
            capacity_grav=cap_g,
            capacity_vol=cap_v,
            energy_grav=avg_v * cap_g,
            energy_vol=avg_v * cap_v,
            max_delta_volume=delta_v,
            stability_charge=float(target_property_vector.get("stability_charge", 0.0) or 0.0),
            stability_discharge=float(target_property_vector.get("stability_discharge", 0.0) or 0.0),
            uncertainty={
                "material_generation": {
                    "uncertainty_proxy": float(1.0 - compatibility_score),
                    "thermodynamic_proxy": float(compatibility.chemical_stability_score),
                    "compatibility_score": float(compatibility_score),
                    "cathode_redox_potential": float(cathode_v),
                    "anode_redox_potential": float(anode_v),
                    "full_cell_voltage": float(full_cell_v),
                    "np_ratio": float(np_ratio),
                    "limiting_electrode": limiting,
                    "electrolyte_window": {
                        "reduction_limit": float(electrolyte_eval.window.reduction_limit),
                        "oxidation_limit": float(electrolyte_eval.window.oxidation_limit),
                    },
                    "electrolyte_alkali_valid": bool(electrolyte_alkali.valid),
                    "electrolyte_window_valid": bool(electrolyte_eval.valid),
                    "sei_expected": bool(sei_expected),
                    "thermal_risk": bool(thermal_risk),
                    "component_compatibility_valid": bool(component_valid),
                    "component_compatibility_reasons": list(component_reasons),
                    "role_assignment_valid": bool(role_assignment_hard_valid),
                    "role_assignment_reasons": list(role_assignment_hard_reasons),
                    "role_assignment_soft_warnings": list(role_assignment_soft_warnings),
                    "battery_assembly_rule_valid": bool(assembly_rule_valid),
                    "battery_assembly_rule_reasons": list(assembly_rule_reasons),
                    "stage6_additive_interphase_valid": bool(stage6_valid),
                    "stage6_additive_interphase_reasons": list(stage6_reasons),
                    "stage7_physics_gates_valid": bool(stage7_valid),
                    "stage7_physics_gates_reasons": list(stage7_reasons),
                    "stage8_voltage_integrity_valid": bool(stage8_valid),
                    "stage8_voltage_integrity_reasons": list(stage8_reasons),
                },
                "material_level": {
                    "capacity_grav": cap_g,
                    "capacity_vol": cap_v,
                    "energy_grav": avg_v * cap_g,
                    "energy_vol": avg_v * cap_v,
                },
                "cell_level": {
                    "capacity_grav": cap_g,
                    "capacity_vol": cap_v,
                    "energy_grav": avg_v * cap_g,
                    "energy_vol": avg_v * cap_v,
                },
            },
            material_level={
                "capacity_grav": cap_g,
                "capacity_vol": cap_v,
                "energy_grav": avg_v * cap_g,
                "energy_vol": avg_v * cap_v,
            },
            cell_level={
                "capacity_grav": cap_g,
                "capacity_vol": cap_v,
                "energy_grav": avg_v * cap_g,
                "energy_vol": avg_v * cap_v,
            },
        )

        assembled = AssembledBatteryCandidate(
            candidate_id=battery_id,
            working_ion=working_ion,
            cathode_formula=cathode_formula,
            anode_formula=anode_formula,
            electrolyte_formula=electrolyte_formula,
            separator_material=separator_material,
            additive_material=additive_material,
            theoretical_voltage_window=(v_low, v_high),
            cathode_redox_potential=float(cathode_v),
            anode_redox_potential=float(anode_v),
            full_cell_voltage=float(full_cell_v),
            np_ratio=float(np_ratio),
            limiting_electrode=limiting,
            theoretical_capacity_cathode=float(cap_cath_theory),
            theoretical_capacity_anode=float(cap_an_theory),
            uncertainty=dict(system.uncertainty or {}),
            provenance={
                "stage": "full_cell_assembler",
                "source_mode": {
                    "cathode": cathode_source_mode,
                    "anode": anode_source_mode,
                    "electrolyte": compatibility.electrolyte.source_mode,
                },
                "role_assignment_engine": {
                    "input_materials": [
                        {
                            "label": "material_A",
                            "input_role_label": "compatibility.cathode",
                            "formula": material_a_formula,
                            "average_redox_potential_v": float(material_a_potential),
                        },
                        {
                            "label": "material_B",
                            "input_role_label": "compatibility.anode",
                            "formula": material_b_formula,
                            "average_redox_potential_v": float(material_b_potential),
                        },
                    ],
                    "sort_order": "ascending",
                    "assigned_roles": {
                        "anode_formula": anode_formula,
                        "cathode_formula": cathode_formula,
                        "swapped_from_input_labels": bool(role_swapped),
                    },
                    "validation": dict(role_assignment_diag),
                    "override_allowed": bool(self.ROLE_ASSIGNMENT_ENGINE.get("override_allowed", False)),
                },
                "compatibility": {
                    "voltage_window_overlap_score": compatibility.voltage_window_overlap_score,
                    "chemical_stability_score": compatibility.chemical_stability_score,
                    "mechanical_strain_risk": compatibility.mechanical_strain_risk,
                    "interface_risk_reason_codes": list(compatibility.interface_risk_reason_codes),
                },
                "chemistry_checks": {
                    "cathode": cath_diag,
                    "anode": an_diag,
                    "full_cell_voltage": float(full_cell_v),
                    "np_ratio": float(np_ratio),
                    "limiting_electrode": limiting,
                    "volume_expansion_threshold": 0.20,
                    "minimum_full_cell_voltage": float(
                        self.BATTERY_ASSEMBLY_RULES["voltage_constraints"]["minimum_cell_voltage_v"]
                    ),
                    "np_ratio_target_range": [1.05, 1.2],
                    "electrolyte_window": {
                        "reduction_limit": float(electrolyte_eval.window.reduction_limit),
                        "oxidation_limit": float(electrolyte_eval.window.oxidation_limit),
                        "source": electrolyte_eval.window.source,
                        "reduction_ok": bool(electrolyte_eval.reduction_ok),
                        "oxidation_ok": bool(electrolyte_eval.oxidation_ok),
                        "valid": bool(electrolyte_eval.valid),
                    },
                    "electrolyte_alkali": {
                        "valid": bool(electrolyte_alkali.valid),
                        "alkali_elements_present": list(electrolyte_alkali.alkali_elements_present),
                        "reasons": list(electrolyte_alkali.reasons),
                    },
                    "interface_flags": {
                        "sei_expected": bool(sei_expected),
                        "thermal_risk": bool(thermal_risk),
                    },
                    "role_assignment_engine": dict(role_assignment_diag),
                    "component_compatibility": dict(component_diag),
                    "battery_assembly_rules": dict(assembly_rule_diag),
                    "stage6_additive_interphase": dict(stage6_diag),
                    "stage7_physics_gates": dict(stage7_diag),
                    "stage8_voltage_integrity": dict(stage8_diag),
                },
                "stack_defaults": dict(stack_defaults),
                "component_selection": {
                    "source_mode": str(selection.get("source_mode") or "stack_defaults"),
                    "selected_electrolyte": str(electrolyte_formula),
                    "selected_separator": str(separator_material),
                    "selected_additive": str(additive_material),
                },
            },
            electrolyte_window={
                "reduction_limit": float(electrolyte_eval.window.reduction_limit),
                "oxidation_limit": float(electrolyte_eval.window.oxidation_limit),
            },
            sei_expected=bool(sei_expected),
            thermal_risk=bool(thermal_risk),
            valid=bool(
                compatibility.hard_valid
                and compatibility.cathode.valid
                and compatibility.anode.valid
                and compatibility.electrolyte.valid
                and cath_valid
                and an_valid
                and role_assignment_hard_valid
                and component_valid
                and assembly_rule_valid
                and stage6_valid
                and stage7_valid
                and stage8_valid
                and len(validation_reasons) == 0
            ),
            valid_reasons=list(compatibility.interface_risk_reason_codes) + list(validation_reasons),
        )

        return assembled, system
