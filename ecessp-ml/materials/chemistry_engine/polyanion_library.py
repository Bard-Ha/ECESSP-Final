from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class PolyanionReport:
    valid: bool
    recognized_units: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class PolyanionLibrary:
    """
    Whitelist of known stable polyanion units.
    """

    ALLOWED_UNITS = ("PO4", "P2O7", "SO4", "CO3", "BO3", "SiO4", "VO4")
    _CENTERS_REQUIRING_WHITELIST = {"P", "S", "B", "Si", "C"}

    _UNIT_PATTERNS = {
        "PO4": re.compile(r"P(?:\d+)?O4\b"),
        "P2O7": re.compile(r"P2O7\b"),
        "SO4": re.compile(r"S(?:\d+)?O4\b"),
        "CO3": re.compile(r"C(?:\d+)?O3\b"),
        "BO3": re.compile(r"B(?:\d+)?O3\b"),
        "SiO4": re.compile(r"Si(?:\d+)?O4\b"),
        "VO4": re.compile(r"V(?:\d+)?O4\b"),
    }

    @staticmethod
    def _clean_formula(formula: str) -> str:
        return str(formula or "").replace(" ", "")

    @classmethod
    def validate_formula(cls, formula: str) -> PolyanionReport:
        text = cls._clean_formula(formula)
        if not text:
            return PolyanionReport(valid=False, reasons=["missing_formula"])

        recognized: list[str] = []
        for unit, pattern in cls._UNIT_PATTERNS.items():
            if pattern.search(text):
                recognized.append(unit)

        # Only enforce strict unit whitelist when likely polyanion centers are present.
        has_polyanion_center = any(center in text for center in cls._CENTERS_REQUIRING_WHITELIST)
        has_oxygen = "O" in text
        if has_polyanion_center and has_oxygen and not recognized:
            return PolyanionReport(
                valid=False,
                recognized_units=[],
                reasons=["unknown_polyanion_cluster"],
            )

        return PolyanionReport(valid=True, recognized_units=recognized, reasons=[])

