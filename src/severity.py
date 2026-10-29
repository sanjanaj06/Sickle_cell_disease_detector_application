"""
severity.py - rule-based severity logic: IBSI, estimated Hb range, VORS.

Pure functions (no I/O, no image code) so they are trivial to unit-test in Phase 8.
Every weight, threshold and lookup table comes from config.py.

    IBSI = 0.7 * SCI + 0.3 * CD                       -> Mild / Moderate / Severe
    Hb range  = table lookup by (severity, age group)
    VORS = 0.40*SpO2 score + 0.30*IBSI + 0.20*Hb score + 0.10*Pain-crisis score

NOTE: these are heuristic indices, not diagnostic measurements. The Hb range is a
lookup from the severity band, not a measured value.
"""
from __future__ import annotations

from typing import Tuple

import config


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return min(hi, max(lo, value))


def get_age_group(age: float) -> str:
    """child 0-12 | adolescent 13-18 | adult 19-40 | older_adult >40."""
    if age is None or age < 0 or age > 120:
        raise ValueError(f"Age must be between 0 and 120, got {age!r}")
    if age < 13:
        return "child"
    if age < 19:
        return "adolescent"
    if age <= 40:
        return "adult"
    return "older_adult"


def normalise_gender(gender: str) -> str:
    g = (gender or "").strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    raise ValueError(
        f"Gender must be 'male' or 'female' (the pain-crisis table only defines these), got {gender!r}"
    )


# ---------------------------------------------------------------------
# IBSI and severity
# ---------------------------------------------------------------------
def compute_ibsi(sci: float, cd: float) -> float:
    """Image Based Severity Index = w1*SCI + w2*CD (both inputs on a 0-100 scale)."""
    if not (0 <= sci <= 100 and 0 <= cd <= 100):
        raise ValueError(f"SCI and CD must be in [0, 100], got SCI={sci}, CD={cd}")
    return config.IBSI_W_SCI * sci + config.IBSI_W_CD * cd


def classify_severity(ibsi: float) -> str:
    ibsi = clamp(ibsi)
    if ibsi >= config.SEVERITY_SEVERE_MIN:
        return "Severe"
    if ibsi >= config.SEVERITY_MODERATE_MIN:
        return "Moderate"
    return "Mild"


# ---------------------------------------------------------------------
# Estimated Hb
# ---------------------------------------------------------------------
def estimate_hb_range(severity: str, age: float) -> Tuple[float, float]:
    """(low, high) in g/dL from the severity x age-group table."""
    return config.HB_RANGES[severity][get_age_group(age)]


def hb_score(hb_low: float, hb_high: float, age: float) -> float:
    """
    Normalised Hb risk score (higher = lower Hb = higher risk):
        mid   = (low + high) / 2
        score = (Hb_max - mid) / (Hb_max - Hb_min) * 100, clamped to [0, 100]
    Hb_min / Hb_max are the lowest and highest Hb values in this age group's column
    of the table (across all three severity bands).
    """
    group = get_age_group(age)
    all_ranges = [config.HB_RANGES[sev][group] for sev in config.HB_RANGES]
    hb_min = min(r[0] for r in all_ranges)
    hb_max = max(r[1] for r in all_ranges)
    mid = (hb_low + hb_high) / 2.0
    return clamp((hb_max - mid) / (hb_max - hb_min) * 100.0)


# ---------------------------------------------------------------------
# Other VORS components
# ---------------------------------------------------------------------
def pain_crisis_score(crises_per_year: float, age: float, gender: str) -> float:
    if crises_per_year < 0:
        raise ValueError("Pain crises per year cannot be negative")
    max_base = config.PAIN_CRISIS_MAX_BASE[get_age_group(age)][normalise_gender(gender)]
    return clamp(crises_per_year / max_base * 100.0)


def spo2_score(spo2: float) -> float:
    if not (0 < spo2 <= 100):
        raise ValueError(f"SpO2 must be in (0, 100], got {spo2}")
    return clamp((100.0 - spo2) / (100.0 - config.SPO2_FLOOR) * 100.0)


# ---------------------------------------------------------------------
# VORS
# ---------------------------------------------------------------------
def classify_vors(vors: float) -> str:
    vors = clamp(vors)
    if vors >= config.VORS_HIGH_MIN:
        return "High"
    if vors >= config.VORS_MODERATE_MIN:
        return "Moderate"
    return "Low"


def compute_vors(ibsi: float, hb_low: float, hb_high: float,
                 pain_crises_per_year: float, spo2: float,
                 age: float, gender: str) -> dict:
    w = config.VORS_WEIGHTS
    components = {
        "spo2": spo2_score(spo2),
        "ibsi": clamp(ibsi),
        "hb": hb_score(hb_low, hb_high, age),
        "pain": pain_crisis_score(pain_crises_per_year, age, gender),
    }
    vors = sum(w[k] * components[k] for k in w)
    return {
        "vors": round(vors, 2),
        "risk": classify_vors(vors),
        "component_scores": {k: round(v, 2) for k, v in components.items()},
    }


# ---------------------------------------------------------------------
# One-call pipeline for inference.py / Flask
# ---------------------------------------------------------------------
def assess(sci: float, cd: float, age: float, gender: str,
           pain_crises_per_year: float, spo2: float) -> dict:
    """SCI + CD + clinical inputs -> IBSI, severity, Hb range, VORS, risk class."""
    ibsi = compute_ibsi(sci, cd)
    severity = classify_severity(ibsi)
    hb_low, hb_high = estimate_hb_range(severity, age)
    vors = compute_vors(ibsi, hb_low, hb_high, pain_crises_per_year, spo2, age, gender)
    return {
        "age_group": get_age_group(age),
        "ibsi": round(ibsi, 2),
        "severity": severity,
        "estimated_hb_range_g_dl": [hb_low, hb_high],
        "vors": vors,   # {"vors": float, "risk": str, "component_scores": {...}}
    }
