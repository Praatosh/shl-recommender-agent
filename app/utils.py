"""
SHL Assessment Recommender - Shared Utilities
==============================================
Centralizes constants and helper functions used across modules.

Design choice: Single source of truth for the type_map avoids
6+ copies scattered through engine.py, prompts.py, schemas.py.
"""

from typing import Dict, List


# ============================================================
# Assessment Type Code Mapping
# ============================================================
# Maps SHL assessment key categories to single-letter codes
# used in the API response's test_type field.
TYPE_MAP: Dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Assessment Exercises": "E",
    "Development & 360": "D",
}


def derive_test_type_code(keys: List[str]) -> str:
    """
    Derive a comma-separated test type code string from assessment keys.

    Args:
        keys: List of SHL assessment category strings (e.g., ["Knowledge & Skills", "Simulations"])

    Returns:
        Comma-separated type codes (e.g., "K,S"). Defaults to "K" if no keys match.

    Why centralize?
        This logic was duplicated 6+ times across the codebase. A single function
        ensures consistent behavior and makes future mapping changes trivial.
    """
    codes: List[str] = []
    for key in keys:
        code = TYPE_MAP.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"
