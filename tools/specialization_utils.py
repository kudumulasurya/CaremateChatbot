"""Utilities to canonicalize medical specializations.

Keep a single source of truth for mapping common typos/variants to canonical names.
"""

CANONICAL_MAP = {
    # lowercase source -> canonical Titlecase
    "pediadrist": "Pediatrician",
    "pediatrist": "Pediatrician",
    "pediatrician": "Pediatrician",
    "cardiologist": "Cardiologist",
    "cardiology": "Cardiologist",
    "ent": "ENT",
    "ear nose throat": "ENT",
    "neurology": "Neurology",
    "neuro": "Neurology",
    "dermatologist": "Dermatologist",
    "derm": "Dermatologist",
    "orthopedist": "Orthopedist",
    "orthopedic": "Orthopedist",
}


def canonicalize_spec(spec: str) -> str:
    """Return the canonical specialization for a given raw spec string.

    If spec is None or empty, returns empty string. If no mapping exists, returns
    the original stripped string.
    """
    if not spec:
        return ""
    s = spec.strip().lower()
    return CANONICAL_MAP.get(s, spec.strip())


def normalize_profile(profile: dict) -> dict:
    """Return a copy of profile where 'specialization' is canonicalized."""
    if not isinstance(profile, dict):
        return profile
    spec = profile.get("specialization")
    canon = canonicalize_spec(spec)
    new = dict(profile)
    if canon:
        new["specialization"] = canon
    return new
