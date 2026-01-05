from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.primes import primes_up_to


RuleDict = Dict[str, object]


def _normalize_residues(values: List[int]) -> List[int]:
    """Return sorted unique residues."""
    return sorted({int(v) for v in values})


def canonical_rule_key(rule: RuleDict) -> str:
    """Return a stable string key for a rule dict."""
    rtype = rule.get("type")

    if rtype == "divisibility":
        d = int(rule.get("d", 0))
        keep_self = bool(rule.get("keep_self", True))
        return f"div:{d}:keep_self={keep_self}"

    if rtype == "keep_residues":
        m = int(rule.get("m", 0))
        allowed_raw = rule.get("allowed_residues", []) or []
        allowed = _normalize_residues([int(x) for x in allowed_raw])  # type: ignore[list-item]
        allowed_str = ",".join(str(a) for a in allowed)
        return f"keepres:m={m}:allowed={allowed_str}"

    if rtype == "mod_class":
        m = int(rule.get("m", 0))
        banned_raw = rule.get("banned_residues", []) or []
        banned = _normalize_residues([int(x) for x in banned_raw])  # type: ignore[list-item]
        banned_str = ",".join(str(b) for b in banned)
        return f"modclass:m={m}:banned={banned_str}"

    # Fallback: include sorted items to avoid accidental collisions.
    items = sorted(rule.items())
    return "other:" + ";".join(f"{k}={v}" for k, v in items)


def _canonicalize_rule(rule: RuleDict) -> RuleDict:
    """Return a shallow canonicalized copy of a rule dict."""
    rtype = rule.get("type")
    canon = dict(rule)
    if rtype == "keep_residues":
        raw = canon.get("allowed_residues", []) or []
        canon["allowed_residues"] = _normalize_residues([int(x) for x in raw])  # type: ignore[list-item]
    elif rtype == "mod_class":
        raw = canon.get("banned_residues", []) or []
        canon["banned_residues"] = _normalize_residues([int(x) for x in raw])  # type: ignore[list-item]
    return canon


@dataclass
class CanonicalProgram:
    rules_canonical: List[RuleDict]
    signature_ordered: str
    signature_unordered: str
    signature_stagewise: str


def canonicalize_program(rules: List[RuleDict]) -> CanonicalProgram:
    """Canonicalize a program and compute signatures.

    - rules_canonical: rules with normalized residue lists.
    - signature_ordered: concatenation of canonical_rule_key in given order.
    - signature_unordered: multiset-style signature ignoring order.
    - signature_stagewise: separates filters vs sieves to capture structure.
    """
    rules_canonical: List[RuleDict] = [_canonicalize_rule(r) for r in rules]

    # Ordered signature.
    keys_ordered: List[str] = [canonical_rule_key(r) for r in rules_canonical]
    signature_ordered = " | ".join(keys_ordered)

    # Unordered / multiset signature.
    counts: Dict[str, int] = {}
    for k in keys_ordered:
        counts[k] = counts.get(k, 0) + 1
    parts_unordered: List[str] = []
    for k in sorted(counts.keys()):
        c = counts[k]
        if c == 1:
            parts_unordered.append(k)
        else:
            parts_unordered.append(f"{k}*{c}")
    signature_unordered = " ; ".join(parts_unordered)

    # Stagewise signature: filters vs sieves.
    filter_rules: List[Tuple[str, RuleDict]] = []
    sieve_rules: List[Tuple[int, str]] = []
    for r in rules_canonical:
        rtype = r.get("type")
        key = canonical_rule_key(r)
        if rtype in {"keep_residues", "mod_class"}:
            filter_rules.append((key, r))
        elif rtype == "divisibility":
            d = int(r.get("d", 0))
            sieve_rules.append((d, key))

    # Filters: unordered signature on filter keys.
    filter_counts: Dict[str, int] = {}
    for key, _ in filter_rules:
        filter_counts[key] = filter_counts.get(key, 0) + 1
    filter_parts: List[str] = []
    for k in sorted(filter_counts.keys()):
        c = filter_counts[k]
        if c == 1:
            filter_parts.append(k)
        else:
            filter_parts.append(f"{k}*{c}")
    filter_sig = " ; ".join(filter_parts)

    # Sieves: sort by divisor d, then ordered signature over keys.
    sieve_rules_sorted = sorted(sieve_rules, key=lambda pair: pair[0])
    sieve_keys_sorted = [k for _, k in sieve_rules_sorted]
    sieve_sig = " | ".join(sieve_keys_sorted)

    signature_stagewise = f"F:[{filter_sig}] S:[{sieve_sig}]"

    return CanonicalProgram(
        rules_canonical=rules_canonical,
        signature_ordered=signature_ordered,
        signature_unordered=signature_unordered,
        signature_stagewise=signature_stagewise,
    )


def detect_sieve_prefix(program_rules: List[RuleDict]) -> Optional[List[int]]:
    """Detect if program is a clean divisibility sieve and return prime divisors used.
    
    Returns:
        List of prime divisors in order if program is all divisibility rules with keep_self=True,
        None otherwise.
    """
    if not program_rules:
        return None
    
    divisors: List[int] = []
    for rule in program_rules:
        if rule.get("type") != "divisibility":
            return None
        if not rule.get("keep_self", True):
            return None
        d = int(rule.get("d", 0))
        if d < 2:
            return None
        divisors.append(d)
    
    return divisors


def expand_sieve_program(prefix_ds: List[int], max_n: int) -> List[RuleDict]:
    """Expand a sieve prefix to include all primes up to sqrt(max_n).
    
    Takes the discovered prefix (e.g. [3, 5, 7, 11, 13]) and continues with
    missing primes up to sqrt(max_n) to create a complete sieve for that range.
    
    Args:
        prefix_ds: Prime divisors discovered during training
        max_n: Maximum number in target range
    
    Returns:
        Full list of DivisibilityRule dicts for complete sieve
    """
    limit = int(max_n ** 0.5) + 1
    all_primes = [p for p in primes_up_to(limit) if p >= 3]
    
    # Start with prefix in order
    used = set(prefix_ds)
    full_ds = list(prefix_ds)
    
    # Add remaining primes not in prefix
    for p in all_primes:
        if p not in used:
            full_ds.append(p)
            used.add(p)
    
    # Convert to rule dicts
    return [
        {"type": "divisibility", "d": d, "keep_self": True}
        for d in full_ds
    ]


def expand_program_if_sieve(program_rules: List[RuleDict], max_n: int) -> Tuple[List[RuleDict], str]:
    """Expand program to full sieve if it's a sieve prefix, otherwise return as-is.
    
    Returns:
        (expanded_rules, replay_mode) where replay_mode is "sieve_expand" or "fixed"
    """
    prefix = detect_sieve_prefix(program_rules)
    if prefix is not None:
        expanded = expand_sieve_program(prefix, max_n)
        return expanded, "sieve_expand"
    return program_rules, "fixed"
