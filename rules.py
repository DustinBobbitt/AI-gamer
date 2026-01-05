from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Set, Type, TypeVar, Union


class Rule(Protocol):
    """Base protocol for all rules."""

    name: str

    def apply(self, numbers: Set[int]) -> Set[int]:
        """Return the subset of `numbers` that should be removed."""

    def to_dict(self) -> Dict[str, Any]:
        ...


RuleT = TypeVar("RuleT", bound="SerializableRule")


class SerializableRule:
    """Base class providing JSON-style (de)serialisation helpers."""

    rule_type: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls: Type[RuleT], data: Dict[str, Any]) -> RuleT:
        raise NotImplementedError


@dataclass(frozen=True)
class DivisibilityRule(SerializableRule):
    """Remove multiples of d (optionally excluding n == d)."""

    d: int
    keep_self: bool = True

    rule_type: str = "divisibility"

    @property
    def name(self) -> str:
        return f"Divisible by {self.d}" + (" (keep self)" if self.keep_self else "")

    def apply(self, numbers: Set[int]) -> Set[int]:
        to_remove: Set[int] = set()
        for n in numbers:
            if n % self.d == 0:
                if self.keep_self and n == self.d:
                    continue
                to_remove.add(n)
        return to_remove

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.rule_type, "d": self.d, "keep_self": self.keep_self}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DivisibilityRule":
        return cls(d=int(data["d"]), keep_self=bool(data.get("keep_self", True)))


@dataclass(frozen=True)
class ModClassRule(SerializableRule):
    """Remove numbers whose residue class modulo m is in banned_residues."""

    m: int
    banned_residues: Set[int]

    rule_type: str = "mod_class"

    @property
    def name(self) -> str:
        residues = ",".join(str(r) for r in sorted(self.banned_residues))
        return f"n % {self.m} in {{{residues}}}"

    def apply(self, numbers: Set[int]) -> Set[int]:
        to_remove: Set[int] = set()
        banned = self.banned_residues
        m = self.m
        for n in numbers:
            if n % m in banned:
                to_remove.add(n)
        return to_remove

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.rule_type,
            "m": self.m,
            "banned_residues": sorted(self.banned_residues),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModClassRule":
        m = int(data["m"])
        residues = {int(r) for r in data["banned_residues"]}
        return cls(m=m, banned_residues=residues)


@dataclass(frozen=True)
class KeepResiduesRule(SerializableRule):
    """Keep only numbers whose residue class modulo m is in allowed_residues.
    
    This is the inverse of ModClassRule - removes numbers NOT in the allowed set.
    Used as a first-cut filter to reduce the search space.
    """

    m: int
    allowed_residues: Set[int]

    rule_type: str = "keep_residues"

    @property
    def name(self) -> str:
        residues = ",".join(str(r) for r in sorted(self.allowed_residues))
        return f"Keep residues {{{residues}}} mod {self.m}"

    def apply(self, numbers: Set[int]) -> Set[int]:
        to_remove: Set[int] = set()
        allowed = self.allowed_residues
        m = self.m
        for n in numbers:
            if n % m not in allowed:
                to_remove.add(n)
        return to_remove

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.rule_type,
            "m": self.m,
            "allowed_residues": sorted(self.allowed_residues),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeepResiduesRule":
        m = int(data["m"])
        residues = {int(r) for r in data["allowed_residues"]}
        return cls(m=m, allowed_residues=residues)


RuleLike = Union[DivisibilityRule, ModClassRule, KeepResiduesRule]


def rule_to_dict(rule: RuleLike) -> Dict[str, Any]:
    return rule.to_dict()


def rule_from_dict(data: Dict[str, Any]) -> RuleLike:
    rule_type = data.get("type")
    if rule_type == "divisibility":
        return DivisibilityRule.from_dict(data)
    if rule_type == "mod_class":
        return ModClassRule.from_dict(data)
    if rule_type == "keep_residues":
        return KeepResiduesRule.from_dict(data)
    raise ValueError(f"Unknown rule type: {rule_type}")


def unique_rule_key(rule: RuleLike) -> str:
    """Stable string key identifying a rule for de-duplication."""
    d = rule.to_dict()
    items: List[str] = []
    for k in sorted(d.keys()):
        items.append(f"{k}={d[k]}")
    return "|".join(items)

