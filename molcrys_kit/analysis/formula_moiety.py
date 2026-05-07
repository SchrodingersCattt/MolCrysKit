"""
Utilities for parsing CIF ``_chemical_formula_moiety`` values.

The parser intentionally keeps moiety fragments unscaled: ``0.5(C3 H8 O1)``
represents a fragment whose stored element counts are still ``C3 H8 O1`` and
whose multiplier is ``0.5``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple
import warnings


@dataclass(frozen=True)
class Fragment:
    """One fragment from a CIF ``_chemical_formula_moiety`` value."""

    multiplier: float
    composition: Dict[str, int]
    charge: int
    raw: str


_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)([0-9]*)")
_MULTIPLIER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*\((.*)\)$")
_CHARGE_RE = re.compile(r"^(?:(\d+))?([+-])$")


def parse_moiety_string(value: Optional[str]) -> Optional[List[Fragment]]:
    """
    Parse a CIF ``_chemical_formula_moiety`` value.

    Returns ``None`` for absent, unknown, or malformed values so callers can
    fall back to the geometric heuristic path.
    """
    if value is None:
        return None

    text = _normalise_moiety_text(str(value))
    if not text or text == "?":
        return None

    try:
        fragments = [_parse_fragment(token) for token in _split_top_level(text)]
    except ValueError as exc:
        warnings.warn(
            f"Could not parse _chemical_formula_moiety={value!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return fragments or None


def heavy_signature(composition: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Return a hashable composition signature with hydrogen removed."""
    return tuple(
        sorted((element, count) for element, count in composition.items() if element != "H")
    )


def match_molecule_to_fragment(
    symbols: List[str], fragments: List[Fragment]
) -> Optional[Fragment]:
    """
    Match a molecule to exactly one moiety fragment by heavy-atom composition.

    Ambiguous or missing matches return ``None`` with a warning, leaving the
    caller free to use the existing heuristic result.
    """
    target = heavy_signature(dict(Counter(s for s in symbols if s != "H")))
    matches = [
        fragment
        for fragment in fragments
        if heavy_signature(fragment.composition) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        warnings.warn(
            "Ambiguous _chemical_formula_moiety fragment match for heavy "
            f"signature {target}: {[m.raw for m in matches]}",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            "No _chemical_formula_moiety fragment matches heavy "
            f"signature {target}",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


def _normalise_moiety_text(value: str) -> str:
    text = value.strip()

    # pymatgen usually strips CIF quoting, but tests and callers may pass the
    # literal field including CIF quotes or semicolon text-block delimiters.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    if text.startswith(";"):
        text = text[1:]
    if text.endswith(";"):
        text = text[:-1]

    return " ".join(text.split())


def _split_top_level(text: str) -> List[str]:
    tokens = []
    current = []
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced closing parenthesis")

        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(char)

    if depth != 0:
        raise ValueError("unbalanced opening parenthesis")

    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _parse_fragment(token: str) -> Fragment:
    raw = token.strip()
    multiplier = 1.0
    body = raw

    multiplier_match = _MULTIPLIER_RE.match(body)
    if multiplier_match:
        multiplier = float(multiplier_match.group(1))
        body = multiplier_match.group(2).strip()

    words = body.split()
    charge = 0
    if words and _CHARGE_RE.match(words[-1]):
        charge = _parse_charge(words.pop())
        body = " ".join(words)

    composition = _parse_composition(body)
    if not composition:
        raise ValueError(f"empty fragment {raw!r}")

    return Fragment(
        multiplier=multiplier,
        composition=composition,
        charge=charge,
        raw=raw,
    )


def _parse_charge(token: str) -> int:
    match = _CHARGE_RE.match(token)
    if not match:
        return 0

    magnitude = int(match.group(1) or "1")
    return magnitude if match.group(2) == "+" else -magnitude


def _parse_composition(text: str) -> Dict[str, int]:
    compact = text.replace(" ", "")
    if not compact:
        return {}

    composition: Dict[str, int] = {}
    consumed = 0
    for match in _ELEMENT_RE.finditer(compact):
        if match.start() != consumed:
            raise ValueError(f"unexpected token near {compact[consumed:]!r}")
        element, count_text = match.groups()
        composition[element] = composition.get(element, 0) + int(count_text or "1")
        consumed = match.end()

    if consumed != len(compact):
        raise ValueError(f"unexpected token near {compact[consumed:]!r}")

    return composition
