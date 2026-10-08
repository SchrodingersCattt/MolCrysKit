"""Structured representation of a generated substitutive name.

The chemistry naming rules produce strings because that is the public API of
MolCrysKit.  Keeping the parts of a name in a small value object gives the
rules and the reverse parser a common, lossless intermediate representation.
This module intentionally does not attempt to be a general IUPAC grammar: the
structure is permissive enough to retain every canonical name emitted by the
package, while the graph parser remains the authority for supported names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


class NamingParseError(ValueError):
    """Raised when a structured name cannot be parsed."""


@dataclass(frozen=True)
class NamePrefix:
    """A detachable prefix and the locants to which it is attached."""

    locants: tuple[int, ...]
    name: str

    def __post_init__(self) -> None:
        if not self.name or any(locant < 1 for locant in self.locants):
            raise NamingParseError("a prefix needs a name and positive locants")


# This spelling is useful to callers that describe the field as a locanted
# prefix.  It is an alias rather than a second class so equality remains
# straightforward.
LocantedPrefix = NamePrefix


_STEREO_PREFIX = re.compile(r"^(?P<descriptors>(?:\([^()]+\)-)+)(?P<body>.+)$")
_STEREO_ITEM = re.compile(r"\(([^()]+)\)-")
_PREFIX = re.compile(
    r"(?P<locants>\d+(?:,\d+)*)-(?:(?P<multiplier>di|tri|\d+-)?"
    r"(?P<name>fluoro|chloro|bromo|iodo|methyl|hydroxy|oxo|amino|nitro))"
)
_SUFFIXES = (
    "carboxylic acid",
    "oic acid",
    "oyl chloride",
    "oyl bromide",
    "oyl fluoride",
    "oyl iodide",
    "amide",
    "nitrile",
    "al",
    "one",
    "ol",
    "benzene",
    "phenol",
    "amine",
    "ene",
    "yne",
    "ane",
)


@dataclass(frozen=True)
class SystematicName:
    """A serializable name assembled from substitutive-name components.

    ``serialized`` is retained internally when parsing.  It makes parsing
    and serialization lossless for names whose grammar is deliberately left
    to a specialized rule (for example ``N-(4-hydroxyphenyl)acetamide``),
    while the public fields still expose the common parent/suffix/prefix
    pieces.  New names can be built without this field and are serialized by
    joining those pieces.
    """

    parent: str = ""
    suffix: str | None = None
    prefixes: tuple[NamePrefix, ...] = ()
    stereochemistry: tuple[str, ...] = ()
    charge: str | None = None
    components: tuple["SystematicName", ...] = ()
    _serialized: str | None = field(default=None, repr=False, compare=False)

    @property
    def main_suffix(self) -> str | None:
        """Alias used by code that calls the principal suffix ``main``."""

        return self.suffix

    @property
    def locanted_prefixes(self) -> tuple[NamePrefix, ...]:
        """Descriptive alias for :attr:`prefixes`."""

        return self.prefixes

    @property
    def stereo_descriptors(self) -> tuple[str, ...]:
        """Descriptive alias for :attr:`stereochemistry`."""

        return self.stereochemistry

    @property
    def charge_suffix(self) -> str | None:
        """Descriptive alias for :attr:`charge`."""

        return self.charge

    def serialize(self) -> str:
        """Return the canonical string represented by this value object."""

        if self._serialized is not None:
            return self._serialized
        if self.components:
            body = " · ".join(component.serialize() for component in self.components)
        else:
            prefix_text = "-".join(
                _serialize_prefix(prefix) for prefix in self.prefixes
            )
            body = f"{prefix_text + '-' if prefix_text else ''}{self.parent}"
            if self.suffix:
                body = _join_parent_suffix(body, self.suffix)
        if self.charge:
            body += self.charge
        if self.stereochemistry:
            body = "".join(f"({item})-" for item in self.stereochemistry) + body
        return body

    @classmethod
    def parse(cls, name: str) -> "SystematicName":
        """Parse a canonical generated name into structured components.

        The method normalizes surrounding whitespace and case, validates
        balanced grouping, and records the exact normalized representation.
        Detailed support checks remain in ``name_conversion`` where the graph
        builders can provide useful valence diagnostics.
        """

        if not isinstance(name, str):
            raise TypeError("name must be a string")
        normalized = " ".join(name.strip().lower().split())
        if not normalized:
            raise NamingParseError("IUPAC name must not be empty")
        # N-substitution locants are conventionally capitalized and this is
        # part of the established MolCrysKit golden spelling.  Other lexical
        # tokens are case-insensitive and remain normalized to lowercase.
        if normalized.startswith("n-("):
            normalized = "N-" + normalized[2:]
        if _unbalanced_parentheses(normalized):
            raise NamingParseError("unbalanced parentheses in IUPAC name")

        # Components are separated by the middle dot emitted by the existing
        # multicomponent naming routine.  Parse each side recursively while
        # preserving the complete string on the outer object.
        if " · " in normalized:
            components = tuple(cls.parse(part) for part in normalized.split(" · "))
            return cls(components=components, _serialized=normalized)

        stereochemistry: tuple[str, ...] = ()
        body = normalized
        stereo_match = _STEREO_PREFIX.match(body)
        if stereo_match:
            stereochemistry = tuple(
                match.group(1) for match in _STEREO_ITEM.finditer(stereo_match.group("descriptors"))
            )
            body = stereo_match.group("body")

        charge = None
        charge_match = re.search(r"(?:\s|^)([+-])$", body)
        if charge_match:
            charge = charge_match.group(1)
            body = body[: charge_match.start(1)].rstrip()

        prefixes, parent_body = _extract_prefixes(body)
        parent, suffix = _split_suffix(parent_body)
        return cls(
            parent=parent,
            suffix=suffix,
            prefixes=tuple(prefixes),
            stereochemistry=stereochemistry,
            charge=charge,
            _serialized=normalized,
        )

    @classmethod
    def from_name(cls, name: str) -> "SystematicName":
        """Backward-compatible constructor spelling for :meth:`parse`."""

        return cls.parse(name)


def _serialize_prefix(prefix: NamePrefix) -> str:
    locants = ",".join(str(locant) for locant in prefix.locants)
    count = len(prefix.locants)
    multiplier = {1: "", 2: "di", 3: "tri"}.get(count, f"{count}-")
    return f"{locants}-{multiplier}{prefix.name}"


def _extract_prefixes(body: str) -> tuple[list[NamePrefix], str]:
    prefixes: list[NamePrefix] = []
    position = 0
    while position < len(body):
        match = _PREFIX.match(body, position)
        if match is None:
            break
        locants = tuple(int(item) for item in match.group("locants").split(","))
        multiplier = match.group("multiplier")
        expected = {None: 1, "di": 2, "tri": 3}.get(multiplier)
        if expected is None:
            try:
                expected = int(multiplier[:-1])
            except (TypeError, ValueError) as exc:
                raise NamingParseError("invalid prefix multiplier") from exc
        if len(locants) != expected:
            raise NamingParseError("prefix multiplier and locants disagree")
        prefixes.append(NamePrefix(locants, match.group("name")))
        position = match.end()
        if position < len(body):
            if body[position] != "-":
                break
            position += 1
    return prefixes, body[position:]


def _split_suffix(body: str) -> tuple[str, str | None]:
    for suffix in _SUFFIXES:
        if suffix == "benzene" and body == "benzene":
            return "benzene", None
        if suffix == "phenol" and body == "phenol":
            return "phenol", None
        if body.endswith(suffix) and len(body) > len(suffix):
            return body[: -len(suffix)], suffix
    return body, None


def _join_parent_suffix(parent: str, suffix: str) -> str:
    """Join lexical parent and suffix without introducing an extra ``e``."""

    if parent.endswith("-"):
        return parent + suffix
    if suffix[:1].isdigit() and parent.endswith("e"):
        return parent[:-1] + "-" + suffix
    if suffix in {
        "ol",
        "al",
        "one",
        "amine",
        "nitrile",
        "oic acid",
        "oyl chloride",
        "oyl bromide",
        "oyl fluoride",
        "oyl iodide",
        "amide",
    } and parent.endswith("e"):
        return parent[:-1] + suffix
    if suffix == "ene" and parent.endswith("e"):
        return parent + suffix[1:]
    return parent + suffix


def _unbalanced_parentheses(text: str) -> bool:
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return True
    return depth != 0


__all__ = ["NamePrefix", "LocantedPrefix", "NamingParseError", "SystematicName"]
