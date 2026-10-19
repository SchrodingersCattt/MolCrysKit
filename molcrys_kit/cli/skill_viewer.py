"""Read-only access to the Agent Skills bundled with MolCrysKit.

This module intentionally uses only the standard library.  The CLI can import
it without importing a renderer, web frontend, TUI, or optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import sys
from importlib import resources
from typing import Any
from urllib.parse import unquote


class SkillError(ValueError):
    """A safe, user-facing error while resolving a bundled skill resource."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    root: Any


_LINK_RE = re.compile(
    r"\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))"
)
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_BUNDLED_SKILL_NAMES = frozenset({
    "analyze-molecular-crystal",
    "operate-molecular-crystal",
    "query-csd-structures",
})


def _skills_root():
    try:
        return resources.files("skills")
    except (ImportError, TypeError) as exc:  # pragma: no cover - broken install
        raise SkillError("bundled skills are not installed") from exc


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return {}
    values: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _description(entry: Any) -> str:
    try:
        return _frontmatter(entry.read_text(encoding="utf-8")).get("description", "")
    except (OSError, UnicodeError) as exc:
        raise SkillError("unable to read bundled skill entrypoint") from exc


def list_skills() -> list[Skill]:
    """Return bundled skills in stable name order."""
    children = sorted(_skills_root().iterdir(), key=lambda item: item.name)
    restrict_to_this_distribution = any(
        child.name in _BUNDLED_SKILL_NAMES for child in children
    )
    result: list[Skill] = []
    for child in children:
        # ``skills`` is a deliberately small top-level data package so the
        # source tree stays canonical.  Filter to this distribution's names
        # when MatterVis and MolCrysKit are installed in the same environment.
        if restrict_to_this_distribution and child.name not in _BUNDLED_SKILL_NAMES:
            continue
        if not child.is_dir():
            continue
        entry = child.joinpath("SKILL.md")
        if entry.is_file():
            result.append(Skill(child.name, _description(entry), child))
    return result


def get_skill(name: str) -> Skill:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise SkillError(f"invalid skill name: {name!r}")
    for skill in list_skills():
        if skill.name == name:
            return skill
    raise SkillError(f"unknown skill: {name!r}")


def _normalise_relative(raw: str, *, label: str = "resource") -> str:
    value = unquote(raw.strip())
    if not value or value in {".", "./"}:
        raise SkillError(f"{label} path is empty")
    if "\\" in value or value.startswith("/") or PurePosixPath(value).is_absolute():
        raise SkillError(f"{label} path escapes the selected skill: {raw!r}")
    parts: list[str] = []
    for part in value.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise SkillError(f"{label} path escapes the selected skill: {raw!r}")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise SkillError(f"{label} path is empty")
    return "/".join(parts)


def _resource(skill: Skill, relative: str):
    relative = _normalise_relative(relative)
    candidate = skill.root.joinpath(*relative.split("/"))
    if isinstance(skill.root, Path) or isinstance(candidate, Path):
        try:
            root_path = Path(skill.root).resolve(strict=True)
            candidate_path = Path(candidate).resolve(strict=False)
            candidate_path.relative_to(root_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SkillError(f"resource path escapes the selected skill: {relative!r}") from exc
    if not candidate.is_file():
        raise SkillError(f"resource not found in skill {skill.name!r}: {relative}")
    return candidate, relative


def _read(skill: Skill, relative: str) -> str:
    resource, _ = _resource(skill, relative)
    try:
        return resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SkillError(f"unable to read resource {relative!r}") from exc


def _local_target(source: str, raw_target: str) -> tuple[str, str | None] | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    if not target or target.startswith("#") or target.startswith("//") or _SCHEME_RE.match(target):
        return None
    path, separator, fragment = target.partition("#")
    source_parent = posixpath.dirname(source)
    joined = posixpath.join(source_parent, unquote(path))
    return _normalise_relative(joined, label="linked resource"), fragment if separator else None


def _links(source: str, text: str) -> list[tuple[str, str, str | None]]:
    result: list[tuple[str, str, str | None]] = []
    for match in _LINK_RE.finditer(text):
        raw = match.group(1) or match.group(2) or ""
        target = _local_target(source, raw)
        if target is not None:
            result.append((raw, target[0], target[1]))
    return result


def _kind(relative: str) -> str:
    return "markdown" if relative.lower().endswith(".md") else "asset"


def tree(skill: Skill) -> list[dict[str, Any]]:
    """Return linked-resource metadata without returning document bodies."""
    rows: list[dict[str, Any]] = []
    visited: set[str] = set()

    def visit(relative: str) -> None:
        if relative in visited:
            return
        visited.add(relative)
        text = _read(skill, relative)
        links: list[dict[str, Any]] = []
        rows.append({"path": relative, "kind": _kind(relative), "links": links})
        if not relative.lower().endswith(".md"):
            return
        for raw, target, fragment in _links(relative, text):
            _resource(skill, target)
            links.append({"target": target, "fragment": fragment, "raw": raw, "kind": _kind(target)})
            if target.lower().endswith(".md"):
                visit(target)
            elif target not in visited:
                visited.add(target)
                rows.append({"path": target, "kind": _kind(target), "links": []})

    visit("SKILL.md")
    return rows


def all_documents(skill: Skill, start: str = "SKILL.md") -> list[dict[str, str]]:
    """Return reachable Markdown resources in deterministic depth-first order."""
    _resource(skill, start)
    documents: list[dict[str, str]] = []
    visited: set[str] = set()

    def visit(relative: str) -> None:
        if relative in visited:
            return
        visited.add(relative)
        text = _read(skill, relative)
        links = _links(relative, text) if relative.lower().endswith(".md") else []
        documents.append({"path": relative, "content": text})
        for _raw, target, _fragment in links:
            _resource(skill, target)
            if target.lower().endswith(".md"):
                visit(target)

    visit(start)
    return documents


def json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def emit_list(*, as_json: bool) -> None:
    skills = list_skills()
    if as_json:
        print(json_payload({
            "schema": "mck.skills.list/v1",
            "skills": [{"name": item.name, "description": item.description} for item in skills],
        }))
        return
    for item in skills:
        print(f"{item.name}\t{item.description}")


def emit_show(skill: Skill, relative: str, *, all_files: bool, as_json: bool) -> None:
    if all_files:
        documents = all_documents(skill, relative)
        if as_json:
            print(json_payload({"schema": "mck.skills.show/v1", "skill": skill.name, "documents": documents}))
            return
        for index, document in enumerate(documents):
            if index:
                sys.stdout.write("\n")
            sys.stdout.write(f"--- {skill.name}/{document['path']} ---\n")
            sys.stdout.write(document["content"])
        return
    content = _read(skill, relative)
    if as_json:
        print(json_payload({"schema": "mck.skills.show/v1", "skill": skill.name, "path": relative, "content": content}))
    else:
        sys.stdout.write(content)


def emit_tree(skill: Skill, *, as_json: bool) -> None:
    rows = tree(skill)
    if as_json:
        print(json_payload({"schema": "mck.skills.tree/v1", "skill": skill.name, "resources": rows}))
        return
    print(skill.name)
    for row in rows:
        print(f"  {row['path']} ({row['kind']})")
        for link in row["links"]:
            suffix = f"#{link['fragment']}" if link["fragment"] else ""
            print(f"    -> {link['target']}{suffix}")


__all__ = [
    "Skill",
    "SkillError",
    "all_documents",
    "emit_list",
    "emit_show",
    "emit_tree",
    "get_skill",
    "list_skills",
]
