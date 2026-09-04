"""Build rigid fragments into independent periodic chains."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from itertools import product
from typing import Mapping, Sequence
import numpy as np
from ase import Atoms
from ..structures.periodic_geometry import ChainSpec, ConnectionRule, FragmentInstance, FragmentTemplate, PeriodicBundle, PeriodicEdge, PeriodicGraph

def _cell(value):
    cell = np.asarray(value, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("cell must be a finite, non-singular 3x3 row-vector cell")
    return cell

def _image_shifts(cell, pbc, radius):
    if radius <= 0: return ((0, 0, 0),)
    n = max(1, int(np.ceil(radius / np.linalg.svd(cell, compute_uv=False)[-1])) + 1)
    return tuple(tuple(int(x) for x in s) for s in product(*[range(-n, n + 1) if pbc[i] else range(0, 1) for i in range(3)]))

class _PeriodicCellList:
    def __init__(self, cell, pbc, radius):
        self.cell, self.inverse, self.pbc, self.radius = cell, np.linalg.inv(cell), tuple(bool(x) for x in pbc), float(radius)
        self.bins = np.maximum(1, np.floor(np.linalg.norm(cell, axis=1) / max(radius, 1e-6)).astype(int))
        self.items = defaultdict(list)
        self.positions = []
        self.shifts = _image_shifts(cell, self.pbc, self.radius)
        self.shift_array = np.asarray(self.shifts, dtype=float)
        self.orthogonal = np.allclose(cell, np.diag(np.diag(cell)), atol=1e-12)
        spans = [max(1, int(np.ceil(self.radius * self.bins[i] / np.linalg.norm(self.cell[i])))) for i in range(3)]
        self.offsets = tuple(product(*[range(-spans[i], spans[i] + 1) if self.pbc[i] else range(0, 1) for i in range(3)]))
    def key(self, frac):
        value = np.asarray(frac, dtype=float).copy()
        for i, periodic in enumerate(self.pbc):
            if periodic: value[i] %= 1.0
        return tuple(int(np.floor(value[i] * self.bins[i])) % self.bins[i] for i in range(3))
    def nearby(self, key):
        for delta in self.offsets:
            value = [key[i] + delta[i] for i in range(3)]
            value = [value[i] % self.bins[i] if self.pbc[i] else value[i] for i in range(3)]
            if all(0 <= value[i] < self.bins[i] for i in range(3)): yield tuple(value)
    def can_insert(self, position, excluded):
        for key in self.nearby(self.key(position @ self.inverse)):
            for index, other in self.items.get(key, ()):
                if index in excluded: continue
                delta = (other - position) @ self.inverse
                if self.orthogonal:
                    wrapped = delta.copy()
                    for axis, periodic in enumerate(self.pbc):
                        if periodic: wrapped[axis] -= np.rint(wrapped[axis])
                    if np.linalg.norm(wrapped @ self.cell) < self.radius: return False
                elif np.min(np.linalg.norm((delta[None, :] + self.shift_array) @ self.cell, axis=1)) < self.radius:
                    return False
        return True
    def insert(self, position):
        index = len(self.positions); value = np.asarray(position, dtype=float); self.positions.append(value); self.items[self.key(value @ self.inverse)].append((index, value)); return index

def _resolve_rule(rules: Sequence[ConnectionRule], left: FragmentTemplate, right: FragmentTemplate):
    """Resolve one explicit rule and its orientation for an actual edge.

    A template may expose more than one port.  Selecting ``ports[0]`` made a
    one-fragment fixture look connected even when its declared input/output
    ports were never used.  The rule is now the source of truth for both
    endpoint port ids; ambiguous declarations fail rather than being guessed.
    """
    matches = []
    for rule in rules:
        if (rule.left_template, rule.right_template) == (left.template_id, right.template_id):
            matches.append((rule, rule.left_port, rule.right_port, False))
        elif (rule.right_template, rule.left_template) == (left.template_id, right.template_id):
            matches.append((rule, rule.right_port, rule.left_port, True))
    unique = {(item[0].rule_id, item[1], item[2], item[3]): item for item in matches}
    matches = list(unique.values())
    if not matches:
        raise ValueError(f"no explicit connection rule for {left.template_id} -> {right.template_id}")
    if len(matches) > 1:
        ids = ", ".join(repr(item[0].rule_id) for item in matches)
        raise ValueError(f"ambiguous connection rules for {left.template_id} -> {right.template_id}: {ids}")
    rule, left_port_id, right_port_id, reversed_rule = matches[0]
    left.port(left_port_id)
    right.port(right_port_id)
    return rule, left_port_id, right_port_id, reversed_rule

def _check(rule, distance):
    if rule.distance_range is not None and not rule.distance_range[0] <= distance <= rule.distance_range[1]: raise ValueError(f"connection {rule.rule_id!r} distance {distance:.6g} A is outside {rule.distance_range} A")

def _check_shift(rule, shift, reversed_rule=False):
    value = tuple(int(x) for x in shift)
    allowed = rule.allowed_image_shifts
    if reversed_rule:
        allowed = tuple(tuple(-x for x in item) for item in allowed)
    if value not in allowed: raise ValueError(f"image shift {value} is not allowed by connection {rule.rule_id!r}")

def _check_ports(rule, left_port, right_port, left_rotation, right_rotation):
    if rule.angle_range_deg is not None:
        if left_port.direction is None or right_port.direction is None:
            raise ValueError(f"connection {rule.rule_id!r} requests an angle range but both port directions are not supplied")
        left=np.asarray(left_port.direction)@left_rotation.T; right=np.asarray(right_port.direction)@right_rotation.T
        cosine=float(np.dot(left,-right)/(np.linalg.norm(left)*np.linalg.norm(right)))
        angle=float(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))
        if not rule.angle_range_deg[0] <= angle <= rule.angle_range_deg[1]: raise ValueError(f"connection {rule.rule_id!r} angle {angle:.6g} deg is outside {rule.angle_range_deg} deg")
    if rule.dihedral_range_deg is not None:
        raise ValueError(f"connection {rule.rule_id!r} requests a dihedral range but no reference planes were supplied")


def _world_port(template: FragmentTemplate, port_id: str, rotation, translation):
    port = template.port(port_id)
    return np.asarray(port.position, dtype=float) @ rotation.T + translation


def _check_edge(rule, left_template, right_template, left_port_id, right_port_id,
                reversed_rule, left_transform, right_transform, shift, cell):
    """Check one edge using the declared ports and integer image shift."""
    _check_shift(rule, shift, reversed_rule)
    left_rotation, left_translation = left_transform
    right_rotation, right_translation = right_transform
    left_port = left_template.port(left_port_id)
    right_port = right_template.port(right_port_id)
    _check_ports(rule, left_port, right_port, left_rotation, right_rotation)
    left_position = _world_port(left_template, left_port_id, left_rotation, left_translation)
    right_position = _world_port(right_template, right_port_id, right_rotation, right_translation)
    distance = float(np.linalg.norm(left_position - (right_position + np.asarray(shift, dtype=float) @ cell)))
    _check(rule, distance)
    return distance


def _resolve_and_check_edge(rules, left_template, right_template, left_transform,
                            right_transform, shift, cell):
    rule, left_port_id, right_port_id, reversed_rule = _resolve_rule(rules, left_template, right_template)
    _check_edge(rule, left_template, right_template, left_port_id, right_port_id,
                reversed_rule, left_transform, right_transform, shift, cell)
    return rule, left_port_id, right_port_id

def _build_once(templates: Mapping[str, FragmentTemplate], rules: Sequence[ConnectionRule], cell: Sequence[Sequence[float]], pbc: Sequence[bool], spec: ChainSpec) -> PeriodicBundle:
    cell = _cell(cell); pbc = tuple(bool(x) for x in pbc)
    if len(pbc) != 3: raise ValueError("pbc must have three values")
    screw_shift = (0, 0, 0)
    if spec.closure == "screw":
        assert spec.screw is not None; screw_shift = spec.screw.closure_shift(cell, pbc, spec.tolerance)
        if len(spec.sequence) > spec.screw.max_instances: raise ValueError("screw expansion exceeds max_instances")
    winding = spec.target_winding if spec.target_winding is not None else screw_shift
    if spec.closure == "screw" and winding != screw_shift: raise ValueError("target_winding is inconsistent with screw closure")
    for name in spec.sequence:
        if name not in templates: raise KeyError(f"unknown template {name!r}")
    if spec.instance_centers is not None: centers = tuple((0., 0., 0.) for _ in range(spec.chain_count))
    elif spec.chain_count > 1:
        rng = np.random.default_rng(spec.seed)
        centers = tuple(tuple(float(x) for x in rng.random(3)) for _ in range(spec.chain_count))
    else: centers = ((0., 0., 0.),)
    index = _PeriodicCellList(cell, pbc, spec.min_distance); positions=[]; symbols=[]; arrays={"atom_id":[],"chain_id":[],"fragment_id":[],"repeat_id":[]}; instances=[]; nodes=[]; edges=[]
    for chain_id in range(spec.chain_count):
        base=np.asarray(centers[chain_id % len(centers)]); transforms=[]; chain_nodes=[]
        for repeat_id, name in enumerate(spec.sequence):
            template=templates[name]
            if spec.instance_centers is not None: frac=np.asarray(spec.instance_centers[repeat_id])+base
            elif spec.closure == "translation" and winding != (0,0,0): frac=base + repeat_id*np.asarray(winding,dtype=float)/len(spec.sequence)
            elif spec.closure == "screw": frac=base
            elif len(spec.sequence)>1: raise ValueError("zero-winding multi-fragment chains require explicit instance_centers")
            else: frac=base
            translation=frac@cell; rotation=np.eye(3)
            if spec.closure == "screw": assert spec.screw is not None; rotation=np.linalg.matrix_power(spec.screw.rotation(),repeat_id); translation=translation+spec.screw.transform(np.zeros(3),repeat_id)
            transformed=np.asarray(template.positions)@rotation.T+translation; iid=f"chain{chain_id}:repeat{repeat_id}"; instances.append(FragmentInstance(iid,name,chain_id,repeat_id,tuple(tuple(float(x) for x in row) for row in rotation),tuple(float(x) for x in translation))); transforms.append((rotation,translation)); chain_nodes.append(iid); nodes.append(iid)
            for local,(symbol,position) in enumerate(zip(template.symbols,transformed)):
                excluded=set()
                for left,right in template.explicit_connections:
                    if local==right and left < local: excluded.add(len(positions)-local+left)
                if not index.can_insert(position,excluded): raise ValueError(f"periodic collision while placing {iid} atom {local}")
                index.insert(position); positions.append(position); symbols.append(symbol); arrays["atom_id"].append(len(positions)-1); arrays["chain_id"].append(chain_id); arrays["fragment_id"].append(name); arrays["repeat_id"].append(repeat_id)
        if spec.closure == "translation":
            for i in range(len(chain_nodes)-1):
                rule, left_port, right_port = _resolve_and_check_edge(
                    rules,
                    templates[spec.sequence[i]],
                    templates[spec.sequence[i + 1]],
                    transforms[i],
                    transforms[i + 1],
                    (0, 0, 0),
                    cell,
                )
                edges.append(PeriodicEdge(
                    chain_nodes[i], chain_nodes[i + 1], (0, 0, 0), rule.rule_id,
                    False, left_port, right_port,
                ))
            rule, left_port, right_port = _resolve_and_check_edge(
                rules,
                templates[spec.sequence[-1]],
                templates[spec.sequence[0]],
                transforms[-1],
                transforms[0],
                winding,
                cell,
            )
            if len(chain_nodes) == 1 and winding == (0, 0, 0):
                left_position = _world_port(templates[spec.sequence[-1]], left_port, *transforms[-1])
                right_position = _world_port(templates[spec.sequence[0]], right_port, *transforms[0])
                if left_port == right_port or np.linalg.norm(left_position - right_position) <= spec.tolerance:
                    raise ValueError(
                        "degenerate one-instance zero-winding closure; "
                        "provide distinct endpoint ports or a non-zero image shift"
                    )
            edges.append(PeriodicEdge(
                chain_nodes[-1], chain_nodes[0], tuple(int(x) for x in winding),
                rule.rule_id, True, left_port, right_port,
            ))
        else:
            for i in range(len(chain_nodes)-1):
                if rules:
                    rule, left_port, right_port = _resolve_and_check_edge(
                        rules,
                        templates[spec.sequence[i]],
                        templates[spec.sequence[i + 1]],
                        transforms[i],
                        transforms[i + 1],
                        (0, 0, 0),
                        cell,
                    )
                    edges.append(PeriodicEdge(
                        chain_nodes[i], chain_nodes[i + 1], (0, 0, 0), rule.rule_id,
                        False, left_port, right_port,
                    ))
                else:
                    edges.append(PeriodicEdge(chain_nodes[i], chain_nodes[i + 1], (0, 0, 0), "screw-step"))
            if rules:
                rule, left_port, right_port = _resolve_and_check_edge(
                    rules,
                    templates[spec.sequence[-1]],
                    templates[spec.sequence[0]],
                    transforms[-1],
                    transforms[0],
                    winding,
                    cell,
                )
                edges.append(PeriodicEdge(
                    chain_nodes[-1], chain_nodes[0], tuple(int(x) for x in winding),
                    rule.rule_id, True, left_port, right_port,
                ))
            else:
                edges.append(PeriodicEdge(
                    chain_nodes[-1], chain_nodes[0], tuple(int(x) for x in winding),
                    "screw-closure", True,
                ))
    atoms=Atoms(symbols=symbols,positions=np.asarray(positions),cell=cell,pbc=pbc)
    for name,values in arrays.items(): atoms.set_array(name,np.asarray(values,dtype="U64" if name=="fragment_id" else int))
    graph=PeriodicGraph(tuple(nodes),tuple(edges),spec.closure)
    if spec.target_winding is not None and graph.winding != spec.target_winding: raise ValueError(f"constructed winding {graph.winding} does not match target {spec.target_winding}")
    frac=np.asarray(positions)@np.linalg.inv(cell)
    for axis,periodic in enumerate(pbc):
        if not periodic and np.any((frac[:,axis] < -spec.tolerance)|(frac[:,axis]>1+spec.tolerance)): raise ValueError("constructed geometry leaves a non-periodic direction")
    return PeriodicBundle(atoms,graph,tuple(instances),{"schema":"mck.periodic_bundle","version":1,"cell":cell.tolist(),"pbc":list(pbc),"closure":spec.closure,"target_winding":list(winding),"seed":spec.seed,"tolerance":spec.tolerance,"min_distance":spec.min_distance,"collision_policy":"periodic_cell_list_hard_reject","chemistry_policy":"geometry_only_no_implicit_chemistry"})

def build_periodic_chains(templates: Mapping[str, FragmentTemplate], rules: Sequence[ConnectionRule], cell: Sequence[Sequence[float]], pbc: Sequence[bool], spec: ChainSpec) -> PeriodicBundle:
    """Build with bounded deterministic retry when random chain packing clashes."""
    attempts = 1 if spec.instance_centers is not None or spec.chain_count == 1 else spec.max_backtracks + 1
    last_error = None
    for attempt in range(attempts):
        try:
            return _build_once(templates, rules, cell, pbc, replace(spec, seed=spec.seed + attempt))
        except ValueError as error:
            if "periodic collision" not in str(error):
                raise
            last_error = error
    raise ValueError(f"could not pack {spec.chain_count} independent chains after {attempts} deterministic attempts") from last_error

__all__=["build_periodic_chains"]
