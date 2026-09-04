"""Geometry-native data structures for periodic fragment chains."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np
Vector3=tuple[float,float,float]
Int3=tuple[int,int,int]
def _v(x,name):
    a=np.asarray(x,dtype=float)
    if a.shape!=(3,) or not np.all(np.isfinite(a)): raise ValueError(f"{name} must be a finite length-3 vector")
    return tuple(float(i) for i in a)
def _i(x,name):
    a=np.asarray(x)
    if a.shape!=(3,) or not np.all(a==np.asarray(a,dtype=int)): raise ValueError(f"{name} must be an integer length-3 vector")
    return tuple(int(i) for i in a)
@dataclass(frozen=True)
class BoundaryPort:
    port_id:str; position:Vector3; faces:tuple[str,...]=(); direction:Vector3|None=None; rule_ids:tuple[str,...]=()
    def __post_init__(self):
        object.__setattr__(self,"position",_v(self.position,"port position"))
        if self.direction is not None:
            d=np.asarray(_v(self.direction,"port direction"));
            if np.linalg.norm(d)==0: raise ValueError("port direction cannot be zero")
            object.__setattr__(self,"direction",tuple(float(i) for i in d))
@dataclass(frozen=True)
class FragmentTemplate:
    template_id:str; symbols:tuple[str,...]; positions:tuple[Vector3,...]; ports:tuple[BoundaryPort,...]; explicit_connections:tuple[tuple[int,int],...]=(); metadata:Mapping[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        if not self.template_id or not self.symbols or len(self.symbols)!=len(self.positions): raise ValueError("template_id, symbols, and positions are required")
        object.__setattr__(self,"positions",tuple(_v(p,"template position") for p in self.positions))
        if len({port.port_id for port in self.ports}) != len(self.ports):
            raise ValueError("fragment port ids must be unique")
        for a,b in self.explicit_connections:
            if not (0<=int(a)<len(self.symbols) and 0<=int(b)<len(self.symbols)): raise ValueError("explicit connection index out of range")
    @property
    def atom_count(self): return len(self.symbols)
    def port(self,port_id):
        for p in self.ports:
            if p.port_id==port_id: return p
        raise KeyError(port_id)
@dataclass(frozen=True)
class ConnectionRule:
    rule_id:str; left_template:str; left_port:str; right_template:str; right_port:str; allowed_image_shifts:tuple[Int3,...]=((0,0,0),); distance_range:tuple[float,float]|None=None; angle_range_deg:tuple[float,float]|None=None; dihedral_range_deg:tuple[float,float]|None=None
    def __post_init__(self):
        object.__setattr__(self,"allowed_image_shifts",tuple(_i(x,"image shift") for x in self.allowed_image_shifts))
        if self.distance_range is not None:
            lo,hi=map(float,self.distance_range)
            if not 0<=lo<=hi: raise ValueError("invalid distance_range")
            object.__setattr__(self,"distance_range",(lo,hi))
@dataclass(frozen=True)
class FragmentInstance:
    instance_id:str; template_id:str; chain_id:int; repeat_id:int; rotation:tuple[Vector3,Vector3,Vector3]; translation:Vector3
    def __post_init__(self):
        identity=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
        if self.rotation != identity:
            r=np.asarray(self.rotation,dtype=float)
            if r.shape!=(3,3) or not np.allclose(r.T@r,np.eye(3),atol=1e-7) or np.linalg.det(r)<=0: raise ValueError("rotation must be a proper orthogonal matrix")
        object.__setattr__(self,"translation",_v(self.translation,"translation"))
@dataclass(frozen=True)
class PeriodicEdge:
    left_node:str; right_node:str; right_image_shift:Int3=(0,0,0); rule_id:str=""; closure:bool=False
    left_port:str|None=None; right_port:str|None=None
    def __post_init__(self): object.__setattr__(self,"right_image_shift",_i(self.right_image_shift,"image shift"))
@dataclass
class PeriodicGraph:
    nodes:tuple[str,...]; edges:tuple[PeriodicEdge,...]; closure:str="translation"
    def __post_init__(self):
        if len(set(self.nodes))!=len(self.nodes): raise ValueError("periodic graph node ids must be unique")
        node_set=set(self.nodes)
        if any(e.left_node not in node_set or e.right_node not in node_set for e in self.edges): raise ValueError("periodic edge references unknown node")
    @property
    def cycle_rank(self): return len(self.edges)-len(self.nodes)+len(self._components())
    def _components(self):
        adj={n:set() for n in self.nodes}
        for e in self.edges: adj[e.left_node].add(e.right_node); adj[e.right_node].add(e.left_node)
        out=[]; unseen=set(self.nodes)
        while unseen:
            root=min(unseen); stack=[root]; c=set()
            while stack:
                n=stack.pop()
                if n not in unseen: continue
                unseen.remove(n); c.add(n); stack.extend(adj[n]&unseen)
            out.append(c)
        return out
    def winding_cycles(self):
        # The builder emits disjoint simple cycles.  Traverse that common case
        # directly; constructing/sorting a full adjacency payload for every
        # atom is unnecessarily expensive at the 1e5-atom target.
        if self.nodes and len(self.edges) == len(self.nodes):
            adjacency={n:[] for n in self.nodes}
            for index,edge in enumerate(self.edges):
                adjacency[edge.left_node].append((edge.right_node,np.asarray(edge.right_image_shift,dtype=int),index))
                adjacency[edge.right_node].append((edge.left_node,-np.asarray(edge.right_image_shift,dtype=int),index))
            if all(len(value)==2 for value in adjacency.values()):
                cycles=[]; unseen=set(self.nodes)
                while unseen:
                    start=min(unseen); current=start; previous_edge=None; total=np.zeros(3,dtype=int)
                    while True:
                        unseen.discard(current)
                        choices=[item for item in adjacency[current] if item[2]!=previous_edge]
                        if not choices: break
                        nxt,shift,edge_index=sorted(choices,key=lambda item:(item[0],item[2]))[0]
                        total+=shift; previous_edge=edge_index; current=nxt
                        if current==start: break
                    cycles.append(tuple(int(x) for x in total))
                return tuple(cycles)
        adj={n:[] for n in self.nodes}
        for k,e in enumerate(self.edges):
            s=e.right_image_shift; adj[e.left_node].append((e.right_node,s,str(k))); adj[e.right_node].append((e.left_node,tuple(-x for x in s),str(k)))
        pot={}; tree=set()
        for root in sorted(self.nodes):
            if root in pot: continue
            pot[root]=np.zeros(3,dtype=int); q=[root]
            while q:
                n=q.pop(0)
                for other,s,k in sorted(adj[n],key=lambda x:(x[0],x[2])):
                    if other not in pot: pot[other]=pot[n]+np.asarray(s); tree.add(k); q.append(other)
        result=[]
        for k,e in enumerate(self.edges):
            if str(k) not in tree: result.append(tuple(int(x) for x in pot[e.left_node]+np.asarray(e.right_image_shift)-pot[e.right_node]))
        return tuple(result)
    @property
    def winding(self):
        w=self.winding_cycles()
        if not w: return (0,0,0)
        return w[0] if all(value == w[0] for value in w) else (0,0,0)
@dataclass(frozen=True)
class ScrewSpec:
    order:int; axis:Vector3; angle_deg:float; center:Vector3=(0.,0.,0.); translation:Vector3=(0.,0.,0.); max_instances:int=100000
    def __post_init__(self):
        if int(self.order)!=self.order or self.order<1: raise ValueError("screw order must be positive")
        object.__setattr__(self,"axis",_v(self.axis,"axis")); object.__setattr__(self,"center",_v(self.center,"center")); object.__setattr__(self,"translation",_v(self.translation,"translation"))
        if np.linalg.norm(self.axis)==0: raise ValueError("screw axis cannot be zero")
    def rotation(self):
        a=np.asarray(self.axis); a=a/np.linalg.norm(a); t=np.deg2rad(self.angle_deg); x,y,z=a; k=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
        return np.eye(3)*np.cos(t)+(1-np.cos(t))*np.outer(a,a)+np.sin(t)*k
    def transform(self,pos,power=1):
        r=np.linalg.matrix_power(self.rotation(),power); c=np.asarray(self.center); tr=np.zeros(3); one=self.rotation()
        for _ in range(power): tr=one@tr+np.asarray(self.translation)
        return (np.asarray(pos)-c)@r.T+c+tr
    def closure_shift(self,cell,pbc,tol=1e-7):
        total=self.transform(np.zeros(3),self.order); rq=self.transform(np.eye(3),self.order)-total
        if not np.allclose(rq,np.eye(3),atol=tol): raise ValueError("screw is not finite-order: R^q is not identity")
        frac=total@np.linalg.inv(np.asarray(cell)); rounded=np.rint(frac).astype(int)
        for i,periodic in enumerate(pbc):
            if periodic and abs(frac[i]-rounded[i])>tol: raise ValueError(f"screw translation is incompatible with the supplied cell; possible integer shift {tuple(int(x) for x in rounded)}")
            if not periodic and abs(frac[i])>tol: raise ValueError("screw translation leaves a non-periodic direction")
        return tuple(int(x) for x in rounded)
@dataclass(frozen=True)
class ChainSpec:
    sequence:tuple[str,...]; chain_count:int=1; closure:str="translation"; target_winding:Int3|None=None; instance_centers:tuple[Vector3,...]|None=None; screw:ScrewSpec|None=None; seed:int=0; max_backtracks:int=64; min_distance:float=.8; tolerance:float=1e-6
    def __post_init__(self):
        if not self.sequence: raise ValueError("sequence cannot be empty")
        if self.chain_count<1 or self.closure not in {"translation","screw"}: raise ValueError("invalid ChainSpec")
        if self.closure=="screw" and self.screw is None: raise ValueError("screw closure requires screw")
        if self.target_winding is not None: object.__setattr__(self,"target_winding",_i(self.target_winding,"target winding"))
        if self.instance_centers is not None:
            c=tuple(_v(x,"instance center") for x in self.instance_centers)
            if len(c)!=len(self.sequence): raise ValueError("instance_centers must match sequence")
            object.__setattr__(self,"instance_centers",c)
@dataclass
class PeriodicBundle:
    atoms:Any; graph:PeriodicGraph; instances:tuple[FragmentInstance,...]; metadata:dict[str,Any]; validation:dict[str,Any]=field(default_factory=dict)
__all__=["BoundaryPort","ChainSpec","ConnectionRule","FragmentInstance","FragmentTemplate","PeriodicBundle","PeriodicEdge","PeriodicGraph","ScrewSpec"]
