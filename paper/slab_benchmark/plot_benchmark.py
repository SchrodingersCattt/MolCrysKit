"""
Plot Multi-System Benchmark Figure
====================================
Generates a 3-panel publication-quality log–log figure showing size-scaling
of MolCrysKit, ASE, and Pymatgen slab generation across three molecular crystal
systems. Each panel shows one system; all panels share the y-axis.

Visual encoding:
  Color  = method  MCK #135788 (deep blue) | ASE #b41f5d (deep rose) | PMG #819c37 (olive)
  One marker style per panel (filled circle for all methods within a panel)

Layout: figsize=(12, 3), sharey=True, fontsize=12, Arial

Data source:
  results/multi_system_benchmark.json
  (committed Bohrium benchmark result mirrored into this paper directory)

Output: results/multi_system_benchmark_figure.pdf / .png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
DATA_FILE = RESULTS_DIR / "multi_system_benchmark.json"

# ── global style ──────────────────────────────────────────────────────────────
FS = 16  # base font size

plt.rcParams.update({
    "font.family":        "Arial",
    "font.size":          FS,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         600,
})

# ── colour encoding ────────────────────────────────────────────────────────────
MCK_COLOR = "#135788"   # JCIM deep blue
ASE_COLOR = "#b41f5d"   # deep rose
PMG_COLOR = "#819c37"   # olive green

SYSTEM_LABELS = {
    "HXACAN": "Acetaminophen", # (HXACAN)
    "HMX":    "$β$-HMX", #  (OCHTET12)
    "DAPM4":  "DAP-M4",  #  (UHILUV02)
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _parse(atom_list, times_list):
    """Return (x, mean, std) for valid (non-None) entries."""
    x, y, e = [], [], []
    for na, times in zip(atom_list, times_list):
        valid = [t for t in (times or []) if t is not None]
        if valid:
            x.append(na)
            y.append(float(np.mean(valid)))
            e.append(float(np.std(valid)))
    return np.asarray(x, float), np.asarray(y, float), np.asarray(e, float)


# ── load + normalise data ──────────────────────────────────────────────────────
with open(DATA_FILE, "r") as _f:
    raw = json.load(_f)

if "systems" in raw:
    systems_data = raw["systems"]
else:
    systems_data = raw

def _normalise(res: dict) -> dict:
    """Convert old flat format to new cubic/linear nested format."""
    if "cubic" in res:
        return res
    natoms = res.get("natoms", res.get("n_atoms", []))
    return {
        "name":        res.get("name", ""),
        "miller":      res.get("miller", []),
        "base_natoms": res.get("base_natoms", 0),
        "cubic":  {"atoms": natoms, "ase": res.get("ase_times", []), "mck": res.get("mck_times", [])},
        "linear": {"atoms": [],    "pmg": []},
    }

systems_data = {k: _normalise(v) for k, v in systems_data.items() if "error" not in v}
system_keys  = list(systems_data.keys())
n_systems    = len(system_keys)

# ── figure: one panel per system, sharey ──────────────────────────────────────
fig, axes = plt.subplots(
    1, n_systems,
    figsize=(14, 3),
    sharey=True,
)
fig.subplots_adjust(wspace=0.45, left=0.07, right=0.70, top=0.92, bottom=0.17)

# Build method legend handles once (shared across all panels)
legend_handles = []
legend_labels  = []

for col_idx, short in enumerate(system_keys):
    res   = systems_data[short]
    ax    = axes[col_idx]
    sname = SYSTEM_LABELS.get(short, short)

    # ── MCK cubic ─────────────────────────────────────────────────────────────
    cx, cy, ce = _parse(res["cubic"]["atoms"], res["cubic"]["mck"])
    if len(cx):
        h = ax.errorbar(
            cx, cy, yerr=ce,
            fmt="o-",
            color=MCK_COLOR,
            markerfacecolor=MCK_COLOR,
            markeredgecolor=MCK_COLOR,
            linewidth=1.5, markersize=5,
            capsize=3, capthick=0.8, elinewidth=0.8,
            zorder=4,
        )
        if col_idx == 0:
            legend_handles.append(h)
            legend_labels.append("MolCrysKit\n(topological)")

    # ── ASE cubic ─────────────────────────────────────────────────────────────
    ax2, ay, ae = _parse(res["cubic"]["atoms"], res["cubic"]["ase"])
    if len(ax2):
        h = ax.errorbar(
            ax2, ay, yerr=ae,
            fmt="o--",
            color=ASE_COLOR,
            markerfacecolor="white",
            markeredgecolor=ASE_COLOR,
            linewidth=1.5, markersize=5,
            capsize=3, capthick=0.8, elinewidth=0.8,
            alpha=0.85, zorder=3,
        )
        if col_idx == 0:
            legend_handles.append(h)
            legend_labels.append("ASE\n(geometric)")

    # ── Pymatgen linear ───────────────────────────────────────────────────────
    px, py, pe = _parse(res["linear"]["atoms"], res["linear"]["pmg"])
    if len(px):
        h = ax.errorbar(
            px, py, yerr=pe,
            fmt="o-.",
            color=PMG_COLOR,
            markerfacecolor="white",
            markeredgecolor=PMG_COLOR,
            linewidth=1.5, markersize=5,
            capsize=3, capthick=0.8, elinewidth=0.8,
            alpha=0.85, zorder=3,
        )
        if col_idx == 0:
            legend_handles.append(h)
            legend_labels.append("Pymatgen\n(geometric+repair)")

    # ── panel formatting ──────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40, 60000)
    ax.set_ylim(0.001, 50000)
    ax.text(0.05, 0.95, sname, transform=ax.transAxes, fontsize=FS, va="top", ha="left")
    ax.set_xlabel("Number of atoms", fontsize=FS)
    ax.yaxis.grid(True, which="major", alpha=0.25, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # y-axis label only on leftmost panel
    if col_idx == 0:
        ax.set_ylabel("Wall time (s)", fontsize=FS)

# ── shared legend — right side, vertical ─────────────────────────────────────
fig.legend(
    legend_handles, legend_labels,
    fontsize=FS - 1,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(0.75, 0.5),
    ncol=1,
    handlelength=1.50,
    labelspacing=1.6
)

# ── save ──────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    out = RESULTS_DIR / f"multi_system_benchmark_figure.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
