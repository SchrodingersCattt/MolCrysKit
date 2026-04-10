"""
Plot CSD Statistics Figure (Revised)
=====================================
Generates a publication-quality figure showing the prevalence of
disorder and missing hydrogens in the CSD.

Style: low-saturation grayscale + hatch patterns, Arial font, italic n.
Single panel: Venn diagram showing overlap between disorder and missing-H,
with the union percentage annotated.

Reads from: results/csd_statistics_raw.json  (re-computes overlap from raw data)
Output:     results/csd_statistics_figure.pdf / .png
"""

import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RAW_FILE    = RESULTS_DIR / "csd_statistics_raw.json"

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'Arial',
    'font.size':         12,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        200,
})

# Grayscale palette
GRAY_LIGHT  = '#CCCCCC'
GRAY_MID    = '#888888'
GRAY_DARK   = '#444444'

# Hatch patterns
HATCH_DISORDER  = '///'
HATCH_MISSING_H = '\\\\\\'
HATCH_BOTH      = 'xxx'

# ── load and recompute from raw data ──────────────────────────────────────────
with open(RAW_FILE, 'r') as f:
    records = json.load(f)

valid = [r for r in records if not r.get('error')]
n = len(valid)

n_disorder  = sum(1 for r in valid if r['has_disorder'])
n_missing_h = sum(1 for r in valid if r['has_missing_h'])
n_union     = sum(1 for r in valid if r['has_disorder'] or r['has_missing_h'])
n_both      = sum(1 for r in valid if r['has_disorder'] and r['has_missing_h'])

n_disorder_only  = sum(1 for r in valid if r['has_disorder'] and not r['has_missing_h'])
n_missing_h_only = sum(1 for r in valid if not r['has_disorder'] and r['has_missing_h'])
n_neither        = sum(1 for r in valid if not r['has_disorder'] and not r['has_missing_h'])

pct_disorder  = 100.0 * n_disorder  / n
pct_missing_h = 100.0 * n_missing_h / n
pct_union     = 100.0 * n_union     / n
pct_both      = 100.0 * n_both      / n

pct_disorder_only  = 100.0 * n_disorder_only  / n
pct_missing_h_only = 100.0 * n_missing_h_only / n
pct_neither        = 100.0 * n_neither        / n

# Wilson score 95% CI (matches QM9 analysis script for consistency)
def wilson_ci(k, n_total, z=1.96):
    """Wilson score interval for proportion k/n_total.
    Returns (lo_pct, hi_pct) as percentages.
    """
    p = k / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, center - margin) * 100, min(1.0, center + margin) * 100

ci_union_lo, ci_union_hi = wilson_ci(n_union, n)

# ── figure: single-panel Venn diagram (wide, flat, with outer ellipse) ────────
from matplotlib.patches import Circle, Ellipse
from matplotlib.patches import Polygon as MplPolygon

# Wide figure; aspect='auto' so xlim/ylim control shape independently
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.set_aspect('auto')
ax.axis('off')

# Coordinate system: wide x, compressed y
ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-1.8, 2.8)   # extra headroom at top for callouts

EDGE = '#333333'
r    = 1.05          # inner circle radius (data-coords)
cx_L = -0.68         # left circle centre x  (Disorder)
cx_R =  0.68         # right circle centre x (Missing H)
cy   =  0.0

# ── Outer ellipse representing "Neither" (the background universe) ─────────────
# Width spans from well left of left circle to well right of right circle
outer_w = 2 * (cx_R + r + 0.55)   # total width
outer_h = 2 * (r + 0.38)          # total height (slightly larger than inner circles)
outer_ellipse = Ellipse(
    (cx_int := (cx_L + cx_R) / 2, cy),
    width=outer_w, height=outer_h,
    facecolor='none', edgecolor='#333333',
    linewidth=1.5, zorder=1,
)
ax.add_patch(outer_ellipse)

# ── Left circle: Disorder (grey mid, ///) ─────────────────────────────────────
circ_L = Circle((cx_L, cy), r, facecolor=GRAY_MID, edgecolor=EDGE,
                linewidth=1.2, zorder=2, hatch=HATCH_DISORDER, alpha=0.85)
ax.add_patch(circ_L)

# ── Right circle: Missing H (grey light, \\\) ─────────────────────────────────
circ_R = Circle((cx_R, cy), r, facecolor=GRAY_LIGHT, edgecolor=EDGE,
                linewidth=1.2, zorder=3, hatch=HATCH_MISSING_H, alpha=0.85)
ax.add_patch(circ_R)

# ── Intersection polygon (white, xxx) ─────────────────────────────────────────
theta_vals = np.linspace(0, 2 * np.pi, 2048)

pts_L = [(cx_L + r * np.cos(t), cy + r * np.sin(t)) for t in theta_vals
         if (cx_L + r * np.cos(t) - cx_R)**2 + (cy + r * np.sin(t))**2 <= r**2 + 1e-9]
pts_R = [(cx_R + r * np.cos(t), cy + r * np.sin(t)) for t in theta_vals
         if (cx_R + r * np.cos(t) - cx_L)**2 + (cy + r * np.sin(t))**2 <= r**2 + 1e-9]

all_pts = pts_L + pts_R
all_pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx_int))
all_pts = np.array(all_pts)

if len(all_pts) > 2:
    inter_patch = MplPolygon(all_pts, closed=True,
                              facecolor='white', edgecolor=EDGE,
                              linewidth=0.8, zorder=4, hatch=HATCH_BOTH)
    ax.add_patch(inter_patch)

# ── Callout style helper ───────────────────────────────────────────────────────
ARROW_STYLE = dict(
    arrowstyle='->', color='#555555', lw=0.9,
    connectionstyle='arc3,rad=0.0',
)

def callout(ax, label, text_xy, arrow_tip_xy, ha='center', fontsize=10):
    ax.annotate(
        label,
        xy=arrow_tip_xy, xycoords='data',
        xytext=text_xy, textcoords='data',
        ha=ha, va='bottom', fontsize=fontsize, color='#222222',
        fontweight='bold',
        arrowprops=ARROW_STYLE,
        zorder=7,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=1),
    )

# Disorder only → label upper-left, tip in left lobe
callout(ax,
        f'Crystallographic disorder\n(disorder only: {pct_disorder_only:.1f}%)',
        text_xy=(-2.5, 2.10),
        arrow_tip_xy=(cx_L - 0.42, cy + 0.15),
        ha='left', fontsize=9.5)

# Missing H only → label upper-right, tip in right lobe
callout(ax,
        f'H inspection required\n(H inspection only: {pct_missing_h_only:.1f}%)',
        text_xy=(0.35, 2.10),
        arrow_tip_xy=(cx_R + 0.42, cy + 0.15),
        ha='left', fontsize=9.5)

# Both → label centre-top, tip in intersection centre
callout(ax,
        f'Both: {pct_both:.1f}%',
        text_xy=(cx_int, 1.72),
        arrow_tip_xy=(cx_int, cy),
        ha='center', fontsize=9.5)

# Neither → label pointing to gap between outer ellipse and inner circles (right side)
ax.annotate(
    f'Neither: {pct_neither:.1f}%',
    xy=(cx_R + r + 0.28, cy),       # gap between right circle edge and outer ellipse
    xycoords='data',
    xytext=(2.35, 1.50), textcoords='data',
    ha='left', va='bottom', fontsize=9.5, color='#444444',
    fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#777777', lw=0.9,
                    connectionstyle='arc3,rad=0.15'),
    zorder=7,
)

# ── Union annotation — text below outer ellipse ───────────────────────────────
ax.text(cx_int, -(outer_h / 2 + 0.22),
        f'Union (any preprocessing needed): {pct_union:.1f}%'
        f'  [95% CI {ci_union_lo:.1f}–{ci_union_hi:.1f}%]',
        ha='center', va='top', fontsize=9.0, color='#444444', zorder=6)

# ── Figure title ──────────────────────────────────────────────────────────────
ax.set_title(
    f'Preprocessing needs in the CSD\n'
    f'($\\it{{n}}$ = {n:,} organic 3D structures)',
    fontsize=12, pad=8,
)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    out = RESULTS_DIR / f"csd_statistics_figure.{ext}"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
