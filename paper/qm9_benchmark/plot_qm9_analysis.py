"""
QM9 Hybridization Validation — Analysis and Figures

Generates:
  1. qm9_confusion_matrix.pdf/png    — confusion matrix heatmap with CI
  2. qm9_failure_analysis.pdf/png    — error cases with structural annotations
  3. qm9_excluded_molecules.pdf/png  — molecules excluded from validation
  4. qm9_statistics.json             — CI estimates and summary

Usage:
    cd R1/experiments/qm9
    python plot_qm9_analysis.py
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# RDKit drawing
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

# ── Constants ────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
RAW_JSON = os.path.join(RESULTS_DIR, 'qm9_real_raw.json')

# CSD-paper grey palette
GREY_DARK  = '#333333'
GREY_MID   = '#777777'
GREY_LIGHT = '#AAAAAA'
GREY_BG    = '#DDDDDD'
RED_ERROR  = '#CC2200'
BLUE_OK    = '#2255CC'

FONT = 'Arial'
plt.rcParams.update({
    'font.family': FONT,
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ── Load data ─────────────────────────────────────────────────────────────────
with open(RAW_JSON) as f:
    data = json.load(f)

n_ok     = data['n_molecules_ok']        # 491
n_failed = data['n_molecules_failed']    # 9 (3D embedding failure)
n_total  = data['n_smiles_input']        # 500
n_atoms  = data['total_atoms_evaluated'] # 4314
n_correct= data['atoms_correct']         # 4251
accuracy = data['overall_accuracy']      # 0.9854

failed_smiles = data['failed_smiles']
failure_cases = data['failure_cases']    # wrong predictions (up to 50)
per_element   = data['per_element']
confusion_raw = data['confusion_matrix'] # {true_hyb: {pred_hyb: count}}
aromatic_data = data.get('aromatic', {}) # aromatic atom statistics


# ── 1. 95% CI (Wilson score interval) ────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for proportion k/n."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


ci_overall = wilson_ci(n_correct, n_atoms)
ci_per_elem = {}
for elem, stats in per_element.items():
    ci_per_elem[elem] = wilson_ci(stats['correct'], stats['total'])

# Aromatic CI
n_arom_atoms = aromatic_data.get('n_aromatic_atoms', 0)
n_arom_correct = aromatic_data.get('n_aromatic_correct', 0)
n_arom_mols = aromatic_data.get('n_aromatic_molecules', 0)
arom_acc = n_arom_correct / n_arom_atoms if n_arom_atoms > 0 else 0.0
ci_aromatic = wilson_ci(n_arom_correct, n_arom_atoms) if n_arom_atoms > 0 else (0, 0)

stats_summary = {
    'n_molecules_sampled': n_total,
    'n_molecules_ok': n_ok,
    'n_molecules_failed': n_failed,
    'n_atoms_evaluated': n_atoms,
    'overall_accuracy': round(accuracy * 100, 2),
    'ci_95_overall': [round(ci_overall[0]*100, 2), round(ci_overall[1]*100, 2)],
    'per_element': {
        elem: {
            'total': per_element[elem]['total'],
            'correct': per_element[elem]['correct'],
            'accuracy_pct': round(per_element[elem]['accuracy']*100, 2),
            'ci_95': [round(ci_per_elem[elem][0]*100, 2),
                      round(ci_per_elem[elem][1]*100, 2)],
        }
        for elem in per_element
    },
    'aromatic': {
        'n_aromatic_molecules': n_arom_mols,
        'n_aromatic_atoms': n_arom_atoms,
        'n_aromatic_correct': n_arom_correct,
        'aromatic_accuracy_pct': round(arom_acc * 100, 2),
        'ci_95': [round(ci_aromatic[0]*100, 2), round(ci_aromatic[1]*100, 2)],
    },
    'failed_smiles': failed_smiles,
    'n_wrong_predictions': n_atoms - n_correct,
    'wrong_prediction_rate_pct': round((1 - accuracy) * 100, 2),
}

with open(os.path.join(RESULTS_DIR, 'qm9_statistics.json'), 'w') as f:
    json.dump(stats_summary, f, indent=2)

print("=== QM9 Statistics ===")
print(f"Overall: {accuracy*100:.2f}% [{ci_overall[0]*100:.2f}%, {ci_overall[1]*100:.2f}%] 95% CI")
for elem, ci in ci_per_elem.items():
    acc = per_element[elem]['accuracy']
    print(f"  {elem}: {acc*100:.1f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
if n_arom_atoms > 0:
    print(f"  Aromatic atoms: {arom_acc*100:.1f}% [{ci_aromatic[0]*100:.1f}%, {ci_aromatic[1]*100:.1f}%]"
          f"  ({n_arom_correct}/{n_arom_atoms} in {n_arom_mols} molecules)")


# ── 2. Confusion matrix heatmap ───────────────────────────────────────────────

hyb_labels = [r'$\it{sp}$', r'$\it{sp}^{2}$', r'$\it{sp}^{3}$']
hyb_keys   = ['sp', 'sp2', 'sp3']

# Build matrix
cm = np.zeros((3, 3), dtype=int)
for ri, true_h in enumerate(hyb_keys):
    for ci2, pred_h in enumerate(hyb_keys):
        cm[ri, ci2] = confusion_raw.get(true_h, {}).get(pred_h, 0)

total_per_row = cm.sum(axis=1, keepdims=True)
cm_norm = cm.astype(float) / total_per_row.clip(1)

# Custom grey-to-dark colormap
cmap = LinearSegmentedColormap.from_list(
    'grey_scale', ['#FFFFFF', '#AAAAAA', '#333333'], N=256
)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5),
                          gridspec_kw={'width_ratios': [1.05, 0.95, 0.75]})

# Left: normalized heatmap
ax = axes[0]
im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')

for ri in range(3):
    for ci2 in range(3):
        val = cm_norm[ri, ci2]
        cnt = cm[ri, ci2]
        color = 'white' if val > 0.55 else GREY_DARK
        ax.text(ci2, ri, f'{val*100:.1f}%\n({cnt})',
                ha='center', va='center', fontsize=8.5,
                color=color, fontweight='bold' if ri == ci2 else 'normal')

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(hyb_labels, fontsize=11)
ax.set_yticklabels(hyb_labels, fontsize=11)
ax.set_xlabel('MolCrysKit prediction', fontsize=11, labelpad=6)
ax.set_ylabel('RDKit reference labels', fontsize=11, labelpad=6)
ax.set_title('(a) Confusion matrix (normalized by row)', fontsize=11, pad=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction')

# Right: per-element accuracy bar with CI
# Exclude F: terminal sp3 by definition (trivial result, n=16 too small)
ax2 = axes[1]
PLOT_ELEMENTS = [e for e in sorted(per_element.keys()) if e != 'F']
elements = PLOT_ELEMENTS
accs  = [per_element[e]['accuracy']*100 for e in elements]
lo    = [per_element[e]['accuracy']*100 - ci_per_elem[e][0]*100 for e in elements]
hi    = [ci_per_elem[e][1]*100 - per_element[e]['accuracy']*100 for e in elements]
ns    = [per_element[e]['total'] for e in elements]

x = np.arange(len(elements))
HATCHES = ['///', '\\\\\\', '...']
bars = ax2.bar(x, accs, color=GREY_MID, width=0.55, zorder=3,
               edgecolor=GREY_DARK, linewidth=0.8)
for bar, hatch in zip(bars, HATCHES):
    bar.set_hatch(hatch)
ax2.errorbar(x, accs, yerr=[lo, hi], fmt='none',
             ecolor=GREY_DARK, capsize=4, capthick=1.0, linewidth=1.0, zorder=4)

# Overall accuracy line + text annotation (no legend)
ax2.axhline(accuracy*100, color=RED_ERROR, linestyle='--', linewidth=1.0, zorder=5)
ax2.text(x[-1] + 0.35, accuracy*100 + 0.3, f'Overall {accuracy*100:.1f}%',
         ha='right', va='bottom', fontsize=8.5, color=RED_ERROR)

# Labels on bars
for xi, (acc, n, bar) in enumerate(zip(accs, ns, bars)):
    ax2.text(xi, acc + hi[xi] + 0.5, f'{acc:.1f}%',
             ha='center', va='bottom', fontsize=9, color=GREY_DARK)
    ax2.text(xi, 87, f'$\\it{{n}}$\u2009=\u2009{n:,}',
             ha='center', va='bottom', fontsize=8, color='white',
             fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(elements, fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_ylim(80, 104)
ax2.set_xlabel('Element', fontsize=11, labelpad=6)
ax2.set_title('(b) Per-element accuracy with 95% CI', fontsize=11, pad=8)
ax2.yaxis.grid(True, linestyle=':', color=GREY_BG, zorder=0)

# Panel (c): Aromatic vs non-aromatic accuracy
ax3 = axes[2]
n_nonarom_atoms = n_atoms - n_arom_atoms
n_nonarom_correct = n_correct - n_arom_correct
nonarom_acc = n_nonarom_correct / n_nonarom_atoms * 100 if n_nonarom_atoms > 0 else 0
ci_nonarom = wilson_ci(n_nonarom_correct, n_nonarom_atoms) if n_nonarom_atoms > 0 else (0, 0)

cat_labels = ['Non-aromatic', 'Aromatic']
cat_accs = [nonarom_acc, arom_acc * 100]
cat_lo = [nonarom_acc - ci_nonarom[0] * 100, arom_acc * 100 - ci_aromatic[0] * 100]
cat_hi = [ci_nonarom[1] * 100 - nonarom_acc, ci_aromatic[1] * 100 - arom_acc * 100]
cat_ns = [n_nonarom_atoms, n_arom_atoms]
cat_colors = [GREY_MID, GREY_DARK]

xc = np.arange(len(cat_labels))
bars_c = ax3.bar(xc, cat_accs, color=cat_colors, width=0.50, zorder=3)
ax3.errorbar(xc, cat_accs, yerr=[cat_lo, cat_hi], fmt='none',
             ecolor='#222222', capsize=4, capthick=1.0, linewidth=1.0, zorder=4)

# Overall accuracy line + text annotation (no legend)
ax3.axhline(accuracy * 100, color=RED_ERROR, linestyle='--', linewidth=1.0, zorder=5)
ax3.text(0.5, accuracy * 100 + 0.3, f'Overall {accuracy * 100:.1f}%',
         ha='center', va='bottom', fontsize=8.5, color=RED_ERROR)

# Labels on bars
for xi, (acc_c, n_c, bar_c) in enumerate(zip(cat_accs, cat_ns, bars_c)):
    yerr_hi = cat_hi[xi]
    ax3.text(xi, acc_c + yerr_hi + 0.5, f'{acc_c:.1f}%',
             ha='center', va='bottom', fontsize=9, color=GREY_DARK)
    ax3.text(xi, 87, f'$\\it{{n}}$\u2009=\u2009{n_c:,}',
             ha='center', va='bottom', fontsize=8, color='white',
             fontweight='bold')

ax3.set_xticks(xc)
ax3.set_xticklabels(cat_labels, fontsize=10)
ax3.set_ylabel('Accuracy (%)', fontsize=11)
ax3.set_ylim(80, 104)
ax3.set_xlabel('Atom category', fontsize=11, labelpad=6)
ax3.set_title('(c) Aromatic atom accuracy', fontsize=11, pad=8)
ax3.yaxis.grid(True, linestyle=':', color=GREY_BG, zorder=0)

plt.tight_layout(pad=1.5)
for ext in ('pdf', 'png'):
    out_path = os.path.join(RESULTS_DIR, f'qm9_confusion_matrix.{ext}')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved: {out_path}")
plt.close()


# ── 3. Failure analysis: wrong predictions ────────────────────────────────────
# Collect unique SMILES with errors from failure_cases
error_cases = []
for fc in failure_cases:
    smi = fc['smiles']
    for wa in fc['wrong_atoms']:
        error_cases.append({
            'smiles': smi,
            'element': wa['element'],
            'true': wa['rdkit_hyb'],
            'pred': wa['mck_hyb'],
        })

# Get unique error SMILES
error_smiles_set = list(dict.fromkeys(fc['smiles'] for fc in failure_cases))
print(f"\nUnique SMILES with prediction errors: {len(error_smiles_set)}")

# Categorize errors
categories = {
    'O sp3→sp2 (strained ether)': [],
    'N sp2→sp3 (distorted amide/imine)': [],
    'C sp2→sp3 (bicyclic)': [],
    'Other': [],
}
for fc in failure_cases:
    for wa in fc['wrong_atoms']:
        e, t, p = wa['element'], wa['rdkit_hyb'], wa['mck_hyb']
        key = None
        if e == 'O' and t == 'sp3' and p == 'sp2':
            key = 'O sp3→sp2 (strained ether)'
        elif e == 'N' and t == 'sp2' and p == 'sp3':
            key = 'N sp2→sp3 (distorted amide/imine)'
        elif e == 'C' and t == 'sp2' and p == 'sp3':
            key = 'C sp2→sp3 (bicyclic)'
        else:
            key = 'Other'
        if fc['smiles'] not in categories[key]:
            categories[key].append(fc['smiles'])

print("\nError categories:")
for cat, smis in categories.items():
    print(f"  {cat}: {len(smis)} cases")


def smiles_to_img(smi, size=(200, 150), highlight_atoms=None, highlight_color=None):
    """Render SMILES to PIL Image."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().addStereoAnnotation = False
    if highlight_atoms and highlight_color:
        atom_colors = {i: highlight_color for i in highlight_atoms}
        drawer.DrawMolecule(mol,
                            highlightAtoms=highlight_atoms,
                            highlightAtomColors=atom_colors)
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = io.BytesIO(drawer.GetDrawingText())
    return Image.open(bio).copy()


# ── Figure: Failure Analysis ──────────────────────────────────────────────────
# Show error cases with error-type annotations (no verdict badges — all are
# definition divergences between topology-based and geometry-based hybridization)

def hyb_label(h):
    """Return compact plain-text hybridization labels for panel annotations."""
    return h if h is not None else 'None'

SMILES_FONT = 'Consolas'
MAX_ERROR_MOLECULES_SHOWN = 20

all_error_smiles = error_smiles_set[:MAX_ERROR_MOLECULES_SHOWN]
n_show = len(all_error_smiles)
ncols = 4
nrows = max(1, math.ceil(n_show / ncols))

# Build annotation dict: smiles → list of wrong-atom records
annot = {}
for fc in failure_cases:
    smi = fc['smiles']
    if smi not in annot:
        annot[smi] = []
    for wa in fc['wrong_atoms']:
        annot[smi].append(wa)

# Two stacked regions per panel: a compact text strip above the molecule image.
fig2 = plt.figure(figsize=(ncols * 3.0, nrows * 2.78))
outer_gs = gridspec.GridSpec(
    nrows, ncols, figure=fig2,
    hspace=0.22, wspace=0.18,
    bottom=0.03, top=0.91,
)

for idx in range(nrows * ncols):
    ri, ci2 = divmod(idx, ncols)
    inner_gs = outer_gs[ri, ci2].subgridspec(
        2, 1, height_ratios=[0.34, 1.66], hspace=0.01
    )
    ax_text = fig2.add_subplot(inner_gs[0, 0])
    ax_img = fig2.add_subplot(inner_gs[1, 0])
    ax_text.axis('off')
    ax_img.axis('off')

    if idx >= n_show:
        continue

    smi = all_error_smiles[idx]

    # Collect error atom indices for highlighting
    errs = annot.get(smi, [])
    highlight_indices = [wa['atom_index'] for wa in errs if 'atom_index' in wa]

    if highlight_indices:
        img = smiles_to_img(
            smi,
            size=(240, 170),
            highlight_atoms=highlight_indices,
            highlight_color=(0.7, 0.7, 0.7, 0.45),
        )
    else:
        img = smiles_to_img(smi, size=(240, 170))

    if img is None:
        ax_text.text(
            0.5, 0.7, 'SMILES parsing error',
            ha='center', va='center', fontsize=8,
            color=RED_ERROR, transform=ax_text.transAxes,
        )
        continue

    ax_img.imshow(img)
    ax_img.set_anchor('N')

    # ── Build compact annotation text ─────────────────────────────────────────
    err_parts = []
    for wa in errs[:3]:
        err_parts.append(
            f"{wa['element']}: RDKit {hyb_label(wa['rdkit_hyb'])} | "
            f"MCK {hyb_label(wa['mck_hyb'])}"
        )
    err_str = '\n'.join(err_parts)
    if len(errs) > 3:
        err_str += f'\n(+{len(errs)-3} more)'

    disp_smi = smi if len(smi) <= 24 else smi[:22] + '…'

    ax_text.text(
        0.5, 0.95, disp_smi,
        ha='center', va='top',
        fontsize=7.6, color=GREY_DARK,
        fontfamily=SMILES_FONT,
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.5, 0.44, err_str,
        ha='center', va='top',
        fontsize=7.0, color=GREY_DARK,
        linespacing=1.0,
        transform=ax_text.transAxes,
    )

# ── Subtitle note ─────────────────────────────────────────────────────────────
n_total_err_atoms = n_atoms - n_correct
n_total_err_mols = len(error_smiles_set)
fig2.suptitle(
    'Representative QM9 molecules containing hybridization discrepancies\n'
    f'({n_show} of {n_total_err_mols} error-containing molecules shown; '
    f'{n_total_err_atoms} discrepant atoms in total)',
    fontsize=11, y=0.97, color=GREY_DARK
)
for ext in ('pdf', 'png'):
    out_path = os.path.join(RESULTS_DIR, f'qm9_failure_analysis.{ext}')
    fig2.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {out_path}")
plt.close()


# ── Figure: Excluded molecules ────────────────────────────────────────────────
if failed_smiles:
    n_fail_show = min(len(failed_smiles), 30)
    ncols_f = min(5, n_fail_show)
    nrows_f = math.ceil(n_fail_show / ncols_f) if ncols_f else 1
    fig3, axs3 = plt.subplots(nrows_f, ncols_f,
                               figsize=(ncols_f * 3.0, nrows_f * 3.2))
    axs3_flat = np.array(axs3).flatten() if n_fail_show > 1 else [axs3]
    for idx, smi in enumerate(failed_smiles[:n_fail_show]):
        ax = axs3_flat[idx]
        img = smiles_to_img(smi, size=(280, 200))
        if img is not None:
            ax.imshow(img)
        ax.axis('off')
        disp_smi = smi if len(smi) <= 26 else smi[:24] + '…'
        ax.set_title(disp_smi, fontsize=8, pad=3, color=RED_ERROR)

    # Turn off remaining axes
    for idx in range(n_fail_show, len(axs3_flat)):
        axs3_flat[idx].axis('off')

    fig3.suptitle(
        f'QM9 molecules excluded from validation ($\\it{{n}}$\u2009=\u2009{len(failed_smiles)})\n'
        'Charged species (zwitterions) — formal charges incompatible with neutral-molecule pipeline',
        fontsize=10.5, y=1.03, color=GREY_DARK
    )
    plt.tight_layout(pad=1.0)
    for ext in ('pdf', 'png'):
        out_path = os.path.join(RESULTS_DIR, f'qm9_excluded_molecules.{ext}')
        fig3.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {out_path}")
    plt.close()
else:
    print("No failed molecules — skipping excluded molecules figure.")


# ── Summary print ────────────────────────────────────────────────────────────
print("\n=== Summary for manuscript ===")
print(f"Overall accuracy: {accuracy*100:.1f}% "
      f"[95% CI: {ci_overall[0]*100:.1f}%–{ci_overall[1]*100:.1f}%]")
for elem in sorted(per_element.keys()):
    acc = per_element[elem]['accuracy']
    ci  = ci_per_elem[elem]
    n   = per_element[elem]['total']
    print(f"  {elem} (n={n:4d}): {acc*100:.1f}% "
          f"[{ci[0]*100:.1f}%–{ci[1]*100:.1f}%]")
print(f"\nExcluded molecules: {n_failed}/{n_total} "
      f"({n_failed/n_total*100:.1f}%)")
print(f"Prediction errors: {n_atoms - n_correct}/{n_atoms} atoms "
      f"({(1-accuracy)*100:.1f}%)")
if n_arom_atoms > 0:
    print(f"\nAromatic atoms: {n_arom_correct}/{n_arom_atoms} = {arom_acc*100:.1f}% "
          f"[95% CI: {ci_aromatic[0]*100:.1f}%–{ci_aromatic[1]*100:.1f}%] "
          f"(in {n_arom_mols} aromatic molecules)")
print("\nError breakdown:")
for cat, smis in categories.items():
    if smis:
        print(f"  {cat}: {len(smis)} molecules")
print("\nAll figures saved to:", RESULTS_DIR)
