"""
Figure 3 — Normalized local sensitivity heatmaps.

Panels:
    (a) S(P_ss; k_i)     — plasma Lp(a)
    (b) S(E_ss; k_i)     — ER apo(a) load
    (c) S(eta_ER; k_i)   — ER secretion efficiency

Sensitivity evaluated at posterior median parameters.
Requires figure2.py to have been run (or pass cached posterior medians).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lpa_model import params, OBS_N, OBS_PR, OBS_ERR
from lpa_model.inference import run_inference, posterior_curves
from lpa_model.model import compute_sensitivity, PARAM_KEYS, PARAM_LABELS
from lpa_model.parameters import N_ARR
from scripts.plot_style import apply_style, COLORS, panel_label

apply_style()
os.makedirs("figures", exist_ok=True)

# ── Get posterior median parameters ───────────────────────────────────────────

print("Running Bayesian inference for posterior medians …")
trace   = run_inference(params, OBS_N, OBS_PR, OBS_ERR)
result  = posterior_curves(trace, params, N_ARR)
params_post = result["params_posterior"]

# ── Compute sensitivity matrices ───────────────────────────────────────────────

print("Computing sensitivity matrices …")
S_P   = compute_sensitivity(N_ARR, params_post, output_key="P")
S_E   = compute_sensitivity(N_ARR, params_post, output_key="E")
S_eta = compute_sensitivity(N_ARR, params_post, output_key="eta")

S_MATS = [S_P, S_E, S_eta]
OUTPUT_LABELS = [
    r"$P_{ss}(n)$  — Plasma Lp(a)",
    r"$E_{ss}(n)$  — ER apo(a) load",
    r"$\eta_{ER}(n)$  — ER secretion efficiency",
]

# ── Colormap ───────────────────────────────────────────────────────────────────

_neg = [COLORS[9], COLORS[8], COLORS[7], COLORS[6]]
_pos = [COLORS[4], COLORS[2], COLORS[1], COLORS[0]]
cmap = LinearSegmentedColormap.from_list(
    "lpa_sensitivity", _neg + ["#ffffff"] + _pos, N=256
)

all_vals  = np.concatenate([s.flatten() for s in S_MATS])
abs_max   = np.ceil(np.nanpercentile(np.abs(all_vals), 95) * 10) / 10

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                         gridspec_kw={"wspace": 0.05})

ims = []
for col, (ax, S_mat, col_label) in enumerate(zip(axes, S_MATS, OUTPUT_LABELS)):
    im = ax.imshow(
        S_mat, aspect="auto",
        cmap=cmap, vmin=-abs_max, vmax=abs_max,
        interpolation="nearest",
    )
    ims.append(im)

    tick_pos = [i for i, n in enumerate(N_ARR) if n in [1, 5, 10, 15, 20, 25, 30, 35, 40]]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(N_ARR[i]) for i in tick_pos], fontsize=9)

    ax.set_yticks(range(len(PARAM_KEYS)))

    # Update pram
    if col == 0:
        ax.set_yticklabels(
            [PARAM_LABELS[pk] for pk in PARAM_KEYS],
            fontsize=9, rotation=0, va="center", ha="right",
        )
    else:
        ax.set_yticklabels([])

    panel_label(ax, f"({'abc'[col]})", x=0.02, y=1.05, size=12)

cbar = fig.colorbar(ims[2], ax=axes[2], fraction=0.035, pad=0.03,
                    orientation="vertical", extend="both")
cbar.set_label(r"$S(Y;\,k_i)$", fontsize=9)
cbar.ax.tick_params(labelsize=8)
ticks = [-abs_max, -abs_max / 2, 0, abs_max / 2, abs_max]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{v:.1f}" for v in ticks])

fig.supxlabel("KIV-2 copy number", fontsize=11, fontweight="bold", color="gray")
plt.tight_layout()
plt.savefig("figures/fig3_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()
print("Figure 3 saved.")
