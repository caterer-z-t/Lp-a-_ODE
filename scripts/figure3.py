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
from scripts.plot_style import apply_style, panel_label

apply_style()
os.makedirs("figures", exist_ok=True)

# ── Get posterior median parameters ───────────────────────────────────────────

print("Running Bayesian inference for posterior medians …")
trace = run_inference(params, OBS_N, OBS_PR, OBS_ERR)
result = posterior_curves(trace, params, N_ARR)
params_post = result["params_posterior"]

# ── Compute sensitivity matrices ───────────────────────────────────────────────

print("Computing sensitivity matrices …")
S_P = compute_sensitivity(N_ARR, params_post, output_key="P")
S_E = compute_sensitivity(N_ARR, params_post, output_key="E")
S_eta = compute_sensitivity(N_ARR, params_post, output_key="eta")

S_MATS = [S_P, S_E, S_eta]
OUTPUT_LABELS = [
    r"$P_{ss}(n)$  — Plasma Lp(a)",
    r"$E_{ss}(n)$  — ER apo(a) load",
    r"$\eta_{ER}(n)$  — ER secretion efficiency",
]

# ── Diverging Colormap ────────────────────────────────────────
_neg_colors = ["#0F2342", "#1C3E6E", "#2A5C7E", "#4C84A0", "#92B8CA"]
_pos_colors = ["#A3CBB1", "#5C9E74", "#296B45", "#184A2C", "#0D2E1A"]

cmap = LinearSegmentedColormap.from_list(
    "coolors_diverging", _neg_colors + ["#FFFFFF"] + _pos_colors, N=256
)

all_vals = np.concatenate([s.flatten() for s in S_MATS])
abs_max = np.ceil(np.nanpercentile(np.abs(all_vals), 95) * 10) / 10

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"wspace": 0.08})

ims = []
for col, (ax, S_mat, col_label) in enumerate(zip(axes, S_MATS, OUTPUT_LABELS)):
    im = ax.imshow(
        S_mat,
        aspect="auto",
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        interpolation="nearest",
    )
    ims.append(im)

    # Subpanel Title
    # ax.set_title(col_label, fontsize=10, pad=10, fontweight="bold", color="#333333")

    tick_pos = [
        i for i, n in enumerate(N_ARR) if n in [1, 5, 10, 15, 20, 25, 30, 35, 40]
    ]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(N_ARR[i]) for i in tick_pos], fontsize=8)

    ax.set_yticks(range(len(PARAM_KEYS)))

    # Y-axis labels only on first panel
    if col == 0:
        ax.set_yticklabels(
            [PARAM_LABELS[pk] for pk in PARAM_KEYS],
            fontsize=9,
            rotation=0,
            va="center",
            ha="right",
        )
    else:
        ax.set_yticklabels([])

    panel_label(ax, f"({chr(97 + col)})", x=-0.02, y=1.08, size=11)

# Colorbar
cbar = fig.colorbar(
    ims[2], ax=axes[2], fraction=0.035, pad=0.03, orientation="vertical", extend="both"
)
cbar.set_label(r"Sensitivity  $S(Y;\,k_i)$", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
cbar.set_ticks(ticks)
cbar.set_ticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"])

fig.supxlabel("KIV-2 copy number", fontsize=10, color="#444444", y=0.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig("figures/fig3_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
# plt.show()
print("Figure 3 saved.")
