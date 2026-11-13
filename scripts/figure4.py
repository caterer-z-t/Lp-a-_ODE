import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lpa_model import OBS_ERR, OBS_N, OBS_PR, params
from lpa_model.inference import posterior_curves, run_inference
from lpa_model.model import PARAM_KEYS, PARAM_LABELS, compute_sensitivity
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
    r"$P_{ss}(n)$ — Plasma Lp(a)",
    r"$E_{ss}(n)$ — ER apo(a) load",
    r"$\eta_{ER}(n)$ — ER secretion efficiency",
]

# ── Diverging Colormap ────────────────────────────────────────────────────────
# Added a subtle off-white midpoint (#F5F7FA) to prevent near-zero values
# from washing out into white figure backgrounds.
_neg_colors = ["#0F2342", "#1C3E6E", "#2A5C7E", "#4C84A0", "#92B8CA"]
_pos_colors = ["#A3CBB1", "#5C9E74", "#296B45", "#184A2C", "#0D2E1A"]

cmap = LinearSegmentedColormap.from_list(
    "coolors_diverging", _neg_colors + ["#F5F7FA"] + _pos_colors, N=256
)

all_vals = np.concatenate([s.flatten() for s in S_MATS])
abs_max = np.ceil(np.nanpercentile(np.abs(all_vals), 95) * 10) / 10
if abs_max == 0:
    abs_max = 1.0  # Fallback guard

# ── Plot ───────────────────────────────────────────────────────────────────────

# # Fixed to 1 row, 3 columns (side-by-side)
# fig, axes = plt.subplots(
#     1, 3, figsize=(14, 4.8), sharey=True, gridspec_kw={"wspace": 0.08}
# )

# ims = []
# tick_pos = [i for i, n in enumerate(N_ARR) if n in [1, 5, 10, 15, 20, 25, 30, 35, 40]]

# for col, (ax, S_mat, col_label) in enumerate(zip(axes, S_MATS, OUTPUT_LABELS)):
#     im = ax.imshow(
#         S_mat,
#         aspect="auto",
#         cmap=cmap,
#         vmin=-abs_max,
#         vmax=abs_max,
#         interpolation="nearest",
#     )
#     ims.append(im)

#     # Subpanel Title
#     ax.set_title(col_label, fontsize=10, pad=10, fontweight="normal", color="#222222")

#     # X-axis formatting
#     ax.set_xticks(tick_pos)
#     ax.set_xticklabels([str(N_ARR[i]) for i in tick_pos], fontsize=8.5)
#     ax.set_xlabel("KIV-2 copy number ($n$)", fontsize=9, labelpad=6)

#     # Y-axis formatting
#     ax.set_yticks(range(len(PARAM_KEYS)))
#     if col == 0:
#         ax.set_yticklabels(
#             [PARAM_LABELS.get(pk, pk) for pk in PARAM_KEYS],
#             fontsize=9,
#             rotation=0,
#             va="center",
#             ha="right",
#         )
#     else:
#         ax.tick_params(axis="y", which="both", length=0)

#     # Panel tag (a), (b), (c)
#     panel_label(ax, f"({chr(97 + col)})", x=-0.03, y=1.06, size=11)

#     # Clean frame styling
#     for spine in ax.spines.values():
#         spine.set_color("#CCCCCC")
#         spine.set_linewidth(0.8)

# # Colorbar (Dedicated axis on the right side)
# cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
# cbar = fig.colorbar(ims[0], cax=cbar_ax, extend="both")
# cbar.set_label(r"Normalized Sensitivity  $S(Y;\,k_i)$", fontsize=9.5, labelpad=8)
# cbar.ax.tick_params(labelsize=8.5)

# # Dynamic tick positions based on abs_max
# cbar_ticks = np.linspace(-abs_max, abs_max, 5)
# cbar.set_ticks(cbar_ticks)
# cbar.set_ticklabels([f"{t:+.1f}" for t in cbar_ticks])

# plt.subplots_adjust(left=0.15, right=0.90, top=0.88, bottom=0.15)
# plt.savefig("figures/fig3_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
# print("Figure 3 saved successfully.")

# ── Plot (3 Rows × 1 Column Stacked with Flipped Y-Axis) ──────────────────────

fig, axes = plt.subplots(
    3, 1, figsize=(7, 10), sharex=True, gridspec_kw={"hspace": 0.25}
)

ims = []
tick_pos = [i for i, n in enumerate(N_ARR) if n in [1, 5, 10, 15, 20, 25, 30, 35, 40]]

for row, (ax, S_mat, col_label) in enumerate(zip(axes, S_MATS, OUTPUT_LABELS)):
    im = ax.imshow(
        S_mat.T,
        origin="lower",  # Flips Y-axis so n=1 starts at the bottom
        aspect="auto",
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        interpolation="nearest",
    )
    ims.append(im)

    # Subpanel Title
    ax.set_title(
        col_label, fontsize=10, pad=8, fontweight="normal", color="#222222", loc="left"
    )

    # X-axis: Model parameters (only show labels & title on bottom subplot)
    ax.set_xticks(range(len(PARAM_KEYS)))
    if row == len(axes) - 1:
        ax.set_xticklabels(
            [PARAM_LABELS.get(pk, pk) for pk in PARAM_KEYS],
            fontsize=8.5,
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlabel("Parameters ($k_i$)", fontsize=9, labelpad=6)

    # Y-axis: KIV-2 copy number (n) on all subplots
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(N_ARR[i]) for i in tick_pos], fontsize=8.5)
    ax.set_ylabel("KIV-2 copy number ($n$)", fontsize=9, labelpad=6)

    # Panel tag (a), (b), (c)
    panel_label(ax, f"({chr(97 + row)})", x=-0.08, y=1.06, size=11)

    # Clean frame styling
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.8)

# Colorbar (Dedicated axis on the right side)
cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.70])
cbar = fig.colorbar(ims[0], cax=cbar_ax, extend="both")
cbar.set_label(r"Normalized Sensitivity  $S(Y;\,k_i)$", fontsize=9.5, labelpad=8)
cbar.ax.tick_params(labelsize=8.5)

# Dynamic tick positions based on abs_max
cbar_ticks = np.linspace(-abs_max, abs_max, 5)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f"{t:+.1f}" for t in cbar_ticks])

plt.subplots_adjust(left=0.12, right=0.84, top=0.95, bottom=0.10)
plt.savefig("figures/fig4_sensitivity_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig4_sensitivity_heatmap.pdf", dpi=600, bbox_inches="tight")
print("Figure 4 saved successfully.")