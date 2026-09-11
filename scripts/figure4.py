"""
Figure 4 — ERAD saturation and ER stress as a function of LPA expression level.

Uses the Michaelis-Menten ERAD model (steady_state_mm) to show how ER load
and plasma Lp(a) respond to changes in transcriptional output across isoforms.

Panels:
    (a) ER load (E_ss / K_M) vs relative LPA expression  [log-log]
    (b) Plasma Lp(a) vs relative LPA expression           [log-log]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lpa_model import params
from lpa_model.model import steady_state_mm
from scripts.plot_style import apply_style, COLORS

apply_style()
os.makedirs("figures", exist_ok=True)

N_BIF      = [10, 20, 30]
BIF_COLORS = ["steelblue", "tomato", "forestgreen"]
ATX_FAC    = np.logspace(-2, 2, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for n_b, col in zip(N_BIF, BIF_COLORS):
    E_arr, P_arr = [], []
    for fac in ATX_FAC:
        p_tmp = {**params, "alpha_tx": params["alpha_tx"] * fac}
        s = steady_state_mm(n_b, p_tmp)
        E_arr.append(s["E"])
        P_arr.append(s["P"])

    E_rel = np.array(E_arr) / params["K_M_ERAD"]
    axes[0].plot(ATX_FAC, E_rel,            color=col, label=f"$n={n_b}$")
    axes[1].plot(ATX_FAC, np.array(P_arr),  color=col, label=f"$n={n_b}$")

axes[0].axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.5,
                label=r"ERAD saturation  ($E = K_M$)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"Relative $LPA$ expression  ($\alpha_{tx}$ / baseline)")
axes[0].set_ylabel(r"ER load  ($E_{ss}\,/\,K_M$)")
axes[0].set_title("ER Stress vs LPA Expression Level")
axes[0].legend(fontsize=10)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel(r"Relative $LPA$ expression  ($\alpha_{tx}$ / baseline)")
axes[1].set_ylabel("Plasma Lp(a)  [a.u.]")
axes[1].set_title("Lp(a) Secretion vs LPA Expression Level")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("figures/fig4_erad_bifurcation.png", dpi=300, bbox_inches="tight")
# plt.show()
print("Figure 4 saved.")
