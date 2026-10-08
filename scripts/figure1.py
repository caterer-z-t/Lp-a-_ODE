"""
Figure 1 — Steady-state model behavior across KIV-2 repeat number.

Panels:
    A  Normalized plasma Lp(a) P_ss(n)/P_ss(14) vs Rader et al. 1994
    B  ER secretion efficiency η_ER(n)
    C  Assembly efficiency η_a(n)
    D  Normalized ER apo(a) load E_ss(n)/E_ss(14)  [log scale]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lpa_model import params, OBS_N, OBS_PR, OBS_ERR
from lpa_model.model import steady_state, chi2_gof
from lpa_model.parameters import N_ARR, REF_N
from scripts.plot_style import apply_style, C, panel_label

apply_style()
os.makedirs("figures", exist_ok=True)

# ── Compute steady states ──────────────────────────────────────────────────────

ref_ss = steady_state(REF_N, params)
ss_all = [steady_state(n, params) for n in N_ARR]

P_ss      = np.array([s["P"]         for s in ss_all])
E_ss      = np.array([s["E"]         for s in ss_all])
eta       = np.array([s["eta"]       for s in ss_all])
eta_assem = np.array([s["eta_assem"] for s in ss_all])

P_norm = P_ss / ref_ss["P"]
E_norm = E_ss / ref_ss["E"]

prior_pred = np.array([steady_state(int(n), params)["P"] / ref_ss["P"] for n in OBS_N])
chi2_stat, p_val = chi2_gof(OBS_PR, prior_pred, OBS_ERR)

# ── Layout — 6.7" × 5.0" targets Cell two-column (170 mm) width ──────────────

fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.0),
                         constrained_layout=True)

XLABEL = r"KIV-2 copy number ($n$)"
XLIM   = (0.5, 40.5)
XTICKS = [1, 5, 10, 15, 20, 25, 30, 35, 40]

for ax in axes.flat:
    ax.set_xlim(XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xlabel(XLABEL)

# ── A: Plasma Lp(a) vs Rader 1994 ────────────────────────────────────────────

ax = axes[0, 0]
ax.plot(N_ARR, P_norm, "o-",
        color=C["plasma_lpa"], ms=3.5, lw=2, zorder=3,
        label="ODE model")
ax.errorbar(OBS_N, OBS_PR, yerr=OBS_ERR, 
            fmt="ks", color=C["data"], ms=4,
            capsize=4,
            zorder=5, label="Rader et al. (1994)")
# ax.set_ylabel(r"$P_{ss}(n)\,/\,P_{ss}(14)$")
ax.set_ylabel(r"$\hat{P}_{\mathrm{ss}}(n)$")
ax.legend(loc="upper right", borderpad=0.6)
ax.text(0.98, 0.15,
        rf"$\chi^2_{{(3)}} = {chi2_stat:.2f},\; p = {p_val:.3f}$",
        transform=ax.transAxes, fontsize=6.5,
        va="bottom", ha="right", color="#555555")
panel_label(ax, "(a)", size=12)

# ── B: ER secretion efficiency ────────────────────────────────────────────────

ax = axes[0, 1]
ax.plot(N_ARR, eta * 100, color=C["er_eff"], lw=1.5)
# ax.axhline(50, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
ax.set_ylabel(r"ER secretion efficiency (%)")
ax.set_ylim(-2, 105)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
panel_label(ax, "(b)", size=12)

# ── C: Assembly efficiency ────────────────────────────────────────────────────

ax = axes[1, 0]
ax.plot(N_ARR, eta_assem * 100, color=C["assem_eff"], lw=1.5)
# ax.axhline(50, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
ax.set_ylabel(r"Assembly efficiency (%)")
ax.set_ylim(-2, 105)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
panel_label(ax, "(c)", size=12)

# ── D: ER apo(a) load (log scale) ────────────────────────────────────────────

ax = axes[1, 1]
ax.semilogy(N_ARR, E_norm, color=C["er_load"], lw=1.5)
ax.set_ylabel(r"$\log(\hat{E}_{\mathrm{ss}}(n))$")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, _: f"{y:g}"
))
panel_label(ax, "(d)", size=12)

plt.savefig("figures/fig1_pss_vs_n.png", dpi=300, bbox_inches="tight")
# plt.show()

print(f"Figure 1 saved.  GOF prior: chi2={chi2_stat:.2f}, p={p_val:.3f}")
