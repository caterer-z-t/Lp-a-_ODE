"""
Figure 2 — Bayesian prior-to-posterior update for ERAD kinetics.

Panels:
    (a) Marginal posterior for lambda (ERAD per-repeat susceptibility)
    (b) Marginal posterior for k_fold0 (ER folding rate)
    (c) Joint 2-D posterior (lambda, k_fold0)
    (d) P_ss(n)/P_ss(14): prior vs posterior median + credible envelopes
    (e) k_ERAD(n) on log scale: prior vs posterior 95% CI
    (f) ER secretion efficiency eta_ER(n): prior vs posterior 95% CI
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lpa_model import params, OBS_N, OBS_PR, OBS_ERR
from lpa_model.model import steady_state, k_fold, k_ERAD_exp, chi2_gof
from lpa_model.inference import run_inference, posterior_curves, truncnorm_samples
from lpa_model.parameters import N_ARR, REF_N
from scripts.plot_style import apply_style, COLORS, panel_label, C

apply_style()
os.makedirs("figures", exist_ok=True)

# ── Color Palette Definitions for Figure 2 ─────────────────────────────────────
COLOR_PRIOR = COLORS[3]
COLOR_POST_LAM = COLORS[7]
COLOR_POST_KF = COLORS[6]
COLOR_POST_MAIN = COLORS[9]

# ── Panel (c) ───────────────
_COOLORS_CMAP = LinearSegmentedColormap.from_list(
    "reflecting_pool_coolors",
    [
        "#122C3D",
        "#22577A",
        "#38A3A5",
        "#57CC99",
        "#80ED99",
        "#C7F9CC",
    ],
    N=256,
)

# ── Run inference ──────────────────────────────────────────────────────────────

print("Running Bayesian inference …")
trace = run_inference(params, OBS_N, OBS_PR, OBS_ERR)
result = posterior_curves(trace, params, N_ARR, ref_n=REF_N)

post_lam = result["post_lam"]
post_kfold = result["post_k_fold0"]
lam_med = result["lam_med"]
kfold_med = result["kfold_med"]
params_post = result["params_posterior"]

# ── Prior samples for overlay ─────────────────────────────────────────────────

prior_lam = truncnorm_samples(mu=0.07, sigma=0.04, lower=0.01, upper=0.25)
prior_kfold = truncnorm_samples(mu=2.0, sigma=1.0, lower=0.2, upper=8.0)

# ── Reference curves ──────────────────────────────────────────────────────────

ref_ss_prior = steady_state(REF_N, params)
ref_ss_post = steady_state(REF_N, params_post)

P_prior_curve = np.array(
    [steady_state(int(n), params)["P"] / ref_ss_prior["P"] for n in N_ARR]
)
P_post_curve = np.array(
    [steady_state(int(n), params_post)["P"] / ref_ss_post["P"] for n in N_ARR]
)

kerad_prior = np.array([params["k_e"] * np.exp(params["lam"] * n) for n in N_ARR])
kerad_post = np.array([params["k_e"] * np.exp(lam_med * n) for n in N_ARR])

kfold_prior_arr = np.array([k_fold(n, params) for n in N_ARR])
kfold_post_arr = np.array([k_fold(n, params_post) for n in N_ARR])

eta_prior = kfold_prior_arr / (kfold_prior_arr + kerad_prior)
eta_post = kfold_post_arr / (kfold_post_arr + kerad_post)

# ── Posterior draw envelopes for panels (e) and (f) ──────────────────────────

rng = np.random.default_rng(1)
draw_idx = rng.choice(len(post_lam), 500, replace=False)

kerad_draws = np.array(
    [[params["k_e"] * np.exp(float(post_lam[i]) * n) for n in N_ARR] for i in draw_idx]
)
eta_draws = np.array(
    [
        [
            (kfold_med / n)
            / ((kfold_med / n) + params["k_e"] * np.exp(float(post_lam[i]) * n))
            for n in N_ARR
        ]
        for i in draw_idx
    ]
)

# ── GOF statistics ────────────────────────────────────────────────────────────

ref_p_prior = steady_state(REF_N, params)["P"]
ref_p_post = steady_state(REF_N, params_post)["P"]

prior_pred = np.array([steady_state(int(n), params)["P"] / ref_p_prior for n in OBS_N])
post_pred = np.array(
    [steady_state(int(n), params_post)["P"] / ref_p_post for n in OBS_N]
)

chi2_prior, p_prior = chi2_gof(OBS_PR, prior_pred, OBS_ERR)
chi2_post, p_post = chi2_gof(OBS_PR, post_pred, OBS_ERR)
print(f"GOF — Prior:  chi2={chi2_prior:.2f}, p={p_prior:.3f}")
print(f"GOF — Posterior:  chi2={chi2_post:.2f}, p={p_post:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    3, 2, figsize=(9, 12), gridspec_kw={"hspace": 0.42, "wspace": 0.35}
)

# (a) lambda prior vs posterior
ax = axes[0, 0]
bins_lam = np.linspace(0.01, 0.25, 55)
ax.hist(
    prior_lam, bins=bins_lam, density=True, alpha=0.45, color=COLOR_PRIOR, label="Prior"
)
ax.hist(
    post_lam,
    bins=bins_lam,
    density=True,
    alpha=0.75,
    color=COLOR_POST_LAM,
    label="Posterior",
)
ax.axvline(params["lam"], color=COLOR_PRIOR, lw=1.8, ls="--")
ax.axvline(lam_med, color=COLOR_POST_LAM, lw=1.8, ls="-")
lo, hi = np.percentile(post_lam, [2.5, 97.5])
ax.axvspan(
    lo, hi, alpha=0.15, color=COLOR_POST_LAM, label=f"95% CI [{lo:.3f}, {hi:.3f}]"
)
ax.set_xlabel(r"$\lambda$  [hr$^{-1}$ repeat$^{-1}$]", fontsize=10)
ax.set_ylabel("Density")
ax.legend(fontsize=9)
panel_label(ax, "(a)", size=12)

# (b) k_fold0 prior vs posterior
ax = axes[0, 1]
bins_kf = np.linspace(0.2, 8.0, 55)
ax.hist(
    prior_kfold,
    bins=bins_kf,
    density=True,
    alpha=0.45,
    color=COLOR_PRIOR,
    label="Prior",
)
ax.hist(
    post_kfold,
    bins=bins_kf,
    density=True,
    alpha=0.75,
    color=COLOR_POST_KF,
    label="Posterior",
)
ax.axvline(params["k_fold0"], color=COLOR_PRIOR, lw=1.8, ls="--")
ax.axvline(kfold_med, color=COLOR_POST_KF, lw=1.8, ls="-")
lo_kf, hi_kf = np.percentile(post_kfold, [2.5, 97.5])
ax.axvspan(
    lo_kf,
    hi_kf,
    alpha=0.15,
    color=COLOR_POST_KF,
    label=f"95% CI [{lo_kf:.2f}, {hi_kf:.2f}]",
)
ax.set_xlabel(r"$k_{fold,0}$  [hr$^{-1}$]", fontsize=10)
ax.set_ylabel("Density")
ax.legend(fontsize=9)
panel_label(ax, "(b)", size=12)

# (c) Joint posterior
ax = axes[1, 0]
h = ax.hist2d(
    post_lam, post_kfold, bins=45, cmap=_COOLORS_CMAP, density=True
)
cbar = fig.colorbar(h[3], ax=ax, fraction=0.04, pad=0.03)
cbar.set_label("Posterior density", fontsize=9)
cbar.ax.tick_params(labelsize=7)
ax.scatter(
    params["lam"],
    params["k_fold0"],
    marker="*",
    s=160,
    color="#E67E22",
    edgecolor="white",
    lw=0.8,
    zorder=5,
    label="Prior point est.",
)
ax.scatter(
    lam_med,
    kfold_med,
    marker="D",
    s=60,
    color="white",
    edgecolor="#122C3D",
    lw=1.2,
    zorder=5,
    label="Posterior median",
)
ax.set_xlabel(r"$\lambda$  [hr$^{-1}$ repeat$^{-1}$]", fontsize=10)
ax.set_ylabel(r"$k_{fold,0}$  [hr$^{-1}$]")
ax.legend(fontsize=8, loc="upper right", facecolor="white", framealpha=0.9)
panel_label(ax, "(c)", size=12)

# (d) P_ss(n) model comparison
ax = axes[1, 1]
ax.fill_between(
    N_ARR,
    result["P_lo95"],
    result["P_hi95"],
    alpha=0.15,
    color=COLOR_POST_MAIN,
    label="Posterior 95% CI",
)
ax.fill_between(
    N_ARR,
    result["P_lo50"],
    result["P_hi50"],
    alpha=0.30,
    color=COLOR_POST_MAIN,
    label="Posterior 50% CI",
)
ax.plot(N_ARR, P_post_curve, color=COLOR_POST_MAIN, lw=2.2, label="Posterior median")
ax.plot(N_ARR, P_prior_curve, color=COLOR_PRIOR, lw=1.8, ls="--", label="Prior model")
ax.errorbar(
    OBS_N,
    OBS_PR,
    yerr=OBS_ERR,
    fmt="o",
    color=C["data"],
    ms=5.5,
    capsize=3.5,
    zorder=6,
    label="Rader 1994",
)
ax.set_yscale("log")
ax.set_xlabel("KIV-2 copy number", fontsize=10)
ax.set_ylabel(r"ln(Relative plasma Lp(a))")
ax.legend(fontsize=8, loc="upper right")
ax.text(
    0.02,
    0.05,
    f"GOF:\nPrior: χ²={chi2_prior:.2f}, p={p_prior:.3f}\nPosterior: χ²={chi2_post:.2f}, p={p_post:.3f}",
    transform=ax.transAxes,
    fontsize=8,
    va="bottom",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.9
    ),
)
panel_label(ax, "(d)", size=12)

# (e) k_ERAD(n)
ax = axes[2, 0]
ax.semilogy(
    N_ARR,
    kerad_prior,
    color=COLOR_PRIOR,
    lw=2,
    ls="--",
    label=rf"Prior ($\lambda$={params['lam']:.2f})",
)
ax.semilogy(
    N_ARR,
    kerad_post,
    color=COLOR_POST_LAM,
    lw=2,
    label=rf"Posterior ($\lambda$={lam_med:.3f})",
)
ax.fill_between(
    N_ARR,
    np.percentile(kerad_draws, 2.5, axis=0),
    np.percentile(kerad_draws, 97.5, axis=0),
    alpha=0.15,
    color=COLOR_POST_LAM,
    label="Posterior 95% CI",
)
ax.set_xlabel("KIV-2 copy number", fontsize=10)
ax.set_ylabel(r"ln($k_{ERAD}(n)$)  [hr$^{-1}$]")
ax.legend(fontsize=8)
panel_label(ax, "(e)", size=12)

# (f) eta_ER(n)
ax = axes[2, 1]
ax.plot(N_ARR, eta_prior * 100, color=COLOR_PRIOR, lw=2, ls="--", label="Prior")
ax.plot(N_ARR, eta_post * 100, color=COLOR_POST_KF, lw=2, label="Posterior")
ax.fill_between(
    N_ARR,
    np.percentile(eta_draws * 100, 2.5, axis=0),
    np.percentile(eta_draws * 100, 97.5, axis=0),
    alpha=0.15,
    color=COLOR_POST_KF,
    label="Posterior 95% CI",
)
ax.set_ylim(0, 105)
ax.set_xlabel("KIV-2 copy number", fontsize=10)
ax.set_ylabel(r"$\eta_{ER}(n)$  [%]")
ax.legend(fontsize=8)
panel_label(ax, "(f)", size=12)

plt.savefig("figures/fig2_prior_posterior_comparison.png", dpi=300, bbox_inches="tight")
# plt.show()
print("Figure 2 saved.")
