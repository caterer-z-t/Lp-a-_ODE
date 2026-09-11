"""
Bayesian inference for ERAD kinetic parameters.

Conditions on isoform-specific apo(a) production rate data from Rader et al.
(1994) to estimate the posterior distributions of lambda (ERAD per-repeat
susceptibility) and k_fold0 (ER folding rate constant).

Sampler: NUTS via PyMC v5, 4 chains × 2000 draws.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import truncnorm


# ── PyTensor steady-state expression (required for PyMC gradient computation) ─

def _P_ss_pytensor(n_val: float, lam, k_fold0, p: dict):
    """Closed-form P_ss using PyTensor ops for a scalar n (enables NUTS)."""
    kt  = p["alpha_tx"] / (p["N0_tx"]   + n_val)
    kl  = p["alpha_tl"] / (p["N0_prot"] + n_val)
    kf  = k_fold0 / n_val
    ke  = p["k_e"] * pt.exp(lam * n_val)
    x   = (n_val / p["K_n_assem"]) ** p["h_assem"]
    ka  = p["k_assem_max"] * x / (1.0 + x)
    kds = p["k_deg_S"]
    M   = kt / p["k_deg_m"]
    E   = kl * M / (kf + ke)
    S   = kf * E / (ka + kds)
    return ka * S / p["k_cl"]


# ── Prior sampling helper ─────────────────────────────────────────────────────

def truncnorm_samples(
    mu: float, sigma: float, lower: float, upper: float,
    n: int = 8000, seed: int = 0,
) -> np.ndarray:
    """Draw samples from a truncated normal matching the PyMC prior spec."""
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=seed)


# ── Main inference function ───────────────────────────────────────────────────

def run_inference(
    p: dict,
    obs_n: np.ndarray,
    obs_pr: np.ndarray,
    obs_err: np.ndarray,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.92,
    random_seed: int = 42,
):
    """
    Run NUTS sampling to estimate lambda and k_fold0.

    Priors
    ------
    lambda   ~ TruncNormal(mu=0.07, sigma=0.04, bounds=[0.01, 0.25])
    k_fold0  ~ TruncNormal(mu=2.0,  sigma=1.0,  bounds=[0.2,  8.0])

    Likelihood
    ----------
    obs_pr ~ Normal(mu = P_ss(n) / P_ss(14), sigma = obs_err)

    Returns
    -------
    trace : arviz.InferenceData
        Full posterior trace with convergence diagnostics.
    """
    ref_n = 14.0

    with pm.Model() as model:
        lam     = pm.TruncatedNormal("lam",     mu=0.07, sigma=0.04,
                                     lower=0.01, upper=0.25)
        k_fold0 = pm.TruncatedNormal("k_fold0", mu=2.0,  sigma=1.0,
                                     lower=0.2,  upper=8.0)

        P_ref  = _P_ss_pytensor(ref_n, lam, k_fold0, p)
        P_pred = pt.stack([
            _P_ss_pytensor(float(n), lam, k_fold0, p) / P_ref
            for n in obs_n
        ])

        pm.Normal("obs", mu=P_pred, sigma=obs_err, observed=obs_pr)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=chains,
            cores=1,
            progressbar=True,
            random_seed=random_seed,
        )

    return trace


# ── Posterior predictive curves ───────────────────────────────────────────────

def posterior_curves(
    trace,
    p: dict,
    n_arr: np.ndarray,
    ref_n: int = 14,
    n_draws: int = 500,
    seed: int = 0,
) -> dict:
    """
    Sample posterior predictive envelopes for P_ss(n) / P_ss(ref_n).

    Parameters
    ----------
    trace : arviz.InferenceData from run_inference().
    p : base parameter dict (non-inferred parameters).
    n_arr : KIV-2 copy numbers to evaluate.
    ref_n : normalization isoform.
    n_draws : number of posterior draws for credible envelopes.
    seed : random seed for draw selection.

    Returns
    -------
    dict with keys:
        post_lam, post_k_fold0  — flattened posterior chains
        P_med, P_lo50, P_hi50, P_lo95, P_hi95  — percentile envelopes
        lam_med, kfold_med      — posterior medians
        params_posterior        — updated parameter dict at posterior medians
    """
    from .model import steady_state

    post_lam    = trace.posterior["lam"].values.flatten()
    post_kfold  = trace.posterior["k_fold0"].values.flatten()

    lam_med   = float(np.median(post_lam))
    kfold_med = float(np.median(post_kfold))
    params_post = {**p, "lam": lam_med, "k_fold0": kfold_med}

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post_lam), size=n_draws, replace=False)

    curves = np.zeros((n_draws, len(n_arr)))
    for i, di in enumerate(idx):
        p_draw = {**p, "lam": float(post_lam[di]), "k_fold0": float(post_kfold[di])}
        ref_p  = steady_state(ref_n, p_draw)["P"]
        curves[i] = [steady_state(int(n), p_draw)["P"] / ref_p for n in n_arr]

    return {
        "post_lam":        post_lam,
        "post_k_fold0":    post_kfold,
        "lam_med":         lam_med,
        "kfold_med":       kfold_med,
        "params_posterior": params_post,
        "P_med":  np.median(curves, axis=0),
        "P_lo50": np.percentile(curves, 25,  axis=0),
        "P_hi50": np.percentile(curves, 75,  axis=0),
        "P_lo95": np.percentile(curves, 2.5, axis=0),
        "P_hi95": np.percentile(curves, 97.5, axis=0),
    }
