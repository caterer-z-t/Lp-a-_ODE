"""
Four-compartment ODE model of hepatic apo(a) biosynthesis.

State variables (all in arbitrary units unless normalized):
    M   : cytoplasmic mRNA
    E   : ER-resident apo(a) protein
    S   : secreted extracellular apo(a)
    P   : circulating plasma Lp(a)

All rate parameters are explicit functions of KIV-2 copy number n.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats as sc

# ── Rate functions ─────────────────────────────────────────────────────────────

def k_tx(n: float, p: dict) -> float:
    """Transcription rate [hr^-1]; limited by RNAP elongation over (N0_tx + n) repeat-equivalents."""
    return p["alpha_tx"] / (p["N0_tx"] + n)


def k_tl(n: float, p: dict) -> float:
    """Translation rate per mRNA [hr^-1]; limited by ribosome transit over (N0_prot + n) kringles."""
    return p["alpha_tl"] / (p["N0_prot"] + n)


def k_fold(n: float, p: dict) -> float:
    """ER processing rate [hr^-1]; n kringle domains fold sequentially."""
    return p["k_fold0"] / n


def k_ERAD_exp(n: float, p: dict) -> float:
    """ERAD rate constant (exponential model) [hr^-1]: k_e * exp(lambda * n)."""
    return p["k_e"] * np.exp(p["lam"] * n)


def k_assem_n(n: float, p: dict) -> float:
    """
    Isoform-length-dependent Lp(a) assembly rate [hr^-1].

    Hill function saturating above n ~ K_n_assem repeats, reflecting the
    minimum kringle scaffold required for productive apo(a)–apoB-100 linkage.
    """
    x = (n / p["K_n_assem"]) ** p["h_assem"]
    return p["k_assem_max"] * x / (1.0 + x)


def ERAD_flux_mm(E: float, n: float, p: dict) -> float:
    """Michaelis-Menten ERAD flux [a.u. hr^-1]; saturates at high ER load."""
    V = p["V_ERAD_max"] * np.exp(p["lam"] * n)
    return V * E / (p["K_M_ERAD"] + E)


# ── ODE system ────────────────────────────────────────────────────────────────

def apoa_ode(t, y, n: float, p: dict, erad_model: str = "exp") -> list:
    """
    RHS of the four-compartment apo(a) biosynthesis ODE.

    Parameters
    ----------
    t : float
        Time [hr] (unused; required by solve_ivp).
    y : array-like
        State vector [M, E, S, P].
    n : float
        KIV-2 copy number.
    p : dict
        Parameter dictionary.
    erad_model : {"exp", "mm"}
        "exp" — first-order exponential ERAD (default).
        "mm"  — Michaelis-Menten saturating ERAD.
    """
    M, E, S, Plp = y

    f_tx    = k_tx(n, p)
    f_tl    = k_tl(n, p) * M
    f_fold  = k_fold(n, p) * E
    ka      = k_assem_n(n, p)
    f_assem = ka * S
    f_deg_S = p["k_deg_S"] * S
    f_ERAD  = k_ERAD_exp(n, p) * E if erad_model == "exp" else ERAD_flux_mm(E, n, p)

    dM   = f_tx  - p["k_deg_m"] * M - f_tl
    dE   = f_tl  - f_fold - f_ERAD
    dS   = f_fold - f_assem - f_deg_S
    dPlp = f_assem - p["k_cl"] * Plp

    return [dM, dE, dS, dPlp]


def solve_ode(n: float, p: dict, t_span=(0, 500), n_points=1000,
              erad_model: str = "exp") -> dict:
    """Integrate the ODE system to steady state; returns time series dict."""
    y0 = [0.0, 0.0, 0.0, 0.0]
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(
        apoa_ode, t_span, y0,
        args=(n, p, erad_model),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8, atol=1e-10,
    )
    return {"t": sol.t, "M": sol.y[0], "E": sol.y[1], "S": sol.y[2], "P": sol.y[3]}


# ── Closed-form steady-state solutions ────────────────────────────────────────

def steady_state(n: float, p: dict) -> dict:
    """
    Closed-form steady state for the exponential ERAD model.

    Derived by setting all time derivatives to zero:
        M_ss = k_tx / k_deg_m
        E_ss = k_tl * M_ss / (k_fold + k_ERAD)
        S_ss = k_fold * E_ss / (k_assem + k_deg_S)
        P_ss = k_assem * S_ss / k_cl

    Returns
    -------
    dict with keys M, E, S, P, eta (ER secretion efficiency),
    eta_assem (assembly efficiency).
    """
    kt = k_tx(n, p)
    kl = k_tl(n, p)
    kf = k_fold(n, p)
    ke = k_ERAD_exp(n, p)
    ka = k_assem_n(n, p)
    kds = p["k_deg_S"]

    M   = kt / p["k_deg_m"]
    E   = kl * M / (kf + ke)
    S   = kf * E / (ka + kds)
    Plp = ka * S / p["k_cl"]

    eta       = kf / (kf + ke)
    eta_assem = ka / (ka + kds)

    return {"M": M, "E": E, "S": S, "P": Plp, "eta": eta, "eta_assem": eta_assem}


def steady_state_mm(n: float, p: dict) -> dict:
    """
    Closed-form steady state for the Michaelis-Menten ERAD model.

    Solves the quadratic: k_fold * E^2 + (k_fold*K_M + V - A)*E - A*K_M = 0,
    where A = k_tl * M_ss is the apo(a) synthesis flux into the ER.
    """
    M  = k_tx(n, p) / p["k_deg_m"]
    A  = k_tl(n, p) * M
    F  = k_fold(n, p)
    V  = p["V_ERAD_max"] * np.exp(p["lam"] * n)
    K  = p["K_M_ERAD"]

    b    = F * K + V - A
    disc = b ** 2 + 4 * F * A * K
    E    = (-b + np.sqrt(disc)) / (2 * F)

    ka  = k_assem_n(n, p)
    kds = p["k_deg_S"]
    S   = F * E / (ka + kds)
    Plp = ka * S / p["k_cl"]

    return {
        "M": M, "E": E, "S": S, "P": Plp,
        "eta": F * E / A,
        "eta_assem": ka / (ka + kds),
        "A": A, "V": V,
    }


# ── Goodness-of-fit ───────────────────────────────────────────────────────────

def chi2_gof(obs: np.ndarray, pred: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """
    Chi-squared goodness-of-fit with one normalization constraint (ddof=1).

    Returns
    -------
    (chi2_stat, p_value)
    """
    chi2_stat = float(np.sum(((obs - pred) / err) ** 2))
    dof = len(obs) - 1
    p_value = float(sc.chi2.sf(chi2_stat, dof))
    return chi2_stat, p_value


# ── Sensitivity analysis ──────────────────────────────────────────────────────

PARAM_KEYS: list[str] = [
    "alpha_tx", "N0_tx", "k_deg_m",
    "alpha_tl", "N0_prot",
    "k_fold0",
    "k_e", "lam",
    "k_assem_max", "K_n_assem", "h_assem",
    "k_deg_S", "k_cl",
]

PARAM_LABELS: dict[str, str] = {
    "alpha_tx":    r"$\alpha_{tx}$",
    "N0_tx":       r"$N_{0,tx}$",
    "k_deg_m":     r"$k_{deg,m}$",
    "alpha_tl":    r"$\alpha_{tl}$",
    "N0_prot":     r"$N_{0,prot}$",
    "k_fold0":     r"$k_{fold,0}$",
    "k_e":         r"$k_e$",
    "lam":         r"$\lambda$",
    "k_assem_max": r"$k_{assem,max}$",
    "K_n_assem":   r"$K_n$",
    "h_assem":     r"$h$",
    "k_deg_S":     r"$k_{deg,S}$",
    "k_cl":        r"$k_{cl}$",
}


def compute_sensitivity(
    n_arr: np.ndarray,
    p: dict,
    output_key: str = "P",
    delta: float = 0.10,
) -> np.ndarray:
    """
    Normalized local sensitivity coefficients via symmetric finite differences.

        S(Y; k_i) = d(ln Y) / d(ln k_i)
                  ≈ [Y(k_i * (1+δ/2)) - Y(k_i * (1-δ/2))] / (δ * Y(k_i))

    Parameters
    ----------
    n_arr : array of KIV-2 copy numbers.
    p : parameter dict (evaluated at posterior median for publication figures).
    output_key : one of "P", "E", "eta".
    delta : fractional perturbation size (default 10%).

    Returns
    -------
    S_mat : ndarray of shape (len(PARAM_KEYS), len(n_arr)).
    """
    S_mat = np.zeros((len(PARAM_KEYS), len(n_arr)))
    for pi, pk in enumerate(PARAM_KEYS):
        base_val = p[pk]
        for ni, n in enumerate(n_arr):
            p_hi = {**p, pk: base_val * (1 + delta / 2)}
            p_lo = {**p, pk: base_val * (1 - delta / 2)}
            Y_hi = steady_state(int(n), p_hi)[output_key]
            Y_lo = steady_state(int(n), p_lo)[output_key]
            Y_0  = steady_state(int(n), p)[output_key]
            if Y_0 != 0 and np.isfinite(Y_0):
                S_mat[pi, ni] = (Y_hi - Y_lo) / (delta * Y_0)
            else:
                S_mat[pi, ni] = np.nan
    return S_mat
