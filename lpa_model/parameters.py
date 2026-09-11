"""
Model parameters and observed data.

All rate constants are in hours^-1 unless noted.
KIV-2 copy number n ranges from 1 to 40 (physiological range).
"""

import numpy as np

# ── Structural constants (LPA gene / ribosome biology) ────────────────────────

_LPA_FIXED_KB  = 90.0   # non-KIV-2 pre-mRNA length            [kb]
_KIV2_KB       = 5.5    # length of one KIV-2 repeat            [kb]
_RNAP_NT_S     = 40.0   # RNA polymerase II elongation rate     [nt/s]
_AA_PER_KRINGLE = 80    # amino acids per kringle domain
_N_RIBOSOMES   = 5      # ribosomes per mRNA (polysome estimate)
_RIB_AA_S      = 4.0    # ribosome elongation rate              [aa/s]

# Derived: non-KIV-2 gene in KIV-2-equivalent repeat units (~16.4)
_N0_TX = _LPA_FIXED_KB / _KIV2_KB

# Derived: inverse of elongation time per KIV-2-equivalent [hr^-1] (~26.2)
_ALPHA_TX = (_RNAP_NT_S * 3600) / (_KIV2_KB * 1000)

# Fixed kringle-equivalent domains: KIV1, KIV3–10, KV, protease domain = 11
_N0_PROT = 11

# Derived: translation rate scaling [hr^-1] (900)
_ALPHA_TL = _N_RIBOSOMES * _RIB_AA_S * 3600 / _AA_PER_KRINGLE

# ── Parameter dictionary ───────────────────────────────────────────────────────

params: dict = {
    # Transcription:  k_tx(n) = alpha_tx / (N0_tx + n)
    # Source: LPA genomic structure (McLean 1987); RNAP elongation (Jonkers 2015)
    "alpha_tx": _ALPHA_TX,   # ~26.2  [hr^-1 · repeat]
    "N0_tx":    _N0_TX,      # ~16.4  [repeats]

    # mRNA decay
    # Source: hepatic mRNA t_1/2 ~ 6 h (Sharova 2009)
    "k_deg_m": np.log(2) / 6.0,   # 0.1155  [hr^-1]

    # Translation:  k_tl(n) = alpha_tl / (N0_prot + n)
    # Source: ribosome elongation (Ingolia 2011); kringle domain size (Schmidt 2016)
    "alpha_tl": _ALPHA_TL,   # 900   [hr^-1]
    "N0_prot":  _N0_PROT,    # 11    [domains]

    # ER folding/processing:  k_fold(n) = k_fold0 / n
    # Sequential co-translational processing of each KIV-2 kringle domain.
    # Calibrated to yield ~4 h ER residence time at n=8 (White 1994 HepG2 data).
    "k_fold0": 2.0,   # [hr^-1]; refined by Bayesian inference

    # ERAD — exponential model:  k_ERAD(n) = k_e * exp(lambda * n)
    # Per-kringle misfolding probability lambda; k_e is the baseline ERAD rate.
    # Both estimated by Bayesian inference conditioned on Rader 1994 data.
    "k_e":  0.10,   # [hr^-1]      prior point estimate
    "lam":  0.07,   # [repeat^-1]  prior point estimate

    # ERAD — Michaelis-Menten (used only for saturation / ER-stress analysis)
    "V_ERAD_max": 5.0,   # [hr^-1]
    "K_M_ERAD":   1.0,   # [a.u.]

    # Lp(a) assembly:  k_assem(n) = k_assem_max * (n/K_n)^h / (1 + (n/K_n)^h)
    # Hill function reflects minimum kringle scaffold needed for apoB-100 linkage.
    # K_n=8 sets half-maximal efficiency at n=8 (Mukamel 2021 attenuation point).
    "k_assem_max": 3.0,   # [hr^-1]
    "K_n_assem":   8.0,   # [repeats] half-maximal assembly
    "h_assem":     3.0,   # Hill coefficient (cooperative structural threshold)
    "k_deg_S":     1.5,   # [hr^-1] unassembled secreted apo(a) clearance

    # Plasma Lp(a) clearance
    # Source: FCR = 0.32 d^-1 (Rader 1994, isoform-invariant)
    "k_cl": 0.32 / 24.0,   # 0.01333  [hr^-1]
}

# ── Observed data (Rader 1994, Table III) ─────────────────────────────────────
# Isoform-specific apo(a) production rates, normalized to n=14 (S2 isoform).
# n values correspond to S1 (7), S2 (14), S3 (20), S4 (26) size classes.

OBS_N   = np.array([7,    14,   20,   26  ])   # KIV-2 copy number
OBS_PR  = np.array([2.15, 1.00, 0.40, 0.18])   # relative production rate
OBS_ERR = np.array([0.40, 0.15, 0.08, 0.05])   # approximate SEM

REF_N = 14   # normalization isoform
N_ARR = np.arange(1, 41)   # full physiological KIV-2 range
