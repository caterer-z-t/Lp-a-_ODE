# Lp(a) ODE Model

A four-compartment mechanistic model of Lp(a) biosynthesis as a function of KIV-2 repeat number.

---

## what it does

Links KIV-2 copy number *n* to circulating Lp(a) through explicit rate functions for transcription, translation, ER folding, ERAD, and assembly. Parameters estimated via Bayesian inference (PyMC/NUTS) conditioned on Rader et al. 1994.

## structure

```
lpa_model/     core model, parameters, inference
scripts/       one file per figure
figures/       generated outputs
```

## usage

```bash
pip install -r requirements.txt
python run_all.py        # regenerates all four figures
```

