# Replication code

Symbolic verification and calibration code for:

> Zhang, Y. and Czerny, A. I. "When Mandates Bind: Regime-Dependent Instrument
> Assignment in Aviation Decarbonisation."

Every symbolic identity in Appendix A of the paper is machine-checked here.
`verify_appendix.py` runs all of them as a single check and reports each
expression that must vanish.

## Requirements

Python 3.12 with:

```
pip install -r requirements.txt
```

Only `sympy` and `numpy` are needed. The scripts read no external data files.

## Running

Run everything and get a pass/fail summary:

```
python run_all.py
```

Or run any script on its own, from the `scripts/` directory:

```
python scripts/verify_appendix.py
```

Each script prints its derivation and the expressions verified to vanish.

## Contents

| Script | Produces |
|---|---|
| `verify_appendix.py` | All Appendix A identities, as a single check |
| `derive_model.py` | Wedge, FOCs, interior s*, Observation 1 (Cournot) |
| `derive_bertrand.py` | Markup identity, mode invariance, Lemmas 5-6 |
| `derive_blp.py` | Berry vs residual inversion identity |
| `derive_uniqueness2.py` | Signs the dominance region; concavity condition |
| `derive_uniqueness3.py` | Frequency-feedback contraction; uniqueness condition |
| `derive_soc2.py` | Reconciles the fixed- and endogenous-outside-option SOCs |
| `derive_twofuel.py` | Two-fuel separation, Proposition 1 |
| `derive_mandate_kkt.py` | KKT regimes I-IV; Regime II crowding-out |
| `derive_equivalence2.py` | Two-fuel iso-cost rules; blend-neutral direction |
| `derive_etsbase.py` | Propositions 2-3 on the EU ETS regulatory base |
| `derive_fuelmarket2.py` | Fuel-market Jacobian, existence, uniqueness condition |
| `derive_welfare2.py` | Result 4 in full, including the forgone-revenue term |
| `derive_segments2.py` | Effective valuation, omega_eff, omega* |
| `passthrough.py` | Subsidy incidence at the corner and in the interior |
| `calibrate_eu.py` | EASA/IATA calibration, external validation |
| `sweep.py` | Threshold grids reported in Section 7 |

This table corresponds to Table 22 in Appendix B of the paper.

## A note on `derive_blp.py`

This script is exploratory rather than confirmatory. It works through whether
respecifying the outside option in the Berry (1994) form changes the inverse
demand, and concludes that it does not: the two specifications give the
identical inversion, so the Cournot dominance failure is a property of
logit-Cournot with an outside option rather than an artifact. Its final block
displays the Bertrand SAF first-order condition but does not solve it in closed
form, because `s` enters both exponentially and polynomially. The interior `s*`
is derived in `derive_bertrand.py` and `derive_twofuel.py`, which divide the
condition through by the share first.

## Data

The scripts require no data files. Calibration inputs quoted in the paper are
third-party and cited at the point of use: EASA reference fuel prices, OAG
schedule data, and the stated-preference estimates discussed in Section 6.2.
OAG schedule data are licensed and cannot be redistributed by the authors.

## License

MIT. See `LICENSE`.
