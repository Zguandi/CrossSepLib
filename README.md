# CrossSepLib

Angular-$C_\ell$ CMB **component separation** for galaxy–tSZ cross-correlation analysis.

CrossSepLib takes multi-frequency galaxy × CMB-temperature cross-spectra and separates
them into physical components (tSZ + foregrounds) via a generalized-least-squares /
marginalized-likelihood solve in harmonic space.

This is a **fresh rewrite** of the component-separation core that currently lives, mixed
with everything else, in `../unWISE_tsz_correlator/scripts/`. The old `scripts/` package
and the notebooks are kept only as reference implementations. `../unwise-tsz.pdf` is the
authority for the math.

---

## Scope

**In scope — the component-separation core:**

1. SED / spectral-response models (tSZ $g(\nu)$, free-free, synchrotron, modified
   blackbody / CIB with $\beta$ and $T$ moments).
2. Bandpass handling: RIMO → Rayleigh-Jeans passband extraction, bandpass-weighted SED
   integration.
3. Beam / pixel-window handling: beam-file reading + interpolation, pixel window
   functions, Gaussian-beam-from-FWHM.
4. Response-matrix ($G$) assembly (delta-frequency and band-averaged; optional beam and
   pixel-window deconvolution).
5. The GLS / marginalized-likelihood solve → separated component spectra $C_\ell^{gS}$
   plus their covariance (the "Fisher" / precision matrix).
6. The SED-nuisance fit layer directly above the solve: fitting $(\beta_0, T_0,
   \alpha_\mathrm{ff})$ by minimizing / sampling the marginalized cleaning $\chi^2$.
7. I/O for the `.npz` schemas that are the contracts with the stages on either side.

**Out of scope — future sibling libraries:**

| Concern | Goes to |
| --- | --- |
| NaMaster $C_\ell$ estimation, analytic Gaussian + jackknife covariance, map / mask preprocessing (PLA maps, apodization, equ→gal rotation) | **`AngClLib`** (future) |
| HOD fit to $C_\ell^{gg}$, GNFW pressure-profile fit to $C_\ell^{gy}$, $\langle bP_e\rangle(z)$, $\tilde Y_{500}$–$M_{500}$ scaling relation | **halo-model library** (future, one library) |

CrossSepLib ships a `spectra` stub module that only names `AngClLib` as the home for that
work and documents/validates the input `.npz` schema, so the boundary is explicit in code.

---

## Analysis flow

```
        (from AngClLib / cluster pipeline)
  C_l^{f g} for f in frequencies   +   block covariance   +   bandpasses, beams, pixwin
                    │
                    ▼
      build response matrix  G(β0, T0, α_ff, signals)         ← sed.py + bandpass_beam.py
      (optionally deconvolve beam × pixel window)             ← separation.py
                    │
                    ▼
      GLS / marginalized-likelihood solve                     ← separation.py
                    │
                    ▼
  cleaned C_l^{gS} (tSZ, β-moment, CIB-amplitude, …)  +  covariance / precision (invF)
                    │
     ┌──────────────┴───────────────┐
     ▼                              ▼
 write multiclean .npz          optional: fit (β0, T0, α_ff)  ← fitting.py
   (→ halo-model library)         by minimizing / sampling the cleaning χ²
```

### $\chi^2$ and the canonical estimator

For a stacked data vector $d = \{C_\ell^{f g}\}_f$ (one block per frequency, over the fit
$\ell$-range) with block covariance $\mathbf{C}$, model $d = B_\ell W_\ell\, G\, s + n$
where $G$ is the frequency-response matrix, $s$ the component spectra, and $B_\ell W_\ell$
the beam × pixel-window:

$$
\chi^2(s) = \left[d - B_\ell W_\ell\, G\, s\right]^{T}\mathbf{C}^{-1}
            \left[d - B_\ell W_\ell\, G\, s\right].
$$

The **canonical estimator** is the marginalized-likelihood / QR solve
(`_marg_loglike_and_bf_qr` in the reference code): it returns the best-fit $\hat s$, the
inverse Fisher matrix $F^{-1}$ (covariance of $\hat s$), $\log\det F$, and the marginal
log-likelihood used by the SED-nuisance fit. A plain normal-equations (`inv`) path is kept
as a cross-check.

---

## Package layout

```
crosseplib/
  config.py          Config object: data root (env var / YAML), cosmology, Planck/ACT
                     frequency + beam tables. Replaces the old wildcard `pathconfig`
                     import and all import-time config.yaml reads.
  sed.py             tSZ g(ν), free-free, synchrotron, MBB/CIB + β/T moment SEDs.
                     Pure python, unit-tested.
  bandpass_beam.py   RIMO→RJ passband extraction, bandpass-weighted SED integration,
                     beam-file reading + interpolation, pixel window functions,
                     Gaussian-beam-from-FWHM.
  separation.py      G-matrix assembly (delta-freq + band-averaged; optional beam/pixwin
                     deconvolution) and the GLS / marginalized-likelihood solve.
  fitting.py         SED-nuisance (β0, T0, α_ff) fit above the solve: response builder,
                     cleaning-χ² wrapper, a minimizer / emcee runner. Generic Fisher /
                     emcee helpers live here too.
  io.py              readers / writers for the .npz contracts (see below).
  spectra.py         STUB — points to future AngClLib; validates the input C_l + cov
                     schema. No NaMaster code.
tests/               pytest units for the pure-python pieces (sed, bandpass_beam,
                     separation linear algebra on synthetic inputs, io round-trips).
examples/            a short end-to-end separation run from a cl/cov .npz.
```

Design rules: no `import *`, no import-time side effects, `cosmo` / config passed
explicitly, `T_CMB` single-sourced from `config`, and the bugs listed in
`../unWISE_tsz_correlator/LATENT_BUGS.md` fixed on the way in.

---

## `.npz` data contracts

These schemas are the real interface between pipeline stages. `io.py` owns them.

**Input — ACT cl/cov** (per-frequency 90/150/220): keys `cl_gy_90 / cl_gy_150 / cl_gy_220`,
covariance blocks `cov90g90g`, `cov90g150g`, … ; plus `ells`.

**Input — Planck cl/cov** (arbitrary frequency set): keys `cell_<f>g`,
`cov_<f1>g<f2>g`, `ells`. Frequencies inferred from the `cell_*g` keys.

**Output — separated / "multiclean" result**: keys `ells`, `cleaned_signals`
(shape `n_signal × n_ell`), `errors`, `signal_names`, `invF` (inverse Fisher /
precision), `beta0`, `T0`, plus free-text `comments`.

---

## Reference implementations

Each new module is derived from the following (kept for reference, not imported):

| New module | Reference in `../unWISE_tsz_correlator/` |
| --- | --- |
| `sed.py` | `scripts/signal_cleansing.py` (SED / g(ν) / blackbody / β,T moment functions) |
| `bandpass_beam.py` | `scripts/signal_cleansing.py::bandpass_weighted_mean`; `scripts/actplanckmaps.py` (`beam_from_fwhm`, `stack_beams*`, `stack_pixwin*`, `beam_deconvolve`); passband extraction in `notebooks/actplanck_multiclean.ipynb` (RIMO cells) |
| `separation.py` | `scripts/multiclean.py::component_cleansing` + `_marg_loglike_and_bf_qr/_inv`; `scripts/signal_cleansing.py` (`threecomponent_gmatrix`, `bandaveraged_gmatrix`, `quadratic_estimate_separation`); `scripts/actplanckmaps.py` (`bandaveraged_gmatrix_plk`, `couple_beam_gmatrix`) |
| `fitting.py` | `notebooks/actplanck_multiclean.ipynb` (`build_responses`, `chisq_multclean`, `save_multclean`, β0/T0 grid scans); `tasks/run_mcmc_multiclean.py`; `scripts/mcmc.py` (`run_emcee_from_chisq`, `estimate_fisher_matrix`) |
| `io.py` | `scripts/signal_cleansing.py::load_cl_cov_results`, `load_gls_result`; `scripts/actplanckmaps.py::load_plk_cl_cov`; `scripts/multiclean.py` result loaders |
| `config.py` | `scripts/pathconfig.py`, `config.yaml` |
| `spectra.py` (stub) | `scripts/gaussian_covariance_nmt.py`, `scripts/actplanckmaps.py`, `tasks/planck_all_cov_pla.py` — all destined for `AngClLib` |

---

## Install & dependencies

```
pip install -e .
```

Core dependencies: `numpy`, `scipy`, `pyyaml`, `astropy` (FITS/RIMO reading),
`healpy` (pixel windows). `emcee` + `corner` for `fitting.py`. NaMaster / pixell are
**not** required by the separation core.

## Reference

`../unwise-tsz.pdf` has the detailed derivation and the analysis it was built for.
