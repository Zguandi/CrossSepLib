"""End-to-end component separation from a Planck cl/cov .npz.

Run on the cluster where the data products live::

    CROSSEPLIB_DATA=/scratch2/guzhao python examples/run_separation.py

This mirrors the "Snippet to carry out actplanck separation" and the
``save_multclean`` cells of ``notebooks/actplanck_multiclean.ipynb``.
"""

from __future__ import annotations

import numpy as np

from crosseplib.bandpass_beam import stack_beams, stack_pixwin
from crosseplib.config import Config
from crosseplib.fitting import build_responses, fit_sed_nuisance, make_cleaning_chisq
from crosseplib.io import load_clcov_planck, save_separation
from crosseplib.separation import separate_components

cfg = Config.default()

# --- inputs (edit paths / bandpass list for your run) ----------------------- #
CLCOV_NPZ = cfg.data_path("results", "planck_allfreq_lowz_with857.npz")
BANDPASS_DIR = cfg.data_path("results", "planck_passbands")
FREQS = list(cfg.planck_freqs)  # 30 ... 857
BANDPASSES = [BANDPASS_DIR / f"{'LFI' if f in cfg.lfi_freqs else 'HFI'}_f{f:03d}_passband_RJ.txt"
              for f in FREQS]
SIGNAL_NAMES = ["tSZ", "free-free", "CIB-amp", "CIB-beta"]
PARAM_NAMES = ["beta0", "T0", "alpha"]
ELL_MIN, ELL_MAX = 200, 2000
OUT_DIR = cfg.data_path("results", "crosseplib_separation")

# --- load ------------------------------------------------------------------- #
ells, cl_dict, cov_dict = load_clcov_planck(CLCOV_NPZ, freqs=FREQS)

mask = (ells >= ELL_MIN) & (ells <= ELL_MAX)
nbins = int(mask.sum())
beams = stack_beams(FREQS, ells[mask], config=cfg)
pixwin = stack_pixwin(FREQS, ells[mask], config=cfg)

# --- fit the SED nuisance parameters -------------------------------------- #
chisq = make_cleaning_chisq(
    cl_dict, cov_dict, FREQS, ells, BANDPASSES, SIGNAL_NAMES, PARAM_NAMES,
    ell_min=ELL_MIN, ell_max=ELL_MAX, beams=beams, pixwin=pixwin, method="qr",
    verbose=True,
)
fit = fit_sed_nuisance(
    chisq, PARAM_NAMES, x0=[1.4, 24.0, -2.6],
    bounds=[(0.1, 3.0), (10.0, 50.0), (-5.0, 0.0)],
)
print("\nSED nuisance fit:")
print(fit.summary())

# --- produce the cleaned spectra at the best fit ------------------------- #
best = dict(zip(PARAM_NAMES, fit.best_fit))
responses = build_responses(best, SIGNAL_NAMES)
result = separate_components(
    cl_dict, cov_dict, FREQS, ells, BANDPASSES, responses,
    signal_names=SIGNAL_NAMES, ell_min=ELL_MIN, ell_max=ELL_MAX,
    beams=beams, pixwin=pixwin, method="qr",
    beta0=best["beta0"], t0=best["T0"], alpha=best["alpha"],
    comments="lowz tSZ+radio+CIB+CIB-beta, crosseplib rehearsal",
)
out = save_separation(OUT_DIR, result)
print(f"\nwrote {out}")
print(f"chi2 = {result.chi2:.1f}   tSZ[:3] = {np.array2string(result.signal('tSZ')[0][:3])}")
