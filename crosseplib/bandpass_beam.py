"""Bandpass and beam / pixel-window handling for the separation G-matrix.

This is the "data handlers" layer of the old CrossSepLib README made concrete:

* read instrument passbands (from a 2-column text file, or extract them from a
  Planck RIMO FITS file) and integrate an SED against them;
* read harmonic beams (ACT-style ``ell b_ell`` text files or Gaussian from FWHM);
* evaluate HEALPix pixel window functions;
* stack per-frequency beams / pixel windows into the vectors the solver needs.

Reference: ``scripts/signal_cleansing.py::bandpass_weighted_mean`` and
``scripts/actplanckmaps.py`` (``beam_from_fwhm``, ``read_act_beam``,
``stack_beams``, ``stack_pixwin*``), plus the RIMO cells of
``notebooks/actplanck_multiclean.ipynb``.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import sed
from .config import Config

__all__ = [
    "load_bandpass",
    "resolve_bandpass",
    "bandpass_weighted_mean",
    "gaussian_beam",
    "read_beam_file",
    "pixel_window",
    "stack_beams",
    "stack_pixwin",
    "extract_rimo_passband",
    "read_rimo_fwhm",
]

Bandpass = tuple[np.ndarray, np.ndarray]  # (freq_ghz, transmission)

# Wavenumber (1/cm) -> frequency (GHz): nu = k * c, with c in cm/s, then /1e9.
_WAVENUMBER_INV_CM_TO_GHZ = sed.C_LIGHT * 100.0 / 1e9  # == 299792458 / 1e7

# ``np.trapezoid`` is the numpy>=2.0 name; ``np.trapz`` the older one.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# bandpasses
# --------------------------------------------------------------------------- #
def load_bandpass(path: str | os.PathLike) -> Bandpass:
    """Load a 2-column ``freq_GHz  transmission`` text file."""
    freq, trans = np.loadtxt(path, unpack=True)
    return np.asarray(freq, dtype=float), np.asarray(trans, dtype=float)


def resolve_bandpass(bandpass: str | os.PathLike | Bandpass) -> Bandpass:
    """Accept either a text-file path or a ``(freq, transmission)`` pair."""
    if isinstance(bandpass, (str, os.PathLike)):
        return load_bandpass(bandpass)
    freq, trans = bandpass
    return np.asarray(freq, dtype=float), np.asarray(trans, dtype=float)


def bandpass_weighted_mean(
    bandpass: str | os.PathLike | Bandpass,
    response: Callable[[np.ndarray], np.ndarray],
    *,
    t_cmb: float = sed.T_CMB,
    nu0: float = 220.0,
) -> float:
    """Bandpass-averaged value of ``response(nu)``.

    Reproduces ``scripts/signal_cleansing.py::bandpass_weighted_mean``: the
    passband is weighted by ``transmission * nu**-2 * [dB/dT|_tcmb / B_nu0]`` and
    the SED is integrated against it with the trapezoid rule.
    """
    freq, trans = resolve_bandpass(bandpass)
    weights = trans * freq**-2.0 * sed.relative_dbdT(freq, t_cmb, nu0)
    values = np.asarray(response(freq), dtype=float)
    return float(_trapz(weights * values, freq) / _trapz(weights, freq))


# --------------------------------------------------------------------------- #
# beams / pixel windows
# --------------------------------------------------------------------------- #
def gaussian_beam(ell: ArrayLike, fwhm_arcmin: float) -> np.ndarray:
    """Harmonic transform of a Gaussian beam of the given FWHM [arcmin].

    Matches ``scripts/actplanckmaps.py::beam_from_fwhm``:
    ``exp(-ell (ell+1) sigma**2 / 2)`` with ``sigma`` from the FWHM.
    """
    ell = np.asarray(ell, dtype=float)
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-ell * (ell + 1.0) * sigma**2 / 2.0)


def read_beam_file(path: str | os.PathLike, ell: ArrayLike) -> np.ndarray:
    """Read an ``ell  b_ell`` beam text file, normalise to ``b_ell[0]``, interpolate.

    Matches ``scripts/actplanckmaps.py::read_act_beam`` + ``stack_act_beams``.
    """
    data = np.loadtxt(path)
    ell_in, beam_in = data[:, 0], data[:, 1] / data[0, 1]
    return np.interp(np.asarray(ell, dtype=float), ell_in, beam_in)


def pixel_window(ell: ArrayLike, nside: int) -> np.ndarray:
    """HEALPix spin-0 pixel window interpolated onto ``ell``."""
    import healpy as hp

    ell = np.asarray(ell, dtype=float)
    pw = hp.pixwin(nside, lmax=int(np.max(ell)) + 100)
    return np.interp(ell, np.arange(len(pw)), pw)


def _per_freq_beam(
    freq: int,
    ell: np.ndarray,
    *,
    fwhm_arcmin: Mapping[int, float] | Sequence[float] | None,
    beam_files: Mapping[int, str] | Sequence[str] | None,
    idx: int,
    config: Config | None,
) -> np.ndarray:
    if beam_files is not None:
        entry = beam_files[freq] if isinstance(beam_files, Mapping) else beam_files[idx]
        return read_beam_file(entry, ell)
    if fwhm_arcmin is not None:
        val = fwhm_arcmin[freq] if isinstance(fwhm_arcmin, Mapping) else fwhm_arcmin[idx]
        return gaussian_beam(ell, val)
    if config is not None:
        return gaussian_beam(ell, config.fwhm_arcmin(freq))
    raise ValueError(
        f"no beam information for frequency {freq}: pass fwhm_arcmin, beam_files, or a Config"
    )


def stack_beams(
    freqs: Sequence[int],
    ell: ArrayLike,
    *,
    fwhm_arcmin: Mapping[int, float] | Sequence[float] | None = None,
    beam_files: Mapping[int, str] | Sequence[str] | None = None,
    config: Config | None = None,
) -> np.ndarray:
    """Concatenate per-frequency :math:`b_\\ell` onto a single vector.

    Provide exactly one of ``beam_files`` (ACT-style text files) or
    ``fwhm_arcmin`` (Gaussian). If neither is given but ``config`` is, Gaussian
    beams are built from ``config.fwhm_arcmin(freq)``.
    """
    ell = np.asarray(ell, dtype=float)
    parts = [
        _per_freq_beam(
            int(f), ell, fwhm_arcmin=fwhm_arcmin, beam_files=beam_files, idx=i, config=config
        )
        for i, f in enumerate(freqs)
    ]
    return np.concatenate(parts)


def stack_pixwin(
    freqs: Sequence[int],
    ell: ArrayLike,
    *,
    nside: int | Mapping[int, int] | None = None,
    config: Config | None = None,
) -> np.ndarray:
    """Concatenate per-frequency pixel windows onto a single vector.

    ``nside`` may be a scalar (same for all frequencies), a mapping
    ``{freq: nside}``, or ``None`` -- in which case ``config.pixwin_nside(freq)``
    is used (1024 for LFI, 2048 for HFI).
    """
    ell = np.asarray(ell, dtype=float)
    parts = []
    for f in freqs:
        f = int(f)
        if isinstance(nside, Mapping):
            ns = nside[f]
        elif nside is not None:
            ns = int(nside)
        elif config is not None:
            ns = config.pixwin_nside(f)
        else:
            raise ValueError(f"no pixel-window nside for frequency {f}: pass nside or a Config")
        parts.append(pixel_window(ell, ns))
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# RIMO extraction (Planck)
# --------------------------------------------------------------------------- #
def extract_rimo_passband(
    rimo_path: str | os.PathLike,
    freq: int,
    *,
    instrument: str = "HFI",
    fmin_ghz: float = 30.0,
    fmax_ghz: float = 1200.0,
) -> Bandpass:
    """Extract a Rayleigh-Jeans-normalised passband from a Planck RIMO FITS file.

    Reproduces the RIMO cells of ``notebooks/actplanck_multiclean.ipynb``:
    read ``WAVENUMBER`` / ``TRANSMISSION`` from the ``BANDPASS_*`` HDU, convert
    the wavenumber to GHz (HFI: ``k[1/cm] * c / 1e7``; LFI: already GHz),
    multiply the transmission by ``nu**2`` to get the RJ weighting, and clip to
    ``[fmin_ghz, fmax_ghz]``.
    """
    from astropy.io import fits

    instrument = instrument.upper()
    if instrument == "HFI":
        hdu_name = f"BANDPASS_F{int(freq):03d}"
    elif instrument == "LFI":
        hdu_name = f"BANDPASS_{int(freq):03d}"
    else:
        raise ValueError("instrument must be 'HFI' or 'LFI'")

    with fits.open(rimo_path) as hdul:
        data = hdul[hdu_name].data
        wavenumber = np.asarray(data["WAVENUMBER"], dtype=float)
        transmission = np.asarray(data["TRANSMISSION"], dtype=float)

    freq_ghz = wavenumber * _WAVENUMBER_INV_CM_TO_GHZ if instrument == "HFI" else wavenumber
    trans_rj = transmission * freq_ghz**2.0

    keep = (freq_ghz > fmin_ghz) & (freq_ghz < fmax_ghz)
    return freq_ghz[keep], trans_rj[keep]


def read_rimo_fwhm(rimo_path: str | os.PathLike, *, instrument: str = "HFI") -> dict[int, float]:
    """Read the per-frequency effective FWHM [arcmin] from a Planck RIMO file."""
    from astropy.io import fits

    instrument = instrument.upper()
    hdu = "MAP_PARAMS" if instrument == "HFI" else "FREQUENCY_MAP_PARAMETERS"
    out: dict[int, float] = {}
    with fits.open(rimo_path) as hdul:
        params = hdul[hdu].data
        for label, fwhm in zip(params["FREQUENCY"], params["FWHM"]):
            digits = "".join(ch for ch in str(label) if ch.isdigit())
            if digits:
                out[int(digits)] = float(fwhm)
    return out
