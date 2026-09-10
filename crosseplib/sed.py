"""Spectral-energy-distribution / frequency-response models.

Pure NumPy, no I/O. Every response here is expressed in the same convention as
the legacy ``scripts/signal_cleansing.py``: a dimensionless factor in
:math:`\\Delta T_{\\rm CMB}` units, normalised at ``nu0`` (default 220 GHz).

Reference: ``../unWISE_tsz_correlator/scripts/signal_cleansing.py`` and
``../unwise-tsz.pdf``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .config import DEFAULT_T_CMB

__all__ = [
    "H_PLANCK",
    "K_BOLTZ",
    "C_LIGHT",
    "T_CMB",
    "dimensionless_x",
    "cmb_to_rj",
    "tsz_g",
    "tsz_response",
    "freefree_response",
    "synchrotron_response",
    "planck_bnu",
    "planck_dbdT",
    "relative_bnu",
    "relative_dbdT",
    "cib_amplitude_response",
    "cib_beta_response",
]

# Physical constants (SI)
H_PLANCK = 6.62607015e-34  # J s
K_BOLTZ = 1.380649e-23  # J / K
C_LIGHT = 299_792_458.0  # m / s

#: Reference CMB temperature for the SED math (see :data:`crosseplib.config.DEFAULT_T_CMB`).
T_CMB = DEFAULT_T_CMB


def dimensionless_x(freq_ghz: ArrayLike, t_cmb: float = T_CMB) -> np.ndarray:
    """:math:`x = h\\nu / k_B T`."""
    freq_hz = np.asarray(freq_ghz, dtype=float) * 1e9
    return freq_hz * H_PLANCK / (K_BOLTZ * t_cmb)


def cmb_to_rj(freq_ghz: ArrayLike, t_cmb: float = T_CMB) -> np.ndarray:
    """Conversion factor from :math:`\\Delta T_{\\rm CMB}` to Rayleigh-Jeans units.

    ``x**2 e**x / (e**x - 1)**2`` -- i.e. the derivative of the Planck function
    with respect to temperature, normalised out of Planck's constant.
    """
    x = dimensionless_x(freq_ghz, t_cmb)
    return x**2 * np.exp(x) / np.expm1(x) ** 2


def tsz_g(freq_ghz: ArrayLike, t_cmb: float = T_CMB) -> np.ndarray:
    """Non-relativistic tSZ spectral function :math:`g(\\nu) = x\\coth(x/2) - 4`."""
    x = dimensionless_x(freq_ghz, t_cmb)
    return x / np.tanh(x / 2.0) - 4.0


def tsz_response(freq_ghz: ArrayLike, *, tempfactor: bool = False, t_cmb: float = T_CMB) -> np.ndarray:
    """tSZ response used to build the G-matrix.

    With ``tempfactor=True`` the response is multiplied by ``t_cmb`` (the
    ``tSZ_integrand(..., tempfactor=True)`` path in the legacy notebooks, which
    produces a :math:`\\mu K` normalised tSZ amplitude).
    """
    g = tsz_g(freq_ghz, t_cmb)
    return g * t_cmb if tempfactor else g


def freefree_response(
    freq_ghz: ArrayLike,
    alpha: float = -2.14,
    nu0: float = 220.0,
    t_cmb: float = T_CMB,
) -> np.ndarray:
    """Free-free (thermal bremsstrahlung) SED in :math:`\\Delta T_{\\rm CMB}` units.

    The ``alpha`` power law is defined in Rayleigh-Jeans brightness temperature;
    dividing by :func:`cmb_to_rj` converts it to CMB units.
    """
    return (np.asarray(freq_ghz, dtype=float) / nu0) ** alpha / cmb_to_rj(freq_ghz, t_cmb)


def synchrotron_response(
    freq_ghz: ArrayLike,
    alpha: float = -3.11,
    nu0: float = 220.0,
    t_cmb: float = T_CMB,
) -> np.ndarray:
    """Synchrotron SED in :math:`\\Delta T_{\\rm CMB}` units (same convention as free-free)."""
    return (np.asarray(freq_ghz, dtype=float) / nu0) ** alpha / cmb_to_rj(freq_ghz, t_cmb)


def planck_bnu(freq_ghz: ArrayLike, temperature: float) -> np.ndarray:
    """Planck function :math:`B_\\nu(T)` in SI units."""
    freq_hz = np.asarray(freq_ghz, dtype=float) * 1e9
    x = freq_hz * H_PLANCK / (K_BOLTZ * temperature)
    return (2.0 * H_PLANCK * freq_hz**3) / C_LIGHT**2 / np.expm1(x)


def planck_dbdT(freq_ghz: ArrayLike, temperature: float) -> np.ndarray:
    """:math:`\\partial B_\\nu / \\partial T` in SI units."""
    freq_hz = np.asarray(freq_ghz, dtype=float) * 1e9
    x = freq_hz * H_PLANCK / (K_BOLTZ * temperature)
    return (
        (2.0 * H_PLANCK**2 * freq_hz**4)
        / (C_LIGHT**2 * K_BOLTZ * temperature**2)
        * np.exp(x)
        / np.expm1(x) ** 2
    )


def relative_bnu(freq_ghz: ArrayLike, temperature: float, nu0: float = 220.0) -> np.ndarray:
    """:math:`B_\\nu(T) / B_{\\nu_0}(T)`."""
    return planck_bnu(freq_ghz, temperature) / planck_bnu(nu0, temperature)


def relative_dbdT(freq_ghz: ArrayLike, temperature: float, nu0: float = 220.0) -> np.ndarray:
    """:math:`(\\partial B_\\nu/\\partial T) / B_{\\nu_0}(T)`."""
    return planck_dbdT(freq_ghz, temperature) / planck_bnu(nu0, temperature)


def cib_amplitude_response(
    freq_ghz: ArrayLike,
    beta0: float,
    t0: float,
    nu0: float = 220.0,
    t_cmb: float = T_CMB,
) -> np.ndarray:
    """Modified-blackbody (CIB) amplitude response in :math:`\\Delta T_{\\rm CMB}` units.

    ``(nu/nu0)**beta0 * [B_nu(t0)/B_nu0(t0)] / [dB_nu/dT|_tcmb / B_nu0(tcmb)]``
    -- the legacy ``B_integrand``.
    """
    freq = np.asarray(freq_ghz, dtype=float)
    bf = relative_bnu(freq, t0, nu0=nu0)
    rbd = relative_dbdT(freq, t_cmb, nu0=nu0)
    return (freq / nu0) ** beta0 * bf / rbd


def cib_beta_response(
    freq_ghz: ArrayLike,
    beta0: float,
    t0: float,
    nu0: float = 220.0,
    t_cmb: float = T_CMB,
) -> np.ndarray:
    """First moment of the CIB SED w.r.t. the spectral index :math:`\\beta`.

    ``cib_amplitude_response * ln(nu/nu0)`` -- the legacy ``beta_integrand``.
    """
    freq = np.asarray(freq_ghz, dtype=float)
    return cib_amplitude_response(freq, beta0, t0, nu0=nu0, t_cmb=t_cmb) * np.log(freq / nu0)
