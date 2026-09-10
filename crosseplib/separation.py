"""Response-matrix assembly and the GLS / marginalised-likelihood solve.

This is the heart of CrossSepLib. Given multi-frequency cross-spectra
:math:`C_\\ell^{fg}`, their block covariance, the instrument bandpasses, and a
set of SED responses, it builds the frequency-response matrix :math:`G` and
solves for the separated component spectra :math:`C_\\ell^{gS}` together with
their covariance (inverse Fisher / precision).

Canonical estimator: the marginalised-likelihood QR solve (``method="qr"``),
equivalent to the legacy ``scripts/multiclean.py::_marg_loglike_and_bf_qr``. A
plain normal-equations path (``method="inv"``) is kept for cross-checks.

Model (per :math:`\\ell`, stacked over frequencies)::

    d = (B_l W_l) . G . s + noise

with block covariance ``Cov``. ``G`` has block structure ``G[i,j] = g_ij * I``
where ``g_ij`` is the bandpass-averaged response of signal ``j`` in frequency
band ``i``.

Reference: ``scripts/multiclean.py`` and ``scripts/signal_cleansing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from . import sed
from .bandpass_beam import Bandpass, bandpass_weighted_mean

__all__ = [
    "GLSResult",
    "SeparationResult",
    "stack_cl_vector",
    "stack_covariance",
    "build_gmatrix",
    "gls_solve",
    "separate_components",
]

Response = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class GLSResult:
    """Output of :func:`gls_solve`.

    ``amplitudes`` is the flat best-fit vector (length ``n_signal * n_ell``);
    ``inv_fisher`` is its covariance. ``chi2`` is the best-fit
    :math:`\\chi^2`; ``log_likelihood`` is the amplitude-marginalised
    log-likelihood ``-0.5 (log|F| + chi2)``.
    """

    amplitudes: np.ndarray
    inv_fisher: np.ndarray
    errors: np.ndarray
    chi2: float
    log_det_fisher: float
    log_likelihood: float


@dataclass
class SeparationResult:
    """Output of :func:`separate_components` -- the ``multiclean`` .npz contract."""

    ells: np.ndarray
    signal_names: list[str]
    cleaned_signals: np.ndarray  # flat, length n_signal * n_ell (legacy layout)
    errors: np.ndarray  # flat, same layout
    inv_fisher: np.ndarray
    chi2: float
    log_det_fisher: float
    log_likelihood: float
    beta0: float | None = None
    t0: float | None = None
    alpha: float | None = None
    comments: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_ell(self) -> int:
        return len(self.ells)

    @property
    def n_signal(self) -> int:
        return len(self.signal_names)

    def signal(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cl, err)`` for one named component."""
        idx = self.signal_names.index(name)
        n = self.n_ell
        sl = slice(idx * n, (idx + 1) * n)
        return self.cleaned_signals[sl], self.errors[sl]

    def signals_2d(self) -> np.ndarray:
        """``cleaned_signals`` reshaped to ``(n_signal, n_ell)``."""
        return self.cleaned_signals.reshape(self.n_signal, self.n_ell)


# --------------------------------------------------------------------------- #
# stacking
# --------------------------------------------------------------------------- #
def _as_mask(ells: np.ndarray, mask: np.ndarray | None, ell_min: float, ell_max: float) -> np.ndarray:
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    return (ells >= ell_min) & (ells <= ell_max)


def stack_cl_vector(
    cl_dict: Mapping[int, np.ndarray],
    freqs: Sequence[int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate ``cl_dict[f]`` (optionally boolean-masked) over ``freqs``."""
    out = []
    for f in freqs:
        cl = np.asarray(cl_dict[f], dtype=float).ravel()
        out.append(cl[mask] if mask is not None else cl)
    return np.concatenate(out)


def stack_covariance(
    cov_dict: Mapping[tuple[int, int], np.ndarray],
    freqs: Sequence[int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the full block covariance from per-pair blocks.

    ``cov_dict`` need only hold the upper triangle: ``(f2, f1)`` is filled from
    ``cov_dict[(f1, f2)].T``.
    """
    rows = []
    for f1 in freqs:
        row = []
        for f2 in freqs:
            if (f1, f2) in cov_dict:
                block = np.asarray(cov_dict[(f1, f2)], dtype=float)
            elif (f2, f1) in cov_dict:
                block = np.asarray(cov_dict[(f2, f1)], dtype=float).T
            else:
                raise KeyError(f"no covariance block for ({f1}, {f2}) or ({f2}, {f1})")
            if mask is not None:
                block = block[np.ix_(mask, mask)]
            row.append(block)
        rows.append(row)
    return np.block(rows)


# --------------------------------------------------------------------------- #
# G-matrix
# --------------------------------------------------------------------------- #
def build_gmatrix(
    g: np.ndarray,
    nbins: int,
    *,
    beams: np.ndarray | None = None,
    pixwin: np.ndarray | None = None,
) -> np.ndarray:
    """Build the block response matrix.

    Parameters
    ----------
    g
        ``(n_freq, n_signal)`` array of bandpass-averaged responses.
    nbins
        Number of :math:`\\ell` bins in the fit range.
    beams, pixwin
        Optional stacked (length ``n_freq * nbins``) beam / pixel-window
        vectors. Each row ``m`` of ``G`` is multiplied by ``beams[m]`` and/or
        ``pixwin[m]`` -- i.e. the model is ``(B_l W_l) G s``.

    Returns
    -------
    ndarray
        ``(n_freq * nbins, n_signal * nbins)`` matrix.
    """
    g = np.asarray(g, dtype=float)
    G = np.kron(g, np.eye(nbins))
    if beams is not None:
        G = G * np.asarray(beams, dtype=float)[:, None]
    if pixwin is not None:
        G = G * np.asarray(pixwin, dtype=float)[:, None]
    return G


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def _solve_qr(d: np.ndarray, G: np.ndarray, cov: np.ndarray) -> GLSResult:
    L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, d)
    A = np.linalg.solve(L, G)

    Q, R = np.linalg.qr(A, mode="reduced")
    c_hat = np.linalg.solve(R, Q.T @ y)

    log_det_fisher = 2.0 * np.sum(np.log(np.abs(np.diag(R))))

    resid = y - A @ c_hat
    chi2 = float(resid @ resid)
    log_like = -0.5 * (log_det_fisher + chi2)

    inv_R = np.linalg.solve(R, np.eye(R.shape[0]))
    inv_fisher = inv_R @ inv_R.T
    errors = np.sqrt(np.sum(inv_R**2, axis=1))
    return GLSResult(c_hat, inv_fisher, errors, chi2, log_det_fisher, log_like)


def _solve_inv(d: np.ndarray, G: np.ndarray, cov: np.ndarray) -> GLSResult:
    inv_c = np.linalg.inv(cov)
    fisher = G.T @ inv_c @ G
    inv_fisher = np.linalg.inv(fisher)

    c_hat = inv_fisher @ G.T @ inv_c @ d
    log_det_fisher = float(np.linalg.slogdet(fisher)[1])

    resid = d - G @ c_hat
    chi2 = float(resid @ inv_c @ resid)
    log_like = -0.5 * (log_det_fisher + chi2)

    errors = np.sqrt(np.diag(inv_fisher))
    return GLSResult(c_hat, inv_fisher, errors, chi2, log_det_fisher, log_like)


def gls_solve(
    d: np.ndarray,
    G: np.ndarray,
    cov: np.ndarray,
    method: str = "qr",
) -> GLSResult:
    """Generalised-least-squares solve of ``d = G s`` under covariance ``cov``.

    ``method="qr"`` (default, canonical) uses a Cholesky + QR factorisation and
    is numerically the safer path; ``method="inv"`` forms the normal equations
    explicitly.
    """
    d = np.asarray(d, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if method == "qr":
        return _solve_qr(d, G, cov)
    if method == "inv":
        return _solve_inv(d, G, cov)
    raise ValueError("method must be 'qr' or 'inv'")


# --------------------------------------------------------------------------- #
# high-level driver
# --------------------------------------------------------------------------- #
def separate_components(
    cl_dict: Mapping[int, np.ndarray],
    cov_dict: Mapping[tuple[int, int], np.ndarray],
    freqs: Sequence[int],
    ells: np.ndarray,
    bandpasses: Sequence[str | Bandpass],
    responses: Sequence[Response],
    *,
    signal_names: Sequence[str] | None = None,
    ell_min: float = 30.0,
    ell_max: float = 1200.0,
    ell_mask: np.ndarray | None = None,
    beams: np.ndarray | None = None,
    pixwin: np.ndarray | None = None,
    method: str = "qr",
    t_cmb: float = sed.T_CMB,
    nu0: float = 220.0,
    beta0: float | None = None,
    t0: float | None = None,
    alpha: float | None = None,
    comments: str = "",
) -> SeparationResult:
    """Run one component separation.

    Parameters
    ----------
    cl_dict, cov_dict
        Per-frequency cross-spectra and per-pair covariance blocks, keyed the
        way :func:`crosseplib.io.load_clcov_planck` returns them.
    freqs
        Frequencies to include, in the order they should be stacked.
    ells
        The full (unmasked) multipole array that ``cl_dict`` / ``cov_dict`` are
        sampled on.
    bandpasses
        One passband per frequency (text-file path or ``(freq, transmission)``).
    responses
        One SED response callable per component (see :mod:`crosseplib.sed` and
        :func:`crosseplib.fitting.build_responses`).
    signal_names
        Names for the components; defaults to ``signal_0, signal_1, ...``.
    ell_min, ell_max, ell_mask
        Fit window. ``ell_mask`` (a boolean array over ``ells``) overrides the
        min/max cut when given.
    beams, pixwin
        Stacked (length ``n_freq * nbins``) beam / pixel-window vectors to
        deconvolve the model with. Build them with
        :func:`crosseplib.bandpass_beam.stack_beams` /
        :func:`crosseplib.bandpass_beam.stack_pixwin`.
    beta0, t0, alpha
        Recorded on the result for provenance (the SED nuisance parameters the
        ``responses`` were built with).
    """
    ells = np.asarray(ells, dtype=float)
    if len(bandpasses) != len(freqs):
        raise ValueError(f"got {len(bandpasses)} bandpasses for {len(freqs)} frequencies")

    mask = _as_mask(ells, ell_mask, ell_min, ell_max)
    ells_sel = ells[mask]
    nbins = int(mask.sum())

    d = stack_cl_vector(cl_dict, freqs, mask)
    cov = stack_covariance(cov_dict, freqs, mask)

    g = np.empty((len(freqs), len(responses)), dtype=float)
    for i, band in enumerate(bandpasses):
        for j, response in enumerate(responses):
            g[i, j] = bandpass_weighted_mean(band, response, t_cmb=t_cmb, nu0=nu0)

    G = build_gmatrix(g, nbins, beams=beams, pixwin=pixwin)

    res = gls_solve(d, G, cov, method=method)

    if signal_names is None:
        names = [f"signal_{j}" for j in range(len(responses))]
    else:
        names = list(signal_names)

    return SeparationResult(
        ells=ells_sel,
        signal_names=names,
        cleaned_signals=res.amplitudes,
        errors=res.errors,
        inv_fisher=res.inv_fisher,
        chi2=res.chi2,
        log_det_fisher=res.log_det_fisher,
        log_likelihood=res.log_likelihood,
        beta0=beta0,
        t0=t0,
        alpha=alpha,
        comments=comments,
        meta={"freqs": list(map(int, freqs)), "method": method, "nu0": nu0, "t_cmb": t_cmb},
    )
