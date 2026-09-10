"""SED-nuisance fitting on top of the separation solve.

The component separation is exact given the foreground SED shape parameters
(:math:`\\beta_0`, :math:`T_0`, the free-free index :math:`\\alpha`). This module
fits *those* by minimising / sampling the cleaning statistic returned by
:func:`crosseplib.separation.separate_components`.

Reference: the ``build_responses`` / ``chisq_multclean`` / ``save_multclean``
cells of ``notebooks/actplanck_multiclean.ipynb``,
``tasks/run_mcmc_multiclean.py``, and ``scripts/mcmc.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from . import sed
from .bandpass_beam import Bandpass
from .separation import Response, SeparationResult, separate_components

__all__ = [
    "build_responses",
    "make_cleaning_chisq",
    "estimate_fisher_matrix",
    "estimate_chisq_hessian",
    "numerical_jacobian",
    "flat_log_prior",
    "log_posterior_chisq",
    "FitResult",
    "fit_sed_nuisance",
]

# canonical component labels
KNOWN_SIGNALS = ("tSZ", "free-free", "Radio", "synchrotron", "CIB-amp", "CIB-beta")


def build_responses(
    params: Mapping[str, float],
    signal_names: Sequence[str],
    *,
    tsz_tempfactor: bool = True,
    nu0: float = 220.0,
    t_cmb: float = sed.T_CMB,
) -> list[Response]:
    """Map ``(params, signal_names)`` to a list of SED response callables.

    Recognised ``signal_names`` (order preserved):

    ==============  ===================================  params used
    label           response                             --
    ==============  ===================================  --
    ``tSZ``         :func:`crosseplib.sed.tsz_response`   --
    ``free-free`` / ``Radio``  :func:`~crosseplib.sed.freefree_response`  ``alpha`` (optional)
    ``synchrotron`` :func:`~crosseplib.sed.synchrotron_response`  --
    ``CIB-amp``     :func:`~crosseplib.sed.cib_amplitude_response`  ``beta0``, ``T0``
    ``CIB-beta``    :func:`~crosseplib.sed.cib_beta_response`       ``beta0``, ``T0``
    ==============  ===================================  --

    ``tsz_tempfactor=True`` reproduces the notebook ``build_responses``; the
    ``tasks/run_mcmc_multiclean.py`` path used ``False``.
    """
    beta0 = params.get("beta0")
    t0 = params.get("T0", params.get("t0"))
    alpha = params.get("alpha", -2.14)

    out: list[Response] = []
    for name in signal_names:
        if name == "tSZ":
            out.append(lambda nu, _tf=tsz_tempfactor: sed.tsz_response(nu, tempfactor=_tf, t_cmb=t_cmb))
        elif name in ("free-free", "Radio"):
            out.append(lambda nu, _a=alpha: sed.freefree_response(nu, alpha=_a, nu0=nu0, t_cmb=t_cmb))
        elif name == "synchrotron":
            out.append(lambda nu: sed.synchrotron_response(nu, nu0=nu0, t_cmb=t_cmb))
        elif name == "CIB-amp":
            if beta0 is None or t0 is None:
                raise ValueError("CIB-amp needs 'beta0' and 'T0' in params")
            out.append(lambda nu, _b=beta0, _t=t0: sed.cib_amplitude_response(nu, _b, _t, nu0=nu0, t_cmb=t_cmb))
        elif name == "CIB-beta":
            if beta0 is None or t0 is None:
                raise ValueError("CIB-beta needs 'beta0' and 'T0' in params")
            out.append(lambda nu, _b=beta0, _t=t0: sed.cib_beta_response(nu, _b, _t, nu0=nu0, t_cmb=t_cmb))
        else:
            raise ValueError(f"unknown signal name {name!r}; known: {KNOWN_SIGNALS}")
    return out


def make_cleaning_chisq(
    cl_dict: Mapping[int, np.ndarray],
    cov_dict: Mapping[tuple[int, int], np.ndarray],
    freqs: Sequence[int],
    ells: np.ndarray,
    bandpasses: Sequence[str | Bandpass],
    signal_names: Sequence[str],
    param_names: Sequence[str],
    *,
    objective: str = "chi2",
    tsz_tempfactor: bool = True,
    verbose: bool = False,
    **separation_kwargs,
) -> Callable[[np.ndarray], float]:
    """Return ``f(theta) -> float`` for the SED nuisance parameters ``param_names``.

    ``objective="chi2"`` returns the best-fit :math:`\\chi^2` (the legacy
    ``chisq_multclean`` behaviour, fed straight to ``scipy.optimize.minimize``);
    ``objective="neglogL"`` returns ``-2 log L`` including the ``log|F|`` term.

    Extra ``separation_kwargs`` (``ell_min``, ``ell_max``, ``beams``, ``pixwin``,
    ``method``, ...) are forwarded to
    :func:`crosseplib.separation.separate_components`.
    """
    param_names = list(param_names)
    signal_names = list(signal_names)

    def chisq(theta: np.ndarray) -> float:
        params = dict(zip(param_names, np.asarray(theta, dtype=float)))
        responses = build_responses(params, signal_names, tsz_tempfactor=tsz_tempfactor)
        res = separate_components(
            cl_dict,
            cov_dict,
            freqs,
            ells,
            bandpasses,
            responses,
            signal_names=signal_names,
            beta0=params.get("beta0"),
            t0=params.get("T0", params.get("t0")),
            alpha=params.get("alpha"),
            **separation_kwargs,
        )
        value = res.chi2 if objective == "chi2" else -2.0 * res.log_likelihood
        if verbose:
            msg = " ".join(f"{n}={v:.4g}" for n, v in zip(param_names, theta))
            print(f"{msg}  {objective}={value:.6g}", end="\r")
        return float(value)

    if objective not in ("chi2", "neglogL"):
        raise ValueError("objective must be 'chi2' or 'neglogL'")
    return chisq


# --------------------------------------------------------------------------- #
# Fisher / Jacobian (finite differences) -- ported from scripts/mcmc.py
# --------------------------------------------------------------------------- #
def estimate_chisq_hessian(theta_best, chisq, epsilon=1e-4, bounds=None) -> np.ndarray:
    """Central-difference Hessian ``d^2 chi2 / dtheta_i dtheta_j``."""
    theta_best = np.asarray(theta_best, dtype=float)
    n = len(theta_best)
    H = np.zeros((n, n))
    f0 = chisq(theta_best)
    steps = epsilon * np.maximum(1.0, np.abs(theta_best))

    for i in range(n):
        hi = steps[i]
        tp, tm = theta_best.copy(), theta_best.copy()
        tp[i] += hi
        tm[i] -= hi
        if bounds is not None:
            lo, hi_b = bounds[i]
            if tp[i] > hi_b or tm[i] < lo:
                raise ValueError(f"parameter {i} too close to a bound for central differences")
        H[i, i] = (chisq(tp) - 2.0 * f0 + chisq(tm)) / hi**2

        for j in range(i + 1, n):
            hj = steps[j]
            tpp, tpm, tmp, tmm = (theta_best.copy() for _ in range(4))
            tpp[i] += hi; tpp[j] += hj
            tpm[i] += hi; tpm[j] -= hj
            tmp[i] -= hi; tmp[j] += hj
            tmm[i] -= hi; tmm[j] -= hj
            hij = (chisq(tpp) - chisq(tpm) - chisq(tmp) + chisq(tmm)) / (4.0 * hi * hj)
            H[i, j] = H[j, i] = hij
    return H


def estimate_fisher_matrix(theta_best, chisq, epsilon=1e-4, bounds=None) -> np.ndarray:
    """Fisher matrix for a Gaussian :math:`\\chi^2` likelihood: ``0.5 * Hessian``."""
    return 0.5 * estimate_chisq_hessian(theta_best, chisq, epsilon=epsilon, bounds=bounds)


def numerical_jacobian(func, x, epsilon=1e-4, bounds=None) -> np.ndarray:
    """Central-difference Jacobian ``J_ij = d func_i / d x_j`` (one-sided near bounds)."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(func(x), dtype=float))
    J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        h = epsilon * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        central = True
        if bounds is not None:
            lo, hi = bounds[j]
            if (lo is not None and xm[j] < lo) or (hi is not None and xp[j] > hi):
                central = False
        if central:
            J[:, j] = (np.asarray(func(xp), dtype=float) - np.asarray(func(xm), dtype=float)) / (2.0 * h)
        else:
            J[:, j] = (np.atleast_1d(np.asarray(func(xp), dtype=float)) - f0) / h
    return J


# --------------------------------------------------------------------------- #
# priors / posterior
# --------------------------------------------------------------------------- #
def flat_log_prior(theta, bounds) -> float:
    for val, (lo, hi) in zip(np.asarray(theta), bounds):
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0


def log_posterior_chisq(theta, chisq_func, bounds) -> float:
    """``log-prior - 0.5 * chi2`` (chi2 already ``-2 log L`` in the caller's units)."""
    lp = flat_log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    try:
        chi2 = chisq_func(theta)
    except Exception:
        return -np.inf
    if not np.isfinite(chi2):
        return -np.inf
    return lp - 0.5 * chi2


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    param_names: list[str]
    best_fit: np.ndarray
    chi2: float
    fisher: np.ndarray | None = None
    covariance: np.ndarray | None = None
    errors: np.ndarray | None = None
    chain: np.ndarray | None = None
    log_prob: np.ndarray | None = None
    minimize_result: object = field(default=None, repr=False)

    def summary(self) -> str:
        lines = []
        errs = self.errors if self.errors is not None else [np.nan] * len(self.best_fit)
        for name, val, err in zip(self.param_names, self.best_fit, errs):
            lines.append(f"  {name:8s} = {val:.5g} +/- {err:.3g}")
        lines.append(f"  chi2 = {self.chi2:.6g}")
        return "\n".join(lines)


def fit_sed_nuisance(
    chisq: Callable[[np.ndarray], float],
    param_names: Sequence[str],
    x0: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    minimizer: str = "L-BFGS-B",
    fisher: bool = True,
    fisher_epsilon: float = 1e-4,
    mcmc: bool = False,
    nwalkers: int = 16,
    nsteps: int = 4000,
    burnin: int = 1000,
    thin: int = 10,
    init_scatter: float = 1e-2,
    random_seed: int | None = None,
    progress: bool = True,
) -> FitResult:
    """Minimise ``chisq`` over the SED nuisance parameters, then optionally sample.

    ``chisq`` is typically the closure from :func:`make_cleaning_chisq`.
    """
    from scipy.optimize import minimize

    param_names = list(param_names)
    res = minimize(chisq, x0=np.asarray(x0, dtype=float), bounds=bounds, method=minimizer)
    best = np.asarray(res.x, dtype=float)

    out = FitResult(param_names=param_names, best_fit=best, chi2=float(chisq(best)), minimize_result=res)

    if fisher:
        try:
            F = estimate_fisher_matrix(best, chisq, epsilon=fisher_epsilon, bounds=bounds)
            cov = np.linalg.inv(F)
            out.fisher = F
            out.covariance = cov
            out.errors = np.sqrt(np.diag(cov))
        except (ValueError, np.linalg.LinAlgError):
            pass

    if mcmc:
        out.chain, out.log_prob = _run_emcee(
            chisq, bounds, start=best, nwalkers=nwalkers, nsteps=nsteps, burnin=burnin,
            thin=thin, init_scatter=init_scatter, random_seed=random_seed, progress=progress,
        )
        p16, p50, p84 = np.percentile(out.chain, [16, 50, 84], axis=0)
        out.best_fit = p50
        if out.errors is None:
            out.errors = 0.5 * (p84 - p16)

    return out


def _run_emcee(chisq, bounds, *, start, nwalkers, nsteps, burnin, thin, init_scatter, random_seed, progress):
    import emcee

    rng = np.random.default_rng(random_seed)
    bounds = np.asarray(bounds, dtype=float)
    ndim = len(bounds)
    if nwalkers <= 2 * ndim:
        raise ValueError(f"nwalkers must exceed 2*ndim = {2 * ndim}")

    lows, highs = bounds[:, 0], bounds[:, 1]
    widths = highs - lows
    start = np.clip(np.asarray(start, dtype=float), lows, highs)

    p0 = np.empty((nwalkers, ndim))
    for i in range(nwalkers):
        trial = start + init_scatter * widths * rng.normal(size=ndim)
        tries = 0
        while np.any(trial < lows) or np.any(trial > highs):
            trial = start + init_scatter * widths * rng.normal(size=ndim)
            tries += 1
            if tries > 100:
                trial = rng.uniform(lows, highs, size=ndim)
                break
        p0[i] = trial

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_chisq, args=(chisq, bounds))
    sampler.run_mcmc(p0, nsteps, progress=progress)
    chain = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    return chain, log_prob
