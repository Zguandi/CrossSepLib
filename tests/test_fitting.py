import numpy as np
import pytest

from crosseplib.bandpass_beam import bandpass_weighted_mean
from crosseplib.fitting import (
    build_responses,
    estimate_fisher_matrix,
    make_cleaning_chisq,
    numerical_jacobian,
    fit_sed_nuisance,
)
from crosseplib.separation import build_gmatrix


def test_build_responses_count_and_labels():
    r = build_responses({"beta0": 1.7, "T0": 10.7, "alpha": -2.1},
                        ["tSZ", "free-free", "CIB-amp", "CIB-beta"])
    assert len(r) == 4
    nu = np.array([100.0, 353.0])
    # CIB-beta == CIB-amp * ln(nu/nu0)
    np.testing.assert_allclose(r[3](nu), r[2](nu) * np.log(nu / 220.0), rtol=1e-12)


def test_build_responses_rejects_unknown():
    with pytest.raises(ValueError):
        build_responses({}, ["mystery-foreground"])


def test_build_responses_cib_needs_params():
    with pytest.raises(ValueError):
        build_responses({"alpha": -2.1}, ["CIB-amp"])


def test_numerical_jacobian_linear():
    A = np.array([[2.0, 0.0], [1.0, -3.0], [0.0, 5.0]])
    J = numerical_jacobian(lambda x: A @ x, np.array([1.0, 1.0]))
    np.testing.assert_allclose(J, A, atol=1e-6)


def test_estimate_fisher_matrix_quadratic():
    # chi2 = (x - x0)^T M (x - x0)  ->  Fisher = M
    M = np.array([[3.0, 1.0], [1.0, 2.0]])
    x0 = np.array([0.5, -0.2])
    chi2 = lambda x: float((x - x0) @ M @ (x - x0))  # noqa: E731
    F = estimate_fisher_matrix(x0 + 0.1, chi2, epsilon=1e-3)
    np.testing.assert_allclose(F, M, atol=1e-4)


def _synthetic_planck(true_params, signal_names, freqs, ells):
    """Forward-model a noise-free data vector for the given SED params."""
    nbins = len(ells)
    responses = build_responses(true_params, signal_names, tsz_tempfactor=False)
    bandpasses = [(np.linspace(f - 1.0, f + 1.0, 11), np.ones(11)) for f in freqs]
    g = np.array([[bandpass_weighted_mean(bp, r) for r in responses] for bp in bandpasses])
    G = build_gmatrix(g, nbins)
    rng = np.random.default_rng(7)
    s_true = np.concatenate([rng.uniform(0.5, 1.5, nbins) for _ in signal_names])
    cl = G @ s_true
    cl_dict = {f: cl[i * nbins:(i + 1) * nbins] for i, f in enumerate(freqs)}
    cov_dict = {
        (f1, f2): (np.eye(nbins) * 1e-8 if f1 == f2 else np.zeros((nbins, nbins)))
        for i, f1 in enumerate(freqs)
        for f2 in freqs[i:]
    }
    return cl_dict, cov_dict, bandpasses


def test_cleaning_chisq_minimised_at_truth():
    freqs = [30, 100, 143, 217, 353, 545]
    ells = np.arange(50, 350, 50, dtype=float)
    signal_names = ["tSZ", "CIB-amp", "CIB-beta"]
    true = {"beta0": 1.7, "T0": 12.0}
    cl_dict, cov_dict, bandpasses = _synthetic_planck(true, signal_names, freqs, ells)

    chisq = make_cleaning_chisq(
        cl_dict, cov_dict, freqs, ells, bandpasses, signal_names,
        param_names=["beta0", "T0"], tsz_tempfactor=False,
        ell_min=0, ell_max=1e9, method="qr",
    )
    at_truth = chisq([1.7, 12.0])
    off = chisq([1.2, 20.0])
    assert at_truth < 1e-6
    assert off > at_truth

    fit = fit_sed_nuisance(chisq, ["beta0", "T0"], x0=[1.4, 16.0],
                           bounds=[(0.5, 3.0), (5.0, 40.0)], fisher=False)
    np.testing.assert_allclose(fit.best_fit, [1.7, 12.0], rtol=2e-2)
