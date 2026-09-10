import numpy as np
import pytest

from crosseplib.separation import (
    build_gmatrix,
    gls_solve,
    separate_components,
    stack_covariance,
    stack_cl_vector,
)


def test_build_gmatrix_block_structure():
    g = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    nbins = 4
    G = build_gmatrix(g, nbins)
    assert G.shape == (3 * nbins, 2 * nbins)
    # block (i, j) must be g[i, j] * I
    np.testing.assert_allclose(G[0:nbins, 0:nbins], 1.0 * np.eye(nbins))
    np.testing.assert_allclose(G[nbins:2 * nbins, nbins:2 * nbins], 4.0 * np.eye(nbins))


def test_build_gmatrix_beam_scales_rows():
    g = np.array([[1.0, 0.0], [0.0, 1.0]])
    nbins = 3
    beams = np.arange(1, 2 * nbins + 1, dtype=float)
    G = build_gmatrix(g, nbins, beams=beams)
    np.testing.assert_allclose(np.diag(G), beams)


@pytest.mark.parametrize("method", ["qr", "inv"])
def test_gls_recovers_truth_identity_cov(method):
    rng = np.random.default_rng(0)
    n_freq, n_sig, nbins = 5, 3, 6
    g = rng.uniform(-2, 2, size=(n_freq, n_sig))
    G = build_gmatrix(g, nbins)
    s_true = rng.normal(size=n_sig * nbins)
    d = G @ s_true
    cov = np.eye(n_freq * nbins)

    res = gls_solve(d, G, cov, method=method)
    np.testing.assert_allclose(res.amplitudes, s_true, atol=1e-8)
    assert res.chi2 == pytest.approx(0.0, abs=1e-8)


def test_gls_qr_and_inv_agree_with_noise():
    rng = np.random.default_rng(1)
    n_freq, n_sig, nbins = 6, 2, 5
    g = rng.uniform(-2, 2, size=(n_freq, n_sig))
    G = build_gmatrix(g, nbins)
    s_true = rng.normal(size=n_sig * nbins)
    A = rng.normal(size=(n_freq * nbins, n_freq * nbins))
    cov = A @ A.T + n_freq * nbins * np.eye(n_freq * nbins)
    d = G @ s_true + np.linalg.cholesky(cov) @ rng.normal(size=n_freq * nbins) * 0.01

    qr = gls_solve(d, G, cov, method="qr")
    inv = gls_solve(d, G, cov, method="inv")
    np.testing.assert_allclose(qr.amplitudes, inv.amplitudes, rtol=1e-6)
    np.testing.assert_allclose(qr.inv_fisher, inv.inv_fisher, rtol=1e-5)
    assert qr.chi2 == pytest.approx(inv.chi2, rel=1e-6)


def test_stack_covariance_uses_transpose_for_missing_pair():
    b1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b12 = np.array([[10.0, 20.0], [30.0, 40.0]])
    b2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    cov_dict = {(90, 90): b1, (90, 150): b12, (150, 150): b2}
    full = stack_covariance(cov_dict, [90, 150])
    np.testing.assert_allclose(full[2:, :2], b12.T)
    np.testing.assert_allclose(full[:2, 2:], b12)


def test_stack_covariance_and_cl_respect_mask():
    cov_dict = {(90, 90): np.diag([1.0, 2.0, 3.0, 4.0])}
    cl_dict = {90: np.array([10.0, 20.0, 30.0, 40.0])}
    mask = np.array([True, False, True, False])
    assert stack_covariance(cov_dict, [90], mask).shape == (2, 2)
    np.testing.assert_allclose(stack_cl_vector(cl_dict, [90], mask), [10.0, 30.0])


def test_separate_components_recovers_injected_spectra():
    # Forward-model with tophat bandpasses, then check we get the signals back.
    rng = np.random.default_rng(2)
    freqs = [30, 100, 143, 217, 353]
    ells = np.arange(50, 400, 50, dtype=float)
    nbins = len(ells)
    signal_names = ["tSZ", "CIB-amp"]

    from crosseplib.fitting import build_responses

    params = {"beta0": 1.7, "T0": 10.7}
    responses = build_responses(params, signal_names, tsz_tempfactor=False)

    # narrow tophat bandpasses around each band centre
    bandpasses = [(np.linspace(f - 1.0, f + 1.0, 9), np.ones(9)) for f in freqs]
    g = np.array([[bwm(bp, r) for r in responses] for bp in bandpasses])
    G = build_gmatrix(g, nbins)

    s_true = np.concatenate([rng.uniform(1, 2, nbins), rng.uniform(0.1, 0.5, nbins)])
    cl_stacked = G @ s_true
    cl_dict = {f: cl_stacked[i * nbins:(i + 1) * nbins] for i, f in enumerate(freqs)}
    cov_dict = {
        (f1, f2): (np.eye(nbins) * 1e-6 if f1 == f2 else np.zeros((nbins, nbins)))
        for i, f1 in enumerate(freqs)
        for f2 in freqs[i:]
    }

    res = separate_components(
        cl_dict, cov_dict, freqs, ells, bandpasses, responses,
        signal_names=signal_names, ell_min=0, ell_max=1e9, method="qr",
    )
    got = res.signals_2d()
    np.testing.assert_allclose(got[0], s_true[:nbins], rtol=1e-4)
    np.testing.assert_allclose(got[1], s_true[nbins:], rtol=1e-4)
    assert res.signal("tSZ")[0].shape == (nbins,)


def bwm(bp, response):
    from crosseplib.bandpass_beam import bandpass_weighted_mean

    return bandpass_weighted_mean(bp, response)
