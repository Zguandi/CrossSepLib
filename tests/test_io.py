import numpy as np
import pytest

from crosseplib.io import load_clcov_act, load_clcov_planck, load_separation, save_separation
from crosseplib.separation import SeparationResult
from crosseplib.spectra import check_clcov_file, expected_planck_keys


def _write_planck_npz(path, freqs, n_ell):
    rng = np.random.default_rng(42)
    payload = {"ells": np.arange(n_ell, dtype=float)}
    for f in freqs:
        payload[f"cell_{f}g"] = rng.normal(size=n_ell)
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i:]:
            payload[f"cov_{f1}g{f2}g"] = rng.normal(size=(n_ell, n_ell))
    np.savez(path, **payload)


def test_load_clcov_planck_infers_freqs(tmp_path):
    freqs = [30, 100, 217]
    p = tmp_path / "planck_allfreq.npz"
    _write_planck_npz(p, freqs, n_ell=8)

    ells, cl_dict, cov_dict = load_clcov_planck(p)
    assert list(cl_dict) == freqs
    assert ells.shape == (8,)
    assert (100, 217) in cov_dict
    assert cov_dict[(30, 100)].shape == (8, 8)


def test_load_clcov_planck_missing_block_raises(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, ells=np.arange(4.0), cell_30g=np.zeros(4), cov_30g30g=np.zeros((4, 4)),
             cell_100g=np.zeros(4))  # no cov_30g100g / cov_100g100g
    with pytest.raises(KeyError):
        load_clcov_planck(p)


def test_load_clcov_act_single_file(tmp_path):
    n_ell = 6
    payload = {"ells": np.arange(n_ell, dtype=float)}
    for f in (90, 150, 220):
        payload[f"cl_gy_{f}"] = np.ones(n_ell) * f
    for i, f1 in enumerate((90, 150, 220)):
        for f2 in (90, 150, 220)[i:]:
            payload[f"cov{f1}g{f2}g"] = np.eye(n_ell)
    p = tmp_path / "cl_cov_act.npz"
    np.savez(p, **payload)

    ells, cl_dict, cov_dict = load_clcov_act(p)
    assert cl_dict[150].shape == (n_ell,)
    assert set(cov_dict) == {(90, 90), (90, 150), (90, 220), (150, 150), (150, 220), (220, 220)}


def test_save_and_load_separation_roundtrip(tmp_path):
    n_ell = 5
    names = ["tSZ", "CIB-amp"]
    res = SeparationResult(
        ells=np.linspace(200, 600, n_ell),
        signal_names=names,
        cleaned_signals=np.arange(2 * n_ell, dtype=float),
        errors=np.ones(2 * n_ell) * 0.1,
        inv_fisher=np.eye(2 * n_ell),
        chi2=12.3,
        log_det_fisher=4.5,
        log_likelihood=-8.4,
        beta0=1.7,
        t0=10.7,
        comments="unit test",
    )
    path = save_separation(tmp_path, res, filename="multiclean_test.npz")
    back = load_separation(path)

    assert back.signal_names == names
    np.testing.assert_allclose(back.cleaned_signals, res.cleaned_signals)
    np.testing.assert_allclose(back.inv_fisher, res.inv_fisher)
    assert back.chi2 == pytest.approx(12.3)
    assert back.beta0 == pytest.approx(1.7)
    assert back.comments == "unit test"

    cl, err = back.signal("CIB-amp")
    assert cl.shape == (n_ell,)


def test_check_clcov_file_reports_ok(tmp_path):
    freqs = [30, 143, 353]
    p = tmp_path / "ok.npz"
    _write_planck_npz(p, freqs, n_ell=7)
    report = check_clcov_file(p)
    assert report["ok"]
    assert report["freqs"] == freqs
    assert report["n_ell"] == 7
    assert report["missing"] == []


def test_expected_planck_keys():
    keys = expected_planck_keys([30, 44])
    assert keys == ["ells", "cell_30g", "cell_44g", "cov_30g30g", "cov_30g44g", "cov_44g44g"]
