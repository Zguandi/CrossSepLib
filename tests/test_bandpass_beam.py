import numpy as np
import pytest

from crosseplib import bandpass_beam as bb


def test_gaussian_beam_starts_at_one_and_decreases():
    ell = np.arange(0, 3000, 50)
    b = bb.gaussian_beam(ell, fwhm_arcmin=10.0)
    assert b[0] == pytest.approx(1.0)
    assert np.all(np.diff(b) <= 0)
    assert b[-1] < b[0]


def test_bandpass_weighted_mean_of_constant_is_that_constant():
    freq = np.linspace(120.0, 180.0, 64)
    trans = np.ones_like(freq)
    val = bb.bandpass_weighted_mean((freq, trans), lambda nu: np.full_like(nu, 3.5))
    assert val == pytest.approx(3.5, rel=1e-10)


def test_bandpass_weighted_mean_narrow_band_approximates_pointwise():
    centre = 143.0
    freq = np.linspace(centre - 0.5, centre + 0.5, 21)
    trans = np.ones_like(freq)
    response = lambda nu: nu**2  # noqa: E731
    val = bb.bandpass_weighted_mean((freq, trans), response)
    assert val == pytest.approx(centre**2, rel=1e-3)


def test_read_beam_file_roundtrip(tmp_path):
    ell_in = np.arange(0, 5000, 10, dtype=float)
    b_in = np.exp(-((ell_in / 2000.0) ** 2))
    path = tmp_path / "beam.txt"
    np.savetxt(path, np.column_stack([ell_in, b_in * 7.0]))  # arbitrary normalisation

    ell = np.array([0.0, 1000.0, 2500.0])
    out = bb.read_beam_file(path, ell)
    assert out[0] == pytest.approx(1.0)  # normalised to first row
    np.testing.assert_allclose(out, np.exp(-((ell / 2000.0) ** 2)), rtol=1e-6)


def test_stack_beams_gaussian_from_fwhm_list():
    ell = np.arange(0, 1000, 100, dtype=float)
    out = bb.stack_beams([100, 143], ell, fwhm_arcmin=[9.68, 7.30])
    assert out.shape == (2 * len(ell),)
    np.testing.assert_allclose(out[: len(ell)], bb.gaussian_beam(ell, 9.68))


def test_stack_beams_requires_beam_info():
    with pytest.raises(ValueError):
        bb.stack_beams([100], np.arange(10.0))
