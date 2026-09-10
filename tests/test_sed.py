import numpy as np
import pytest

from crosseplib import sed


def test_tsz_g_null_near_217ghz():
    # The non-relativistic tSZ spectral function crosses zero at ~217 GHz.
    assert sed.tsz_g(90.0) < 0.0
    assert sed.tsz_g(350.0) > 0.0
    zero = sed.tsz_g(217.0)
    assert abs(zero) < 0.05


def test_tsz_g_small_frequency_limit():
    # x coth(x/2) -> 2 as x -> 0, so g -> -2.
    assert sed.tsz_g(1e-3) == pytest.approx(-2.0, abs=1e-4)


def test_tsz_response_tempfactor():
    g = sed.tsz_g(150.0)
    assert sed.tsz_response(150.0, tempfactor=False) == pytest.approx(g)
    assert sed.tsz_response(150.0, tempfactor=True) == pytest.approx(g * sed.T_CMB)


def test_cmb_to_rj_low_frequency_limit():
    # x^2 e^x / (e^x - 1)^2 -> 1 as x -> 0.
    assert sed.cmb_to_rj(1e-3) == pytest.approx(1.0, abs=1e-5)


def test_cib_beta_is_amp_times_log():
    nu = np.array([100.0, 143.0, 353.0, 545.0])
    amp = sed.cib_amplitude_response(nu, beta0=1.7, t0=10.7)
    beta = sed.cib_beta_response(nu, beta0=1.7, t0=10.7)
    np.testing.assert_allclose(beta, amp * np.log(nu / 220.0), rtol=1e-12)


def test_cib_amplitude_normalised_at_nu0():
    # At nu = nu0 the (nu/nu0)**beta0 and relative_bnu factors are 1; the
    # response reduces to 1 / relative_dbdT(nu0, T_CMB).
    val = sed.cib_amplitude_response(220.0, beta0=1.7, t0=10.7, nu0=220.0)
    expected = 1.0 / sed.relative_dbdT(220.0, sed.T_CMB, nu0=220.0)
    assert val == pytest.approx(expected, rel=1e-12)


def test_freefree_matches_manual_formula():
    nu = np.array([30.0, 100.0, 353.0])
    manual = (nu / 220.0) ** -2.14 / sed.cmb_to_rj(nu)
    np.testing.assert_allclose(sed.freefree_response(nu), manual, rtol=1e-12)


def test_planck_derivative_positive():
    nu = np.linspace(30, 900, 20)
    assert np.all(sed.planck_dbdT(nu, 20.0) > 0)
