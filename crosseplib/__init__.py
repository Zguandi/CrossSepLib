"""CrossSepLib -- angular-C_ell CMB component separation for galaxy-tSZ analysis.

See ``README.md`` for scope and the pipeline overview. Typical use::

    from crosseplib.io import load_clcov_planck, save_separation
    from crosseplib.fitting import build_responses
    from crosseplib.separation import separate_components

    ells, cl_dict, cov_dict = load_clcov_planck("planck_allfreq_lowz.npz")
    responses = build_responses({"beta0": 1.7, "T0": 10.7, "alpha": -2.14},
                                ["tSZ", "free-free", "CIB-amp", "CIB-beta"])
    result = separate_components(cl_dict, cov_dict, freqs, ells,
                                 bandpasses, responses,
                                 signal_names=["tSZ", "free-free", "CIB-amp", "CIB-beta"],
                                 ell_min=200, ell_max=2000, method="qr")
    save_separation("out/", result)
"""

from __future__ import annotations

from .config import Config, DEFAULT_T_CMB
from .separation import (
    GLSResult,
    SeparationResult,
    build_gmatrix,
    gls_solve,
    separate_components,
    stack_covariance,
    stack_cl_vector,
)
from .fitting import (
    FitResult,
    build_responses,
    fit_sed_nuisance,
    make_cleaning_chisq,
)
from .io import (
    load_clcov_act,
    load_clcov_planck,
    load_separation,
    save_separation,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DEFAULT_T_CMB",
    "GLSResult",
    "SeparationResult",
    "FitResult",
    "build_gmatrix",
    "gls_solve",
    "separate_components",
    "stack_covariance",
    "stack_cl_vector",
    "build_responses",
    "make_cleaning_chisq",
    "fit_sed_nuisance",
    "load_clcov_planck",
    "load_clcov_act",
    "load_separation",
    "save_separation",
]
