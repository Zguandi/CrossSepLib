"""Stub: angular power spectra + covariance live in a future ``AngClLib``.

CrossSepLib deliberately does **not** estimate :math:`C_\\ell` or build the
Gaussian / jackknife covariance. That NaMaster-based machinery -- plus the map
and mask preprocessing that feeds it -- is being moved into a separate library,
``AngClLib``. Its reference implementations are:

* ``scripts/gaussian_covariance_nmt.py``  -- workspaces, decoupling, Gaussian cov
* ``scripts/actplanckmaps.py``            -- Planck/ACT map + mask readers
* ``scripts/jacknife_namaster.py``        -- k-means region jackknife
* ``tasks/planck_all_cov_pla.py``         -- the memory-light all-frequency driver

This module only defines / checks the ``.npz`` schema that CrossSepLib expects
as **input**, so the boundary between the two libraries is explicit in code.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

__all__ = ["expected_planck_keys", "check_clcov_file", "AngClLibNotAvailable"]


class AngClLibNotAvailable(NotImplementedError):
    """Raised by the placeholder spectrum entry points."""


def compute_cross_spectra(*_args, **_kwargs):
    """Placeholder -- use ``AngClLib`` (see module docstring)."""
    raise AngClLibNotAvailable(
        "C_ell estimation is out of scope for CrossSepLib; it belongs to AngClLib. "
        "Reference: scripts/gaussian_covariance_nmt.py."
    )


def compute_gaussian_covariance(*_args, **_kwargs):
    """Placeholder -- use ``AngClLib`` (see module docstring)."""
    raise AngClLibNotAvailable(
        "Gaussian covariance is out of scope for CrossSepLib; it belongs to AngClLib. "
        "Reference: scripts/gaussian_covariance_nmt.py, tasks/planck_all_cov_pla.py."
    )


def expected_planck_keys(freqs: Sequence[int]) -> list[str]:
    """The full set of keys a Planck-style cl/cov ``.npz`` must contain."""
    freqs = [int(f) for f in freqs]
    keys = ["ells"]
    keys += [f"cell_{f}g" for f in freqs]
    keys += [f"cov_{f1}g{f2}g" for i, f1 in enumerate(freqs) for f2 in freqs[i:]]
    return keys


def check_clcov_file(
    path: str | os.PathLike,
    freqs: Sequence[int] | None = None,
) -> dict:
    """Validate an input cl/cov ``.npz`` against the expected schema.

    Returns a report dict with ``ok``, ``present``, ``missing``, ``n_ell`` and
    the inferred ``freqs``. Raises nothing -- inspect the report.
    """
    data = np.load(path)
    present = set(data.files)

    if freqs is None:
        freqs = sorted(
            int(k[len("cell_") : -1])
            for k in present
            if k.startswith("cell_") and k.endswith("g") and k[len("cell_") : -1].isdigit()
        )
    freqs = [int(f) for f in freqs]

    expected = expected_planck_keys(freqs)
    missing = [k for k in expected if k not in present]
    n_ell = int(len(data["ells"])) if "ells" in present else None

    shape_problems = []
    for f in freqs:
        key = f"cell_{f}g"
        if key in present and n_ell is not None and data[key].ravel().shape[0] != n_ell:
            shape_problems.append(f"{key} has length {data[key].ravel().shape[0]}, expected {n_ell}")
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i:]:
            key = f"cov_{f1}g{f2}g"
            if key in present and n_ell is not None and data[key].shape != (n_ell, n_ell):
                shape_problems.append(f"{key} has shape {data[key].shape}, expected {(n_ell, n_ell)}")

    return {
        "ok": not missing and not shape_problems,
        "freqs": freqs,
        "n_ell": n_ell,
        "present": sorted(present),
        "missing": missing,
        "shape_problems": shape_problems,
    }
