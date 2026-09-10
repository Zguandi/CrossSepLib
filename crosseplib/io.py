""".npz readers / writers for the inter-stage data contracts.

Three schemas cross the CrossSepLib boundary:

**Planck cl/cov input** (arbitrary frequency set) -- produced upstream by the
NaMaster pipeline (future ``AngClLib``)::

    ells                 (n_ell,)
    cell_<f>g            (n_ell,)          for each frequency f
    cov_<f1>g<f2>g       (n_ell, n_ell)    for each pair f1 <= f2

**ACT cl/cov input** (frequencies 90 / 150 / 220)::

    ells                 (n_ell,)          optional
    cl_gy_<f>            (n_ell,) or (1, n_ell)
    cov<f1>g<f2>g        (n_ell, n_ell)    for each pair f1 <= f2

**Separation output** (``multiclean_*.npz``)::

    ells                 (n_ell,)
    signal_names         (n_signal,)  str
    cleaned_signals      (n_signal * n_ell,)   flat, signal-major
    errors               (n_signal * n_ell,)
    invF                 (n_signal*n_ell, n_signal*n_ell)
    chisq, logL, logdetF scalars
    beta0, T0, alpha     scalars (may be absent)
    comments             str
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Sequence

import numpy as np

from .separation import SeparationResult

__all__ = [
    "load_clcov_planck",
    "load_clcov_act",
    "save_separation",
    "load_separation",
    "ACT_FREQS",
]

ACT_FREQS = (90, 150, 220)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_clcov_planck(
    path: str | os.PathLike,
    freqs: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Load a Planck-style ``cell_<f>g`` / ``cov_<f1>g<f2>g`` file.

    Frequencies are inferred from the ``cell_*g`` keys when ``freqs`` is None.
    Returns ``(ells, cl_dict, cov_dict)`` in the layout
    :func:`crosseplib.separation.separate_components` expects.
    """
    data = np.load(path)
    ells = np.asarray(data["ells"], dtype=float)

    if freqs is None:
        found = []
        for key in data.files:
            if key.startswith("cell_") and key.endswith("g"):
                try:
                    found.append(int(key[len("cell_") : -1]))
                except ValueError:
                    pass
        freqs = sorted(set(found))
    freqs = [int(f) for f in freqs]

    cl_dict: dict[int, np.ndarray] = {}
    for f in freqs:
        key = f"cell_{f}g"
        if key in data:
            cl_dict[f] = np.asarray(data[key], dtype=float).ravel()
    missing = [f for f in freqs if f not in cl_dict]
    if missing:
        raise KeyError(f"{path}: missing cell_<f>g for frequencies {missing}")

    cov_dict: dict[tuple[int, int], np.ndarray] = {}
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i:]:
            key = f"cov_{f1}g{f2}g"
            if key in data:
                cov_dict[(f1, f2)] = np.asarray(data[key], dtype=float)
    _check_cov_complete(cov_dict, freqs, path)
    return ells, cl_dict, cov_dict


def load_clcov_act(
    path: str | os.PathLike | None = None,
    *,
    cl_path: str | os.PathLike | None = None,
    cov_path: str | os.PathLike | None = None,
    freqs: Sequence[int] = ACT_FREQS,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Load ACT-style ``cl_gy_<f>`` / ``cov<f1>g<f2>g`` data.

    Either pass a single ``path`` holding both, or ``cl_path`` + ``cov_path``.
    A single path with ``cl_`` / ``cov_`` in its name is split automatically
    (matching the legacy ``load_cl_cov_res_plcact`` convention).
    """
    freqs = [int(f) for f in freqs]

    if path is not None and cl_path is None and cov_path is None:
        name = os.fspath(path)
        with np.load(name) as probe:
            has_cov = any(k.startswith("cov") for k in probe.files)
            has_cl = any(k.startswith(("cl_gy_", "cell_", "cl_")) for k in probe.files)
        if has_cov and has_cl:
            cl_path = cov_path = name  # single self-contained file
        elif has_cl:
            cl_path = name
            cov_path = _sibling(name, "cl", "cov")
        elif has_cov:
            cov_path = name
            cl_path = _sibling(name, "cov", "cl")
        else:
            raise KeyError(f"{name}: no cl_gy_<f> / cov<f1>g<f2>g keys found")

    if cl_path is None or cov_path is None:
        raise ValueError("pass either `path`, or both `cl_path` and `cov_path`")

    cl_data = np.load(cl_path)
    cov_data = np.load(cov_path)

    ells = None
    for src in (cl_data, cov_data):
        if "ells" in src.files:
            ells = np.asarray(src["ells"], dtype=float)
            break

    cl_dict: dict[int, np.ndarray] = {}
    for f in freqs:
        for key in (f"cl_gy_{f}", f"cell_{f}g", f"cl_{f}"):
            if key in cl_data.files:
                cl_dict[f] = np.asarray(cl_data[key], dtype=float).ravel()
                break
    missing = [f for f in freqs if f not in cl_dict]
    if missing:
        raise KeyError(f"{cl_path}: missing cl_gy_<f> for frequencies {missing}")

    if ells is None:
        ells = np.arange(len(next(iter(cl_dict.values()))), dtype=float)

    cov_dict: dict[tuple[int, int], np.ndarray] = {}
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i:]:
            for key in (f"cov{f1}g{f2}g", f"cov_{f1}g{f2}g"):
                if key in cov_data.files:
                    cov_dict[(f1, f2)] = np.asarray(cov_data[key], dtype=float)
                    break
    _check_cov_complete(cov_dict, freqs, cov_path)
    return ells, cl_dict, cov_dict


def _sibling(path: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` in the file's basename only."""
    head, tail = os.path.split(os.fspath(path))
    return os.path.join(head, tail.replace(old, new, 1))


def _check_cov_complete(cov_dict: dict, freqs: Sequence[int], path) -> None:
    need = {(f1, f2) for i, f1 in enumerate(freqs) for f2 in list(freqs)[i:]}
    have = set(cov_dict)
    if not need <= have:
        raise KeyError(f"{path}: missing covariance blocks {sorted(need - have)}")


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def save_separation(
    out_dir: str | os.PathLike,
    result: SeparationResult,
    *,
    filename: str | None = None,
    timestamp: bool = True,
) -> str:
    """Write a :class:`SeparationResult` to a ``multiclean``-schema ``.npz``.

    Returns the path written. If ``filename`` is None a name
    ``multiclean_<YYYYMMDDThhmmss>.npz`` is used.
    """
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"multiclean_{stamp}.npz"
    elif timestamp and not filename.endswith(".npz"):
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"{filename}_{stamp}.npz"
    path = os.path.join(out_dir, filename)

    payload = dict(
        ells=result.ells,
        signal_names=np.asarray(result.signal_names, dtype=object),
        cleaned_signals=result.cleaned_signals,
        errors=result.errors,
        invF=result.inv_fisher,
        chisq=result.chi2,
        logL=result.log_likelihood,
        logdetF=result.log_det_fisher,
        comments=result.comments,
    )
    for key, val in (("beta0", result.beta0), ("T0", result.t0), ("alpha", result.alpha)):
        if val is not None:
            payload[key] = val
    np.savez(path, **payload)
    return path


def load_separation(path: str | os.PathLike) -> SeparationResult:
    """Read a ``multiclean``-schema ``.npz`` back into a :class:`SeparationResult`."""
    data = np.load(path, allow_pickle=True)
    names = [str(s) for s in data["signal_names"]]

    def _get(key, default=None):
        return data[key].item() if key in data.files else default

    return SeparationResult(
        ells=np.asarray(data["ells"], dtype=float),
        signal_names=names,
        cleaned_signals=np.asarray(data["cleaned_signals"], dtype=float),
        errors=np.asarray(data["errors"], dtype=float),
        inv_fisher=np.asarray(data["invF"], dtype=float),
        chi2=float(_get("chisq", np.nan)),
        log_det_fisher=float(_get("logdetF", np.nan)),
        log_likelihood=float(_get("logL", np.nan)),
        beta0=_get("beta0"),
        t0=_get("T0"),
        alpha=_get("alpha"),
        comments=str(_get("comments", "")),
    )
