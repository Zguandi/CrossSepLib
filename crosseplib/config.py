"""Configuration for CrossSepLib.

A single explicit ``Config`` object replaces the old ``scripts/pathconfig.py``
wildcard import and the import-time ``config.yaml`` reads scattered through the
legacy code. Nothing here touches the filesystem at import time.

The YAML schema understood by :meth:`Config.from_yaml` is the one used by the
legacy repo's ``config.yaml``::

    physics:   {T_CMB: ...}
    planck:    {freq_GHz: [...], beams_fwhm_arcmin: [...],
                lfi_freqs_GHz: [...], hfi_freqs_GHz: [...],
                used_freqs_GHz: [...]}
    act:       {freq_GHz: [...]}
    cosmology: {Omega_c: ..., Omega_b: ..., h: ..., sigma8: ..., n_s: ...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import yaml

# Reference SED temperature. The legacy ``scripts/signal_cleansing.py`` used
# 2.7255 for every published separation result; ``config.yaml`` and the GNFW
# configs carry 2.725458 (Fixsen 2009). We keep 2.7255 as the default so results
# reproduce, and expose it so callers can override. See
# ``../unWISE_tsz_correlator/LATENT_BUGS.md``.
DEFAULT_T_CMB = 2.7255

_DEFAULT_PLANCK_FREQS = (30, 44, 70, 100, 143, 217, 353, 545, 857)
_DEFAULT_PLANCK_FWHM_ARCMIN = (
    33.10265212, 27.94348615, 13.07645961, 9.682, 7.303,
    5.021, 4.944, 4.831, 4.638,
)
_DEFAULT_LFI_FREQS = (30, 44, 70)
_DEFAULT_HFI_FREQS = (100, 143, 217, 353, 545, 857)
_DEFAULT_ACT_FREQS = (90, 150, 220)
_DEFAULT_COSMOLOGY = {
    "Omega_c": 0.264,
    "Omega_b": 0.0493,
    "h": 0.6736,
    "sigma8": 0.8111,
    "n_s": 0.9649,
}

# Environment variable pointing at the directory that holds the maps / .npz
# products. Absent on machines that only run the pure-python parts.
DATA_ROOT_ENV = "CROSSEPLIB_DATA"


@dataclass(frozen=True)
class Config:
    """Immutable analysis configuration.

    Parameters
    ----------
    t_cmb
        CMB monopole temperature [K] used by the SED models.
    planck_freqs, planck_fwhm_arcmin
        Planck frequency list [GHz] and matching Gaussian-beam FWHM [arcmin].
    lfi_freqs, hfi_freqs
        Planck LFI / HFI frequency subsets (used to pick pixel-window nside).
    act_freqs
        ACT frequency list [GHz].
    cosmology
        Cosmological parameters (kept for downstream libraries; unused here).
    data_root
        Directory holding data products. Falls back to ``$CROSSEPLIB_DATA``.
    """

    t_cmb: float = DEFAULT_T_CMB
    planck_freqs: tuple[int, ...] = _DEFAULT_PLANCK_FREQS
    planck_fwhm_arcmin: tuple[float, ...] = _DEFAULT_PLANCK_FWHM_ARCMIN
    lfi_freqs: tuple[int, ...] = _DEFAULT_LFI_FREQS
    hfi_freqs: tuple[int, ...] = _DEFAULT_HFI_FREQS
    act_freqs: tuple[int, ...] = _DEFAULT_ACT_FREQS
    cosmology: dict = field(default_factory=lambda: dict(_DEFAULT_COSMOLOGY))
    data_root: Path | None = None

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def default(cls) -> "Config":
        """Return a config populated from the legacy ``config.yaml`` values."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | os.PathLike, *, data_root: str | os.PathLike | None = None) -> "Config":
        """Build a :class:`Config` from a legacy-style ``config.yaml``."""
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        physics = raw.get("physics", {}) or {}
        planck = raw.get("planck", {}) or {}
        act = raw.get("act", {}) or {}
        cosmo = raw.get("cosmology", {}) or {}

        kwargs: dict = {}
        if "T_CMB" in physics:
            kwargs["t_cmb"] = float(physics["T_CMB"])
        if "freq_GHz" in planck:
            kwargs["planck_freqs"] = tuple(int(f) for f in planck["freq_GHz"])
        if "beams_fwhm_arcmin" in planck:
            kwargs["planck_fwhm_arcmin"] = tuple(float(x) for x in planck["beams_fwhm_arcmin"])
        if "lfi_freqs_GHz" in planck:
            kwargs["lfi_freqs"] = tuple(int(f) for f in planck["lfi_freqs_GHz"])
        if "hfi_freqs_GHz" in planck:
            kwargs["hfi_freqs"] = tuple(int(f) for f in planck["hfi_freqs_GHz"])
        if "freq_GHz" in act:
            kwargs["act_freqs"] = tuple(int(f) for f in act["freq_GHz"])
        if cosmo:
            kwargs["cosmology"] = dict(cosmo)

        cfg = cls(**kwargs)
        if data_root is not None:
            cfg = cfg.with_data_root(data_root)
        return cfg

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def with_data_root(self, data_root: str | os.PathLike) -> "Config":
        """Return a copy with ``data_root`` set."""
        return replace(self, data_root=Path(data_root))

    def resolve_data_root(self) -> Path:
        """Return the data root, consulting ``$CROSSEPLIB_DATA`` as a fallback."""
        if self.data_root is not None:
            return self.data_root
        env = os.environ.get(DATA_ROOT_ENV)
        if env:
            return Path(env)
        raise RuntimeError(
            f"No data root configured. Pass Config(data_root=...) or set ${DATA_ROOT_ENV}."
        )

    def data_path(self, *parts: str | os.PathLike) -> Path:
        """Join ``parts`` onto the resolved data root."""
        return self.resolve_data_root().joinpath(*map(str, parts))

    def fwhm_arcmin(self, freq: int) -> float:
        """Gaussian-beam FWHM [arcmin] for a Planck frequency."""
        try:
            return self.planck_fwhm_arcmin[self.planck_freqs.index(int(freq))]
        except ValueError as exc:
            raise KeyError(f"frequency {freq} not in planck_freqs {self.planck_freqs}") from exc

    def pixwin_nside(self, freq: int) -> int:
        """HEALPix nside whose pixel window applies to a given Planck frequency."""
        f = int(freq)
        if f in self.lfi_freqs:
            return 1024
        if f in self.hfi_freqs:
            return 2048
        raise KeyError(f"frequency {freq} is neither LFI {self.lfi_freqs} nor HFI {self.hfi_freqs}")

    def is_act(self, freqs: Sequence[int]) -> bool:
        return all(int(f) in self.act_freqs for f in freqs)

    def is_planck(self, freqs: Sequence[int]) -> bool:
        return all(int(f) in self.planck_freqs for f in freqs)
