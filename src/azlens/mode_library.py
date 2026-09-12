"""Reusable projected-axis-ratio mode library for population field evaluation.

The population calculation evaluates many halos with different projected axis
ratios.  This module wraps the already validated Fourier--Green multipole layer
in a stored q--x library and performs the publication interpolation explicitly.
The public scientific path rejects out-of-domain evaluations by default instead
of silently clipping radii to the library bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .multipoles import EllipticalModeGrid, build_stored_elliptical_mode_grid
from .solver_settings import POPULATION_MODE_SETTINGS, SolverSettings
from .ring_fields import RingFields

FloatArray = NDArray[np.float64]

_COEFF_NAMES = (
    "kappa_cos", "kappa_sin",
    "gamma_plus_cos", "gamma_plus_sin",
    "gamma_cross_cos", "gamma_cross_sin",
)


@dataclass(frozen=True)
class ModeLibrary:
    """Stored unit-amplitude q--x multipole coefficient library."""

    grid: EllipticalModeGrid

    @classmethod
    def build(cls, settings: SolverSettings = POPULATION_MODE_SETTINGS) -> "ModeLibrary":
        """Build a unit-kappa_s library from a named solver setting."""
        return cls(build_stored_elliptical_mode_grid(settings, kappa_s=1.0))

    @property
    def q_grid(self) -> FloatArray:
        return np.asarray(self.grid.q_grid, dtype=np.float64)

    @property
    def x_grid(self) -> FloatArray:
        return np.asarray(self.grid.radius, dtype=np.float64)

    @property
    def m_max(self) -> int:
        return int(self.grid.kappa_cos.shape[-1] - 1)

    @property
    def radial_interpolation(self) -> str:
        return str(self.grid.radial_interpolation)

    def save_npz(self, path: str | Path, *, extra_metadata: dict | None = None) -> None:
        """Save the stored coefficient library with a compact JSON metadata field."""
        target = Path(path)
        metadata = {
            "settings_name": self.grid.settings_name,
            "radial_interpolation": self.grid.radial_interpolation,
            "m_max": self.m_max,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        np.savez_compressed(
            target,
            q_grid=self.q_grid,
            radius=self.x_grid,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            **{name: getattr(self.grid, name) for name in _COEFF_NAMES},
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "ModeLibrary":
        """Load a stored coefficient library written by :meth:`save_npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            grid = EllipticalModeGrid(
                q_grid=np.asarray(data["q_grid"], dtype=np.float64),
                radius=np.asarray(data["radius"], dtype=np.float64),
                settings_name=str(metadata.get("settings_name", "loaded")),
                radial_interpolation=str(metadata["radial_interpolation"]),
                **{name: np.asarray(data[name], dtype=np.float64) for name in _COEFF_NAMES},
            )
        return cls(grid)

    def evaluator_for_q(self, q_perp: ArrayLike) -> "ModeLibraryEvaluator":
        """Return an evaluator with per-halo projected axis ratios fixed."""
        return ModeLibraryEvaluator(self, q_perp)


@dataclass(frozen=True)
class ModeLibraryEvaluator:
    """True-center field evaluator for a supplied per-halo q_perp array."""

    library: ModeLibrary
    q_perp: ArrayLike
    bounds_policy: str = "raise"

    def __post_init__(self) -> None:
        q = np.asarray(self.q_perp, dtype=np.float64).reshape(-1)
        if q.size < 1 or np.any(~np.isfinite(q)) or np.any(q <= 0.0) or np.any(q > 1.0):
            raise ValueError("q_perp must contain finite values in (0,1]")
        qmin, qmax = self.library.q_grid[0], self.library.q_grid[-1]
        if q.min() < qmin or q.max() > qmax:
            raise ValueError("q_perp lies outside the mode-library q range")
        if self.bounds_policy not in ("raise", "clip_for_legacy_comparison"):
            raise ValueError("bounds_policy must be 'raise' or 'clip_for_legacy_comparison'")
        object.__setattr__(self, "q_perp", q)

    def _prepare(self, x: ArrayLike, phi: ArrayLike, kappa_s: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        radii = np.asarray(x, dtype=np.float64)
        angles = np.asarray(phi, dtype=np.float64)
        if radii.shape != angles.shape or radii.ndim != 2 or radii.shape[0] != self.q_perp.size:
            raise ValueError("x and phi must have shape (n_halo,n_phi) matching q_perp")
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0) or np.any(~np.isfinite(angles)):
            raise ValueError("x and phi must be finite with positive radii")
        x_nodes = self.library.x_grid
        if self.bounds_policy == "raise":
            if radii.min() < x_nodes[0] or radii.max() > x_nodes[-1]:
                raise ValueError("x values lie outside the mode-library range; the public path does not silently clip")
            clipped = radii
        else:
            clipped = np.clip(radii, x_nodes[0], x_nodes[-1])
        amplitude = np.asarray(kappa_s, dtype=np.float64)
        try:
            amplitude = np.broadcast_to(amplitude, radii.shape)
        except ValueError as exc:
            raise ValueError("kappa_s must broadcast to x shape") from exc
        if np.any(~np.isfinite(amplitude)) or np.any(amplitude < 0.0):
            raise ValueError("kappa_s must be finite and non-negative")
        return clipped, angles, amplitude, np.broadcast_to(self.q_perp[:, None], radii.shape)

    def _interpolate_array(self, coefficient: FloatArray, x: FloatArray, q: FloatArray) -> FloatArray:
        q_nodes = self.library.q_grid
        x_nodes = self.library.x_grid
        flat_x = x.reshape(-1)
        flat_q = q.reshape(-1)
        if self.library.radial_interpolation == "x":
            xp = x_nodes
            points = flat_x
        elif self.library.radial_interpolation == "logx":
            xp = np.log(x_nodes)
            points = np.log(flat_x)
        else:
            raise ValueError("unsupported library radial interpolation")

        if q_nodes.size == 1:
            if np.any(np.abs(flat_q - q_nodes[0]) > 1.0e-13):
                raise ValueError("single-q library received a different q")
            out = np.empty((flat_x.size, coefficient.shape[-1]), dtype=np.float64)
            for m in range(coefficient.shape[-1]):
                out[:, m] = np.interp(points, xp, coefficient[0, :, m])
            return out.reshape(x.shape + (coefficient.shape[-1],))

        lo = np.searchsorted(q_nodes, flat_q, side="right") - 1
        lo = np.clip(lo, 0, q_nodes.size - 2)
        hi = lo + 1
        denom = q_nodes[hi] - q_nodes[lo]
        w = np.where(denom > 0.0, (flat_q - q_nodes[lo]) / denom, 0.0)
        out = np.empty((flat_x.size, coefficient.shape[-1]), dtype=np.float64)
        unique_lo = np.unique(lo)
        for m in range(coefficient.shape[-1]):
            lower = np.empty(flat_x.size, dtype=np.float64)
            upper = np.empty(flat_x.size, dtype=np.float64)
            for idx in unique_lo:
                mask = lo == idx
                lower[mask] = np.interp(points[mask], xp, coefficient[idx, :, m])
                upper[mask] = np.interp(points[mask], xp, coefficient[idx + 1, :, m])
            out[:, m] = (1.0 - w) * lower + w * upper
        return out.reshape(x.shape + (coefficient.shape[-1],))

    def _field_from_coefficients(self, x: FloatArray, phi: FloatArray, cosine: FloatArray, sine: FloatArray) -> FloatArray:
        ac = self._interpolate_array(cosine, x, np.broadcast_to(self.q_perp[:, None], x.shape))
        bs = self._interpolate_array(sine, x, np.broadcast_to(self.q_perp[:, None], x.shape))
        modes = np.arange(cosine.shape[-1], dtype=np.float64)
        phase = phi[..., None] * modes
        return np.asarray(np.sum(ac * np.cos(phase) + bs * np.sin(phase), axis=-1), dtype=np.float64)

    def evaluate(self, x: ArrayLike, phi: ArrayLike, kappa_s: ArrayLike) -> RingFields:
        """Evaluate true-center fields at q_perp, x, and phi before spin rotation."""
        radii, angles, amplitude, _ = self._prepare(x, phi, kappa_s)
        grid = self.library.grid
        return RingFields(
            kappa=self._field_from_coefficients(radii, angles, grid.kappa_cos, grid.kappa_sin) * amplitude,
            gamma_plus=self._field_from_coefficients(radii, angles, grid.gamma_plus_cos, grid.gamma_plus_sin) * amplitude,
            gamma_cross=self._field_from_coefficients(radii, angles, grid.gamma_cross_cos, grid.gamma_cross_sin) * amplitude,
        )
