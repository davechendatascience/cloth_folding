"""Cloth error functional J: X -> R_{>=0}  (spec Sec. 0.1, 2.1).

J is the Lyapunov-like functional whose descent the whole pipeline is designed
to enforce:

    J(x) = l1 * (1 - IOU(x, x_target))
         + l2 * edge_gap(x)
         + l3 * wrinkle_global(x)

Design requirements from the spec, and how each is met here:

* ``J`` is continuous.
    - The occupancy mask is built by *soft* Gaussian splatting, not hard
      binning, so the map verts -> occupancy is smooth.
    - soft-IoU uses min/max, which are continuous (piecewise-linear).
    - edge_gap and the wrinkle term are compositions of continuous maps.

* ``J(x) = 0`` iff ``x`` is at the folded target.
    - Each of the three terms is >= 0 and vanishes exactly when the
      corresponding descriptor of ``x`` matches the target's.
    - Note the wrinkle term is written as ``|R(x) - R(x_target)|`` rather than
      ``R(x)``. A folded garment is *not* flat -- it has curvature along the
      fold line -- so penalising absolute roughness would put the zero of J at
      a flat (unfolded) sheet, contradicting ``J(x)=0 <=> x in G``. Measuring
      roughness *relative to the target* keeps the zero where it belongs.

* ``J`` increases with misalignment, wrinkles and poor overlap. Each term is
  monotone in its respective defect.

J is defined on a low-dimensional descriptor of the state (cloth vertices), so
the goal set is the preimage ``G = J^{-1}({0})``, exactly as in Sec. 0.1.

This module is pure torch -- it has no Isaac Lab / LeHome dependency, so it can
be unit-tested and tuned without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ClothFunctionalCfg:
    """Weights and discretisation for J."""

    # --- term weights (lambda_1, lambda_2, lambda_3 in the spec) ---
    lambda_iou: float = 1.0
    lambda_edge: float = 1.0
    lambda_wrinkle: float = 0.1

    # --- occupancy rasterisation (top-down xy projection) ---
    grid_res: int = 64
    grid_bounds: Tuple[float, float, float, float] = (-0.4, 0.4, -0.4, 0.4)
    """(x_min, x_max, y_min, y_max) of the rasterised workspace, metres."""
    splat_sigma: float = 0.015
    """Std-dev of the Gaussian splat, metres. Controls how smooth J is in the
    vertex positions: larger sigma => smaller Lipschitz constant, blurrier IoU."""

    # --- edge gap ---
    edge_indices: Optional[Tuple[int, ...]] = None
    """Vertex indices treated as garment edge landmarks. ``None`` => use the
    boundary of the structured grid (see :meth:`ClothErrorFunctional.__init__`)."""
    edge_scale: float = 0.10
    """Normaliser, metres. edge_gap is reported in units of this length."""

    # --- wrinkle ---
    wrinkle_scale: float = 1.0e-3
    """Normaliser for the mean-squared discrete Laplacian, m^2."""

    # --- structure ---
    grid_shape: Optional[Tuple[int, int]] = None
    """(rows, cols) of the structured cloth mesh. Required for the wrinkle term
    (it defines the discrete Laplacian stencil) and for the default edge set."""

    def __post_init__(self) -> None:
        if self.grid_res < 2:
            raise ValueError("grid_res must be >= 2")
        if self.splat_sigma <= 0.0:
            raise ValueError("splat_sigma must be > 0 for J to be continuous")
        x0, x1, y0, y1 = self.grid_bounds
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"degenerate grid_bounds: {self.grid_bounds}")
        for name in ("lambda_iou", "lambda_edge", "lambda_wrinkle"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0 (J must be non-negative)")


class ClothErrorFunctional(nn.Module):
    """Computes J(x) and its three components for a batch of cloth states.

    Args:
        cfg: functional configuration.
        target_verts: ``(N, 3)`` or ``(B, N, 3)`` folded-goal vertex positions.
            May also be supplied per-call to :meth:`forward`.
    """

    def __init__(
        self,
        cfg: ClothFunctionalCfg,
        target_verts: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Grid cell centres for soft rasterisation.
        x0, x1, y0, y1 = cfg.grid_bounds
        xs = torch.linspace(x0, x1, cfg.grid_res)
        ys = torch.linspace(y0, y1, cfg.grid_res)
        self.register_buffer("_xs", xs, persistent=False)
        self.register_buffer("_ys", ys, persistent=False)

        # Edge landmark indices.
        idx = cfg.edge_indices
        if idx is None and cfg.grid_shape is not None:
            idx = tuple(_grid_boundary_indices(*cfg.grid_shape))
        edge_idx = torch.as_tensor(idx, dtype=torch.long) if idx else torch.empty(0, dtype=torch.long)
        self.register_buffer("_edge_idx", edge_idx, persistent=False)

        if target_verts is not None:
            self.set_target(target_verts)
        else:
            self.register_buffer("_target", torch.empty(0), persistent=False)

    # ------------------------------------------------------------------ target

    def set_target(self, target_verts: torch.Tensor) -> None:
        """Set the folded-goal configuration x_target."""
        t = torch.as_tensor(target_verts)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() != 3 or t.shape[-1] != 3:
            raise ValueError(f"target_verts must be (N,3) or (B,N,3), got {tuple(t.shape)}")
        self.register_buffer("_target", t.to(self._xs.device, self._xs.dtype), persistent=False)

    @property
    def has_target(self) -> bool:
        return self._target.numel() > 0

    # ------------------------------------------------------------------ pieces

    def soft_occupancy(self, verts: torch.Tensor) -> torch.Tensor:
        """Differentiable top-down occupancy mask in [0, 1].

        Splats each vertex as an isotropic Gaussian in the xy-plane and converts
        accumulated density to occupancy via ``1 - exp(-density)``, which is
        smooth, monotone, and saturates at 1.

        Args:
            verts: ``(B, N, 3)``.
        Returns:
            ``(B, R, R)`` occupancy, indexed ``[b, ix, iy]``.
        """
        sigma = self.cfg.splat_sigma
        two_s2 = 2.0 * sigma * sigma

        # Separable Gaussian: (B,N,R) each, contracted into (B,R,R).
        dx = verts[..., 0:1] - self._xs.view(1, 1, -1)  # (B,N,R)
        dy = verts[..., 1:2] - self._ys.view(1, 1, -1)  # (B,N,R)
        wx = torch.exp(-(dx * dx) / two_s2)
        wy = torch.exp(-(dy * dy) / two_s2)
        density = torch.einsum("bnx,bny->bxy", wx, wy)
        return -torch.expm1(-density)  # == 1 - exp(-density), stable near 0

    def _iou_term(self, verts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = self.soft_occupancy(verts)
        b = self.soft_occupancy(target)
        inter = torch.minimum(a, b).flatten(1).sum(-1)
        union = torch.maximum(a, b).flatten(1).sum(-1)
        iou = inter / union.clamp_min(1e-8)
        return (1.0 - iou).clamp_min(0.0)

    def _edge_term(self, verts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._edge_idx.numel() == 0:
            return verts.new_zeros(verts.shape[0])
        v = verts.index_select(1, self._edge_idx)
        t = target.index_select(1, self._edge_idx)
        gap = torch.linalg.vector_norm(v - t, dim=-1).mean(-1)
        return gap / self.cfg.edge_scale

    def _roughness(self, verts: torch.Tensor) -> torch.Tensor:
        """Mean squared discrete Laplacian over interior mesh vertices."""
        if self.cfg.grid_shape is None:
            return verts.new_zeros(verts.shape[0])
        rows, cols = self.cfg.grid_shape
        b = verts.shape[0]
        g = verts.view(b, rows, cols, 3)
        # 5-point stencil on the interior.
        lap = (
            g[:, :-2, 1:-1, :]
            + g[:, 2:, 1:-1, :]
            + g[:, 1:-1, :-2, :]
            + g[:, 1:-1, 2:, :]
            - 4.0 * g[:, 1:-1, 1:-1, :]
        )
        return (lap * lap).sum(-1).flatten(1).mean(-1)

    def _wrinkle_term(self, verts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Relative to the target's own curvature -- see module docstring.
        d = self._roughness(verts) - self._roughness(target)
        return d.abs() / self.cfg.wrinkle_scale

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        verts: torch.Tensor,
        target_verts: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        """Evaluate J.

        Args:
            verts: ``(B, N, 3)`` (or ``(N, 3)``) current cloth vertices.
            target_verts: optional override of the stored target.
            return_components: also return the three unweighted terms.
        Returns:
            ``J`` of shape ``(B,)``, optionally with a component dict.
        """
        v = torch.as_tensor(verts)
        if v.dim() == 2:
            v = v.unsqueeze(0)
        if v.dim() != 3 or v.shape[-1] != 3:
            raise ValueError(f"verts must be (N,3) or (B,N,3), got {tuple(v.shape)}")

        if target_verts is not None:
            t = torch.as_tensor(target_verts, dtype=v.dtype, device=v.device)
            if t.dim() == 2:
                t = t.unsqueeze(0)
        elif self.has_target:
            t = self._target.to(v.dtype).to(v.device)
        else:
            raise RuntimeError("no target set; pass target_verts or call set_target()")

        if t.shape[0] == 1 and v.shape[0] > 1:
            t = t.expand_as(v)
        if t.shape != v.shape:
            raise ValueError(f"target shape {tuple(t.shape)} != verts shape {tuple(v.shape)}")

        c = self.cfg
        term_iou = self._iou_term(v, t)
        term_edge = self._edge_term(v, t)
        term_wrinkle = self._wrinkle_term(v, t)

        j = c.lambda_iou * term_iou + c.lambda_edge * term_edge + c.lambda_wrinkle * term_wrinkle
        j = j.clamp_min(0.0)  # guard against fp round-off at the goal

        if return_components:
            return j, {"iou": term_iou, "edge_gap": term_edge, "wrinkle": term_wrinkle}
        return j


def _grid_boundary_indices(rows: int, cols: int) -> list[int]:
    """Flat indices of the boundary ring of a rows x cols structured mesh."""
    out = []
    for r in range(rows):
        for c in range(cols):
            if r in (0, rows - 1) or c in (0, cols - 1):
                out.append(r * cols + c)
    return out


def make_flat_cloth(
    rows: int,
    cols: int,
    size: Tuple[float, float] = (0.3, 0.3),
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A flat rectangular sheet, ``(rows*cols, 3)``, row-major."""
    sx, sy = size
    cx, cy, cz = center
    ys = torch.linspace(cy - sy / 2, cy + sy / 2, rows, device=device, dtype=dtype)
    xs = torch.linspace(cx - sx / 2, cx + sx / 2, cols, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gz = torch.full_like(gx, cz)
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def make_folded_target(
    rows: int,
    cols: int,
    size: Tuple[float, float] = (0.3, 0.3),
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    thickness: float = 0.004,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A once-folded sheet: the left half reflected onto the right half.

    The fold line is the vertical mid-line (constant x). Folded-over vertices
    are lifted by ``thickness`` so the two layers do not coincide exactly,
    which is what a real folded garment looks like to the occupancy mask.
    """
    flat = make_flat_cloth(rows, cols, size, center, device, dtype).view(rows, cols, 3)
    cx = center[0]
    folded = flat.clone()
    left = flat[..., 0] < cx
    folded[..., 0] = torch.where(left, 2.0 * cx - flat[..., 0], flat[..., 0])
    folded[..., 2] = torch.where(left, flat[..., 2] + thickness, flat[..., 2])
    return folded.reshape(-1, 3)
