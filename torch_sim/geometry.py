"""Geometry utilities for internal coordinates.

Pure PyTorch implementations of angle, dihedral, and distance calculations
and their Cartesian derivatives (Wilson B-matrices). Ported from ASE's
``ase.geometry.geometry`` module.
"""

from __future__ import annotations

import torch

from torch_sim.transforms import minimum_image_displacement


def conditional_find_mic(
    vectors: list[torch.Tensor],
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Apply minimum image convention to vectors if cell/pbc are provided.

    Args:
        vectors: List of displacement vector tensors, each of shape ``(n, 3)``.
        cell: Unit cell in **row-vector** convention ``(3, 3)`` (ASE convention).
            Converted internally to column-vector convention for
            :func:`minimum_image_displacement`.
        pbc: Periodic boundary conditions ``(3,)`` bool tensor.

    Returns:
        Tuple of (wrapped_vectors, vector_lengths) where each element is a
        list matching the input.
    """
    if (cell is None) != (pbc is None):
        raise ValueError("cell and pbc must both be set or both be None")

    wrapped: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    for v in vectors:
        if cell is not None:
            # minimum_image_displacement expects column-vector cell
            v = minimum_image_displacement(dr=v, cell=cell.mT, pbc=pbc)
        vlen = torch.linalg.norm(v, dim=-1)
        wrapped.append(v)
        lengths.append(vlen)
    return wrapped, lengths


def get_angles(
    v0: torch.Tensor,
    v1: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute angles between pairs of vectors in degrees.

    Args:
        v0: First vectors ``(n, 3)``.
        v1: Second vectors ``(n, 3)``.
        cell: Row-vector unit cell ``(3, 3)`` or ``None``.
        pbc: Periodic boundary conditions ``(3,)`` or ``None``.

    Returns:
        Angles in degrees ``(n,)``.
    """
    (v0, v1), (nv0, nv1) = conditional_find_mic([v0, v1], cell, pbc)

    if (nv0 <= 0).any() or (nv1 <= 0).any():
        raise ZeroDivisionError("Undefined angle")

    v0n = v0 / nv0[:, None]
    v1n = v1 / nv1[:, None]
    cos = (v0n * v1n).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def get_angles_derivatives(
    v0: torch.Tensor,
    v1: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derivatives of angles w.r.t. Cartesian coordinates (degrees).

    Args:
        v0: First vectors ``(n, 3)``.
        v1: Second vectors ``(n, 3)``.
        cell: Row-vector unit cell ``(3, 3)`` or ``None``.
        pbc: Periodic boundary conditions ``(3,)`` or ``None``.

    Returns:
        Derivatives ``(n, 3, 3)`` — ``[n_constraints, 3_atoms, 3_xyz]``.
    """
    (v0, v1), (nv0, nv1) = conditional_find_mic([v0, v1], cell, pbc)

    angles = torch.deg2rad(get_angles(v0, v1, cell=cell, pbc=pbc))
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    if (sin_angles == 0.0).any():
        raise ZeroDivisionError("Singularity for derivative of a planar angle")

    product = nv0 * nv1
    # derivatives by atom 0
    deriv_d0 = (
        -(v1 / product[:, None] - v0 * (cos_angles / nv0**2)[:, None])
        / sin_angles[:, None]
    )
    # derivatives by atom 2
    deriv_d2 = (
        -(v0 / product[:, None] - v1 * (cos_angles / nv1**2)[:, None])
        / sin_angles[:, None]
    )
    # derivatives by atom 1
    deriv_d1 = -(deriv_d0 + deriv_d2)
    derivs = torch.stack((deriv_d0, deriv_d1, deriv_d2), dim=1)
    return torch.rad2deg(derivs)


def get_dihedrals(
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute dihedral angles in degrees [0, 360).

    Args:
        v0: Vectors a0->a1 ``(n, 3)``.
        v1: Vectors a1->a2 ``(n, 3)``.
        v2: Vectors a2->a3 ``(n, 3)``.
        cell: Row-vector unit cell ``(3, 3)`` or ``None``.
        pbc: Periodic boundary conditions ``(3,)`` or ``None``.

    Returns:
        Dihedral angles in degrees ``(n,)`` in ``[0, 360)``.
    """
    (v0, v1, v2), (_, nv1, _) = conditional_find_mic([v0, v1, v2], cell, pbc)

    v1n = v1 / nv1[:, None]
    # projection of v0, v2 onto plane perpendicular to v1
    v = -v0 - torch.einsum("ij,ij,ik->ik", -v0, v1n, v1n)
    w = v2 - torch.einsum("ij,ij,ik->ik", v2, v1n, v1n)

    undefined_v = (v == 0.0).all(dim=1)
    undefined_w = (w == 0.0).all(dim=1)
    if undefined_v.any() or undefined_w.any():
        raise ZeroDivisionError("Undefined dihedral for planar inner angle")

    x = (v * w).sum(-1)
    y = (torch.cross(v1n, v, dim=1) * w).sum(-1)
    dihedrals = torch.atan2(y, x)  # [-pi, pi]
    dihedrals = dihedrals % (2 * torch.pi)  # [0, 2*pi]
    return torch.rad2deg(dihedrals)


def get_dihedrals_derivatives(
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derivatives of dihedral angles w.r.t. Cartesian coordinates (degrees).

    Args:
        v0: Vectors a0->a1 ``(n, 3)``.
        v1: Vectors a1->a2 ``(n, 3)``.
        v2: Vectors a2->a3 ``(n, 3)``.
        cell: Row-vector unit cell ``(3, 3)`` or ``None``.
        pbc: Periodic boundary conditions ``(3,)`` or ``None``.

    Returns:
        Derivatives ``(n, 4, 3)`` — ``[n_constraints, 4_atoms, 3_xyz]``.
    """
    (v0, v1, v2), (nv0, nv1, nv2) = conditional_find_mic([v0, v1, v2], cell, pbc)

    v0n = v0 / nv0[:, None]
    v1n = v1 / nv1[:, None]
    v2n = v2 / nv2[:, None]

    normal_v01 = torch.cross(v0n, v1n, dim=1)
    normal_v12 = torch.cross(v1n, v2n, dim=1)

    cos_psi01 = (v0n * v1n).sum(-1)
    sin_psi01 = torch.sin(torch.arccos(cos_psi01.clamp(-1.0, 1.0)))
    cos_psi12 = (v1n * v2n).sum(-1)
    sin_psi12 = torch.sin(torch.arccos(cos_psi12.clamp(-1.0, 1.0)))

    if (sin_psi01 == 0.0).any() or (sin_psi12 == 0.0).any():
        raise ZeroDivisionError(
            "Undefined derivative for undefined dihedral with planar inner angle"
        )

    deriv_d0 = -normal_v01 / (nv0 * sin_psi01**2)[:, None]
    deriv_d3 = normal_v12 / (nv2 * sin_psi12**2)[:, None]
    deriv_d1 = (
        ((nv1 + nv0 * cos_psi01) / nv1)[:, None] * -deriv_d0
        + (cos_psi12 * nv2 / nv1)[:, None] * deriv_d3
    )
    deriv_d2 = (
        -((nv1 + nv2 * cos_psi12) / nv1)[:, None] * deriv_d3
        - (cos_psi01 * nv0 / nv1)[:, None] * -deriv_d0
    )

    derivs = torch.stack((deriv_d0, deriv_d1, deriv_d2, deriv_d3), dim=1)
    return torch.rad2deg(derivs)


def get_distances_derivatives(
    v0: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derivatives of distances w.r.t. Cartesian coordinates.

    Args:
        v0: Displacement vectors ``(n, 3)``.
        cell: Row-vector unit cell ``(3, 3)`` or ``None``.
        pbc: Periodic boundary conditions ``(3,)`` or ``None``.

    Returns:
        Derivatives ``(n, 2, 3)`` — ``[n_distances, 2_atoms, 3_xyz]``.
    """
    (v0,), (dists,) = conditional_find_mic([v0], cell, pbc)

    if (dists <= 0.0).any():
        raise ZeroDivisionError("Singularity for derivative of a zero distance")

    derivs_d0 = -v0 / dists[:, None]
    derivs_d1 = -derivs_d0
    return torch.stack((derivs_d0, derivs_d1), dim=1)
