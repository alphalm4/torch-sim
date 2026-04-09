"""Constraints for molecular dynamics simulations.

This module implements constraints inspired by ASE's constraint system,
adapted for the torch-sim framework with support for batched operations
and PyTorch tensors.

The constraints affect degrees of freedom counting and modify forces, momenta,
and positions during MD simulations.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import torch


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch_sim.state import SimState


class Constraint(ABC):
    """Base class for all constraints in torch-sim.

    This is the abstract base class that all constraints must inherit from.
    It defines the interface that constraints must implement to work with
    the torch-sim MD system.
    """

    @abstractmethod
    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get the number of degrees of freedom removed by this constraint.

        Args:
            state: The simulation state

        Returns:
            Number of degrees of freedom removed by this constraint
        """

    @abstractmethod
    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Adjust positions to satisfy the constraint.

        This method should modify new_positions in-place to ensure the
        constraint is satisfied.

        Args:
            state: Current simulation state
            new_positions: Proposed new positions to be adjusted
        """

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Adjust momenta to satisfy the constraint.

        This method should modify momenta in-place to ensure the constraint
        is satisfied. By default, it calls adjust_forces with the momenta.

        Args:
            state: Current simulation state
            momenta: Momenta to be adjusted
        """
        # Default implementation: treat momenta like forces
        self.adjust_forces(state, momenta)

    @abstractmethod
    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Adjust forces to satisfy the constraint.

        This method should modify forces in-place to ensure the constraint
        is satisfied.

        Args:
            state: Current simulation state
            forces: Forces to be adjusted
        """

    def adjust_stress(  # noqa: B027
        self, state: SimState, stress: torch.Tensor
    ) -> None:
        """Adjust stress tensor to satisfy the constraint.

        Default is a no-op. Override in subclasses that need stress symmetrization.

        Args:
            state: Current simulation state
            stress: Stress tensor to be adjusted in-place
        """

    def adjust_cell(  # noqa: B027
        self, state: SimState, cell: torch.Tensor
    ) -> None:
        """Adjust cell to satisfy the constraint.

        Default is a no-op. Override in subclasses that need cell symmetrization.

        Args:
            state: Current simulation state
            cell: Cell tensor to be adjusted in-place (column vector convention)
        """

    @abstractmethod
    def select_constraint(
        self, atom_mask: torch.Tensor, system_mask: torch.Tensor
    ) -> None | Self:
        """Update the constraint to account for atom and system masks.

        Args:
            atom_mask: Boolean mask for atoms to keep
            system_mask: Boolean mask for systems to keep
        """

    @abstractmethod
    def select_sub_constraint(self, atom_idx: torch.Tensor, sys_idx: int) -> None | Self:
        """Select a constraint for a given atom and system index.

        Args:
            atom_idx: Atom indices for a single system
            sys_idx: System index for a single system

        Returns:
            Constraint for the given atom and system index
        """

    @abstractmethod
    def reindex(self, atom_offset: int, system_offset: int) -> Self:
        """Return a copy with indices shifted to global coordinates.

        Called during state concatenation to adjust indices before merging.

        Args:
            atom_offset: Offset to add to atom indices
            system_offset: Offset to add to system indices
        """

    @classmethod
    @abstractmethod
    def merge(cls, constraints: list[Constraint]) -> Self:
        """Merge multiple already-reindexed constraints into one.

        Constraints must have global (absolute) indices — call ``reindex``
        first. Subclasses override this to handle type-specific data.

        Args:
            constraints: Constraints to merge (all same type, already reindexed)
        """

    @abstractmethod
    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a copy with all internal tensors moved to *device*/*dtype*.

        Float tensors are cast to *dtype*; integer/bool tensors are only moved
        to *device*.
        """


def _cumsum_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Cumulative sum with a leading zero, e.g. [3, 2, 4] -> [0, 3, 5, 9]."""
    return torch.cat(
        [torch.zeros(1, device=tensor.device, dtype=tensor.dtype), tensor.cumsum(dim=0)]
    )


def _mask_constraint_indices(idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cumsum_atom_mask = torch.cumsum(~mask, dim=0)
    new_indices = idx - cumsum_atom_mask[idx]
    mask_indices = torch.where(mask)[0]
    drop_indices = ~torch.isin(idx, mask_indices)
    return new_indices[~drop_indices]


class AtomConstraint(Constraint):
    """Base class for constraints that act on specific atom indices.

    This class provides common functionality for constraints that operate
    on a subset of atoms, identified by their indices.
    """

    def __init__(
        self,
        atom_idx: torch.Tensor | list[int] | None = None,
        atom_mask: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Initialize indexed constraint.

        Args:
            atom_idx: Indices of atoms to constrain. Can be a tensor or list of integers.
            atom_mask: Boolean mask for atoms to constrain.

        Raises:
            ValueError: If both indices and mask are provided, or if indices have
                       wrong shape/type
        """
        if atom_idx is not None and atom_mask is not None:
            raise ValueError("Provide either atom_idx or atom_mask, not both.")
        if atom_mask is not None:
            atom_mask = torch.as_tensor(atom_mask)
            atom_idx = torch.where(atom_mask)[0]

        # Convert to tensor if needed
        atom_idx = torch.as_tensor(atom_idx)

        # Ensure we have the right shape and type
        atom_idx = torch.atleast_1d(atom_idx)
        if atom_idx.ndim != 1:
            raise ValueError(
                "atom_idx has wrong number of dimensions. "
                f"Got {atom_idx.ndim}, expected ndim <= 1"
            )

        if torch.is_floating_point(atom_idx):
            raise ValueError(
                f"Indices must be integers or boolean mask, not dtype={atom_idx.dtype}"
            )

        self.atom_idx = atom_idx.long()

    def get_indices(self) -> torch.Tensor:
        """Get the constrained atom indices.

        Returns:
            Tensor of atom indices affected by this constraint
        """
        return self.atom_idx.clone()

    def select_constraint(
        self,
        atom_mask: torch.Tensor,
        system_mask: torch.Tensor,  # noqa: ARG002
    ) -> None | Self:
        """Update the constraint to account for atom and system masks.

        Args:
            atom_mask: Boolean mask for atoms to keep
            system_mask: Boolean mask for systems to keep
        """
        indices = self.atom_idx.clone()
        indices = _mask_constraint_indices(indices, atom_mask)
        if len(indices) == 0:
            return None
        return type(self)(indices)

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,
        sys_idx: int,  # noqa: ARG002
    ) -> None | Self:
        """Select a constraint for a given atom and system index.

        Args:
            atom_idx: Atom indices for a single system
            sys_idx: System index for a single system
        """
        mask = torch.isin(self.atom_idx, atom_idx)
        masked_indices = self.atom_idx[mask]
        new_atom_idx = masked_indices - atom_idx.min()
        if len(new_atom_idx) == 0:
            return None
        return type(self)(new_atom_idx)

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with atom indices shifted by atom_offset."""
        return type(self)(self.atom_idx + atom_offset)

    @classmethod
    def merge(cls, constraints: list[Constraint]) -> Self:
        """Merge by concatenating already-reindexed atom indices."""
        atom_constraints = [
            constraint for constraint in constraints if isinstance(constraint, cls)
        ]
        if not atom_constraints:
            raise ValueError(
                f"{cls.__name__}.merge requires at least one {cls.__name__}."
            )
        return cls(torch.cat([constraint.atom_idx for constraint in atom_constraints]))

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,  # noqa: ARG002
    ) -> Self:
        """Return a copy with atom indices moved to *device*."""
        return type(self)(self.atom_idx.to(device=device))


class SystemConstraint(Constraint):
    """Base class for constraints that act on specific system indices.

    This class provides common functionality for constraints that operate
    on a subset of systems, identified by their indices.
    """

    def __init__(
        self,
        system_idx: torch.Tensor | list[int] | None = None,
        system_mask: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Initialize indexed constraint.

        Args:
            system_idx: Indices of systems to constrain.
                Can be a tensor or list of integers.
            system_mask: Boolean mask for systems to constrain.

        Raises:
            ValueError: If both indices and mask are provided, or if indices have
                       wrong shape/type
        """
        if system_idx is not None and system_mask is not None:
            raise ValueError("Provide either system_idx or system_mask, not both.")
        if system_mask is not None:
            system_idx = torch.where(torch.as_tensor(system_mask))[0]

        # Convert to tensor if needed
        system_idx = torch.as_tensor(system_idx)

        # Ensure we have the right shape and type
        system_idx = torch.atleast_1d(system_idx)
        if system_idx.ndim != 1:
            raise ValueError(
                "system_idx has wrong number of dimensions. "
                f"Got {system_idx.ndim}, expected ndim <= 1"
            )

        # Check for duplicates
        if len(system_idx) != len(torch.unique(system_idx)):
            raise ValueError("Duplicate system indices found in SystemConstraint.")

        if torch.is_floating_point(system_idx):
            raise ValueError(
                f"Indices must be integers or boolean mask, not dtype={system_idx.dtype}"
            )

        self.system_idx = system_idx.long()

    def select_constraint(
        self,
        atom_mask: torch.Tensor,  # noqa: ARG002
        system_mask: torch.Tensor,
    ) -> None | Self:
        """Update the constraint to account for atom and system masks.

        Args:
            atom_mask: Boolean mask for atoms to keep
            system_mask: Boolean mask for systems to keep
        """
        system_idx = self.system_idx.clone()
        system_idx = _mask_constraint_indices(system_idx, system_mask)
        if len(system_idx) == 0:
            return None
        return type(self)(system_idx)

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,  # noqa: ARG002
        sys_idx: int,
    ) -> None | Self:
        """Select a constraint for a given atom and system index.

        Args:
            atom_idx: Atom indices for a single system
            sys_idx: System index for a single system
        """
        return type(self)(torch.tensor([0])) if sys_idx in self.system_idx else None

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with system indices shifted by system_offset."""
        return type(self)(self.system_idx + system_offset)

    @classmethod
    def merge(cls, constraints: list[Constraint]) -> Self:
        """Merge by concatenating already-reindexed system indices."""
        system_constraints = [
            constraint for constraint in constraints if isinstance(constraint, cls)
        ]
        if not system_constraints:
            raise ValueError(
                f"{cls.__name__}.merge requires at least one {cls.__name__}."
            )
        return cls(
            torch.cat([constraint.system_idx for constraint in system_constraints])
        )

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,  # noqa: ARG002
    ) -> Self:
        """Return a copy with system indices moved to *device*."""
        return type(self)(self.system_idx.to(device=device))


def merge_constraints(
    constraint_lists: list[list[Constraint]],
    num_atoms_per_state: torch.Tensor,
    num_systems_per_state: torch.Tensor | None = None,
) -> list[Constraint]:
    """Merge constraints from multiple states into a single list.

    Each constraint is first reindexed to global coordinates (via ``reindex``),
    then constraints of the same type are merged (via ``merge``).

    Args:
        constraint_lists: List of lists of constraints, one list per state
        num_atoms_per_state: Number of atoms per state
        num_systems_per_state: Number of systems per state. Falls back to 1
            per state if not provided.

    Returns:
        List of merged constraints
    """
    from collections import defaultdict

    # Calculate cumulative offsets for atoms and systems
    device, dtype = num_atoms_per_state.device, num_atoms_per_state.dtype
    atom_offsets = _cumsum_with_zero(num_atoms_per_state[:-1])
    if num_systems_per_state is None:
        num_systems_per_state = torch.ones(
            len(constraint_lists), device=device, dtype=dtype
        )
    system_offsets = _cumsum_with_zero(num_systems_per_state[:-1])

    # Reindex each constraint to global coordinates, then group by type
    grouped: dict[type[Constraint], list[Constraint]] = defaultdict(list)
    for state_idx, constraint_list in enumerate(constraint_lists):
        a_off = int(atom_offsets[state_idx].item())
        s_off = int(system_offsets[state_idx].item())
        for constraint in constraint_list:
            grouped[type(constraint)].append(constraint.reindex(a_off, s_off))

    return [ctype.merge(cs) for ctype, cs in grouped.items()]


class FixAtoms(AtomConstraint):
    """Constraint that fixes specified atoms in place.

    This constraint prevents the specified atoms from moving by:
    - Resetting their positions to original values
    - Setting their forces to zero
    - Removing 3 degrees of freedom per fixed atom

    Examples:
        Fix atoms with indices [0, 1, 2]:
        >>> constraint = FixAtoms(atom_idx=[0, 1, 2])

        Fix atoms using a boolean mask:
        >>> mask = torch.tensor([True, True, True, False, False])
        >>> constraint = FixAtoms(mask=mask)
    """

    def __init__(
        self,
        atom_idx: torch.Tensor | list[int] | None = None,
        atom_mask: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Initialize FixAtoms constraint and check for duplicate indices."""
        super().__init__(atom_idx=atom_idx, atom_mask=atom_mask)
        # Check duplicates
        if len(self.atom_idx) != len(torch.unique(self.atom_idx)):
            raise ValueError("Duplicate atom indices found in FixAtoms constraint.")

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        Each fixed atom removes 3 degrees of freedom (x, y, z motion).

        Args:
            state: Simulation state

        Returns:
            Number of degrees of freedom removed (3 * number of fixed atoms)
        """
        sys_idx = state.system_idx
        if sys_idx is None:
            raise ValueError("FixAtoms requires system_idx to be set")
        fixed_atoms_system_idx = torch.bincount(
            sys_idx[self.atom_idx], minlength=state.n_systems
        )
        return 3 * fixed_atoms_system_idx

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Reset positions of fixed atoms to their current values.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        new_positions[self.atom_idx] = state.positions[self.atom_idx]

    def adjust_forces(
        self,
        state: SimState,  # noqa: ARG002
        forces: torch.Tensor,
    ) -> None:
        """Set forces on fixed atoms to zero.

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        forces[self.atom_idx] = 0.0

    def __repr__(self) -> str:
        """String representation of the constraint."""
        if len(self.atom_idx) <= 10:
            indices_str = self.atom_idx.tolist()
        else:
            indices_str = f"{self.atom_idx[:5].tolist()}...{self.atom_idx[-5:].tolist()}"
        return f"FixAtoms(indices={indices_str})"


class FixCom(SystemConstraint):
    """Constraint that fixes the center of mass of all atoms per system.

    This constraint prevents the center of mass from moving by:
    - Adjusting positions to maintain center of mass position
    - Removing center of mass velocity from momenta
    - Adjusting forces to remove net force
    - Removing 3 degrees of freedom (center of mass translation)

    The constraint is applied to all atoms in the system.
    """

    coms: torch.Tensor | None = None

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        Fixing center of mass removes 3 degrees of freedom (x, y, z translation).

        Args:
            state: Simulation state

        Returns:
            Always returns 3 (center of mass translation degrees of freedom)
        """
        affected_systems = torch.zeros(state.n_systems, dtype=torch.long)
        affected_systems[self.system_idx] = 1
        return 3 * affected_systems

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Adjust positions to maintain center of mass position.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        if state.system_idx is None:
            raise ValueError("FixCom requires state with system_idx")
        system_idx = state.system_idx
        dtype = state.positions.dtype
        system_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0, system_idx, state.masses
        )
        if self.coms is None:
            self.coms = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
                0,
                system_idx.unsqueeze(-1).expand(-1, 3),
                state.masses.unsqueeze(-1) * state.positions,
            )
            self.coms /= system_mass.unsqueeze(-1)

        new_com = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            system_idx.unsqueeze(-1).expand(-1, 3),
            state.masses.unsqueeze(-1) * new_positions,
        )
        new_com /= system_mass.unsqueeze(-1)
        displacement = torch.zeros(state.n_systems, 3, dtype=dtype)
        displacement[self.system_idx] = (
            -new_com[self.system_idx] + self.coms[self.system_idx]
        )
        new_positions += displacement[system_idx]

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Remove center of mass velocity from momenta.

        Args:
            state: Current simulation state
            momenta: Momenta to be adjusted in-place
        """
        if state.system_idx is None:
            raise ValueError("FixCom requires state with system_idx")
        system_idx = state.system_idx
        # Compute center of mass momenta
        dtype = momenta.dtype
        com_momenta = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            system_idx.unsqueeze(-1).expand(-1, 3),
            momenta,
        )
        system_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0, system_idx, state.masses
        )
        velocity_com = com_momenta / system_mass.unsqueeze(-1)
        velocity_change = torch.zeros(state.n_systems, 3, dtype=dtype)
        velocity_change[self.system_idx] = velocity_com[self.system_idx]
        momenta -= velocity_change[system_idx] * state.masses.unsqueeze(-1)

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Remove net force to prevent center of mass acceleration.

        This implements the constraint from Eq. (3) and (7) in
        https://doi.org/10.1021/jp9722824

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        if state.system_idx is None:
            raise ValueError("FixCom requires state with system_idx")
        system_idx = state.system_idx
        dtype = state.positions.dtype
        system_square_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0,
            system_idx,
            torch.square(state.masses),
        )
        lmd = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            system_idx.unsqueeze(-1).expand(-1, 3),
            forces * state.masses.unsqueeze(-1),
        )
        lmd /= system_square_mass.unsqueeze(-1)
        forces_change = torch.zeros(state.n_systems, 3, dtype=dtype)
        forces_change[self.system_idx] = lmd[self.system_idx]
        forces -= forces_change[system_idx] * state.masses.unsqueeze(-1)

    def __repr__(self) -> str:
        """String representation of the constraint."""
        return f"FixCom(system_idx={self.system_idx})"

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a copy with tensors moved to *device*/*dtype*."""
        new = type(self)(self.system_idx.to(device=device))
        if self.coms is not None:
            new.coms = self.coms.to(device=device, dtype=dtype)
        return new


class FixInternals(Constraint):
    """Constraint that fixes internal coordinates (bonds, angles, dihedrals).

    Fixes bond lengths, bond angles, dihedral (torsion) angles, and linear
    combinations of bond lengths using Jacobian-based iterative position
    correction and QR-based force projection.  Ported from ASE's
    ``FixInternals`` constraint.

    Each constrained system can have an independent set of constraint
    definitions.  Per-system data is stored as parallel lists indexed by a
    local constraint index, following the same pattern as
    :class:`FixSymmetry`.
    """

    system_idx: torch.Tensor
    bond_indices: list[torch.Tensor]
    bond_targets: list[torch.Tensor]
    angle_indices: list[torch.Tensor]
    angle_targets: list[torch.Tensor]
    dihedral_indices: list[torch.Tensor]
    dihedral_targets: list[torch.Tensor]
    combo_indices: list[list[torch.Tensor]]
    combo_coefs: list[list[torch.Tensor]]
    combo_targets: list[torch.Tensor]
    mic: bool
    epsilon: float

    def __init__(
        self,
        system_idx: torch.Tensor,
        *,
        bond_indices: list[torch.Tensor] | None = None,
        bond_targets: list[torch.Tensor] | None = None,
        angle_indices: list[torch.Tensor] | None = None,
        angle_targets: list[torch.Tensor] | None = None,
        dihedral_indices: list[torch.Tensor] | None = None,
        dihedral_targets: list[torch.Tensor] | None = None,
        combo_indices: list[list[torch.Tensor]] | None = None,
        combo_coefs: list[list[torch.Tensor]] | None = None,
        combo_targets: list[torch.Tensor] | None = None,
        mic: bool = False,
        epsilon: float = 1e-7,
    ) -> None:
        """Initialize FixInternals constraint.

        Args:
            system_idx: Indices of constrained systems ``(n_constrained,)``.
            bond_indices: Per-system bond atom pairs, each ``(n_bonds, 2)`` long.
            bond_targets: Per-system target bond lengths, each ``(n_bonds,)`` float.
            angle_indices: Per-system angle atom triples, each ``(n_angles, 3)`` long.
            angle_targets: Per-system target angles in degrees, each ``(n_angles,)``.
            dihedral_indices: Per-system dihedral atom quads, each ``(n_dihedrals, 4)``
                long.
            dihedral_targets: Per-system target dihedrals in degrees,
                each ``(n_dihedrals,)``.
            combo_indices: Per-system, per-combo bond pair indices.
            combo_coefs: Per-system, per-combo linear coefficients.
            combo_targets: Per-system target combo values, each ``(n_combos,)``.
            mic: Whether to use minimum image convention for periodic systems.
            epsilon: Convergence tolerance for iterative position adjustment.
        """
        self.system_idx = torch.as_tensor(system_idx).long()
        n = len(self.system_idx)

        def _default_list(val: list | None, n: int, empty_fn):  # noqa: ANN001
            return val if val is not None else [empty_fn() for _ in range(n)]

        self.bond_indices = _default_list(
            bond_indices, n, lambda: torch.empty(0, 2, dtype=torch.long)
        )
        self.bond_targets = _default_list(
            bond_targets, n, lambda: torch.empty(0)
        )
        self.angle_indices = _default_list(
            angle_indices, n, lambda: torch.empty(0, 3, dtype=torch.long)
        )
        self.angle_targets = _default_list(
            angle_targets, n, lambda: torch.empty(0)
        )
        self.dihedral_indices = _default_list(
            dihedral_indices, n, lambda: torch.empty(0, 4, dtype=torch.long)
        )
        self.dihedral_targets = _default_list(
            dihedral_targets, n, lambda: torch.empty(0)
        )
        self.combo_indices = _default_list(combo_indices, n, list)
        self.combo_coefs = _default_list(combo_coefs, n, list)
        self.combo_targets = _default_list(
            combo_targets, n, lambda: torch.empty(0)
        )
        self.mic = mic
        self.epsilon = epsilon

    # -- helpers ---------------------------------------------------------------

    def _n_constraints_per_system(self) -> list[int]:
        """Return total number of constraints for each constrained system."""
        counts = []
        for ci in range(len(self.system_idx)):
            c = (
                len(self.bond_targets[ci])
                + len(self.angle_targets[ci])
                + len(self.dihedral_targets[ci])
                + len(self.combo_targets[ci])
            )
            counts.append(c)
        return counts

    @staticmethod
    def _finalize_jacobian(
        pos: torch.Tensor,
        indices: torch.Tensor,
        derivs: torch.Tensor,
        n_atoms_per_internal: int,
    ) -> torch.Tensor:
        """Build the full Cartesian Jacobian row(s) from per-atom derivatives.

        Args:
            pos: Positions ``(n_atoms, 3)`` for the current system.
            indices: Atom indices ``(n_internals, n_atoms_per_internal)`` long.
            derivs: Derivatives ``(n_internals, n_atoms_per_internal, 3)``.
            n_atoms_per_internal: 2 (bond), 3 (angle), or 4 (dihedral).

        Returns:
            Jacobian rows ``(n_internals, n_atoms * 3)``.
        """
        n_internals = indices.shape[0]
        n_atoms = pos.shape[0]
        jac = torch.zeros(
            n_internals, n_atoms, 3, device=pos.device, dtype=pos.dtype
        )
        for j in range(n_atoms_per_internal):
            # indices[:, j] gives which atom each internal's j-th slot refers to
            idx = indices[:, j]  # (n_internals,)
            jac[torch.arange(n_internals, device=pos.device), idx] = derivs[:, j]
        return jac.reshape(n_internals, n_atoms * 3)

    def _bond_jacobian_and_sigma(
        self, ci: int, pos: torch.Tensor, cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (jacobian_rows, sigma) for bond constraints of system *ci*."""
        from torch_sim.geometry import (
            conditional_find_mic,
            get_distances_derivatives,
        )

        idx = self.bond_indices[ci]  # (nb, 2)
        targets = self.bond_targets[ci].to(dtype=pos.dtype, device=pos.device)
        if len(idx) == 0:
            return torch.empty(0, pos.shape[0] * 3, device=pos.device, dtype=pos.dtype), \
                torch.empty(0, device=pos.device, dtype=pos.dtype)
        vecs = pos[idx[:, 1]] - pos[idx[:, 0]]  # (nb, 3)
        derivs = get_distances_derivatives(vecs, cell=cell, pbc=pbc)  # (nb, 2, 3)
        jac = self._finalize_jacobian(pos, idx, derivs, 2)
        (vecs_mic,), (dists,) = conditional_find_mic([vecs], cell=cell, pbc=pbc)
        sigma = dists - targets
        return jac, sigma

    def _angle_jacobian_and_sigma(
        self, ci: int, pos: torch.Tensor, cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch_sim.geometry import get_angles, get_angles_derivatives

        idx = self.angle_indices[ci]  # (na, 3)
        targets = self.angle_targets[ci].to(dtype=pos.dtype, device=pos.device)
        if len(idx) == 0:
            return torch.empty(0, pos.shape[0] * 3, device=pos.device, dtype=pos.dtype), \
                torch.empty(0, device=pos.device, dtype=pos.dtype)
        v0 = pos[idx[:, 0]] - pos[idx[:, 1]]
        v1 = pos[idx[:, 2]] - pos[idx[:, 1]]
        derivs = get_angles_derivatives(v0, v1, cell=cell, pbc=pbc)  # (na, 3, 3)
        jac = self._finalize_jacobian(pos, idx, derivs, 3)
        sigma = get_angles(v0, v1, cell=cell, pbc=pbc) - targets
        return jac, sigma

    def _dihedral_jacobian_and_sigma(
        self, ci: int, pos: torch.Tensor, cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch_sim.geometry import get_dihedrals, get_dihedrals_derivatives

        idx = self.dihedral_indices[ci]  # (nd, 4)
        targets = self.dihedral_targets[ci].to(dtype=pos.dtype, device=pos.device)
        if len(idx) == 0:
            return torch.empty(0, pos.shape[0] * 3, device=pos.device, dtype=pos.dtype), \
                torch.empty(0, device=pos.device, dtype=pos.dtype)
        v0 = pos[idx[:, 1]] - pos[idx[:, 0]]
        v1 = pos[idx[:, 2]] - pos[idx[:, 1]]
        v2 = pos[idx[:, 3]] - pos[idx[:, 2]]
        derivs = get_dihedrals_derivatives(v0, v1, v2, cell=cell, pbc=pbc)
        jac = self._finalize_jacobian(pos, idx, derivs, 4)
        values = get_dihedrals(v0, v1, v2, cell=cell, pbc=pbc)
        sigma = (values - targets + 180) % 360 - 180  # minimum dihedral convention
        return jac, sigma

    def _combo_jacobian_and_sigma(
        self, ci: int, pos: torch.Tensor, cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch_sim.geometry import conditional_find_mic, get_distances_derivatives

        targets = self.combo_targets[ci].to(dtype=pos.dtype, device=pos.device)
        combo_idx_list = self.combo_indices[ci]
        combo_coef_list = self.combo_coefs[ci]
        n_combos = len(combo_idx_list)
        if n_combos == 0:
            return torch.empty(0, pos.shape[0] * 3, device=pos.device, dtype=pos.dtype), \
                torch.empty(0, device=pos.device, dtype=pos.dtype)
        n_atoms = pos.shape[0]
        jac_rows = []
        sigmas = []
        for k in range(n_combos):
            pair_idx = combo_idx_list[k].to(device=pos.device)  # (n_pairs, 2)
            coefs = combo_coef_list[k].to(dtype=pos.dtype, device=pos.device)
            vecs = pos[pair_idx[:, 1]] - pos[pair_idx[:, 0]]
            derivs = get_distances_derivatives(vecs, cell=cell, pbc=pbc)  # (np, 2, 3)
            # build per-combo full Jacobian
            sub_jac = self._finalize_jacobian(pos, pair_idx, derivs, 2)  # (np, 3N)
            # weighted sum
            jac_row = (coefs[:, None] * sub_jac).sum(0)  # (3N,)
            jac_rows.append(jac_row)
            # value
            (vecs_mic,), (dists,) = conditional_find_mic([vecs], cell=cell, pbc=pbc)
            value = (coefs * dists).sum()
            sigmas.append(value - targets[k])
        return torch.stack(jac_rows), torch.stack(sigmas)

    def _all_jacobians_and_sigmas(
        self, ci: int, pos: torch.Tensor, cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated Jacobian matrix and sigma vector for system *ci*."""
        parts = [
            self._bond_jacobian_and_sigma(ci, pos, cell, pbc),
            self._angle_jacobian_and_sigma(ci, pos, cell, pbc),
            self._dihedral_jacobian_and_sigma(ci, pos, cell, pbc),
            self._combo_jacobian_and_sigma(ci, pos, cell, pbc),
        ]
        jacs = [j for j, _ in parts if j.shape[0] > 0]
        sigs = [s for _, s in parts if s.shape[0] > 0]
        if not jacs:
            n3 = pos.shape[0] * 3
            return torch.empty(0, n3, device=pos.device, dtype=pos.dtype), \
                torch.empty(0, device=pos.device, dtype=pos.dtype)
        return torch.cat(jacs, dim=0), torch.cat(sigs, dim=0)

    # -- Constraint interface --------------------------------------------------

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Each bond/angle/dihedral/combo constraint removes 1 DOF."""
        counts = self._n_constraints_per_system()
        dof = torch.zeros(state.n_systems, dtype=torch.long, device=state.device)
        for ci, si in enumerate(self.system_idx):
            dof[si] = counts[ci]
        return dof

    @torch.no_grad()
    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Iteratively correct positions to satisfy internal coordinate constraints.

        Uses a Jacobian-based (Wilson B-matrix) Newton-Raphson scheme, following
        the same algorithm as ASE's ``FixInternals.adjust_positions``.
        """
        cumsum = _cumsum_with_zero(state.n_atoms_per_system)
        for ci, si in enumerate(self.system_idx):
            si_val = si.item()
            start, end = cumsum[si_val].item(), cumsum[si_val + 1].item()
            old_pos = state.positions[start:end]
            cur_pos = new_positions[start:end]
            n_atoms_sys = end - start
            masses = state.masses[start:end].repeat_interleave(3)  # (3N,)
            cell = state.row_vector_cell[si_val] if self.mic else None
            pbc = state.pbc if self.mic else None

            # initial Jacobians at old positions
            init_jacs, _ = self._all_jacobians_and_sigmas(ci, old_pos, cell, pbc)
            if init_jacs.shape[0] == 0:
                continue

            converged = False
            for _iteration in range(50):
                jacs, sigmas = self._all_jacobians_and_sigmas(ci, cur_pos, cell, pbc)
                if sigmas.abs().max() < self.epsilon:
                    converged = True
                    break
                # correct each constraint independently (Gauss-Seidel style)
                for row in range(jacs.shape[0]):
                    j_old = init_jacs[row]  # Jacobian at old positions
                    j_new = jacs[row]  # Jacobian at current positions
                    j_mass = j_old / masses
                    lamda = -sigmas[row] / (j_mass @ j_new)
                    delta = lamda * j_mass
                    cur_pos = cur_pos + delta.reshape(n_atoms_sys, 3)
                    # recompute for next constraint
                    jacs, sigmas = self._all_jacobians_and_sigmas(
                        ci, cur_pos, cell, pbc
                    )

            if not converged:
                import warnings
                residual = (
                    sigmas.abs().max().item()
                    if sigmas.numel() > 0 and torch.isfinite(sigmas).all()
                    else float("nan")
                )
                msg = (
                    f"FixInternals.adjust_positions did not converge for "
                    f"system {si_val} (residual={residual:.3e}, "
                    f"epsilon={self.epsilon:.1e}). Reverting to old positions "
                    f"for this system."
                )
                if any(
                    t > 175.0 or t < 5.0
                    for t in self.angle_targets[ci].tolist()
                ):
                    msg += (
                        " This may be caused by an almost planar angle."
                        " Fixing planar angles is not supported."
                    )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                # Revert to pre-adjustment positions rather than writing
                # potentially NaN/Inf/runaway values back into the state.
                cur_pos = old_pos.clone()
            elif cur_pos.isnan().any() or cur_pos.isinf().any():
                import warnings
                warnings.warn(
                    f"FixInternals.adjust_positions produced NaN/Inf for "
                    f"system {si_val}; reverting to old positions.",
                    RuntimeWarning, stacklevel=2,
                )
                cur_pos = old_pos.clone()

            new_positions[start:end] = cur_pos

    @torch.no_grad()
    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Project out constraint, translation, and rotation components from forces.

        Uses QR decomposition following ASE's ``FixInternals.adjust_forces``.
        """
        cumsum = _cumsum_with_zero(state.n_atoms_per_system)
        for ci, si in enumerate(self.system_idx):
            si_val = si.item()
            start, end = cumsum[si_val].item(), cumsum[si_val + 1].item()
            pos = state.positions[start:end]  # (N, 3)
            f = forces[start:end]  # (N, 3)
            n_atoms_sys = end - start
            cell = state.row_vector_cell[si_val] if self.mic else None
            pbc = state.pbc if self.mic else None

            jacs, _ = self._all_jacobians_and_sigmas(ci, pos, cell, pbc)
            n_constr = jacs.shape[0]
            if n_constr == 0:
                continue

            # Normalize each constraint Jacobian
            for row in range(n_constr):
                norm = torch.linalg.norm(jacs[row])
                if norm > 0:
                    jacs[row] /= norm

            # Build translation + rotation constraint vectors (6 vectors)
            tr_rot = torch.zeros(6, n_atoms_sys, 3, device=pos.device, dtype=pos.dtype)
            tr_rot[0, :, 0] = 1.0  # tx
            tr_rot[1, :, 1] = 1.0  # ty
            tr_rot[2, :, 2] = 1.0  # tz
            center = pos.mean(dim=0)
            d = pos - center
            tr_rot[3, :, 1] = -d[:, 2]  # rx
            tr_rot[3, :, 2] = d[:, 1]
            tr_rot[4, :, 0] = d[:, 2]   # ry
            tr_rot[4, :, 2] = -d[:, 0]
            tr_rot[5, :, 0] = -d[:, 1]  # rz
            tr_rot[5, :, 1] = d[:, 0]
            tr_rot_flat = tr_rot.reshape(6, n_atoms_sys * 3)
            # normalize
            for i in range(6):
                norm = torch.linalg.norm(tr_rot_flat[i])
                if norm > 0:
                    tr_rot_flat[i] /= norm

            # Assemble: constraint Jacobians first, then tr/rot
            all_vecs = torch.cat([jacs, tr_rot_flat], dim=0)  # (n_constr+6, 3N)
            # QR decomposition
            aa, _bb = torch.linalg.qr(all_vecs.T)  # aa: (3N, n_constr+6)
            # Build projection matrix and project
            proj = aa @ aa.T  # (3N, 3N)
            ff = f.reshape(-1)
            ff_proj = proj @ ff
            forces[start:end] -= (proj @ ff_proj).reshape(n_atoms_sys, 3)

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Adjust momenta by projecting out constrained components."""
        self.adjust_forces(state, momenta)

    # -- State manipulation ----------------------------------------------------

    def _get_all_atom_indices(self, ci: int) -> torch.Tensor:
        """Return all unique atom indices referenced by constraints of system *ci*."""
        parts = []
        if len(self.bond_indices[ci]) > 0:
            parts.append(self.bond_indices[ci].flatten())
        if len(self.angle_indices[ci]) > 0:
            parts.append(self.angle_indices[ci].flatten())
        if len(self.dihedral_indices[ci]) > 0:
            parts.append(self.dihedral_indices[ci].flatten())
        for combo_idx in self.combo_indices[ci]:
            if len(combo_idx) > 0:
                parts.append(combo_idx.flatten())
        if not parts:
            return torch.empty(0, dtype=torch.long)
        return torch.unique(torch.cat(parts))

    def _remap_indices(self, ci: int, mapping: dict[int, int]) -> dict:
        """Return remapped constraint data for system *ci* using *mapping*."""
        def remap(t: torch.Tensor) -> torch.Tensor:
            out = t.clone()
            for old, new in mapping.items():
                out[t == old] = new
            return out

        return {
            "bond_indices": remap(self.bond_indices[ci]) if len(self.bond_indices[ci]) else self.bond_indices[ci],
            "bond_targets": self.bond_targets[ci],
            "angle_indices": remap(self.angle_indices[ci]) if len(self.angle_indices[ci]) else self.angle_indices[ci],
            "angle_targets": self.angle_targets[ci],
            "dihedral_indices": remap(self.dihedral_indices[ci]) if len(self.dihedral_indices[ci]) else self.dihedral_indices[ci],
            "dihedral_targets": self.dihedral_targets[ci],
            "combo_indices": [remap(c) if len(c) else c for c in self.combo_indices[ci]],
            "combo_coefs": self.combo_coefs[ci],
            "combo_targets": self.combo_targets[ci],
        }

    def select_constraint(
        self,
        atom_mask: torch.Tensor,  # noqa: ARG002
        system_mask: torch.Tensor,
    ) -> Self | None:
        """Select constraint for systems matching the mask."""
        keep = torch.where(system_mask)[0]
        mask = torch.isin(self.system_idx, keep)
        if not mask.any():
            return None
        local_idx = mask.nonzero(as_tuple=False).flatten().tolist()
        return type(self)(
            _mask_constraint_indices(self.system_idx[mask], system_mask),
            bond_indices=[self.bond_indices[i] for i in local_idx],
            bond_targets=[self.bond_targets[i] for i in local_idx],
            angle_indices=[self.angle_indices[i] for i in local_idx],
            angle_targets=[self.angle_targets[i] for i in local_idx],
            dihedral_indices=[self.dihedral_indices[i] for i in local_idx],
            dihedral_targets=[self.dihedral_targets[i] for i in local_idx],
            combo_indices=[self.combo_indices[i] for i in local_idx],
            combo_coefs=[self.combo_coefs[i] for i in local_idx],
            combo_targets=[self.combo_targets[i] for i in local_idx],
            mic=self.mic,
            epsilon=self.epsilon,
        )

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,  # noqa: ARG002
        sys_idx: int,
    ) -> Self | None:
        """Select constraint for a single system.

        Bond/angle/dihedral/combo indices are already stored as per-system
        LOCAL indices, so no atom-index shifting is needed.  We only need
        to pick out the ``local``-th entry of the per-system index/target
        lists and reset ``system_idx`` to ``[0]`` for the new single-system
        state.
        """
        if sys_idx not in self.system_idx:
            return None
        local = (self.system_idx == sys_idx).nonzero(as_tuple=True)[0].item()

        return type(self)(
            torch.tensor([0], device=self.system_idx.device),
            bond_indices=[self.bond_indices[local].clone()],
            bond_targets=[self.bond_targets[local]],
            angle_indices=[self.angle_indices[local].clone()],
            angle_targets=[self.angle_targets[local]],
            dihedral_indices=[self.dihedral_indices[local].clone()],
            dihedral_targets=[self.dihedral_targets[local]],
            combo_indices=[[c.clone() for c in self.combo_indices[local]]],
            combo_coefs=[self.combo_coefs[local]],
            combo_targets=[self.combo_targets[local]],
            mic=self.mic,
            epsilon=self.epsilon,
        )

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with system_idx shifted for state concatenation.

        Bond/angle/dihedral/combo indices are per-system local indices (not
        global), so they must NOT be shifted by atom_offset.  Only system_idx
        needs adjustment.  adjust_forces/adjust_positions always slice
        positions per-system before indexing.
        """
        return type(self)(
            self.system_idx + system_offset,
            bond_indices=[bi.clone() for bi in self.bond_indices],
            bond_targets=list(self.bond_targets),
            angle_indices=[ai.clone() for ai in self.angle_indices],
            angle_targets=list(self.angle_targets),
            dihedral_indices=[di.clone() for di in self.dihedral_indices],
            dihedral_targets=list(self.dihedral_targets),
            combo_indices=[[c.clone() for c in ci] for ci in self.combo_indices],
            combo_coefs=[list(cc) for cc in self.combo_coefs],
            combo_targets=list(self.combo_targets),
            mic=self.mic,
            epsilon=self.epsilon,
        )

    @classmethod
    def merge(cls, constraints: list[Constraint]) -> Self:
        """Merge multiple already-reindexed FixInternals constraints."""
        fix_internals = [c for c in constraints if isinstance(c, cls)]
        if not fix_internals:
            raise ValueError("FixInternals.merge requires at least one FixInternals.")
        if any(c.mic != fix_internals[0].mic for c in fix_internals[1:]):
            raise ValueError("Cannot merge FixInternals with different mic settings.")
        return cls(
            torch.cat([c.system_idx for c in fix_internals]),
            bond_indices=[bi for c in fix_internals for bi in c.bond_indices],
            bond_targets=[bt for c in fix_internals for bt in c.bond_targets],
            angle_indices=[ai for c in fix_internals for ai in c.angle_indices],
            angle_targets=[at for c in fix_internals for at in c.angle_targets],
            dihedral_indices=[di for c in fix_internals for di in c.dihedral_indices],
            dihedral_targets=[dt for c in fix_internals for dt in c.dihedral_targets],
            combo_indices=[ci for c in fix_internals for ci in c.combo_indices],
            combo_coefs=[cc for c in fix_internals for cc in c.combo_coefs],
            combo_targets=[ct for c in fix_internals for ct in c.combo_targets],
            mic=fix_internals[0].mic,
            epsilon=min(c.epsilon for c in fix_internals),
        )

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a copy with all tensors moved to *device*/*dtype*."""
        def move_long(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device) if len(t) > 0 else t

        def move_float(t: torch.Tensor) -> torch.Tensor:
            kw = {}
            if device is not None:
                kw["device"] = device
            if dtype is not None:
                kw["dtype"] = dtype
            return t.to(**kw) if len(t) > 0 else t

        return type(self)(
            self.system_idx.to(device=device),
            bond_indices=[move_long(bi) for bi in self.bond_indices],
            bond_targets=[move_float(bt) for bt in self.bond_targets],
            angle_indices=[move_long(ai) for ai in self.angle_indices],
            angle_targets=[move_float(at) for at in self.angle_targets],
            dihedral_indices=[move_long(di) for di in self.dihedral_indices],
            dihedral_targets=[move_float(dt) for dt in self.dihedral_targets],
            combo_indices=[
                [move_long(c) for c in ci] for ci in self.combo_indices
            ],
            combo_coefs=[
                [move_float(c) for c in cc] for cc in self.combo_coefs
            ],
            combo_targets=[move_float(ct) for ct in self.combo_targets],
            mic=self.mic,
            epsilon=self.epsilon,
        )

    @classmethod
    def from_definitions(
        cls,
        state: SimState,
        *,
        bonds: list[tuple[float | None, list[int]]] | None = None,
        angles_deg: list[tuple[float | None, list[int]]] | None = None,
        dihedrals_deg: list[tuple[float | None, list[int]]] | None = None,
        bondcombos: list[tuple[float | None, list[list]]] | None = None,
        mic: bool = False,
        epsilon: float = 1e-7,
        system_indices: list[int] | None = None,
    ) -> Self:
        """Create FixInternals from human-readable definitions.

        Constraint definitions follow ASE's format.  If a target value is
        ``None``, the current value from *state* is used.

        Args:
            state: Simulation state to read current values from.
            bonds: ``[(target, [i, j]), ...]``.  Target is a distance.
            angles_deg: ``[(target_deg, [i, j, k]), ...]``.
            dihedrals_deg: ``[(target_deg, [i, j, k, l]), ...]``.
            bondcombos: ``[(target, [[i, j, coef], ...]), ...]``.
            mic: Minimum image convention.
            epsilon: Convergence tolerance.
            system_indices: Which systems to apply constraints to.
                If ``None``, all systems get the same constraints.
        """
        from torch_sim.geometry import (
            conditional_find_mic,
            get_angles,
            get_dihedrals,
        )

        bonds = bonds or []
        angles_deg = angles_deg or []
        dihedrals_deg = dihedrals_deg or []
        bondcombos = bondcombos or []

        if system_indices is None:
            system_indices = list(range(state.n_systems))

        cumsum = _cumsum_with_zero(state.n_atoms_per_system)
        device = state.device
        dtype = state.dtype

        all_bond_idx, all_bond_tgt = [], []
        all_angle_idx, all_angle_tgt = [], []
        all_dih_idx, all_dih_tgt = [], []
        all_combo_idx, all_combo_coef, all_combo_tgt = [], [], []

        cell = None
        pbc = None
        if mic:
            pbc = state.pbc

        for si in system_indices:
            start = cumsum[si].item()
            pos = state.positions  # global positions; indices are global

            if mic:
                cell = state.row_vector_cell[si]

            # Bonds
            bi = torch.tensor(
                [b[1] for b in bonds], dtype=torch.long, device=device
            ).reshape(-1, 2) if bonds else torch.empty(0, 2, dtype=torch.long, device=device)
            bt = []
            for target, indices in bonds:
                if target is None:
                    v = pos[indices[1]] - pos[indices[0]]
                    (v_mic,), (d,) = conditional_find_mic(
                        [v.unsqueeze(0)], cell=cell, pbc=pbc
                    )
                    target = d.item()
                bt.append(target)
            bt_tensor = torch.tensor(bt, dtype=dtype, device=device) if bt else torch.empty(0, dtype=dtype, device=device)
            all_bond_idx.append(bi)
            all_bond_tgt.append(bt_tensor)

            # Angles
            ai = torch.tensor(
                [a[1] for a in angles_deg], dtype=torch.long, device=device
            ).reshape(-1, 3) if angles_deg else torch.empty(0, 3, dtype=torch.long, device=device)
            at = []
            for target, indices in angles_deg:
                if target is None:
                    v0 = pos[indices[0]] - pos[indices[1]]
                    v1 = pos[indices[2]] - pos[indices[1]]
                    target = get_angles(
                        v0.unsqueeze(0), v1.unsqueeze(0), cell=cell, pbc=pbc
                    ).item()
                at.append(target)
            at_tensor = torch.tensor(at, dtype=dtype, device=device) if at else torch.empty(0, dtype=dtype, device=device)
            all_angle_idx.append(ai)
            all_angle_tgt.append(at_tensor)

            # Dihedrals
            di = torch.tensor(
                [d[1] for d in dihedrals_deg], dtype=torch.long, device=device
            ).reshape(-1, 4) if dihedrals_deg else torch.empty(0, 4, dtype=torch.long, device=device)
            dt = []
            for target, indices in dihedrals_deg:
                if target is None:
                    dv0 = pos[indices[1]] - pos[indices[0]]
                    dv1 = pos[indices[2]] - pos[indices[1]]
                    dv2 = pos[indices[3]] - pos[indices[2]]
                    target = get_dihedrals(
                        dv0.unsqueeze(0), dv1.unsqueeze(0), dv2.unsqueeze(0),
                        cell=cell, pbc=pbc,
                    ).item()
                dt.append(target)
            dt_tensor = torch.tensor(dt, dtype=dtype, device=device) if dt else torch.empty(0, dtype=dtype, device=device)
            all_dih_idx.append(di)
            all_dih_tgt.append(dt_tensor)

            # Bondcombos
            sys_combo_idx = []
            sys_combo_coef = []
            ct = []
            for target, combo_def in bondcombos:
                pairs = torch.tensor(
                    [[d[0], d[1]] for d in combo_def],
                    dtype=torch.long, device=device,
                )
                coefs = torch.tensor(
                    [d[2] for d in combo_def], dtype=dtype, device=device,
                )
                sys_combo_idx.append(pairs)
                sys_combo_coef.append(coefs)
                if target is None:
                    vecs = pos[pairs[:, 1]] - pos[pairs[:, 0]]
                    (vecs_mic,), (dists,) = conditional_find_mic(
                        [vecs], cell=cell, pbc=pbc
                    )
                    target = (coefs * dists).sum().item()
                ct.append(target)
            ct_tensor = torch.tensor(ct, dtype=dtype, device=device) if ct else torch.empty(0, dtype=dtype, device=device)
            all_combo_idx.append(sys_combo_idx)
            all_combo_coef.append(sys_combo_coef)
            all_combo_tgt.append(ct_tensor)

        return cls(
            torch.tensor(system_indices, dtype=torch.long, device=device),
            bond_indices=all_bond_idx,
            bond_targets=all_bond_tgt,
            angle_indices=all_angle_idx,
            angle_targets=all_angle_tgt,
            dihedral_indices=all_dih_idx,
            dihedral_targets=all_dih_tgt,
            combo_indices=all_combo_idx,
            combo_coefs=all_combo_coef,
            combo_targets=all_combo_tgt,
            mic=mic,
            epsilon=epsilon,
        )

    @classmethod
    def from_per_system_definitions(
        cls,
        state: SimState,
        definitions: list[dict],
        *,
        mic: bool = False,
        epsilon: float = 1e-7,
    ) -> Self:
        """Create FixInternals with independent constraint definitions per system.

        Unlike ``from_definitions`` (which applies the same constraints to all
        systems), this method takes a list of definition dicts — one per system
        — so each system can have different constraint types, atom indices, and
        target values.

        Args:
            state: Simulation state (positions, cell, etc.).
            definitions: One dict per constrained system.  Each dict may contain:
                - ``"dihedrals_deg"``: ``[(target_or_None, [i,j,k,l]), ...]``
                - ``"bonds"``: ``[(target_or_None, [i,j]), ...]``
                - ``"angles_deg"``: ``[(target_or_None, [i,j,k]), ...]``
                - ``"bondcombos"``: ``[(target_or_None, [[i,j,coef], ...]), ...]``
                - ``"system_index"``: int (default: position in the list)
                Atom indices are **local** (0-based within each system).
                If a target is ``None``, the current value from *state* is used.
            mic: Minimum image convention for periodic systems.
            epsilon: Convergence tolerance.

        Returns:
            FixInternals instance with per-system constraints.

        Example:
            >>> # Batch of 3 conformers, each with a different dihedral target
            >>> defs = [
            ...     {"dihedrals_deg": [(None, [6, 10, 11, 8])]},  # auto-measure
            ...     {"dihedrals_deg": [(210.0, [6, 10, 11, 8])]},
            ...     {"dihedrals_deg": [(225.0, [6, 10, 11, 8])]},
            ... ]
            >>> constraint = FixInternals.from_per_system_definitions(state, defs, mic=True)
        """
        from torch_sim.geometry import (
            conditional_find_mic,
            get_angles,
            get_dihedrals,
        )

        device = state.device
        dtype = state.dtype
        cumsum = _cumsum_with_zero(state.n_atoms_per_system)

        system_indices = []
        all_bond_idx, all_bond_tgt = [], []
        all_angle_idx, all_angle_tgt = [], []
        all_dih_idx, all_dih_tgt = [], []
        all_combo_idx, all_combo_coef, all_combo_tgt = [], [], []

        cell = None
        pbc = None
        if mic:
            pbc = state.pbc

        for i, defn in enumerate(definitions):
            si = defn.get("system_index", i)
            system_indices.append(si)
            start = cumsum[si].item()
            pos = state.positions  # global positions

            if mic:
                cell = state.row_vector_cell[si]

            bonds = defn.get("bonds", [])
            angles_deg = defn.get("angles_deg", [])
            dihedrals_deg = defn.get("dihedrals_deg", [])
            bondcombos = defn.get("bondcombos", [])

            # User provides local (0-based) indices.  We need global indices
            # only for measuring current values from the global positions tensor.
            # Stored indices must be LOCAL because adjust_positions/adjust_forces
            # slice to per-system positions before indexing.
            def to_global(indices: list[int]) -> list[int]:
                return [idx + start for idx in indices]

            # Bonds — store LOCAL indices, use global only for measurement
            bi = torch.tensor(
                [b[1] for b in bonds], dtype=torch.long, device=device
            ).reshape(-1, 2) if bonds else torch.empty(0, 2, dtype=torch.long, device=device)
            bt = []
            for target, indices in bonds:
                if target is None:
                    g = to_global(indices)
                    v = pos[g[1]] - pos[g[0]]
                    (v_mic,), (d,) = conditional_find_mic(
                        [v.unsqueeze(0)], cell=cell, pbc=pbc
                    )
                    target = d.item()
                bt.append(target)
            bt_t = torch.tensor(bt, dtype=dtype, device=device) if bt else torch.empty(0, dtype=dtype, device=device)
            all_bond_idx.append(bi)
            all_bond_tgt.append(bt_t)

            # Angles — store LOCAL indices
            ai = torch.tensor(
                [a[1] for a in angles_deg], dtype=torch.long, device=device
            ).reshape(-1, 3) if angles_deg else torch.empty(0, 3, dtype=torch.long, device=device)
            at = []
            for target, indices in angles_deg:
                if target is None:
                    g = to_global(indices)
                    v0 = pos[g[0]] - pos[g[1]]
                    v1 = pos[g[2]] - pos[g[1]]
                    target = get_angles(
                        v0.unsqueeze(0), v1.unsqueeze(0), cell=cell, pbc=pbc
                    ).item()
                at.append(target)
            at_t = torch.tensor(at, dtype=dtype, device=device) if at else torch.empty(0, dtype=dtype, device=device)
            all_angle_idx.append(ai)
            all_angle_tgt.append(at_t)

            # Dihedrals — store LOCAL indices
            di = torch.tensor(
                [d[1] for d in dihedrals_deg], dtype=torch.long, device=device
            ).reshape(-1, 4) if dihedrals_deg else torch.empty(0, 4, dtype=torch.long, device=device)
            dt = []
            for target, indices in dihedrals_deg:
                if target is None:
                    g = to_global(indices)
                    dv0 = pos[g[1]] - pos[g[0]]
                    dv1 = pos[g[2]] - pos[g[1]]
                    dv2 = pos[g[3]] - pos[g[2]]
                    target = get_dihedrals(
                        dv0.unsqueeze(0), dv1.unsqueeze(0), dv2.unsqueeze(0),
                        cell=cell, pbc=pbc,
                    ).item()
                dt.append(target)
            dt_t = torch.tensor(dt, dtype=dtype, device=device) if dt else torch.empty(0, dtype=dtype, device=device)
            all_dih_idx.append(di)
            all_dih_tgt.append(dt_t)

            # Bondcombos — store LOCAL indices
            sys_combo_idx = []
            sys_combo_coef = []
            ct = []
            for target, combo_def in bondcombos:
                # Store local indices
                pairs = torch.tensor(
                    [[d[0], d[1]] for d in combo_def],
                    dtype=torch.long, device=device,
                )
                coefs = torch.tensor(
                    [d[2] for d in combo_def], dtype=dtype, device=device,
                )
                sys_combo_idx.append(pairs)
                sys_combo_coef.append(coefs)
                if target is None:
                    # Use global indices for measurement
                    g_pairs = torch.tensor(
                        [to_global([d[0], d[1]]) for d in combo_def],
                        dtype=torch.long, device=device,
                    )
                    vecs = pos[g_pairs[:, 1]] - pos[g_pairs[:, 0]]
                    (vecs_mic,), (dists,) = conditional_find_mic(
                        [vecs], cell=cell, pbc=pbc
                    )
                    target = (coefs * dists).sum().item()
                ct.append(target)
            ct_t = torch.tensor(ct, dtype=dtype, device=device) if ct else torch.empty(0, dtype=dtype, device=device)
            all_combo_idx.append(sys_combo_idx)
            all_combo_coef.append(sys_combo_coef)
            all_combo_tgt.append(ct_t)

        return cls(
            torch.tensor(system_indices, dtype=torch.long, device=device),
            bond_indices=all_bond_idx,
            bond_targets=all_bond_tgt,
            angle_indices=all_angle_idx,
            angle_targets=all_angle_tgt,
            dihedral_indices=all_dih_idx,
            dihedral_targets=all_dih_tgt,
            combo_indices=all_combo_idx,
            combo_coefs=all_combo_coef,
            combo_targets=all_combo_tgt,
            mic=mic,
            epsilon=epsilon,
        )

    def __repr__(self) -> str:
        counts = self._n_constraints_per_system()
        total = sum(counts)
        return (
            f"FixInternals(n_systems={len(self.system_idx)}, "
            f"n_constraints={total}, mic={self.mic})"
        )


def count_degrees_of_freedom(
    state: SimState, constraints: list[Constraint] | None = None
) -> torch.Tensor:
    """Count per-system degrees of freedom with compatibility checks.

    This helper computes one DOF value per system. When ``constraints`` are
    supplied, it validates that they are compatible with ``state`` before
    counting.

    Args:
        state: Simulation state
        constraints: Constraints to evaluate. If ``None``, returns unconstrained
            DOF (3 * n_atoms_per_system). Use ``state.get_number_of_degrees_of_freedom()``
            to count with state-attached constraints.

    Returns:
        Degrees of freedom per system as a tensor of shape (n_systems,)
    """
    if constraints is not None:
        validate_constraints(constraints, state)
    return torch.clamp(_dof_per_system(state, constraints), min=0)


def _dof_per_system(
    state: SimState, constraints: list[Constraint] | None = None
) -> torch.Tensor:
    """Compute unconstrained-minus-removed DOF per system."""
    dof_per_system = 3 * state.n_atoms_per_system
    if constraints is not None:
        for constraint in constraints:
            dof_per_system -= constraint.get_removed_dof(state)
    return dof_per_system


def check_no_index_out_of_bounds(
    indices: torch.Tensor, max_state_indices: int, constraint_name: str
) -> None:
    """Check that constraint indices are within bounds of the state."""
    if (len(indices) > 0) and (indices.max() >= max_state_indices):
        raise ValueError(
            f"Constraint {constraint_name} has indices up to "
            f"{indices.max()}, but state only has {max_state_indices} "
            "atoms"
        )


def validate_constraints(constraints: list[Constraint], state: SimState) -> None:
    """Validate constraints for potential issues and incompatibilities.

    This function checks for:
    1. Overlapping atom indices across multiple constraints
    2. AtomConstraints spanning multiple systems (requires state)
    3. Mixing FixCom with other constraints (warning only)

    Args:
        constraints: List of constraints to validate
        state: SimState to check against

    Raises:
        ValueError: If constraints are invalid or span multiple systems

    Warns:
        UserWarning: If constraints may lead to unexpected behavior
    """
    if not constraints:
        return

    indexed_constraints = []
    has_com_constraint = False

    for constraint in constraints:
        if isinstance(constraint, AtomConstraint):
            indexed_constraints.append(constraint)

            # Validate that atom indices exist in state if provided
            check_no_index_out_of_bounds(
                constraint.atom_idx, state.n_atoms, type(constraint).__name__
            )
        elif isinstance(constraint, SystemConstraint):
            check_no_index_out_of_bounds(
                constraint.system_idx, state.n_systems, type(constraint).__name__
            )

        if isinstance(constraint, FixCom):
            has_com_constraint = True

    # Check for overlapping atom indices
    if len(indexed_constraints) > 1:
        all_indices = torch.cat([c.atom_idx for c in indexed_constraints])
        unique_indices = torch.unique(all_indices)
        if len(unique_indices) < len(all_indices):
            msg = (
                "Multiple constraints are acting on the same atoms. "
                "This may lead to unexpected behavior."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            logger.warning(msg)

    # Warn about COM constraint with fixed atoms
    if has_com_constraint and indexed_constraints:
        msg = (
            "Using FixCom together with other constraints may lead to "
            "unexpected behavior. The center of mass constraint is applied "
            "to all atoms, including those that may be constrained by other means."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)


class FixSymmetry(SystemConstraint):
    """Preserve spacegroup symmetry during optimization.

    Symmetrizes forces/momenta as rank-1 tensors and stress/cell deformation
    as rank-2 tensors using the crystal's symmetry operations. Each system in
    a batch can have different symmetry operations.

    Forces and stress are always symmetrized. Position and cell symmetrization
    can be toggled via ``adjust_positions`` and ``adjust_cell``.
    """

    rotations: list[torch.Tensor]
    symm_maps: list[torch.Tensor]
    reference_cells: list[torch.Tensor] | None
    do_adjust_positions: bool
    do_adjust_cell: bool
    max_cumulative_strain: float

    def __init__(
        self,
        rotations: list[torch.Tensor],
        symm_maps: list[torch.Tensor],
        system_idx: torch.Tensor | None = None,
        *,
        adjust_positions: bool = True,
        adjust_cell: bool = True,
        reference_cells: list[torch.Tensor] | None = None,
        max_cumulative_strain: float = 0.5,
    ) -> None:
        """Initialize FixSymmetry constraint.

        Args:
            rotations: Rotation tensors per system, each (n_ops, 3, 3).
            symm_maps: Atom mapping tensors per system, each (n_ops, n_atoms).
            system_idx: System indices (defaults to 0..n_systems-1).
            adjust_positions: Whether to symmetrize position displacements.
            adjust_cell: Whether to symmetrize cell/stress adjustments.
            reference_cells: Initial refined cells (row vectors) per system for
                cumulative strain tracking. If None, cumulative check is skipped.
            max_cumulative_strain: Maximum allowed cumulative strain from the
                reference cell. If exceeded, the cell update is clamped to
                keep the structure within this strain envelope.
        """
        n_systems = len(rotations)
        if len(symm_maps) != n_systems:
            raise ValueError(
                f"rotations and symm_maps length mismatch: "
                f"{n_systems} vs {len(symm_maps)}"
            )
        if system_idx is None:
            device = rotations[0].device if rotations else torch.device("cpu")
            system_idx = torch.arange(n_systems, device=device)
        if len(system_idx) != n_systems:
            raise ValueError(
                f"system_idx length ({len(system_idx)}) != n_systems ({n_systems})"
            )
        if reference_cells is not None and len(reference_cells) != n_systems:
            raise ValueError(
                f"reference_cells length ({len(reference_cells)}) "
                f"!= n_systems ({n_systems})"
            )

        super().__init__(system_idx=system_idx)
        self.rotations = rotations
        self.symm_maps = symm_maps
        self.reference_cells = reference_cells
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell
        self.max_cumulative_strain = max_cumulative_strain

    @classmethod
    def from_state(
        cls,
        state: SimState,
        symprec: float = 0.01,
        *,
        adjust_positions: bool = True,
        adjust_cell: bool = True,
        refine_symmetry_state: bool = True,
        angle_tolerance: float | None = None,
    ) -> Self:
        """Create from SimState, optionally refining to ideal symmetry first.

        Warning:
            When ``refine_symmetry_state=True`` (default), the input state is
            **mutated in-place** to have ideal symmetric positions and cell.

        Args:
            state: SimState containing one or more systems.
            symprec: Symmetry precision for moyopy.
            adjust_positions: Whether to symmetrize position displacements.
            adjust_cell: Whether to symmetrize cell/stress adjustments.
            refine_symmetry_state: Whether to refine positions/cell to ideal values.
            angle_tolerance: Angle tolerance in radians for moyopy symmetry
                detection. If None, moyopy uses its default behaviour.
        """
        try:
            import moyopy  # noqa: F401
        except ImportError:
            raise ImportError(
                "moyopy required for FixSymmetry: pip install moyopy"
            ) from None

        from torch_sim.symmetrize import prep_symmetry, refine_and_prep_symmetry

        rotations, symm_maps, reference_cells = [], [], []
        cumsum = _cumsum_with_zero(state.n_atoms_per_system)

        for sys_idx in range(state.n_systems):
            start, end = cumsum[sys_idx].item(), cumsum[sys_idx + 1].item()
            cell = state.row_vector_cell[sys_idx]
            pos, nums = state.positions[start:end], state.atomic_numbers[start:end]

            if refine_symmetry_state:
                # Single moyopy call: refine + get symmetry ops in one pass
                cell, pos, rots, smap = refine_and_prep_symmetry(
                    cell,
                    pos,
                    nums,
                    symprec=symprec,
                    angle_tolerance=angle_tolerance,
                )
                state.cell[sys_idx] = cell.mT  # row→column vector convention
                state.positions[start:end] = pos
            else:
                rots, smap = prep_symmetry(
                    cell,
                    pos,
                    nums,
                    symprec=symprec,
                    angle_tolerance=angle_tolerance,
                )

            rotations.append(rots)
            symm_maps.append(smap)
            # Store the refined cell as the reference for cumulative strain tracking
            reference_cells.append(state.row_vector_cell[sys_idx].clone())

        return cls(
            rotations,
            symm_maps,
            system_idx=torch.arange(state.n_systems, device=state.device),
            adjust_positions=adjust_positions,
            adjust_cell=adjust_cell,
            reference_cells=reference_cells,
        )

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Symmetrize forces according to crystal symmetry."""
        self._symmetrize_rank1(state, forces)

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Symmetrize position displacements (skipped if do_adjust_positions=False)."""
        if not self.do_adjust_positions:
            return
        displacement = new_positions - state.positions
        self._symmetrize_rank1(state, displacement)
        new_positions[:] = state.positions + displacement

    def adjust_stress(self, state: SimState, stress: torch.Tensor) -> None:
        """Symmetrize stress tensor in-place.

        Always runs (like adjust_forces), independent of do_adjust_cell.
        """
        from torch_sim.symmetrize import symmetrize_rank2

        dtype = stress.dtype
        for ci, si in enumerate(self.system_idx):
            rots = self.rotations[ci].to(dtype=dtype)
            stress[si] = symmetrize_rank2(state.row_vector_cell[si], stress[si], rots)

    def adjust_cell(self, state: SimState, cell: torch.Tensor) -> None:
        """Symmetrize cell deformation gradient in-place.

        Computes ``F = inv(cell) @ new_cell_row``, symmetrizes ``F - I`` as a
        rank-2 tensor, then reconstructs ``cell @ (sym(F-I) + I)``.

        Also checks cumulative strain from the initial reference cell. If the
        total deformation exceeds ``max_cumulative_strain``, the update is
        clamped to prevent phase transitions that would break the symmetry
        constraint (e.g. hexagonal → tetragonal cell collapse).

        Args:
            state: Current simulation state.
            cell: Cell tensor (n_systems, 3, 3) in column vector convention.

        Raises:
            RuntimeError: If deformation gradient contains NaN or Inf.
        """
        if not self.do_adjust_cell:
            return

        from torch_sim.symmetrize import symmetrize_rank2

        identity = torch.eye(3, device=state.device, dtype=state.dtype)
        for ci, si in enumerate(self.system_idx):
            cur_cell = state.row_vector_cell[si]
            new_row = cell[si].mT  # column → row convention

            # Per-step deformation: clamp large steps to avoid ill-conditioned
            # symmetrization while still making progress. The cumulative strain
            # guard below is the real safety net against phase transitions.
            deform_delta = torch.linalg.solve(cur_cell, new_row) - identity
            max_delta = torch.abs(deform_delta).max().item()
            if not math.isfinite(max_delta):
                raise RuntimeError(
                    f"FixSymmetry: deformation gradient is {max_delta}, "
                    f"cell may be singular or ill-conditioned."
                )
            if max_delta > 0.25:
                deform_delta = deform_delta * (0.25 / max_delta)

            # Symmetrize the per-step deformation
            rots = self.rotations[ci].to(dtype=state.dtype)
            sym_delta = symmetrize_rank2(cur_cell, deform_delta, rots)
            proposed_cell = cur_cell @ (sym_delta + identity)

            # Cumulative strain check against reference cell
            if self.reference_cells is not None:
                ref_cell = self.reference_cells[ci].to(
                    device=state.device, dtype=state.dtype
                )
                cumulative_strain = torch.linalg.solve(ref_cell, proposed_cell) - identity
                max_cumulative = torch.abs(cumulative_strain).max().item()
                if max_cumulative > self.max_cumulative_strain:
                    scale = self.max_cumulative_strain / max_cumulative
                    proposed_cell = ref_cell @ (cumulative_strain * scale + identity)

            cell[si] = proposed_cell.mT  # back to column convention

    def _symmetrize_rank1(self, state: SimState, vectors: torch.Tensor) -> None:
        """Symmetrize a rank-1 tensor in-place for each constrained system."""
        from torch_sim.symmetrize import symmetrize_rank1

        cumsum = _cumsum_with_zero(state.n_atoms_per_system)
        dtype = vectors.dtype
        for ci, si in enumerate(self.system_idx):
            start, end = cumsum[si].item(), cumsum[si + 1].item()
            vectors[start:end] = symmetrize_rank1(
                state.row_vector_cell[si],
                vectors[start:end],
                self.rotations[ci].to(dtype=dtype),
                self.symm_maps[ci],
            )

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Returns zero - constrains direction, not DOF count."""
        return torch.zeros(state.n_systems, dtype=torch.long, device=state.device)

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with system indices shifted by system_offset."""
        return type(self)(
            list(self.rotations),
            list(self.symm_maps),
            self.system_idx + system_offset,
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=list(self.reference_cells) if self.reference_cells else None,
            max_cumulative_strain=self.max_cumulative_strain,
        )

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a copy with tensors moved to *device*/*dtype*."""
        return type(self)(
            [r.to(device=device, dtype=dtype) for r in self.rotations],
            [s.to(device=device) for s in self.symm_maps],
            self.system_idx.to(device=device),
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=(
                [c.to(device=device, dtype=dtype) for c in self.reference_cells]
                if self.reference_cells is not None
                else None
            ),
            max_cumulative_strain=self.max_cumulative_strain,
        )

    @classmethod
    def merge(cls, constraints: list[Constraint]) -> Self:
        """Merge by concatenating rotations, symm_maps, and system indices."""
        fix_sym_constraints = [c for c in constraints if isinstance(c, FixSymmetry)]
        if not fix_sym_constraints:
            raise ValueError("Cannot merge empty constraint list")
        if any(
            c.do_adjust_positions != fix_sym_constraints[0].do_adjust_positions
            or c.do_adjust_cell != fix_sym_constraints[0].do_adjust_cell
            or c.max_cumulative_strain != fix_sym_constraints[0].max_cumulative_strain
            for c in fix_sym_constraints[1:]
        ):
            raise ValueError(
                "Cannot merge FixSymmetry constraints with different "
                "adjust_positions/adjust_cell/max_cumulative_strain settings"
            )
        rotations = [r for c in fix_sym_constraints for r in c.rotations]
        symm_maps = [s for c in fix_sym_constraints for s in c.symm_maps]
        system_idx = torch.cat([c.system_idx for c in fix_sym_constraints])
        # Merge reference cells if all constraints have them
        ref_cells = None
        if all(c.reference_cells is not None for c in fix_sym_constraints):
            ref_cells = []
            for c in fix_sym_constraints:
                refs = c.reference_cells
                if refs is not None:
                    ref_cells.extend(refs)
        return cls(
            rotations,
            symm_maps,
            system_idx=system_idx,
            adjust_positions=fix_sym_constraints[0].do_adjust_positions,
            adjust_cell=fix_sym_constraints[0].do_adjust_cell,
            reference_cells=ref_cells,
            max_cumulative_strain=fix_sym_constraints[0].max_cumulative_strain,
        )

    def select_constraint(
        self,
        atom_mask: torch.Tensor,  # noqa: ARG002
        system_mask: torch.Tensor,
    ) -> Self | None:
        """Select constraint for systems matching the mask."""
        keep = torch.where(system_mask)[0]
        mask = torch.isin(self.system_idx, keep)
        if not mask.any():
            return None
        local_idx = mask.nonzero(as_tuple=False).flatten().tolist()
        ref_cells = (
            [self.reference_cells[idx] for idx in local_idx]
            if self.reference_cells
            else None
        )
        return type(self)(
            [self.rotations[idx] for idx in local_idx],
            [self.symm_maps[idx] for idx in local_idx],
            _mask_constraint_indices(self.system_idx[mask], system_mask),
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=ref_cells,
            max_cumulative_strain=self.max_cumulative_strain,
        )

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,  # noqa: ARG002
        sys_idx: int,
    ) -> Self | None:
        """Select constraint for a single system."""
        if sys_idx not in self.system_idx:
            return None
        local = (self.system_idx == sys_idx).nonzero(as_tuple=True)[0].item()
        ref_cells = [self.reference_cells[local]] if self.reference_cells else None
        return type(self)(
            [self.rotations[local]],
            [self.symm_maps[local]],
            torch.tensor([0], device=self.system_idx.device),
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=ref_cells,
            max_cumulative_strain=self.max_cumulative_strain,
        )

    def __repr__(self) -> str:
        """String representation."""
        n_ops = [r.shape[0] for r in self.rotations]
        ops = str(n_ops) if len(n_ops) <= 3 else f"[{n_ops[0]}, ..., {n_ops[-1]}]"
        return (
            f"FixSymmetry(n_systems={len(self.rotations)}, n_ops={ops}, "
            f"adjust_positions={self.do_adjust_positions}, "
            f"adjust_cell={self.do_adjust_cell})"
        )
