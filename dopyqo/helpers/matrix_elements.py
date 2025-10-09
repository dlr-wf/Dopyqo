from dataclasses import dataclass
import numpy as np
import dopyqo


@dataclass
class MatrixElements:
    """Dataclass to store matrix elements"""

    h_pq_kin: np.ndarray | None = None
    """Kinetic energy part of the one-electron matrix elements"""
    h_pq_pp: np.ndarray | None = None
    """Pseudopotential part of the one-electron matrix elements"""
    h_pq_core: np.ndarray | None = None
    """Effective single-particle potential from the frozen core approximation part of the one-electron matrix elements"""
    h_pqrs: np.ndarray | None = None
    r"""Two-electron matrix elements (electron-electron interaction) defined as
    h_pqrs = ∫∫ ψ*_p(r_1) ψ*_q(r_2) ψ_r(r_2) ψ_s(r_1) 1/|r_1-r_2| dr_1 dr_2
    """
    h_pq_ewald: np.ndarray | None = None
    """Nuclear-nuclear interaction energy multiplied with the overlap matrix as one-electron matrix elements"""
    energy_frozen_core: float | None = None
    """Frozen core energy"""
    energy_ewald: float | None = None
    """Nuclear-nuclear interaction energy"""
    energy_e_self: float | None = None
    """Electron self-energy"""
    transform_matrix: np.ndarray | None = None
    """Transformation matrix to transform matrix elements into a different basis. Used for Wannier transformations"""

    @property
    def h_pq(self) -> np.ndarray:
        """One-electron matrix elements: h_pq = h_pq_kin + h_pq_pp + h_pq_core.
        h_pq is transformed with transform_matrix if transform_matrix is not None.
        Transformation is done with dopyqo.transform_one_body_matrix function.
        """
        if self.h_pq_kin is None or self.h_pq_pp is None or self.h_pq_core is None:
            return None
        if self.transform_matrix is not None:
            return dopyqo.transform_one_body_matrix(self.h_pq_kin + self.h_pq_pp + self.h_pq_core, self.transform_matrix)
        return self.h_pq_kin + self.h_pq_pp + self.h_pq_core
