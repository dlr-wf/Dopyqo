import itertools
import logging
import os
from collections import Counter
from functools import partial

import numpy as np
from scipy.special import erfc

from dopyqo import calc_pseudo_pot as cpp
from dopyqo.colors import *
from dopyqo.helpers.atoms import elements_to_atomic_number
from dopyqo.pseudopot import Pseudopot


def gradient_nuclear_repulsion_energy_ewald_wrt_atom_positions(
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    lattice_vectors: np.ndarray,
    lattice_vectors_reciprocal: np.ndarray,
    cell_volume: float,
    gcutrho: int,
    sigma: float | None = None,
    rust_impl: bool = False,
) -> np.ndarray:
    """
    Compute the analytical gradient of the nuclear repulsion energy using
    the Ewald summation method for periodic systems.

    This function returns the gradient on every atom in `atom_positions`/`atomic_numbers`
    due to Coulombic nuclear repulsion in a periodic cell. Selective relaxation (updating
    only some atoms) is not handled here - it is a purely downstream choice: compute the
    gradient for all atoms and simply don't apply it to the atoms that should stay fixed.
    Therefore for the gradient only the short-range and long-range interactions are considered,
    as they are the ones that directly depend on the atomic positions. The other two
    terms of the Ewald sum (self-interaction and background) do not contribute to the gradient
    with respect to atomic positions, therefore they are considered as constant and will be 0 in the
    gradient calculation. The analytical gradients are based on equation 18 for the Ewald energy
    in the dopyqo paper.
    Optionally, if dopyqo-rs is available, a faster implementation can be used.

    Parameters:
    -----------
      atom_positions (np.ndarray):
        Cartesian positions (each atom described by x,y,z) of all atoms of the analysed system.
        Shape: (n_atoms, 3). Units: must be in Bohr for the gradients (must be consistent with lattice_vectors).

      atomic_numbers (np.ndarray):
        Nuclear charges (atomic numbers, Z) for all atoms (`atom_positions`).
        Shape: (n_atoms,). Integer values.

      lattice_vectors (np.ndarray):
        Lattice vectors a,b,c defining the periodic cell with each vector described by x,y,z.
        Shape: (3, 3), where each row is a lattice vector in Cartesian coordinates.
        Units must match `atom_positions` and need to be in Bohr.

      lattice_vectors_reciprocal (np.ndarray):
        Reciprocal lattice vectors.
        Shape: (3, 3), where each row is a reciprocal lattice vector.

      cell_volume (float):
        Volume of the unit cell. Units consistent with `lattice_vectors` and here in Bohr^3.

      gcutrho (int):
        Cutoff for the magnitude of reciprocal lattice vectors G used in the long-range sum.
        All G with |G| ≤ gcutrho are included (except G = 0).

      sigma (float | None, optional):
        Ewald splitting parameter sigma. If None, an automatic selection loop reduces sigma from 2.8
        until an upper bound estimate for the reciprocal-space error is ≤ 1e-7.
        Must be positive. Larger sigma shifts more weight to real space; smaller sigma shifts more to reciprocal space.
        Copied from the implimentation in dopyqo, wich it self is based on the implementaion from Quantum ESPRESSO.

      rust_impl (bool, optional):
        If True and the Rust extension `dopyqo_rs` is importable, use `dopyqo_rs.ewald_forces`
        to compute the gradients. Falls back to Python if import fails.

    Returns:
    --------
      np.ndarray:
        Gradients of nuclear repulsion energy on every atom.
        Shape: (n_atoms, 3). Units consistent with input units.
        The gradient are a matrix each entry corespons to a atom and contains
        the gradient wrt x,y,z cartesian coordinates.
        Example: [[d/dx1, d/dy1, d/dz1], ...] for each atom.

    Notes:
    ------
      - The majority o9f the code was copied from the dopyqo package, with adjustments to
        properly represent the analytical gradient.
      - The function assumes all inputs are in Cartesian coordinates and in Bohr
        units for consistency.
    Assumptions:
      - Inputs are consistent and in Cartesian coordinates.
      - Lattice vectors form a valid cell (non-zero volume).
      - Units are pased consistently (Bohr for lengths, Bohr^3 for volume).

    Mathematical Expression:
    -------------------------
    The equations for the gradients g are based on the definition of the Ewald summation  for nuclear repulsion energy
    from the dopyqo paper, specifically equation 18. The gradient with respect to atomic positions R_K
    involves two main contributions:

      - Short-range (real space):
        g_{K,short} =-Z_{K}\sum_{N}\sum_{T}(R_{K}-R_{N}-T) \cdot \frac{1}{|R_{K}-R_{N}-T|^{3}} \cdot \left( \frac{2\sqrt{\sigma}}{\sqrt{\pi}} \cdot \exp{-(|R_{K}-R_{N}-T|\sqrt{\sigma})^{2}} \cdot |R_{K}-R_{N}-T| + erfc(|R_{K}-R_{N}-T|\sqrt{\sigma}) \right)
      - Long-range (reciprocal space):
        g_{K,long} = -Z_{K} \cdot \frac{4\pi}{V} \cdot \sum_{G \neq 0} \frac{\vec{G}}{G^{2}} \cdot \exp{\left(-\frac{G^{2}}{4\sigma}\right)} \cdot \sum_{N} Z_{N} \cdot \sin(\vec{G} \cdot (R_{K}-R_{N}))

      where:
        - K indexes the target atoms for which forces are computed (subset).
        - N indexes all atoms present in the periodic system (including the subset).
        - T runs over lattice translation vectors.
        - G, the , runs over reciprocal lattice vectors within a cutoff.
        - \vec{G} is the reciprocal lattice vector,
        - Z are nuclear charges, R are atomic positions, V is cell volume, \sigma is the Ewald splitting parameter.

    As the remaining two terms of the Ewald summation (self-interaction and background) do not depend on atomic positions,
    they do not contribute to the gradient and are therefore omitted in the calculation.

    Both the short and long term of the gradient are combiend to get the total gradient for atom K
    g_{K,short} and g_{K,long} both are vectors with 3 components (x,y,z) and the total gradient is given by:
      g_K = g_{K,short} + g_{K,long}

    Where:
      - K indexes the target atoms for which forces are computed (subset).
      - N indexes all atoms present in the periodic system (including the subset).
      - T runs over lattice translation vectors.
      - G runs over reciprocal lattice vectors within a cutoff.
      - Z are nuclear charges, R are atomic positions, V is cell volume, σ is the Ewald splitting parameter.

    Refferences:
    ------------
    - Equations 18 in the dopyqo paper:
        Schultheis E et al. Many-body post-processing of density functional calculations using the variational quantum eigensolver for Bader charge analysis.
        arXiv [quant-ph]. Published online 14 October 2025. doi:10.48550/arXiv.2510.12887
    """

    def getVec(R_K, R_N, T):
        vec = R_K - R_N - T
        return vec

    if rust_impl:
        try:
            import dopyqo_rs as calc_rs
        except ImportError:
            print(f"{ORANGE}Ewald warning: Could not import dopyqo_rs package. Falling back to python implementation.{RESET_COLOR}")
            rust_impl = False  # Set such that variable accurately represents if rust implementation is used
        else:  # No exception
            logging.info("Using Rust implementation.")
            return calc_rs.ewald_forces(
                atom_positions,
                atomic_numbers,
                lattice_vectors,
                lattice_vectors_reciprocal,
                cell_volume,
                gcutrho,
                sigma,
            )

    if sigma is None:
        sigma = 2.8
        charge = np.sum(atomic_numbers)
        gcutm = gcutrho**2

        # choose sigma in order to have convergence in the sum over G
        # upperbound is a safe upper bound for the error in the sum over G
        while True:
            if sigma <= 0.0:
                raise RuntimeError("Optimal sigma for Ewald sum not found!")
            upperbound = 2.0 * charge**2 * np.sqrt(sigma / np.pi) * erfc(np.sqrt(gcutm / 4.0 / sigma))
            if upperbound > 1e-7:
                sigma = sigma - 0.1
            else:
                break
    logging.info("sigma %f", sigma)

    num_atoms = atom_positions.shape[0]
    full_gradients = np.zeros_like(atom_positions)
    for kth_Atom in range(num_atoms):
        Z_K = atomic_numbers[kth_Atom]
        R_K = atom_positions[kth_Atom]
        logging.info("Computing gradient for atom %d with Z=%d at position %s", kth_Atom, Z_K, R_K)
        grad_e_tot = None
        grad_e_short = 0.0  # real-space sum
        grad_e_long = 0.0  # reciprocal-space sum

        alat = np.linalg.norm(lattice_vectors[0], ord=2)
        t_vec_max_norm = 4.0 / np.sqrt(sigma) / alat
        b_norms = np.linalg.norm(lattice_vectors_reciprocal, ord=2, axis=1)
        n_max_x, n_max_y, n_max_z = (b_norms * t_vec_max_norm).astype(int) + 2

        # Generate list or translation vectors ordered by their norm
        # n_unordered = itertools.product(range(-n_max, n_max + 1), repeat=3)
        # logging.debug("Generating unordered translation vectors...")
        t_vecs_unordered = [
            [np.dot(np.array([nx, ny, nz]), lattice_vectors), [nx, ny, nz]]
            for nx, ny, nz in itertools.product(
                range(-n_max_x, n_max_x + 1),
                range(-n_max_y, n_max_y + 1),
                range(-n_max_z, n_max_z + 1),
            )
            # for nx, ny, nz in n_unordered
        ]

        t_vecs = sorted(t_vecs_unordered, key=lambda x: np.linalg.norm(x[0], ord=2))

        #  ------------------------------------------------------------------------------------------
        #  Gradients term for the SHORT-RANGE interaction of the Ewald sum wrt to the atomic positions
        #  ------------------------------------------------------------------------------------------
        # -Z_K \sum_T \sum_N Z_N (R_K-R_N-T)/|R_K-R_N-T|^3 [(2*sqrt(sigma)/sqrt(pi))*exp(-sigma*|R_K-R_N-T|^2)*|R_K-R_N-T| + erfc(sqrt(sigma)*|R_K-R_N-T|)]

        for t_vec, (nx, ny, nz) in t_vecs:  # \sum_T
            for N, pos_N in enumerate(atom_positions):  # \sum_N
                Z_N = atomic_numbers[N]
                R_N = pos_N
                vec = getVec(R_K, R_N, t_vec)  # (R_K-R_N-T) vector
                r = np.linalg.norm(vec, ord=2)  # norm of the (R_K-R_N-T) vector
                if np.all(t_vec == 0.0) and np.array_equal(R_N, R_K):  # skip self-interaction (dividing by zero)
                    continue
                grad_e_short += (
                    Z_N
                    * vec
                    * ((((2 * np.sqrt(sigma)) / (np.sqrt(np.pi))) * np.exp(-((r * np.sqrt(sigma)) ** 2)) * r + erfc(r * np.sqrt(sigma))) / (r**3))
                )
        grad_e_short *= -Z_K

        #  ------------------------------------------------------------------------------------------
        #  Gradients term for the LONG-RANGE interaction of the Ewald sum wrt to the atomic positions
        #  ------------------------------------------------------------------------------------------
        # -Z_K (4*pi/V)  \sum_G!=0 (G_vec/G^2) exp(-G^2/4*sigma) \sum_N Z_N sin(G_vec*(R_K-R_N))

        # Estimate size of Miller indices grid using the cutoff-energy and the real-space lattice
        # vectors a_i (a_i . b_j = 2*pi*delta_ij). For G = kx*b1+ky*b2+kz*b3, we have
        # a_i . G = 2*pi*k_i, so by Cauchy-Schwarz |k_i| <= |a_i|*|G|/(2*pi) <= |a_i|*gcutrho/(2*pi).
        mill_max = max(round(abs(gcutrho * np.linalg.norm(a_i) / (2 * np.pi))) + 2 for a_i in lattice_vectors)
        x, y, z = np.meshgrid(
            np.arange(-mill_max, mill_max + 1),
            np.arange(-mill_max, mill_max + 1),
            np.arange(-mill_max, mill_max + 1),
            indexing="ij",
        )
        mill_rho = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        k_vecs = np.einsum("ij, kj -> ki", lattice_vectors_reciprocal.T, mill_rho)
        norms = np.linalg.norm(k_vecs, axis=1)
        k_vecs = k_vecs[norms <= gcutrho]

        G_vecs = k_vecs
        for G_vec in G_vecs:
            G2 = np.linalg.norm(G_vec, ord=2) ** 2
            for N, pos_N in enumerate(atom_positions):
                if np.isclose(G2, 0.0):
                    continue
                Z_N = atomic_numbers[N]
                R_N = pos_N
                grad_e_long += (G_vec / G2) * np.exp(-G2 / (4 * sigma)) * Z_N * np.sin(np.dot(G_vec, (R_K - R_N)))

        V = cell_volume
        grad_e_long *= -Z_K * ((4 * np.pi) / V)
        grad_e_tot = grad_e_short + grad_e_long
        full_gradients[kth_Atom] = grad_e_tot
    return full_gradients


def gradient_v_loc_pw(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    pseudopot: Pseudopot,
) -> np.ndarray:
    """
    Compute the gradient of the local pseudopotential term in plane-wave representation
    with respect to the position of a single atom in cartesian coordinates (x,y,z).

    This function calculates the derivative of the local pseudopotential matrix elements
    (in the plane-wave basis) with respect to the atomic position of a single atom.
    The calculation is performed for a single atom at a given position, using the provided
    reciprocal lattice vectors and the pseudopotential object.
    The analytical gradient is absed on equation 12 from the dopyqo paper.

    Parameters
    ----------
    p : np.ndarray
        Array of reciprocal lattice vectors (plane waves), shape (#waves, 3).
    c_ip : np.ndarray
        Coefficient matrix for the plane-wave basis, shape (n_bands, #waves).
    cell_volume : float
        Volume of the simulation cell in Bohr^3.
    atom_positions : np.ndarray
        Array of atomic positions for the atom of interest, shape (1, 3).
        Should contain only the coordinates of the atom for which the gradient is calculated.
        The atom is represented by its Cartesian coordinates (x, y, z) in Bohr.
    pseudopot : Pseudopot
        Pseudopotential object containing the local potential information.

    Returns
    -------
    np.ndarray
        The gradient tensor of the local pseudopotential term, shape (#atoms, 3, n_bands, n_bands),
        where #atoms is typically 1 for this function. The tensor contains the gradient components
        for each Cartesian direction and each pair of Kohn-Sham orbitals n_bands.

    Raises:
    -------
        ValueError
        If atom_positions does not have shape (1, 3).

    Note:
    -----
    - This function assumes that atom_positions contains only one atom's coordinates.
    - The gradient is calculated analytically based on the provided pseudopotential and plane-wave basis.
    - The function cals teh dopyco implementation for the local pseudopotential matrix elements
      and then computes the gradient based on that result. Therfore the local pseudopotential
      matrix elements are calculated for an atom at the origin (0,0,0) to obtain the R independant part of the local potential.
      The expeonential term and the gradient factor are then applied to obtain the full gradient with respect to the actual atom position.

    Mathematical Expression:
    ------------------------
    The analytical gradientint for the local pseudopotential <G_1|V_{loc}|G_2> term with respect to atomic positions is based on
    equation 12 from the dopyqo paper. The gradient is computed as follows:

    \frac{\partial}{\partial R} <G_1|V_{loc}|G_2>=<G_1|V_{loc}|G_2>\cdot (-i (G_1-G_2))

    where R is the atomic position,
    G_1 and G_2 are the reciprocal lattice vectors (plane waves),
    and V_{loc} is the local pseudopotential operator.

    Refferences:
    -----------
       -Equations 12 from the dopyqo paper
         Schultheis E et al. Many-body post-processing of density functional calculations using the variational quantum eigensolver for Bader charge analysis.
         arXiv [quant-ph]. Published online 14 October 2025. doi:10.48550/arXiv.2510.12887
    """
    atom_position_origin = np.array([[0.0, 0.0, 0.0]])  # only for one atom at origin # Used to compute the R independant part of the local potential
    local_pp_part = cpp.v_loc_pw(
        p=p,
        cell_volume=cell_volume,
        atom_positions=atom_position_origin,
        pseudopot=(pseudopot),
    )

    p_prime = p
    p_minus_p_prime = p[None] - p_prime[:, None]  # shape (#waves, #waves, 3)

    g_minus_g_prime_dot_R = np.sum(
        p_minus_p_prime[None] * atom_positions[:, None, None], axis=3
    )  # sum over 3D-coordinates, shape (#atoms, #waves, #waves)

    exp_term = np.exp(+1j * g_minus_g_prime_dot_R)  # use same +i phase as v_loc_pw

    gradient_term = +1j * p_minus_p_prime  # +i*(p-p') is the gradient factor
    gradient_term = gradient_term.transpose(2, 0, 1)

    gradient = (
        local_pp_part[np.newaxis, np.newaxis, :, :] * exp_term[:, np.newaxis, :, :] * gradient_term[np.newaxis, :, :, :]
    )  #  shape (#atoms,3,#waves,#waves)

    result = c_ip.conj() @ gradient @ c_ip.T

    return result


def gradient_v_nl_pw(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    pseudopot: Pseudopot,
) -> np.ndarray:
    r"""
    Compute the gradient of the non-local pseudopotential term in plane-wave representation
    with respect to atomic positions.

    This function calculates the derivative of the non-local pseudopotential matrix elements
    (in the plane-wave basis) with respect to the atomic positions for a set of atoms.
    The calculation is performed for each atom at a given position, using the provided
    reciprocal lattice vectors and the pseudopotential object.

    Parameters
    ----------
    p : np.ndarray
        Array of reciprocal lattice vectors (plane waves), shape (#waves, 3).
    c_ip : np.ndarray
        Coefficient matrix for the plane-wave basis, shape (n_bands, #waves).
    cell_volume : float
        Volume of the simulation cell in Bohr^3.
    atom_positions : np.ndarray
        Array of atomic positions for the atom(s) of interest, shape (n_atoms, 3).
        Each atom n is represented by its Cartesian coordinates (x, y, z) in Bohr.
    pseudopot : Pseudopot
        Pseudopotential object containing the non-local potential information.

    Returns
    -------
    np.ndarray
        The gradient tensor of the non-local pseudopotential term, shape (n_atoms, 3, n_bands, n_bands),
        where #atoms is typically 1 for this function. The tensor contains the gradient components
        for each Cartesian direction and each pair of Kohn-Sham orbitals n_bands.

    Raises:
    -------
        ValueError
        If atom_positions does not have shape (n_atoms, 3).

    Note:
    -----
     - This function assumes that atom_positions contains the coordinates of multiple atoms.
     - The gradient is calculated analytically based on the provided pseudopotential and plane-wave basis.
     - The function calls the dopyqo implementation for the non-local pseudopotential matrix elements
       and then computes the gradient based on that result. Therefore the non-local pseudopotential
       matrix elements are calculated for an atom at the origin (0,0,0) to obtain the R independant part of the non-local potential.
       The exponential term and the gradient factor are then applied to obtain the full gradient with respect to the actual atom positions.


    Mathematical Expression:
    ------------------------
    The analytical gradient for the non-local pseudopotential <G_1|V_{nl}|G_2> term with respect to atomic positions is based on
    equation 16 from the dopyqo paper. The gradient is computed as follows:

        \frac{\partial}{\partial R} <G_1|V_{nl}|G_2>=<G_1|V_{nl}|G_2>\cdot (-i (G_1-G_2))

    where R is the atomic position,
    G_1 and G_2 are the reciprocal lattice vectors (plane waves),
    and V_{nl} is the non-local pseudopotential operator.

    Refferences:
    -----------
       -Equations 16 from the dopyqo paper
         Schultheis E et al. Many-body post-processing of density functional calculations using the variational quantum eigensolver for Bader charge analysis.
         arXiv [quant-ph]. Published online 14 October 2025. doi:10.48550/arXiv.2510.12887
    """

    atom_position_origin = np.array([[0.0, 0.0, 0.0]])
    nl_pp_part = cpp.v_nl_pw(
        p=p,
        cell_volume=cell_volume,
        atom_positions=atom_position_origin,
        pseudopot=pseudopot,
    )  # Calcualtion done for only for one atom, therfore using origin (0,0,0)

    p_prime = p
    p_minus_p_prime = p[:, None] - p_prime[None]  # shape (#waves, #waves, 3)
    g_minus_g_prime_dot_R = np.sum(
        p_minus_p_prime[None] * atom_positions[:, None, None], axis=3
    )  # sum over 3D-coordinates, shape (#atoms, #waves, #waves)

    exp_term = np.exp(-1j * g_minus_g_prime_dot_R)

    gradient_term = -1j * p_minus_p_prime
    gradient_term = gradient_term.transpose(2, 0, 1)

    gradient = nl_pp_part[np.newaxis, np.newaxis, :, :] * exp_term[:, np.newaxis, :, :] * gradient_term[np.newaxis, :, :, :]

    result = c_ip.conj() @ gradient @ c_ip.T
    return result


def calc_gradient_pps_wrt_atom_positions(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    pseudopots: list[Pseudopot],
    save_filename: str | None = None,
    rust_impl: bool = False,
    n_threads: int = 1,
) -> np.ndarray:
    """
    Calculate the total gradient of the pseudopotential (local + non-local) with respect to atomic positions for all atoms N.

    This function computes the sum of the gradients of the local and non-local pseudopotential
    terms for each atom in the system, using the provided plane-wave basis, atomic positions,
    atomic numbers, and a list of pseudopotential objects. The function checks for consistency
    between the atomic positions and pseudopotential assignments, and can optionally save the
    computed gradients to disk.

    Parameters
    ----------
    p : np.ndarray
        Array of reciprocal lattice vectors (plane waves), shape (#waves, 3).
    c_ip : np.ndarray
        Coefficient matrix for the plane-wave basis, shape (n_bands, #waves).
    cell_volume : float
        Volume of the simulation cell in Bohr^3.
    atom_positions : np.ndarray
        Array of atomic positions for all atoms, shape (n_atoms, 3).
        With n_atoms being the number of atoms and each row being defined
        by the (x,y,z) coordinates in Bohr.
    atomic_numbers : np.ndarray
        Array of atomic numbers Z for all atoms, shape (n_atoms,).
    pseudopots : list[Pseudopot]
        List of Pseudopot objects, one for each atomic species present.
    save_filename : str or None, optional
        If provided, the computed gradient for each atom will be saved to this file.
    rust_impl : bool, optional
        If True and the Rust extension `dopyqo_rs` is importable, use `dopyqo_rs.v_loc_forces`/
        `dopyqo_rs.v_nl_forces` to compute the gradients. Falls back to Python if import fails.
    n_threads : int, optional
        Number of threads to use for parallel computation (if Rust implementation `dopyqo_rs` is used).

    Returns
    -------
    np.ndarray
        Array of gradients of the pseudopotential with respect to atomic positions, shape (n_atoms, 3, #waves, #waves),
        where n_atoms is the number of atoms, 3 is the Cartesian directions, and #waves is the number of plane waves.

    Mathematical Expression:
    ------------------------
    The total gradient of the pseudopotential with respect to atomic positions are based on equations 11,12 and 16 from the dopyqo paper.
    For the detailed mathematical derivation, please refer to the documentation of functions `gradient_v_loc_pw` and `gradient_v_nl_pw`.
    Both terms of teh pseudopotential gradeient are calcualted seperatly and then summed up:
        g_pp = g_loc + g_nl


    Note:
    -----
    - The function assumes that the pseudopotentials provided correspond to the atomic numbers present in the system.
    - The function checks for duplicate pseudopotentials and ensures that each atomic species has a corresponding pseudopotential.

    Raises:
    -------
    AssertionError
        If there are inconsistencies in the input data, such as mismatched dimensions
        or missing pseudopotentials for certain atomic species.

    Refferences:
    -----------
       -Equations 11,12 and 16 from the dopyqo paper
         Schultheis E et al. Many-body post-processing of density functional calculations using the variational quantum eigensolver for Bader charge analysis.
         arXiv [quant-ph]. Published online 14 October 2025. doi:10.48550/arXiv.2510.12887

    """
    n_threads = int(n_threads)
    assert n_threads > 0, f"Number of threads needs to be positive but is {n_threads}!"

    if rust_impl:
        try:
            import dopyqo_rs as calc_rs
        except ImportError:
            print(f"{ORANGE}Import warning: Could not import dopyqo_rs package. Falling back to python implementation.{RESET_COLOR}")
            gradient_v_loc_pw_func = gradient_v_loc_pw
            gradient_v_nl_pw_func = gradient_v_nl_pw
            rust_impl = False  # Set such that variable accurately represents if rust implementation is used
        else:  # No exception
            gradient_v_loc_pw_func = None
            gradient_v_nl_pw_func = None
            gradient_v_loc_func = partial(calc_rs.v_loc_forces, n_threads=n_threads)
            gradient_v_nl_func = partial(calc_rs.v_nl_forces, n_threads=n_threads)
            logging.info(f"Using Rust implementation with {n_threads} threads.")
            if save_filename is not None:
                logging.info(f"\tPseudopotential will not be saved to {save_filename}!")
    else:
        gradient_v_loc_pw_func = gradient_v_loc_pw
        gradient_v_nl_pw_func = gradient_v_nl_pw

    # --------------------------------- CHECKING FOR INVALID INPUTS ---------------------------------

    total_gradients_list = []

    assert c_ip.shape[1] == p.shape[0], f"c_ip and p arrays have different number of plane waves ({c_ip.shape[1]} vs. {p.shape[0]})!"
    assert atom_positions.shape[0] == len(atomic_numbers), (
        "Atomic numbers array contains different number of atoms than atom positions array "
        + f"({len(atomic_numbers)} vs. {atom_positions.shape[0]})!"
    )

    # Mapping atomic numbers to list of atom positions
    atoms_dict = {num: [] for num in atomic_numbers}
    for i, atomic_num in enumerate(atomic_numbers):
        atoms_dict[atomic_num].append(atom_positions[i])
    atoms_dict = {key: np.array(val) for key, val in atoms_dict.items()}
    pp_dict = {pp.atomic_number: pp for pp in pseudopots}

    # Checking for duplicate PPs
    atomic_nums_pp_duplicates = [
        atomic_num_pp for atomic_num_pp, count_val in Counter([pp.atomic_number for pp in pseudopots]).items() if count_val > 1
    ]
    atomic_el_pp_duplicates = [elements_to_atomic_number[x] for x in atomic_nums_pp_duplicates]
    assert len(atomic_nums_pp_duplicates) == 0, "More than one pseudopotential given for atoms " + ", ".join(
        str(x) + f" (Z={atomic_nums_pp_duplicates[i]})" for i, x in enumerate(atomic_el_pp_duplicates)
    )

    # Checking for atoms without PP
    atomic_num_wo_pp = [atomic_num for atomic_num in atoms_dict.keys() if atomic_num not in pp_dict.keys()]
    atomic_el_wo_pp = [elements_to_atomic_number[x] for x in atomic_num_wo_pp]
    assert len(atomic_num_wo_pp) == 0, "No pseudopotentials given for atoms " + ", ".join(
        str(x) + f" (Z={atomic_num_wo_pp[i]})" for i, x in enumerate(atomic_el_wo_pp)
    )

    # Delete all PPs with no corresponding atoms
    pp_dict = {key: val for key, val in pp_dict.items() if key in atoms_dict.keys()}

    ############################ CALCULATING PPs ############################

    # Loop over all atoms in the order of atom_positions and get the corresponding pseudopotential
    for i, atom_pos in enumerate(atom_positions):
        gradient_v_loc_mat = None
        gradient_v_nl_mat = None
        atomic_num = atomic_numbers[i]
        pp = pp_dict[atomic_num]
        logging.info(f"Atom {i}: position={atom_pos}, atomic_number={atomic_num}, pseudopotential={pp}")

        logging.info(
            "Calculating PP for element %s (Z=%i)...",
            elements_to_atomic_number[atomic_num],
            atomic_num,
        )

        pos = atom_pos.reshape(1, 3)  # Shape (1,3) as only one atom

        logging.info("Calculating local PP...")
        if rust_impl:
            res_loc = gradient_v_loc_func(p, c_ip, cell_volume, pos, pp)
        else:
            res_loc = gradient_v_loc_pw_func(p, c_ip, cell_volume, pos, pp)

        if gradient_v_loc_mat is None:
            gradient_v_loc_mat = res_loc.copy()
        else:
            gradient_v_loc_mat += res_loc

        logging.info("Calculating non-local PP...")
        if rust_impl:
            res_nl = gradient_v_nl_func(p, c_ip, cell_volume, pos, pp)
        else:
            res_nl = gradient_v_nl_pw_func(p, c_ip, cell_volume, pos, pp)

        if gradient_v_nl_mat is None:
            gradient_v_nl_mat = res_nl.copy()
        else:
            gradient_v_nl_mat += res_nl

        gradient_v_pp = gradient_v_loc_mat + gradient_v_nl_mat

        if save_filename is not None and not rust_impl:
            save_folder = os.path.join(*os.path.split(save_filename)[:-1])
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_filename), gradient_v_pp)

        new_gradient_v_pp = gradient_v_pp.copy()

        total_gradients_list.append(new_gradient_v_pp[0])

    return np.array(total_gradients_list)


def gradient_frozen_core_energy_wrt_atom_positions(
    p: np.ndarray,
    c_ip_core: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    pseudopots: list[Pseudopot],
    rust_impl: bool = False,
    n_threads: int = 1,
) -> np.ndarray:
    r"""
    Calculate the gradient of the frozen-core energy with respect to atomic positions for all atoms N.

    The kinetic-energy and core-core two-electron (Hartree/exchange) contributions to the frozen-core
    energy do not depend on atom positions (they only depend on the fixed core-orbital wavefunctions),
    so the entire derivative reduces to the pseudopotential part evaluated on the core orbitals:
    E_frozen_core = 2 * Tr(h_pq_pp_core) + (atom-position-independent terms)
    => d(E_frozen_core)/d(atom position) = 2 * Tr(d(h_pq_pp_core)/d(atom position))

    This matches equation 25 of the dopyqo paper and the reference implementation in
    code/GeometryOptimization_ErikHansen/analytical_gradient/gradients.py.

    Parameters
    ----------
    p : np.ndarray
        Array of reciprocal lattice vectors (plane waves), shape (#waves, 3).
    c_ip_core : np.ndarray
        Coefficient matrix for the core (frozen) orbitals, shape (n_core_orbitals, #waves).
    cell_volume : float
        Volume of the simulation cell in Bohr^3.
    atom_positions : np.ndarray
        Array of atomic positions for all atoms, shape (n_atoms, 3), in Bohr.
    atomic_numbers : np.ndarray
        Array of atomic numbers Z for all atoms, shape (n_atoms,).
    pseudopots : list[Pseudopot]
        List of Pseudopot objects, one for each atomic species present.
    rust_impl : bool, optional
        If True and the Rust extension `dopyqo_rs` is importable, use `dopyqo_rs.v_loc_forces`/
        `dopyqo_rs.v_nl_forces` to compute the gradients. Falls back to Python if import fails.
    n_threads : int, optional
        Number of threads to use for parallel computation (if Rust implementation `dopyqo_rs` is used).

    Returns
    -------
    np.ndarray
        Derivative of the frozen-core energy with respect to each Cartesian atom position, shape (n_atoms, 3).

    References
    ----------
       - Equation 25 from the dopyqo paper
         Schultheis E et al. Many-body post-processing of density functional calculations using the variational quantum eigensolver for Bader charge analysis.
         arXiv [quant-ph]. Published online 14 October 2025. doi:10.48550/arXiv.2510.12887
    """
    d_h_pq_pp_core = calc_gradient_pps_wrt_atom_positions(
        p,
        c_ip_core,
        cell_volume,
        atom_positions,
        atomic_numbers,
        pseudopots,
        rust_impl=rust_impl,
        n_threads=n_threads,
    )
    return 2 * np.trace(d_h_pq_pp_core, axis1=-2, axis2=-1).real
