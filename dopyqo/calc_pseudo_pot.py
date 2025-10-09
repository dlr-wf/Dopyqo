import os
import logging
from functools import partial
import multiprocessing
from warnings import warn
from collections import Counter
import numpy as np
from scipy.special import (
    erf,
    spherical_jn,
    sph_harm,
)
from scipy.integrate import simpson
import dopyqo
from dopyqo.pseudopot import Pseudopot
from dopyqo.colors import *


def sph_harm_real(m, n, theta, phi):
    r"""Compute real spherical harmonics.
    The real spherical harmonics are defined as
    $$
    Y_{nm}(\theta, \phi) =
    \begin{cases}
    \sqrt2 (-1)^m \Im{Y_n^{|m|}}&\text{if } m<0\\
    Y_n^0 &\text{if } m=0\\
    \sqrt2 (-1)^m \Re{Y_n^m}&\text{if } m>0
    \end{cases}
    $$
    where $Y_n^m(\theta, \phi)$ are the complex spherical harmonics; see [scipy.special.sph_harm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html).

    Args:
    m: array_like
        Order of the harmonic (int); must have |m| <= n.
    n: array_like
        Degree of the harmonic (int); must have n >= 0. This is often denoted by l (lower case L) in descriptions of spherical harmonics.
    theta: array_like
        Azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
    phi: array_like
        Polar (colatitudinal) coordinate; must be in [0, pi].
    out: ndarray, optional
        Optional output array for the function values

    Returns:
        y_mn: complex scalar or ndarray
            The harmonic sampled at theta and phi.
    """
    if m < 0:
        return np.sqrt(2) * (-1) ** m * sph_harm(m, n, theta, phi).imag
    elif m > 0:
        return np.sqrt(2) * (-1) ** m * sph_harm(m, n, theta, phi).real
    return sph_harm(m, n, theta, phi)


def cart_to_sph(xyz: np.ndarray) -> np.ndarray:
    r"""Transforms cartesian x,y,z coordinates to spherical r, \theta (polar angle), \phi (azimuthal angle) coordinates

    Args:
        xyz (np.ndarray): Array of cartesian coordinates. Shape (N, 3)

    Returns:
        np.ndarray: Array of spherical coordinates. Shape (N, 3)
    """
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2  # x^2+y^2

    # r = sqrt{x^2+y^2+z^2}
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)

    # \theta = arctan(\sqrt{x^2+y^2}/z) taking into account the correct quadrant
    # for elevation angle defined from Z-axis down
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])

    # \phi = arctan(y/x) taking into account the correct quadrant
    # \phi in range [-\pi, \pi]
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    # put \phi into range [0, 2\pi], by adding 2\pi to negative angles
    ptsnew[:, 2] = np.where(ptsnew[:, 2] < 0.0, ptsnew[:, 2] + 2 * np.pi, ptsnew[:, 2])
    return ptsnew


def v_loc_pw(
    p: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    pseudopot: Pseudopot,
) -> np.ndarray:
    r"""Calculates the matrix of the local pseudopotential V_loc(p, p') in the plane wave basis in Hartree atomic units

    V_loc(p, p') = 4\pi/V \int_0^\infty dr r (V_loc(r)+Z erf(r)/r) sin((p-p')r)/(p-p') - 4\pi/V Z e^{-(p-p')^2/4}/(p-p')^2
    which is the Fourier transform of V_loc(r) + Z erf(r)/r - Z erf(r)/r

    A term -Z erf(r)/r is subtracted (-[-Z erf(r)/r]=+Z erf(r)/r) in real space (thus making the function short-ranged)
    and added again in G space
    V_loc^short(p, p') = 4\pi/V \int_0^\infty dr r (V_loc(r)+Z erf(r)/r) sin((p-p')r)/(p-p')
    which is continious for p-p'=0 but the p-p'=0 term is calculated differently (see below)

    The Fourier transform of Z erf(r)/r is
    4\pi/V Z e^{-(p-p')^2/4}/(p-p')^2

    The p-p'=0 limit for V_loc is the p-p'=0 limit of
    4\pi/V \int_0^\infty dr r^2 (V_loc(r)+Z/r)
    since the p-p'=0 limit for the -Z/r part of V_loc is cancelled by the electronic background
    Because V_loc does not behave exactly like -Z/r, we still have to calculate the p-p'=0 limit
    for V_loc+Z/r

    As a reference see the implementation in Quantum ESPRESSO in its soruce q-e/upflib/vloc_mod.f90

    Args:
        p (np.ndarray): reciprocal space vectors. Shape (#waves, 3)
        cell_volume (float): Volume V of the real-space unit cell
        atom_positions (np.ndarray): Array of the coordinates R_I of every atom described
                                     by the pseudopotential. Shape (#atoms, 3)
        pseudopot (Pseudopot): Pseudopotential object containing z_valence, V_loc(r), and the radial grid

    Returns:
        np.ndarray:  V_loc(p, p')
                     where p, p' take all values from input array p
    """
    # /0.5 to convert from Hartree to Rydberg since the PP is given in Rydberg atomic units
    z_valence = pseudopot.z_valence / 0.5
    v_loc_r = pseudopot.pp_local
    r_grid = pseudopot.r_grid

    p_prime = p
    p_minus_p_prime = p[:, None] - p_prime[None]  # shape (#waves, #waves, 3)
    # p_prime[None] is of shape (1, #waves, 3) while p[:, None] is of shape (#waves, 1, 3)
    # so p_minus_p_prime[i,j] = p[i] - p_prime[j]

    p_minus_p_prime_dot_R = np.sum(
        p_minus_p_prime[None] * atom_positions[:, None, None], axis=3
    )  # sum over 3D-coordinates, shape (#atoms, #waves, #waves)
    # \sum_I 4\pi/V e^{-i (p-p').R_I}
    prefactor = 4 * np.pi / cell_volume * np.sum(np.exp(-1j * p_minus_p_prime_dot_R), axis=0)  # sum over atoms, shape (#waves, #waves)

    # Rydberg to Hartree since the PP is given in Rydberg atomic units
    prefactor = prefactor / 2

    # Fourier transform of V_loc(r) + Z erf(r)/r (short-range part of V_loc) is
    # 4\pi/V \int dr r (V_loc(r)+Z erf(r)/r) sin((p-p')r)/(p-p')
    p_norms = np.linalg.norm(p_minus_p_prime, ord=2, axis=-1)
    p_norms_unique, unique_inv = np.unique(p_norms, return_inverse=True)
    # p_norms obviously has a lot of repeating values, only calculate V_loc for unique p_norms
    # Now p_norms.flatten() is equal to p_norms_unique[unique_inv])
    # consequently p_norms is equal to p_norms_unique[unique_inv].reshape(p_norms.shape)

    integrands = np.zeros(p_norms_unique.shape + (len(r_grid.r),), dtype=np.float64)
    mask = p_norms_unique > 1e-8
    # removal of erf(r)/r term
    integrands[mask, :] = (
        (r_grid.r * v_loc_r + z_valence * erf(r_grid.r))
        * np.sin(p_norms_unique[mask][:, np.newaxis] * r_grid.r)
        / p_norms_unique[mask][:, np.newaxis]
    )
    # p_norm = 0 limit, since it is continuous
    # integrands[~mask, :] = r_grid.r * (r_grid.r * v_loc_r + z_valence * erf(r_grid.r))
    f_v_plus_erf = simpson(y=integrands, x=r_grid.r)

    # Fourier transform of Z erf(r)/r is 4\pi/V Z e^{-(p-p')^2/4}/(p-p')^2 which is added in G-space
    f_erf = np.zeros_like(p_norms_unique, dtype=np.complex128)  # set G=0 to zero
    f_erf[mask] = z_valence * np.exp(-p_norms_unique[mask] ** 2 / 4) / p_norms_unique[mask] ** 2

    f_v = f_v_plus_erf - f_erf

    # Fourier transform of V_loc(r)+Z/r for G=0
    p_zero_lim = simpson(y=r_grid.r * (r_grid.r * v_loc_r + z_valence), x=r_grid.r)
    p_zero_lim_mat = np.zeros_like(p_norms_unique, dtype=np.complex128)  # set G!=0 to zero
    p_zero_lim_mat[~mask] = p_zero_lim
    # print(f"p_zero_lim: {p_zero_lim}")

    v_loc_mat = f_v + p_zero_lim_mat
    # for i, val in enumerate(v_loc_mat):
    #     print(
    #         f"i: {i+1}\ngl: {p_norms_unique[i]**2}, vloc: {4*np.pi/cell_volume*val}\n"
    #     )

    return prefactor * (v_loc_mat[unique_inv].reshape(p_norms.shape))


def v_loc(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    pseudopot: Pseudopot,
    save_filename_pw: str | None = None,
) -> np.ndarray:
    r"""Calculates the matrix of the local pseudopotential V_loc(i, i) in the Kohn-Sham basis
    V_loc(i, j) = \sum_{p, p'} c_{ip} V_loc(p, p') c_{jp'}

    See v_loc_pw for the calculation of V_loc(p, p')

    Args:
        p (np.ndarray): reciprocal space vectors. Shape (#waves, 3)
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis
        cell_volume (float): Volume V of the real-space unit cell
        atom_positions (np.ndarray): Array of the coordinates R_I of every atom described
                                     by the pseudopotential. Shape (#atoms, 3)
        pseudopot (Pseudopot): Pseudopotential object containing z_valence, V_loc(r), and the radial grid

    Returns:
        np.ndarray:  V_loc(i, j') = \sum_{p,p'} c_{p,i}^* c_{p',j} V_loc(p, p')
                     where p, p' take all values from input array p
                     where i, i' represent Kohn-Sham band indices
    """
    v_loc_pw_mat = v_loc_pw(p, cell_volume, atom_positions, pseudopot)
    if save_filename_pw is not None:
        np.save(save_filename_pw, v_loc_pw_mat)
    return c_ip.conj() @ v_loc_pw_mat @ c_ip.T


def v_nl_pw(
    p: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    pseudopot: Pseudopot,
) -> np.ndarray:
    r"""Calculates the matrix of the non-local pseudopotential in the plane wave basis in Hartree atomic units
    V_nl(p, p') = (4\pi)^2/V \sum_I e^(-i (p-p') . R_I) \sum_{lm} \sum_{ij} Y_{lm}(p) Y*_{lm}(p') F^i_l(p) D_{ij} F^j_l(p')
                = (4\pi)^2/V \sum_I e^(-i (p-p') . R_I) \sum_{ij} D_{ij} \sum_l [\sum_m Y_{lm}(p) Y*_{lm}(p')] F^i_l(p) F^j_l(p')
    where
    F^i_l(p) = \int dr r^2 \beta^i_l(r) j_l(pr)
    where j_l(x) is the spherical Bessel function

    As a reference see https://docs.abinit.org/theory/pseudopotentials/

    Args:
        p (np.ndarray): reciprocal space vectors. Shape (#waves, 3)
        cell_volume (float): Volume V of the real-space unit cell
        atom_positions (np.ndarray): Array of the coordinates R_I of every atom described
                                     by the pseudopotential. Shape (#atoms, 3)
        pseudopot (Pseudopot): Pseudopotential object containing D_ij, \beta^i_l(r), and the radial grid

    Returns:
        np.ndarray: V_nl(p, p')
                    where p, p' take all values from input array p and
    """
    d_ij = pseudopot.pp_dij
    beta_projectors = pseudopot.pp_betas
    r_grid = pseudopot.r_grid

    assert len(beta_projectors) == len(np.unique([x.idx for x in beta_projectors])), "Beta projectors are invalid! Their indices are not unique."
    beta_projectors = sorted(beta_projectors, key=lambda x: x.idx)

    # Check that D_ij does not connect beta projectors of different angular momentum
    assert (
        d_ij.shape == (len(beta_projectors),) * 2
    ), f"Number of beta projectors ({len(beta_projectors)}) does not match shape of D_ij ({d_ij.shape})."
    for i, bp_i in enumerate(beta_projectors):
        for j, bp_j in enumerate(beta_projectors):
            if bp_i.angular_momentum != bp_j.angular_momentum:
                assert np.isclose(d_ij[i, j], 0.0), (
                    f"D_ij at i={i}, j={j} (D_{i},{j}={d_ij[i,j]}) connects beta projectors "
                    + f"of different angular momentum, l_i={bp_i.angular_momentum} and l_j={bp_j.angular_momentum}, "
                    + "which should not be the case."
                )

    p_prime = p

    p_minus_p_prime = p[:, None] - p_prime[None]  # shape (#waves, #waves, 3)

    p_minus_p_prime_dot_R = np.sum(
        p_minus_p_prime[None] * atom_positions[:, None, None], axis=3
    )  # sum over 3D-coordinates, shape (#atoms, #waves, #waves)
    # (4\pi)^2/V \sum_I e^(-i (p-p') . R_I)
    prefactor = (4 * np.pi) ** 2 / cell_volume * np.sum(np.exp(-1j * p_minus_p_prime_dot_R), axis=0)  # sum over atoms, shape (#waves, #waves)

    # Rydberg to Hartree since the PP is given in Rydberg atomic units
    prefactor = prefactor / 2

    # Set negative y-axis to axis from which to measure polar angle, instead of z-axis
    # x, y, z -> x, z, -y
    # To make coordinate system right-handed again, set x -> -x
    # Use coordinates -x, z, -y
    # This is the convention used in Quantum ESPRESSO and reproduces
    # the non-local pp matrix elements calculated in Quantum ESPRESSO for Be
    # see QE source code in upflib/init_us_2_acc.f90 line 112 and following,
    # especially the variable vkb_:
    #   vkb = (4pi/sqrt(omega)).Y_lm(q).f_l(q).(i^l).S(q)
    #   vq = f_l(q) = 4\pi/sqrt(omega) int r beta(r) j_l(qr)
    #   ylm are real spherical harmonics in the above described coordinate system.
    #       Order of m values in ylm for l=1: -1, +1, 0
    #   pref = (-i)^l
    #   sk = e^(-i * p.R) where R is the position of one atom
    # NOTE: This coordinate transformation still results in a right-handed
    #       coordinate system and does not change the energies of the Hamiltonian
    p_cart = p.copy()
    p_cart = p_cart[:, [0, 2, 1]]  # xyz to xzy
    p_cart[:, 0] *= -1  # to -x, z, y
    p_cart[:, 2] *= -1  # to -x, z, -y

    p_sph = cart_to_sph(p_cart)
    p_r = p_sph[:, 0]  # Radii p vectors
    p_theta = p_sph[:, 1]  # Theta (polar) angles of p vectors
    p_phi = p_sph[:, 2]  # Phi (azimuthal) angles of p vectors

    # f_il_p = f_i(p) = \int dr r^2 \beta^i_l(r) j_l(pr)
    # Note that l is uniquely defined by i, so l=l_i and
    # f_il_p only depends on i and p
    f_il_p = np.zeros((len(beta_projectors), len(p)))
    for i, bp in enumerate(beta_projectors):
        # \int dr r^2 \beta^i_l(r) j_l(Gr), where bp.values is r \beta^i_l(r)
        integrands = [r_grid.r * bp.values * spherical_jn(n=bp.angular_momentum, z=p_r_val * r_grid.r) for p_r_val in p_r]

        f_il_p[i] = simpson(y=integrands, x=r_grid.r)

    # Precompute spherical harmonics Y_{lm}(p) and Y*_{lm}(p')
    y_lm_p = {}
    for l_val in np.unique([bp.angular_momentum for bp in beta_projectors]):
        # There are different conventions for the meanings of the input arguments theta and phi.
        # In SciPy theta is the azimuthal angle and phi is the polar angle.
        # It is common to see the opposite convention, that is, theta as the polar angle
        # and phi as the azimuthal angle.
        # logging.info(f"m takes values {list(range(-l, l + 1))} for l={l}")

        # Complex spherical harmonics
        # y_lm_g[l_val] = np.array(
        #     [sph_harm(m, l_val, p_phi, p_theta) for m in range(-l_val, l_val + 1)]
        # )
        # !!!: sph_harm_real is deprecated since scipy version 1.15.0:
        #      "This function is deprecated and will be removed in SciPy 1.17.0.
        #           Please use scipy.special.sph_harm_y instead."
        #      This code was written for scipy version 1.13.0.
        # !!!: scipy.special.sph_harm_y redefines theta and phi:
        #      theta is the polar angle and phi is the azimuthal angle
        #      For sph_harm_real:
        #      theta is the azimuthal angle and phi is the polar angle

        # NOTE: Real spherical harmonics can be used for Hamiltonians without spin-orbit coupling
        # since then the Hamiltonian and the energies do not depend on the l and m quantum numbers
        y_lm_p[l_val] = np.array([sph_harm_real(m, l_val, p_phi, p_theta) for m in range(-l_val, l_val + 1)])

    v_nl_mat = np.zeros((len(p), len(p_prime)), dtype=np.complex128)

    # m_sums(l)(p, p') = \sum_m Y_lm(p) Y*_lm(p')
    m_sums = {key: np.sum(val[:, :, None] * val[:, None, :].conj(), axis=0) for key, val in y_lm_p.items()}

    for i, bp_i in enumerate(beta_projectors):
        assert bp_i.idx - 1 == i
        l_i = bp_i.angular_momentum
        f_il_p_i = f_il_p[i, :, None]

        for j, bp_j in enumerate(beta_projectors):
            assert bp_j.idx - 1 == j
            l_j = bp_j.angular_momentum
            f_jl_p_j = f_il_p[j, None, :]

            if l_i != l_j:
                continue

            v_nl_mat += d_ij[i, j] * (f_il_p_i @ f_jl_p_j) * m_sums[l_i]

    return prefactor * v_nl_mat


def v_nl(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    pseudopot: Pseudopot,
    save_filename_pw: str | None = None,
) -> np.ndarray:
    r"""Calculates the matrix of the non-local pseudopotential in the Kohn-Sham basis in Hartree atomic units
    V_nl(i,j) = \sum_{p, p'} c_{ip} V_nl(p, p') c_{jp'}

    See v_nl_pw for the calculation of V_nl(p, p')

    Args:
        p (np.ndarray): reciprocal space vectors. Shape (#waves, 3)
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis
        cell_volume (float): Volume V of the real-space unit cell
        atom_positions (np.ndarray): Array of the coordinates R_I of every atom described
                                     by the pseudopotential. Shape (#atoms, 3)
        pseudopot (Pseudopot): Pseudopotential object containing D_ij, \beta^i_l(r), and the radial grid

    Returns:
        np.ndarray: V_nl(i, j') = \sum_{p,p'} c_{p,i}^* c_{p',j} V_nl(p, p')
                    where p, p' take all values from input array p and
                    where i, i' represent Kohn-Sham band indices
    """
    v_nl_pw_mat = v_nl_pw(p, cell_volume, atom_positions, pseudopot)
    if save_filename_pw is not None:
        np.save(save_filename_pw, v_nl_pw_mat)
    return c_ip.conj() @ v_nl_pw_mat @ c_ip.T


def calc_pps(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    pseudopots: list[Pseudopot],
    save_filename: str | None = None,
    rust_impl: bool = True,
    n_threads: int = 1,
) -> np.ndarray:
    r"""Calculate sum of all pseudopotentials in Hartree atomic units.
    Calls v_nl and v_loc after checking arguments.

    Args:
        p (np.ndarray): Array of momentum vectors. Shape (#waves, 3)
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis. Shape (#bands, #waves)
        atom_positions (np.ndarray): 2D array of positions of each atom. Shape (#atoms, 3)
        atomic_numbers (np.ndarray): 1D array of atomic number of each atom. Shape (#atoms,)
        cell_volume (float): Cell volume of the computationen real space cell.
        pseudopot (list[Pseudopot]): Pseudopotential objects for all atom types

    Returns:
        np.ndarray: Nuclear interaction matrix
    """
    n_threads = int(n_threads)
    assert n_threads > 0, f"Number of threads needs to be positive but is {n_threads}!"
    if rust_impl:
        try:
            import dopyqo_rs as calc_rs
        except ImportError:
            print(f"{ORANGE}Import warning: Could not import dopyqo_rs package. Falling back to python implementation.{RESET_COLOR}")
            v_loc_pw_func = v_loc_pw
            v_nl_pw_func = v_nl_pw
            rust_impl = False  # Set such that variable accurately represents if rust implementation is used
        else:  # No exception
            v_loc_pw_func = None
            v_nl_pw_func = None
            if n_threads > 1:
                v_loc_func = partial(calc_rs.v_loc_par, n_threads=n_threads)
                v_nl_func = partial(calc_rs.v_nl_par, n_threads=n_threads)
                logging.info(f"Using Rust implementation with {n_threads} threads.")
            else:
                v_loc_func = calc_rs.v_loc
                v_nl_func = calc_rs.v_nl
                logging.info("Using Rust implementation.")
            if save_filename is not None:
                logging.info(f"\tPseudopotential will not be saved to {save_filename}!")
    else:
        v_loc_pw_func = v_loc_pw
        v_nl_pw_func = v_nl_pw

    # --------------------------------- CHECKING FOR INVALID INPUTS ---------------------------------
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
    atomic_el_pp_duplicates = [dopyqo.elements_to_atomic_number[x] for x in atomic_nums_pp_duplicates]
    assert len(atomic_nums_pp_duplicates) == 0, "More than one pseudopotential given for atoms " + ", ".join(
        str(x) + f" (Z={atomic_nums_pp_duplicates[i]})" for i, x in enumerate(atomic_el_pp_duplicates)
    )

    # Checking for atoms without PP
    atomic_num_wo_pp = [atomic_num for atomic_num in atoms_dict.keys() if atomic_num not in pp_dict.keys()]
    atomic_el_wo_pp = [dopyqo.elements_to_atomic_number[x] for x in atomic_num_wo_pp]
    assert len(atomic_num_wo_pp) == 0, "No pseudopotentials given for atoms " + ", ".join(
        str(x) + f" (Z={atomic_num_wo_pp[i]})" for i, x in enumerate(atomic_el_wo_pp)
    )

    # Delete all PPs with no corresponding atoms
    pp_dict = {key: val for key, val in pp_dict.items() if key in atoms_dict.keys()}

    ############################ CALCULATING PPs ############################
    v_nl_mat = 0.0
    v_loc_mat = 0.0
    for atomic_num, pp in pp_dict.items():
        logging.info(
            "Calculating PP for element %s (Z=%i)...",
            dopyqo.elements_to_atomic_number[atomic_num],
            atomic_num,
        )
        pos = atoms_dict[atomic_num]
        # logging.info(f"Atomic positions: {pos}")
        logging.info("Calculating local PP...")
        if rust_impl:
            v_loc_mat += v_loc_func(p, c_ip, cell_volume, pos, pp)
        else:
            v_loc_mat += v_loc_pw_func(p, cell_volume, pos, pp)
        logging.info("Calculating non-local PP...")
        if rust_impl:
            v_nl_mat += v_nl_func(p, c_ip, cell_volume, pos, pp)
        else:
            v_nl_mat += v_nl_pw_func(p, cell_volume, pos, pp)

    v_pp = v_loc_mat + v_nl_mat

    if save_filename is not None and not rust_impl:
        save_folder = os.path.join(*os.path.split(save_filename)[:-1])
        os.makedirs(save_folder, exist_ok=True)
        np.save(os.path.join(save_filename), v_pp)

    if rust_impl:
        return v_pp

    return c_ip.conj() @ v_pp @ c_ip.T


def calc_pps_concurrent(
    p: np.ndarray,
    c_ip: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    pseudopots: list[Pseudopot],
    save_filename: str | None = None,
    rust_impl: bool = True,
) -> np.ndarray:
    r"""Calculate sum of all pseudopotentials in Hartree atomic units.
    Calls v_nl and v_loc after checking arguments.

    Args:
        p (np.ndarray): Array of momentum vectors
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis
        atom_positions (np.ndarray): 2D array of positions of each atom. Shape (#atoms, 3)
        atomic_numbers (np.ndarray): 1D array of atomic number of each atom. Shape (#atoms,)
        cell_volume (float): Cell volume of the computationen real space cell.
        pseudopot (list[Pseudopot]): Pseudopotential objects for all atom types

    Returns:
        np.ndarray: Nuclear interaction matrix
    """
    if rust_impl:
        try:
            import dopyqo_rs as calc_rs
        except ImportError:
            warn("Could not import dopyqo_rs package. Falling back to python implementation.")
            v_loc_pw_func = v_loc_pw
            v_nl_pw_func = v_nl_pw
            rust_impl = False  # Set such that variable accurately represents if rust implementation is used
        else:  # No exception
            v_loc_pw_func = None
            v_nl_pw_func = None
            v_loc_func = calc_rs.v_loc
            v_nl_func = calc_rs.v_nl
            logging.info("Using Rust implementation.")
            if save_filename is not None:
                logging.info(f"\tPseudopotential will not be saved to {save_filename}!")
    else:
        v_loc_pw_func = v_loc_pw
        v_nl_pw_func = v_nl_pw

    # --------------------------------- CHECKING FOR INVALID INPUTS ---------------------------------
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
    atomic_el_pp_duplicates = [dopyqo.elements_to_atomic_number[x] for x in atomic_nums_pp_duplicates]
    assert len(atomic_nums_pp_duplicates) == 0, "More than one pseudopotential given for atoms " + ", ".join(
        str(x) + f" (Z={atomic_nums_pp_duplicates[i]})" for i, x in enumerate(atomic_el_pp_duplicates)
    )

    # Checking for atoms without PP
    atomic_num_wo_pp = [atomic_num for atomic_num in atoms_dict.keys() if atomic_num not in pp_dict.keys()]
    atomic_el_wo_pp = [dopyqo.elements_to_atomic_number[x] for x in atomic_num_wo_pp]
    assert len(atomic_num_wo_pp) == 0, "No pseudopotentials given for atoms " + ", ".join(
        str(x) + f" (Z={atomic_num_wo_pp[i]})" for i, x in enumerate(atomic_el_wo_pp)
    )

    # Delete all PPs with no corresponding atoms
    pp_dict = {key: val for key, val in pp_dict.items() if key in atoms_dict.keys()}

    ############################ CALCULATING PPs ############################
    # Helper function to calculate local and non-local potentials (with multiprocessing)
    def calculate_loc(atomic_num, pp, atoms_dict, p, c_ip, cell_volume, rust_impl, queue):
        logging.info(
            "Calculating PP for element %s (Z=%i)...",
            dopyqo.elements_to_atomic_number[atomic_num],
            atomic_num,
        )
        pos = atoms_dict[atomic_num]
        logging.info(f"Atomic positions: {pos}")

        # Calculate local PP
        logging.info("Calculating local PP...")
        if rust_impl:
            v_loc_mat = v_loc_func(p, c_ip, cell_volume, pos, pp)
        else:
            v_loc_mat = v_loc_pw_func(p, cell_volume, pos, pp)

        queue.put(v_loc_mat)

    def calculate_nl(atomic_num, pp, atoms_dict, p, c_ip, cell_volume, rust_impl, queue):
        logging.info(
            "Calculating PP for element %s (Z=%i)...",
            dopyqo.elements_to_atomic_number[atomic_num],
            atomic_num,
        )
        pos = atoms_dict[atomic_num]
        logging.info(f"Atomic positions: {pos}")

        # Calculate non-local PP
        logging.info("Calculating non-local PP...")
        if rust_impl:
            v_nl_mat = v_nl_func(p, c_ip, cell_volume, pos, pp)
        else:
            v_nl_mat = v_nl_pw_func(p, cell_volume, pos, pp)

        queue.put(v_nl_mat)

    def calculate_pp(atomic_num, pp, atoms_dict, p, c_ip, cell_volume, rust_impl, queue):
        logging.info(
            "Calculating PP for element %s (Z=%i)...",
            dopyqo.elements_to_atomic_number[atomic_num],
            atomic_num,
        )
        pos = atoms_dict[atomic_num]
        logging.info(f"Atomic positions: {pos}")

        v_loc_mat = 0  # Initialize local matrix for this process
        v_nl_mat = 0  # Initialize non-local matrix for this process

        # Calculate local PP
        logging.info("Calculating local PP...")
        if rust_impl:
            v_loc_mat += v_loc_func(p, c_ip, cell_volume, pos, pp)
        else:
            v_loc_mat += v_loc_pw_func(p, cell_volume, pos, pp)

        # Calculate non-local PP
        logging.info("Calculating non-local PP...")
        if rust_impl:
            v_nl_mat += v_nl_func(p, c_ip, cell_volume, pos, pp)
        else:
            v_nl_mat += v_nl_pw_func(p, cell_volume, pos, pp)

        # return v_loc_mat, v_nl_mat
        queue.put(v_loc_mat + v_nl_mat)

    # data_queue = [multiprocessing.Queue() for _ in range(2 * len(pp_dict.keys()))]
    data_queue = [multiprocessing.Queue() for _ in range(len(pp_dict.keys()))]
    process_list = []

    for i, (atomic_num, pp) in enumerate(pp_dict.items()):
        # process_loc = multiprocessing.Process(
        #     target=calculate_loc, args=(atomic_num, pp, atoms_dict, p, c_ip, cell_volume, rust_impl, data_queue[2 * i])
        # )
        # process_loc.start()
        # process_list.append(process_loc)
        # process_nl = multiprocessing.Process(
        #     target=calculate_nl, args=(atomic_num, pp, atoms_dict, p, c_ip, cell_volume, rust_impl, data_queue[2 * i + 1])
        # )
        # process_nl.start()
        # process_list.append(process_nl)

        process = multiprocessing.Process(target=calculate_pp, args=(atomic_num, pp, atoms_dict, p, c_ip, cell_volume, rust_impl, data_queue[i]))
        process.start()
        process_list.append(process)

    for proc in process_list:
        proc.join()

    data = [q_data for q in data_queue if (q_data := q.get())[1] is not None]
    v_pp = sum(data)

    if save_filename is not None and not rust_impl:
        save_folder = os.path.join(*os.path.split(save_filename)[:-1])
        os.makedirs(save_folder, exist_ok=True)
        np.save(os.path.join(save_filename), v_pp)

    if rust_impl:
        return v_pp

    return c_ip.conj() @ v_pp @ c_ip.T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from wfc import Wfc

    logging.basicConfig(level=logging.INFO)

    # We first investigate fully non-local norm-conserving pseudopotentials.
    # This is specified by pseudo_type = "NC" in the upf-file.

    # See equations (3.18) and onwards in "Planewaves, Pseudopotentials,
    # and the LAPW Method: Second Edition" from David J. Singh and Lars Nordstrom
    # for an explanation of the local and non-local part of the Kleinman-Bylander
    # transformed pseudopotential

    # Further see https://docs.abinit.org/theory/pseudopotentials/
    # and https://www.tcm.phy.cam.ac.uk/~jry20/gipaw/tutorial_pp.pdf

    base_folder = os.path.join("qe_files", "H2")
    dat_file = os.path.join(base_folder, "H2.save", "wfc1.dat")
    xml_file = os.path.join(base_folder, "H2.save", "data-file-schema.xml")

    base_folder = os.path.join("qe_files", "He")
    dat_file = os.path.join(base_folder, "He.save", "wfc1.dat")
    xml_file = os.path.join(base_folder, "He.save", "data-file-schema.xml")

    # base_folder = os.path.join("qe_files", "H2_15")
    # dat_file = os.path.join(base_folder, "H2.save", "wfc1.dat")
    # xml_file = os.path.join(base_folder, "H2.save", "data-file-schema.xml")

    # base_folder = os.path.join("qe_files", "H2_pos_-1.9")
    # dat_file = os.path.join(base_folder, "H2.save", "wfc1.dat")
    # xml_file = os.path.join(base_folder, "H2.save", "data-file-schema.xml")

    # base_folder = os.path.join("qe_files", "LiH")
    # dat_file = os.path.join(base_folder, "LiH.save", "wfc1.dat")
    # xml_file = os.path.join(base_folder, "LiH.save", "data-file-schema.xml")

    base_folder = os.path.join("qe_files", "Be")
    dat_file = os.path.join(base_folder, "Be.save", "wfc1.dat")
    xml_file = os.path.join(base_folder, "Be.save", "data-file-schema.xml")

    # base_folder = os.path.join("qe_files", "BeH")
    # dat_file = os.path.join(base_folder, "BeH.save", "wfc1.dat")
    # xml_file = os.path.join(base_folder, "BeH.save", "data-file-schema.xml")

    # base_folder = os.path.join("qe_files", "BeH_2")
    # dat_file = os.path.join(base_folder, "BeH.save", "wfc1.dat")
    # xml_file = os.path.join(base_folder, "BeH.save", "data-file-schema.xml")

    # base_folder = os.path.join("qe_files", "Mg")
    # dat_file = os.path.join(base_folder, "Mg.save", "wfc1.dat")
    # xml_file = os.path.join(base_folder, "Mg.save", "data-file-schema.xml")

    # Active space selection and options
    active_electrons = 2
    active_orbitals = 2
    binary_occupations = False

    print("Reading data from files...")
    wfc1_ncpp = Wfc.from_file(dat_file, xml_file)
    # wfc1_ncpp.atomic_numbers = wfc1_ncpp.atomic_numbers[:1]
    # wfc1_ncpp.atom_positions_hartree = wfc1_ncpp.atom_positions_hartree[:1]

    # wfc1_ncpp.atomic_numbers = wfc1_ncpp.atomic_numbers[1:]
    # wfc1_ncpp.atom_positions_hartree = wfc1_ncpp.atom_positions_hartree[1:]

    # wfc1_ncpp.atomic_numbers = np.array([1])
    # wfc1_ncpp.atom_positions_hartree = np.array([[0.5, 0.5, 0.5]])
    # wfc1_ncpp.cell_volume = 1

    wfc1_ncpp.k_plus_G = wfc1_ncpp.k_plus_G[:10]
    wfc1_ncpp.evc = np.eye(wfc1_ncpp.k_plus_G.shape[0])

    pp_base_folder = os.path.join(os.path.expanduser("~"), "Pseudopotentials")
    xml_file_H = os.path.join(pp_base_folder, "H_ONCV_PBE-1.2.upf")
    xml_file_He = os.path.join(pp_base_folder, "He_ONCV_PBE-1.2.upf")
    xml_file_Li = os.path.join(pp_base_folder, "Li_ONCV_PBE-1.2.upf")
    xml_file_Be = os.path.join(pp_base_folder, "Be_ONCV_PBE-1.2.upf")
    xml_file_Mg = os.path.join(pp_base_folder, "Mg_ONCV_PBE-1.2.upf")

    pp_H = Pseudopot(xml_file_H)
    pp_He = Pseudopot(xml_file_He)
    pp_Li = Pseudopot(xml_file_Li)
    pp_Be = Pseudopot(xml_file_Be)
    pp_Mg = Pseudopot(xml_file_Mg)

    pseudopots = [pp_H, pp_He, pp_Li, pp_Be, pp_Mg]

    pp = pp_He

    v_nl_r = np.zeros_like(pp.pp_mesh_r)
    for i in range(pp.pp_nonlocal["PP_DIJ"].shape[0]):
        for j in range(pp.pp_nonlocal["PP_DIJ"].shape[1]):
            v_nl_r += (
                pp.pp_nonlocal["PP_DIJ"][i, j]
                * pp.pp_nonlocal[f"PP_BETA.{i+1}"]
                / np.array(pp.pp_mesh_r)
                * pp.pp_nonlocal[f"PP_BETA.{j+1}"]
                / np.array(pp.pp_mesh_r)
            )

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6), constrained_layout=True)
    # local part, long-ranged and behaving like -Z/r for r \to \infty
    ax[0][0].plot(pp.pp_mesh_r, pp.pp_local, label="local")
    cut = 40
    # -Z/r; /0.5 to convert from Hartree to Rydberg
    ax[0][0].plot(pp.pp_mesh_r[cut:], -pp.z_valence / pp.pp_mesh_r[cut:] / 0.5, "--", label="-Z/r")
    ax[0][0].plot(
        pp.pp_mesh_r,
        pp.pp_local + pp.z_valence * erf(pp.pp_mesh_r) / pp.pp_mesh_r / 0.5,
        ":",
        label="local + erf(r)/r",
    )
    # ax[0][0].plot(pp_mesh_r, (pp_mesh_r*pp_local+z_valence*erf(pp_mesh_r)/0.5)*np.sin(pp_mesh_r),
    #                ":", label="(r*local + erf(r)) sin(r)")
    ax[0][0].plot(
        pp.pp_mesh_r[15:],
        pp.pp_local[15:] + pp.z_valence / pp.pp_mesh_r[15:] / 0.5,
        ":",
        label="local + 1/r",
    )
    # ax[0][0].plot(pp_mesh_r[15:], pp_local[15:]-z_valence/pp_mesh_r[15:]/0.5,
    #                ":", label="local - 1/r")
    ax[0][0].plot(
        pp.pp_mesh_r,
        pp.pp_mesh_r * (pp.pp_mesh_r * pp.pp_local + pp.z_valence / 0.5),
        ":",
        label="r * (r * local + Z)",
    )
    ax[0][0].plot(
        pp.pp_mesh_r,
        pp.pp_mesh_r * (pp.pp_mesh_r * pp.pp_local + pp.z_valence / 0.5 * erf(pp.pp_mesh_r)),
        ":",
        label="r * (r * local + Z erf)",
    )
    # ax[0][0].plot(pp_mesh_r, pp_local*pp_mesh_r**2,
    #                ":", label="r^2 * local")
    # ax[0][0].plot(pp_mesh_r[40:], pp_local[40:]+(-z_valence/pp_mesh_r[40:]/0.5), label="local")
    ax[0][0].set_title("Local (long-ranged)")
    ax[0][0].legend()
    ax[0][0].grid()
    for key, val in pp.pp_nonlocal.items():
        if "BETA" in key:
            # r_i \beta(r_i) -> \beta(r_i)
            val = np.array(val) / np.array(pp.pp_mesh_r)
            ax[0][1].plot(pp.pp_mesh_r, pp.pp_mesh_r**2 * val, label=f"r^2 {key.split('_')[-1]}")  # , label=key.split(".")[-1])
            ax[0][1].set_title("Nonlocal Beta (short-ranged)")
            ax[0][1].legend()
            ax[0][1].grid(True)
        elif "DIJ" in key:
            # val = np.array(val).reshape((int(np.sqrt(len(val))),) * 2, order="C")
            im = ax[1][0].imshow(val, cmap="coolwarm")
            ax[1][0].set_title("Nonlocal D_ij")
            fig.colorbar(im)
        else:
            ax[1][0].plot(val, label=key)
            ax[1][0].set_title("Nonlocal D_ij")
            ax[1][0].grid()
    ax[1][1].plot(pp.pp_mesh_r, v_nl_r + pp.pp_local, label="v_nl + v_loc")
    ax[1][1].plot(pp.pp_mesh_r, -pp.z_valence / pp.pp_mesh_r / 0.5, "--", label="-Z/r")
    ax[1][1].plot(pp.pp_mesh_r, v_nl_r, ":", label="v_nl")
    ax[1][1].legend()
    ax[1][1].set_title("Full Nonlocal + Local")
    ax[1][1].set_yscale("symlog")
    ax[1][1].grid()

    # ax[1][1].plot(pp_rhoatom, label="rho atom")
    # ax[1][1].set_title("Rho Atom")
    # ax[1][1].grid()

    # for axis in ax.flatten():
    #     axis.grid()
    #     axis.legend()

    fig.show()

    # v_nl_mat = v_nl(
    #     wfc1_ncpp.k_plus_G,
    #     wfc1_ncpp.evc,
    #     wfc1_ncpp.cell_volume,
    #     wfc1_ncpp.atom_positions_hartree,
    #     pp,
    # )

    # v_loc_mat = v_loc(
    #     wfc1_ncpp.k_plus_G,
    #     wfc1_ncpp.evc,
    #     wfc1_ncpp.cell_volume,
    #     wfc1_ncpp.atom_positions_hartree,
    #     pp,
    # )

    atom_positions = wfc1_ncpp.atom_positions_hartree
    # atom_positions = np.array(
    #     [[0.0, -1., -4.0],[0.0, 1., 1.0], [0.0, -1., -1], [0.0, 1., 4]]
    # )
    atomic_numbers = wfc1_ncpp.atomic_numbers
    # atomic_numbers = np.array([4, 4, 1, 1])

    pps = calc_pps(
        p=wfc1_ncpp.k_plus_G,
        c_ip=wfc1_ncpp.evc,
        cell_volume=wfc1_ncpp.cell_volume,
        atom_positions=atom_positions,
        atomic_numbers=atomic_numbers,
        pseudopots=pseudopots,
    )

    # assert np.allclose(pps, v_nl_mat + v_loc_mat)

    import calc_matrix_elements

    iUj = calc_matrix_elements.iUj(
        wfc1_ncpp.k_plus_G,
        wfc1_ncpp.evc,
        atom_positions,
        atomic_numbers,
        wfc1_ncpp.cell_volume,
    )

    vmin = np.min([np.min(pps.real), np.min(-iUj.real)])
    vmax = np.max([np.max(pps.real), np.max(-iUj.real)])

    fig, ax = plt.subplots(ncols=3, figsize=(9, 3), constrained_layout=True)
    im0 = ax[0].imshow((pps).real, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax[0].set_title("PP")
    im1 = ax[1].imshow(-iUj.real, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax[1].set_title("iUj")
    im2 = ax[2].imshow(np.abs(-iUj - pps).real, cmap="coolwarm")
    ax[2].set_title("Diff.")
    fig.colorbar(im0)
    fig.colorbar(im1)
    fig.colorbar(im2)

    # plt.figure()
    # plt.title("iUj")
    # plt.imshow(iUj.real, cmap="coolwarm")
    # plt.colorbar()

    # plt.figure()
    # plt.title("Diff")
    # plt.imshow(np.abs(pps - iUj).real, cmap="coolwarm")
    # plt.colorbar()

    print(atom_positions)
    print(atomic_numbers)
