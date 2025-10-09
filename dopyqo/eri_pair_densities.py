import sys
import time
import logging
from math import prod
from contextlib import nullcontext
import numpy as np
from numba import njit
from dopyqo import calc_matrix_elements
from dopyqo.pseudopot import Pseudopot
from dopyqo import calc_pseudo_pot
from dopyqo.colors import *

# Calculate ERIs via h_ijkl = 4\pi \sum_p (\rho*_il(p) \rho_jk(p))/p²
# with \rho_ij(p)=\int dr \rho_ij(r) e^(-ipr) which is the Fourier transform of
# \rho_ij(r) where \rho_ij(r)=\psi*_i(r)\psi_j(r). Therefore \rho_ij(p)
# is the convolution between \psi*_i(p) and \psi_j(p): \psi*_i(p) * \psi_j(p)
# where \psi_i(p) is the Fourier transform of \psi_i(r)

# The calculation steps are inspired by WEST: https://west-code.org/, https://github.com/west-code-development/West
# Especially the code in the compute_eri_vc function: https://github.com/west-code-development/West/blob/master/Wfreq/solve_eri.f90#L327
# Publication related to WEST:
# Large Scale GW Calculations, M. Govoni and G. Galli, J. Chem. Theory Comput. 11, 2680 (2015)
# GPU Acceleration of Large-Scale Full-Frequency GW Calculations, V. Yu and M. Govoni, J. Chem. Theory Comput. 18, 4690 (2022)


@njit
def make_one_over_p_norm_squared_array(
    one_over_p_norm_squared_array: np.ndarray,
    b: np.ndarray,
    fft_grid: np.ndarray,
):
    """Calculate |p|² for each grid point
    Its important to calculate 1/|p|² for the whole momentum grid not only for
    momentum present in the used Miller indices, because
    the density \rho_ij(p) may extend further than the Miller indices,
    since \rho_ij(p) is the result of a convolution between two p-space wavefunctions
    """
    nx, ny, nz = one_over_p_norm_squared_array.shape
    half_nx = fft_grid[0] // 2
    half_ny = fft_grid[1] // 2
    half_nz = fft_grid[2] // 2

    for i in range(nx):
        x = i - half_nx
        px = b[0] * x
        for j in range(ny):
            y = j - half_ny
            py = b[1] * y
            for k in range(nz):
                z = k - half_nz
                p_norm = np.linalg.norm(px + py + b[2] * z, ord=2)
                p_norm_2 = p_norm**2
                if p_norm_2 < 1e-8:
                    one_over_p_norm_squared_array[i, j, k] = 0.0
                    continue

                one_over_p_norm_squared_array[i, j, k] = 1.0 / p_norm_2

    return one_over_p_norm_squared_array


def pair_density(
    c_ip_array: np.ndarray,
    c_jp_array: np.ndarray | None = None,
    use_gpu: bool = True,
) -> np.ndarray:
    r"""Calculate pair density in reciprocal space via Fourier transforms
    Calculates \rho_ij(p)=\psi*_i(p) * \psi_j(p) (* is convolution) which is the
    Fourier transform of \psi*_i(r) * \psi_j(r) (* is standard multiplication)

    The argument c_ip_array is \psi_i(p) for a p-grid in reciprocal space.
    The inverse Fourier transform of \psi_i(p) (or c_ip_array) is \psi_i(r) in real space.
    To calculate \psi*_i(r) * \psi_j(r) (* is standard multiplication) we inverse Fourier
    transform both \psi_i(p) and \psi_j(p) resulting in \psi_i(r) and \psi_j(r)
    and then perform the standard multiplication \psi*_i(r) * \psi_j(r) (* is standard multiplication).
    We then perform a Fourier transform of the calculated real space pair density
    \psi*_i(r) * \psi_j(r) (* is standard multiplication) and return the result.

    Args:
        c_ip_array (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals \psi_i(p) in the plane wave basis,
                                 shape (#bands, #grid_size, #grid_size, #grid_size)
        c_jp_array (np.ndarray | None): Same as c_ip_array but the coefficients describing \psi_j(p)
                                        instead of \psi_i(p) in \rho_ij(p)=\psi*_i(p) * \psi_j(p) (* is convolution).
                                        If None, this is set to c_ip_array

    Returns:
        np.ndarray: Pair density in reciprocal space
    """
    using_cupy = False

    if use_gpu:
        try:
            import cupy as cp
        except ImportError:
            print(f"{ORANGE}Import warning: Could not import cupy package. Falling back to numpy.{RESET_COLOR}")
            xp = np
        else:  # No exception
            xp = cp
            using_cupy = True
    else:
        xp = np

    my_context = nullcontext()
    if using_cupy:
        my_context = cp.cuda.Device(cp.cuda.runtime.getDeviceCount() - 1)  # Use last available GPU
    with my_context:
        c_jp_array_not_given = False
        if c_jp_array is None:
            c_jp_array_not_given = True
            # c_jp_array = c_ip_array.copy()
        # print(f"{c_jp_array_not_given=}")

        dtype = xp.complex128

        if not c_jp_array_not_given:
            assert c_ip_array.ndim == c_jp_array.ndim == 4
            assert c_ip_array.shape[1:] == c_jp_array.shape[1:]

        nbands_i = c_ip_array.shape[0]
        nbands_j = nbands_i
        if not c_jp_array_not_given:
            nbands_j = c_jp_array.shape[0]
        ngrid = c_ip_array.shape[1:]
        if not c_jp_array_not_given:
            assert ngrid == c_jp_array.shape[1:]

        # rho_ij_p = xp.zeros((nbands_i, nbands_j, *ngrid), dtype)
        rho_ij_p = np.zeros((nbands_i, nbands_j, *ngrid), dtype)

        for i in range(nbands_i):
            logging.info("Calculating pair density %s/%s ...", i + 1, nbands_i)
            c_ip_shifted = xp.fft.ifftshift(c_ip_array[i])
            psi_r_i_conj = xp.fft.ifftn(c_ip_shifted).conj()
            #
            if c_jp_array_not_given:
                range_start = i
            else:
                range_start = 0
            #
            for j in range(range_start, nbands_j):
                if c_jp_array_not_given:
                    c_jp_shifted = xp.fft.ifftshift(c_ip_array[j])
                else:
                    c_jp_shifted = xp.fft.ifftshift(c_jp_array[j])
                psi_r_j = xp.fft.ifftn(c_jp_shifted)

                # psi*_i(r) . psi_j(r), where . is the standard multiplication
                # Same as psi_i(p) * psi_j(p), where * is the convolution operation
                rho_ij_r = psi_r_i_conj * psi_r_j

                rho_ij_p_val = xp.fft.fftshift(xp.fft.fftn(rho_ij_r))

                if using_cupy:
                    rho_ij_p_val = xp.asnumpy(rho_ij_p_val)
                rho_ij_p[i, j, :] = rho_ij_p_val
                if c_jp_array_not_given and i != j:
                    # \rho_ji(p)=\rho*_ij(-p)
                    rho_ij_p[j, i, :] = np.flip(rho_ij_p[i, j, :].conj(), (0, 1, 2))

    # if using_cupy:
    #     return xp.asnumpy(rho_ij_p)
    return rho_ij_p


def pair_density_real_space(
    c_ip_array: np.ndarray,
    c_jp_array: np.ndarray | None = None,
) -> np.ndarray:
    r"""Calculate pair density in reciprocal space via Fourier transforms
    Calculates \psi*_i(r) * \psi_j(r) (* is standard multiplication)

    The argument c_ip_array is \psi_i(p) for a p-grid in reciprocal space.
    The inverse Fourier transform of \psi_i(p) (or c_ip_array) is \psi_i(r) in real space.
    To calculate \psi*_i(r) * \psi_j(r) (* is standard multiplication) we inverse Fourier
    transform both \psi_i(p) and \psi_j(p) resulting in \psi_i(r) and \psi_j(r)
    and then perform the standard multiplication \psi*_i(r) * \psi_j(r) (* is standard multiplication).
    We return the calculated real space pair density.

    Args:
        c_ip_array (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals \psi_i(p) in the plane wave basis,
                                 shape (#bands, #grid_size, #grid_size, #grid_size)
        c_jp_array (np.ndarray | None): Same as c_ip_array but the coefficients describing \psi_j(p)
                                        instead of \psi_i(p) in \rho_ij(p)=\psi*_i(p) * \psi_j(p) (* is convolution).
                                        If None, this is set to c_ip_array

    Returns:
        np.ndarray: Pair density in real space
    """
    single_coeffs = False
    if c_jp_array is None:
        single_coeffs = True
        c_jp_array = c_ip_array.copy()

    dtype = np.complex128

    assert c_ip_array.ndim == c_jp_array.ndim == 4
    if not single_coeffs:
        assert c_ip_array.shape[1:] == c_jp_array.shape[1:]

    nbands_i = c_ip_array.shape[0]
    nbands_j = nbands_i
    if not single_coeffs:
        nbands_j = c_jp_array.shape[0]
    ngrid = c_ip_array.shape[1:]
    if not single_coeffs:
        assert ngrid == c_jp_array.shape[1:]

    rho_ij_r_mat = np.zeros((nbands_i, nbands_j, *ngrid), dtype)

    for i in range(nbands_i):
        logging.info("Calculating pair density %s/%s ...", i + 1, nbands_i)
        c_ip_shifted = np.fft.ifftshift(c_ip_array[i])
        psi_r_i_conj = np.fft.ifftn(c_ip_shifted).conj()
        for j in range(nbands_j):
            c_jp_shifted = np.fft.ifftshift(c_jp_array[j])
            psi_r_j = np.fft.ifftn(c_jp_shifted)

            # psi*_i(r) . psi_j(r), where . is the standard multiplication
            # Same as psi_i(p) * psi_j(p), where * is the convolution operation
            rho_ij_r = psi_r_i_conj * psi_r_j

            rho_ij_r_mat[i, j, :] = rho_ij_r

    return rho_ij_r_mat


def eri(
    c_ip: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    fft_grid: np.ndarray | None = None,
    use_gpu: bool = True,
) -> np.ndarray:
    r"""Calculate Electron Repulsion Integrals (ERIs) via pair densities in the physicists' index order
    We calculate

    .. math::

        h_{ijkl}=4\pi \sum_{p \neq 0} \rho_{il}(-p)\rho_{jk}(p)/|p|²
                =4\pi \sum_{p \neq 0} \rho*_{li}(p)\rho_{jk}(p)/|p|²

    where ijkl are indexing spatial orbitals
    Since the momenta p and reciprocal space wavefunctions \psi_i(p) (given as c_ip)
    are given as a list, we need to transfer \psi_i(p) to a 3D grid.
    Then \psi_i(p) is represented on a 3D reciprocal space grid.
    We calculate 1/|p|^2 on all grid points resulting in an infinite value at the
    center of the grid. This infinite value is set to zero, therefore,
    technically we later perform a sum over all momenta p except the zero momenta.
    Note that there are different techniques handling with this singularity
    by e.g. using a resolution-of-identity.

    Note that h_{ijkl} does not explicitly depend on the k-point but only implicitly
    depends on the k-point via the plane wave coefficients c_ip.
    Any explicit dependecy is cancelled out since the k-point contributes a phase
    to each KS-orbital (e^(ikr)) but since the pair densities are a product of two
    orbitals, where one in complex conjugated and the other is not, the phases
    of the two orbitals cancel out ([e^(ikr)]* e^(ikr) = e^(-ikr) e^(ikr) = 1)

    Args:
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        b (np.ndarray): Array of reciprocal lattice vectors, shape (3, 3)
        mill (np.ndarray): Array of Miller indices, shape (#waves, 3)
        fft_grid_size (np.ndarray | None): Edge lengths of the use FFT grid. Has to be larger or equal to grid
            where wavefunctions are defined on. Defaults to None, for which the wavefunction
            grid is used.

    Returns:
        np.ndarray: ERIs in reciprocal space
    """
    nbands, _nwaves = c_ip.shape  # number of DFT bands and number of plane waves

    assert mill.ndim == 2 and mill.shape[1] == 3, f"Array mill should contain 3D Miller indices but has shape {mill.shape}!"
    assert b.ndim == 2 and b.shape[0] == b.shape[1] == 3, f"Array b should contain 3 3D reciprocal lattice vectors but has shape {b.shape}!"

    max_min = mill.max(axis=0) - mill.min(axis=0)
    if fft_grid is not None:
        assert isinstance(fft_grid, np.ndarray), f"fft_grid ({fft_grid}, type {type(fft_grid)}) is not a numpy array!"
        assert fft_grid.shape == (3,), f"fft_grid (shape {fft_grid.shape}) has to have shape (3,), one grid length for each direction!"
        assert np.all(fft_grid >= 0), f"fft_grid {fft_grid} has values smaller than zero!"
        assert np.all(fft_grid >= max_min), f"fft_grid ({fft_grid}) needs to be larger or equal to wavefunction grid ({max_min})!"

        max_min = fft_grid.copy()

    # Grid needs to have odd edge length values so that the zero momentum is in the center of the grid
    for i, val in enumerate(max_min):
        if val % 2 == 0:
            max_min[i] += 1

    logging.info("FFT grid: %s", max_min)

    # Initialize 3D array for each DFT band with zero
    # We want to define \psi_i(p) as a matrix on the whole momentum grid for each i, instead of a list
    # *max_min is number of grid points for given maximum and minimum momenta and given grid spacing
    c_ip_array = np.zeros(
        (
            nbands,
            *max_min,
        ),
        dtype=c_ip.dtype,
    )

    # Set \psi_i(p) on given grid points
    assert mill.shape[0] == c_ip.shape[1]
    for idx, mill_idx in enumerate(mill):
        x, y, z = mill_idx
        i, j, k = (
            x + max_min[0] // 2,
            y + max_min[1] // 2,
            z + max_min[2] // 2,
        )
        c_ip_array[:, i, j, k] = c_ip[:, idx]

    start_time = time.perf_counter()
    one_over_p_norm_squared_array = np.zeros(max_min)
    one_over_p_norm_squared_array = make_one_over_p_norm_squared_array(one_over_p_norm_squared_array, b, max_min)
    norm_mat_time = time.perf_counter() - start_time

    # Calculate pair density \rho_ij(p) in reciprocal space
    start_time = time.perf_counter()
    rho_ij_p = pair_density(c_ip_array=c_ip_array, use_gpu=use_gpu)  # , c_jp_array=c_ip_array)
    pair_density_time = time.perf_counter() - start_time
    # Initialize ERI array
    # TODO: We do not need to calculate all matrix elements due to symmetries
    eri_mat = np.zeros((nbands, nbands, nbands, nbands), dtype=c_ip.dtype)

    start_time = time.perf_counter()
    # Calculate ERI matrix and rescale Fourier transforms
    # h_ijkl = 4\pi \sum_{p \neq 0} \rho*_li(p)\rho_jk(p)/|p|²
    # h_ijkl = 4\pi \sum_{p \neq 0} \rho_il(-p)\rho_jk(p)/|p|²
    eri_mat = (
        4
        * np.pi
        * np.einsum(
            "ilxyz, jkxyz, xyz -> ijkl",  # Physicists' order
            # * prod(max_min) for rescaling of Fourier transforms (order ~1e6)
            # np.flip(rho_ij_p, (2, 3, 4)) * prod(max_min),  # \rho_il(-p)
            rho_ij_p.conj().swapaxes(0, 1) * prod(max_min),  # \rho*_li(p) = \rho_il(-p)
            rho_ij_p * prod(max_min),
            one_over_p_norm_squared_array,
            optimize=True,  # larger memory footprint but faster
        )
    )
    einsum_time = time.perf_counter() - start_time
    # print(f"Time summary: 1/|p|^2: {norm_mat_time:.3f}s | pair density: {pair_density_time:.3f}s | einsum: {einsum_time:.3f}s")
    return eri_mat


def get_frozen_core_energy_eri(
    c_ip_core: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    cell_volume: float,
    fft_grid: np.ndarray | None = None,
    rho_ij_p: np.ndarray | None = None,
    one_over_p_norm_squared_array: np.ndarray | None = None,
    use_gpu: bool = True,
) -> float:
    r"""Calculate frozen core energy ERI part \sum_{ij}^{\mathrm{frozen}} (2h_{iijj} - h_{ijji})

    Args:
        p (np.ndarray): Array of momentum vectors, shape (#waves, 3)
        c_ip_core (np.ndarray): Array of coefficients describing the active Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        b (np.ndarray): Array of reciprocal lattice vectors, shape (3, 3)
        mill (np.ndarray): Array of Miller indices, shape (#waves, 3)
        cell_volume (float): Cell volume of the computationen real space cell.
        atom_positions (np.ndarray): 1D array of positions of each atom
        atomic_numbers (np.ndarray): 1D array of atomic number of each atom
        occupations_core (np.ndarray): Array of occupations of the core orbitals.
        rho_ij_p (np.ndarray | None): Array of pair densities of shape (#bands, #bands, #nwaves_fft),
            where #nwaves_fft is the number of points in momentum space that are used in the Fourier transforms.
            Is calculated from c_ip_core if None. If not None one_over_p_norm_squared_array has to be given, too.
            Defaults to None.
        one_over_p_norm_squared_array (np.ndarray | None): Array of 1/|p|^2 of shape (#nwaves_fft),
            where #nwaves_fft is the same as for rho_ij_p. Is calculated from b and mill if None.
            If not None rho_ij_p has to be given, too. Defaults to None.

    Returns:
        np.ndarray: ERIs in reciprocal space
    """
    max_min = mill.max(axis=0) - mill.min(axis=0)
    if fft_grid is not None:
        assert isinstance(fft_grid, np.ndarray), f"fft_grid ({fft_grid}, type {type(fft_grid)}) is not a numpy array!"
        assert fft_grid.shape == (3,), f"fft_grid (shape {fft_grid.shape}) has to have shape (3,)"
        assert np.all(fft_grid >= 0), f"fft_grid {fft_grid} has values smaller than zero!"
        assert np.all(fft_grid >= max_min), f"fft_grid ({fft_grid}) needs to be larger or equalt than wavefunction grid ({max_min})!"

        max_min = fft_grid.copy()

    # Grid needs to have odd edge length values so that the zero momentum is in the center of the grid
    for i, val in enumerate(max_min):
        if val % 2 == 0:
            max_min[i] += 1

    logging.info("FFT grid: %s", max_min)

    pre_calc = False  # Are the pair densities and 1/p² values given, if not calculated them
    if rho_ij_p is not None or one_over_p_norm_squared_array is not None:
        assert (
            rho_ij_p is not None and one_over_p_norm_squared_array is not None
        ), "Both rho_ij_p and one_over_p_norm_squared_array have to be given if one is given, but only one is given!"
        assert rho_ij_p.shape[0] == rho_ij_p.shape[1] == rho_ij_p.shape[0] == c_ip_core.shape[0]
        assert one_over_p_norm_squared_array.shape[0] == rho_ij_p.shape[2]
        pre_calc = True

    if not pre_calc:
        nbands, _nwaves = c_ip_core.shape  # number of DFT bands and number of plane waves

        assert mill.ndim == 2 and mill.shape[1] == 3, f"Array mill should contain 3D Miller indices but has shape {mill.shape}!"
        assert b.ndim == 2 and b.shape[0] == b.shape[1] == 3, f"Array b should contain 3 3D reciprocal lattice vectors but has shape {b.shape}!"

        # Initialize 3D array for each DFT band with zero
        # We want to define \psi_i(p) as a matrix on the whole momentum grid for each i, instead of a list
        c_ip_array = np.zeros(
            (
                nbands,
                # Number of grid points for given maximum and minimum momenta and given grid spacing
                *max_min,
            ),
            dtype=c_ip_core.dtype,
        )
        # Set \psi_i(p) on given grid points
        for idx, mill_idx in enumerate(mill):
            x, y, z = mill_idx
            i, j, k = (
                x + max_min[0] // 2,
                y + max_min[1] // 2,
                z + max_min[2] // 2,
            )
            c_ip_array[:, i, j, k] = c_ip_core[:, idx]

        one_over_p_norm_squared_array = np.zeros(max_min)
        one_over_p_norm_squared_array = make_one_over_p_norm_squared_array(one_over_p_norm_squared_array, b, max_min)

        # Calculate pair density \rho_ij(p) in reciprocal space
        rho_ij_p = pair_density(c_ip_array=c_ip_array, use_gpu=use_gpu)

    logging.info("Calc einsums (in get_frozen_core_energy_eri)...")
    # 4\pi \sum_{ij} \sum_{p \neq 0} (2\rho_ij(-p)\rho_ij(p)-\rho_ii(-p)\rho_jj(p))/|p|²
    #   and we get \rho_il(-p) from \rho_il(p) via inverting the x,y,z axes.
    #   alternatively one could use \rho_il(-p)=\rho*_li(p), which cannot be easily performed
    #       if i and l belong to different spaces, e.g. i belongs to the core and l belongs
    #       to an active space. And np.flip(...) seems to be faster than .conj() here.
    # 2 g_iijj - g_ijji, like in Yalouz et al, but they use chemists order
    # We use physicist order, therefore we use 2 g_ijji - g_ijij
    rho_ij_p_prod = rho_ij_p * prod(max_min)
    rho_ij_p_flip_prod = np.flip(rho_ij_p_prod, (2, 3, 4))
    eri_energy = (
        4
        * np.pi
        * (
            2
            * np.einsum(
                "iixyz, jjxyz, xyz -> ij",  # ijji
                rho_ij_p_flip_prod,
                rho_ij_p_prod,
                one_over_p_norm_squared_array,
                optimize=True,  # larger memory footprint but faster
            ).sum()
            - np.einsum(
                "ijxyz, jixyz, xyz -> ij",  # ijij
                rho_ij_p_flip_prod,
                rho_ij_p_prod,
                one_over_p_norm_squared_array,
                optimize=True,  # larger memory footprint but faster
            ).sum()
        )
    )

    # Rescaling of Fourier transforms
    return eri_energy / cell_volume


def get_frozen_core_energy(
    p: np.ndarray,
    c_ip_core: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    occupations_core: np.ndarray,
    fft_grid: np.ndarray | None = None,
    use_gpu: bool = True,
) -> float:
    r"""Calculate frozen core energy ERI part
    2\sum_i^{\mathrm{frozen}} h_ii + \sum_{ij}^{\mathrm{frozen}} (2h_{iijj} - h_{ijji})

    Args:
        p (np.ndarray): Array of momentum vectors, shape (#waves, 3)
        c_ip_core (np.ndarray): Array of coefficients describing the active Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        b (np.ndarray): Array of reciprocal lattice vectors, shape (3, 3)
        mill (np.ndarray): Array of Miller indices, shape (#waves, 3)
        cell_volume (float): Cell volume of the computationen real space cell.
        atom_positions (np.ndarray): 1D array of positions of each atom
        atomic_numbers (np.ndarray): 1D array of atomic number of each atom
        occupations_core (np.ndarray): Array of occupations of the core orbitals.

    Returns:
        np.ndarray: ERIs in reciprocal space
    """
    eri_energy = get_frozen_core_energy_eri(c_ip_core, b, mill, cell_volume, fft_grid, use_gpu=use_gpu)

    # Take only occupied core states into account
    occ = occupations_core.astype(bool)

    iTj_core = calc_matrix_elements.iTj(p, c_ip_core)[occ, :][:, occ]

    # Repulsion between electrons and nuclei
    iUj_core = calc_matrix_elements.iUj(p, c_ip_core, atom_positions, atomic_numbers, cell_volume)[occ, :][:, occ]

    # Frozen core energy
    h_pq_core_trace: float = (iTj_core - iUj_core).trace().real
    frozen_core_energy: float = 2 * h_pq_core_trace + eri_energy

    return frozen_core_energy


def get_frozen_core_energy_pp(
    p: np.ndarray,
    c_ip_core: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    cell_volume: float,
    atom_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    occupations_core: np.ndarray,
    pseudopots: list[Pseudopot],
    fft_grid: np.ndarray | None = None,
    use_gpu: bool = True,
) -> float:
    r"""Calculate frozen core energy ERI part
    2\sum_i^{\mathrm{frozen}} h_ii + \sum_{ij}^{\mathrm{frozen}} (2h_{iijj} - h_{ijji})

    Args:
        p (np.ndarray): Array of momentum vectors, shape (#waves, 3)
        c_ip_core (np.ndarray): Array of coefficients describing the active Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        b (np.ndarray): Array of reciprocal lattice vectors, shape (3, 3)
        mill (np.ndarray): Array of Miller indices, shape (#waves, 3)
        cell_volume (float): Cell volume of the computationen real space cell.
        atom_positions (np.ndarray): 1D array of positions of each atom
        atomic_numbers (np.ndarray): 1D array of atomic number of each atom
        occupations_core (np.ndarray): Array of occupations of the core orbitals.

    Returns:
        np.ndarray: ERIs in reciprocal space
    """
    eri_energy = get_frozen_core_energy_eri(c_ip_core, b, mill, cell_volume, fft_grid, use_gpu=use_gpu)

    # Take only occupied core states into account
    occ = occupations_core.astype(bool)

    iTj_core = calc_matrix_elements.iTj(p, c_ip_core)[occ, :][:, occ]

    # Repulsion between electrons and nuclei
    pp_core = calc_pseudo_pot.calc_pps(p, c_ip_core, cell_volume, atom_positions, atomic_numbers, pseudopots)[occ, :][:, occ]

    # Frozen core energy
    h_pq_core_trace: float = (iTj_core + pp_core).trace().real
    frozen_core_energy: float = 2 * h_pq_core_trace + eri_energy

    return frozen_core_energy


def get_frozen_core_energy_given_pp(
    p: np.ndarray,
    c_ip_core: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    cell_volume: float,
    pp_core: np.ndarray,
    occupations_core: np.ndarray,
    fft_grid: np.ndarray | None = None,
    use_gpu: bool = True,
) -> float:
    r"""Calculate frozen core energy ERI part
    2\sum_i^{\mathrm{frozen}} h_ii + \sum_{ij}^{\mathrm{frozen}} (2h_{iijj} - h_{ijji})

    Args:
        p (np.ndarray): Array of momentum vectors, shape (#waves, 3)
        c_ip_core (np.ndarray): Array of coefficients describing the active Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        b (np.ndarray): Array of reciprocal lattice vectors, shape (3, 3)
        mill (np.ndarray): Array of Miller indices, shape (#waves, 3)
        cell_volume (float): Cell volume of the computationen real space cell.
        pp_core (np.ndarray): Pseudopotential matrix for the core orbitals
        occupations_core (np.ndarray): Array of occupations of the core orbitals.

    Returns:
        np.ndarray: ERIs in reciprocal space
    """
    eri_energy = get_frozen_core_energy_eri(c_ip_core, b, mill, cell_volume, fft_grid, use_gpu=use_gpu)

    # Take only occupied core states into account
    occ = occupations_core.astype(bool)

    iTj_core = calc_matrix_elements.iTj(p, c_ip_core)[occ, :][:, occ]

    # Frozen core energy
    h_pq_core_trace: float = (iTj_core + pp_core).trace().real
    frozen_core_energy: float = 2 * h_pq_core_trace + eri_energy

    return frozen_core_energy


def get_frozen_core_pot(
    c_ip_core: np.ndarray,
    c_ip_active: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    fft_grid: np.ndarray | None = None,
    calc_eri_energy: bool = False,
    cell_volume: float | None = None,
    use_gpu: bool = True,
) -> np.ndarray | tuple[np.ndarray, float]:
    r"""Calculate frozen core effective single particle potential V_{tu}=\sum_{i}^{\mathrm{frozen}} (2h_{tuii} - h_{tiiu}).
    The notation is taken from Saad Yalouz et al 2021 Quantum Sci. Technol. 6 024004, Appendix B. Frozen core Hamiltonian,
    i.e. t,u are active orbital indices and i are core orbital indices and chemists index order is used.

    Using physicists' index order this becomes V_{tu}=\sum_{i}^{\mathrm{frozen}} (2h_{tiiu} - h_{tiui}).

    Args:
        c_ip_core (np.ndarray): Array of coefficients describing the core Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        c_ip_active (np.ndarray): Array of coefficients describing the active Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)
        b (np.ndarray): Array of reciprocal lattice vectors, shape (3, 3)
        mill (np.ndarray): Array of Miller indices, shape (#waves, 3)
        calc_eri_energy (bool): Defaults to False.
        cell_volume (float | None): Has to be given if calc_eri_energy is True. Defaults to None.

    Returns:
        np.ndarray | tuple[np.ndarray, float]: ERIs in reciprocal space or
            tuple of (ERIs in reciprocal space, frozen core energy) if calc_eri_energy is True
    """
    if calc_eri_energy:
        assert cell_volume is not None

    # number of core/active DFT bands and number of plane waves
    nbands_core, _nwaves = c_ip_core.shape
    nbands_active, _nwaves = c_ip_active.shape
    dtype = np.complex128

    assert mill.ndim == 2 and mill.shape[1] == 3, f"Array mill should contain 3D Miller indices but has shape {mill.shape}!"
    assert b.ndim == 2 and b.shape[0] == b.shape[1] == 3, f"Array b should contain 3 3D reciprocal lattice vectors but has shape {b.shape}!"

    max_min = mill.max(axis=0) - mill.min(axis=0)
    if fft_grid is not None:
        assert isinstance(fft_grid, np.ndarray), f"fft_grid ({fft_grid}, type {type(fft_grid)}) is not a numpy array!"
        assert fft_grid.shape == (3,), f"fft_grid (shape {fft_grid.shape}) has to have shape (3,)"
        assert np.all(fft_grid >= 0), f"fft_grid {fft_grid} has values smaller than zero!"
        assert np.all(fft_grid >= max_min), f"fft_grid ({fft_grid}) needs to be larger or equalt than wavefunction grid ({max_min})!"

        max_min = fft_grid.copy()

    # Grid needs to have odd edge length values so that the zero momentum is in the center of the grid
    for i, val in enumerate(max_min):
        if val % 2 == 0:
            max_min[i] += 1

    logging.info("FFT grid: %s", max_min)

    logging.info("Init c_ip arrays...")
    # Initialize 3D array for each DFT band with zero
    # We want to define \psi_i(p) as a matrix on the whole momentum grid for each i, instead of a list
    c_ip_core_array = np.zeros(
        (
            nbands_core,
            # Number of grid points for given maximum and minimum Miller index
            *max_min,
        ),
        dtype=dtype,
    )
    c_ip_active_array = np.zeros(
        (
            nbands_active,
            *max_min,
        ),
        dtype=dtype,
    )
    logging.info("Set c_ip arrays...")
    # Set \psi_i(p) on given grid points
    for idx, mill_idx in enumerate(mill):
        x, y, z = mill_idx
        i, j, k = (
            x + max_min[0] // 2,
            y + max_min[1] // 2,
            z + max_min[2] // 2,
        )
        c_ip_core_array[:, i, j, k] = c_ip_core[:, idx]
        c_ip_active_array[:, i, j, k] = c_ip_active[:, idx]

    logging.info("Set p_norm_squared_array...")
    one_over_p_norm_squared_array = np.zeros(max_min)
    one_over_p_norm_squared_array = make_one_over_p_norm_squared_array(one_over_p_norm_squared_array, b, max_min)

    logging.info("Calc pair densities...")
    # Calculate pair density \rho_ij(p) in reciprocal space
    logging.info("Calc pair density core...")
    rho_ij_p_core = pair_density(c_ip_array=c_ip_core_array, use_gpu=use_gpu)
    logging.info("Calc pair density active...")
    rho_tu_p_active = pair_density(c_ip_array=c_ip_active_array, use_gpu=use_gpu)
    logging.info("Calc pair density active-core...")
    rho_ti_p_active_core = pair_density(c_ip_array=c_ip_active_array, c_jp_array=c_ip_core_array, use_gpu=use_gpu)
    # Initialize ERI array
    # TODO: We do not need to calculate all matrix elements due to symmetries
    pot_mat = np.zeros((nbands_active, nbands_active), dtype=dtype)

    logging.info("Calc einsums...")
    # 2 g_tuii - g_tiiu, like in Yalouz et al, but they use chemists order
    # We use physicist order, therefore we use 2 h_tiiu - h_tiui
    # 2 h_tiiu - h_tiui = 2 (4\pi \sum_{p \neq 0} \rho_{tu}(-p)\rho_{ii}(p)/|p|²) - 4\pi \sum_{p \neq 0} \rho_{ti}(-p)\rho_{iu}(p)/|p|²
    #                   = 4\pi \sum_{p \neq 0} [2 \rho_{tu}(-p)\rho_{ii}(p) - \rho_{ti}(-p)\rho_{iu}(p)] /|p|²
    # frozen core potential matrix
    # h_pq_core = 4\pi \sum_{i} \sum_{p \neq 0} [2 \rho_{tu}(-p)\rho_{ii}(p) - \rho_{ti}(-p)\rho_{iu}(p)] /|p|²

    # h_{ijkl}=4\pi \sum_{p \neq 0} \rho_{il}(-p)\rho_{jk}(p)/|p|²
    # h_{tiiu}=4\pi \sum_{p \neq 0} \rho_{tu}(-p)\rho_{ii}(p)/|p|²
    h_tiiu = (
        4
        * np.pi
        * np.einsum(
            "tuxyz, iixyz, xyz -> tiu",  # tiiu, Physicists' order
            np.flip(rho_tu_p_active, (2, 3, 4)) * prod(max_min),
            rho_ij_p_core * prod(max_min),
            one_over_p_norm_squared_array,
            optimize=True,  # larger memory footprint but faster
        )
    )
    # h_{ijkl}=4\pi \sum_{p \neq 0} \rho_{il}(-p)\rho_{jk}(p)/|p|²
    # h_{tiui}=4\pi \sum_{p \neq 0} \rho_{ti}(-p)\rho_{iu}(p)/|p|²
    rho_ti_p_active_core_flip_prod = np.flip(rho_ti_p_active_core, (2, 3, 4)) * prod(max_min)
    h_tiui = (
        4
        * np.pi
        * np.einsum(
            "tixyz, iuxyz, xyz -> tiu",  # tiui, Physicists' order
            rho_ti_p_active_core_flip_prod,
            # \rho_iu(p) = \rho*_ui(-p)
            # swapaxes to bring indices in correct order, np.flip(...) and conj() do p->-p and \rho->\rho*
            #   but not ui -> iu. Regardless of writing \rho_iu(p) or \rho*_ui(-p).
            #   i always runs only over core orbitals while u always runs over active orbitals
            rho_ti_p_active_core_flip_prod.conj().swapaxes(0, 1),  # \rho_iu(p) = \rho*_ui(-p)
            one_over_p_norm_squared_array,
            optimize=True,  # larger memory footprint but faster
        )
    )

    pot_mat = 2 * h_tiiu.sum(1) - h_tiui.sum(1)

    if calc_eri_energy:
        eri_energy = get_frozen_core_energy_eri(
            c_ip_core=c_ip_core,
            b=b,
            mill=mill,
            cell_volume=cell_volume,
            fft_grid=max_min,
            rho_ij_p=rho_ij_p_core,
            one_over_p_norm_squared_array=one_over_p_norm_squared_array,
            use_gpu=use_gpu,
        )
        return pot_mat, eri_energy

    return pot_mat


def get_frozen_core_pot_and_energy_given_pp(
    c_ip_core: np.ndarray,
    c_ip_active: np.ndarray,
    b: np.ndarray,
    mill: np.ndarray,
    p: np.ndarray,
    cell_volume: float,
    pp_core: np.ndarray,
    occupations_core: np.ndarray,
    fft_grid: np.ndarray | None = None,
    use_gpu: bool = True,
) -> tuple[np.ndarray, float]:
    pot_mat, eri_energy = get_frozen_core_pot(
        c_ip_core=c_ip_core,
        c_ip_active=c_ip_active,
        b=b,
        mill=mill,
        fft_grid=fft_grid,
        calc_eri_energy=True,
        cell_volume=cell_volume,
        use_gpu=use_gpu,
    )

    # Take only occupied core states into account
    occ = occupations_core.astype(bool)

    iTj_core = calc_matrix_elements.iTj(p, c_ip_core)[occ, :][:, occ]

    # Frozen core energy
    h_pq_core_trace: float = (iTj_core + pp_core).trace().real
    frozen_core_energy: float = 2 * h_pq_core_trace + eri_energy

    return pot_mat / cell_volume, frozen_core_energy
