import os
import sys
import numpy as np
from dopyqo.colors import *
from dopyqo import DopyqoConfig


def read_u_mat(filename: str) -> dict[tuple[float, float, float], np.ndarray]:
    # Initialize a list to store the data for each k-point
    kpt_to_u = {}

    # Open file for reading
    with open(filename, "r", encoding="utf-8") as f:
        # Read the first line containing the date and time of creation of the file
        _creation_time = f.readline().strip()

        # Read the second line containing the number of k-points (num_kpts) and two times the number of Wannier functions (num_wann)
        num_kpts, num_wann, _num_wann = map(int, f.readline().split())

        # Skip the third line which is empty
        f.readline()

        # Read the data for each k-point
        for _ in range(num_kpts):
            # Read the k-point coordinates (in fractional coordinates of the reciprocal lattice vectors)
            kpt = tuple(map(float, f.readline().split()))

            # Initialize a matrix to store the U(k) matrix elements
            u_mat = np.zeros(shape=(num_wann, num_wann), dtype=np.complex128)

            # Read the matrix elements (real and imaginary parts) of U(k) in column-major order,
            # i.e. cycling over rows first and then columns
            # Note: \psi^{(k)}_{n, wannier} = U^{(k)}_{mn} \psi^{(k)}_{m, KS}
            for col in range(num_wann):
                for row in range(num_wann):
                    real, imag = map(float, f.readline().split())
                    u_mat[row, col] = complex(real, imag)

            # Append the data for this k-point to the list
            # kpt_to_u[kpt] = u_mat
            # NOTE: We have to transform with U*^T instead of with U to replicate
            #       the XSF data of the real-space Wannier orbitals (test for real-wavefunctions at gamma-point).
            kpt_to_u[kpt] = u_mat.T.conj()

            # Skip empty line between k-points
            f.readline()

    return kpt_to_u


# def get_u_mat(config: DopyqoConfig, kpoint: np.ndarray, nbnd: int, orbital_indices_active: list[int]):
#     # Read Wannier90 transform matrix
#     transform_matrix = read_u_mat(config.wannier_umat)[tuple(kpoint)]
#     if not check_unitarity_u(transform_matrix):
#         print(f"{RED}Wannier error: Transformation matrix is not unitary{RESET_COLOR}")
#         sys.exit(1)
#     if config.active_orbitals != transform_matrix.shape[0]:
#         print(
#             f"{RED}Wannier error: Number of active orbitals ({config.active_orbitals}) "
#             + f"does not match number of Wannier orbitals ({transform_matrix.shape[0]})!{RESET_COLOR}"
#         )
#         sys.exit(1)
#     ######################### READING WANNIER INPUT FILE #########################
#     if config.wannier_input_file is not None:
#         results = {}
#         keys = ["num_wann", "exclude_bands"]

#         with open(config.wannier_input_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()

#                 for key in keys:
#                     if line.startswith(key):
#                         if "=" in line:
#                             _, sep, val = line.partition("=")
#                         elif ":" in line:
#                             _, sep, val = line.partition(":")
#                         else:  # whitespace split
#                             parts = line.split(None, 1)
#                             val = parts[1] if len(parts) == 2 else None
#                         results[key] = val.strip()
#                         break

#         _num_wann = int(results["num_wann"])
#         exclude_bands = []
#         if "exclude_bands" in results:
#             for x in results["exclude_bands"].split(","):
#                 if "-" not in x:
#                     exclude_bands.append(int(x))
#                 else:
#                     start_stop = [int(val) for val in x.split("-")]
#                     start = start_stop[0]
#                     stop = start_stop[1] + 1
#                     exclude_bands.extend(list(range(start, stop)))
#             exclude_bands = [x - 1 for x in exclude_bands]  # Wannier90 1-indexing to python 0-indexing
#         orbital_indices_active_wannier = list(set(exclude_bands) ^ set(range(nbnd)))
#         if orbital_indices_active != orbital_indices_active_wannier:
#             print(
#                 f"{RED}Wannier error: Orbitals in the active space ({orbital_indices_active}) are not the "
#                 + f"same orbitals transformed with Wannier90 ({orbital_indices_active_wannier})!{RESET_COLOR}"
#             )
#             sys.exit(1)


def read_hr_dat(filename: str) -> dict[tuple[float, float, float], np.ndarray]:
    # Open file for reading
    with open(filename, "r", encoding="utf-8") as f:
        # Read the first line containing the date and time of creation of the file
        _creation_time = f.readline().strip()

        num_wann = int(f.readline().strip())
        nrpts = int(f.readline().strip())

        degeneracies = []
        for _ in range(nrpts // 15 + 1):
            degeneracies.append(f.readline().split())

        hr_mat_ev = np.zeros(shape=(num_wann, num_wann, nrpts), dtype=np.complex128)

        for i in range(num_wann**2 * nrpts):
            _r_1, _r_2, _r_3, m, n, real, imag = map(float, f.readline().split())
            hr_mat_ev[int(m) - 1, int(n) - 1, i // num_wann**2] = complex(real, imag)

    ev_to_hartree = 0.036749308136649
    hr_mat_hartree = hr_mat_ev * ev_to_hartree
    return hr_mat_hartree


# From https://github.com/hungpham2017/mcu/blob/master/mcu/wannier90/utils.py#L53
# referenced here: https://lists.quantum-espresso.org/pipermail/wannier/2020-March/001726.html
def read_U_matrix(filename):
    """Read seedname_u.mat file"""

    with open(filename, "r", encoding="utf-8") as file:
        data = file.read().split("\n")
        nkpts, nwann, nband = np.int64(data[1].split())
        temp = data[2:-1]
        block_length = nband * nwann + 2
        kpts = []
        U_kpts = []
        for kpt_th in range(nkpts):
            Uk = temp[(kpt_th * block_length) : (kpt_th * block_length + block_length)]
            kpts.append(np.float64(Uk[1].split()))
            U = np.asarray([np.float64(line.split()[0]) + 1j * np.float64(line.split()[1]) for line in Uk[2:]])
            U_kpts.append(U.reshape(nwann, nband).T)

        kpts = np.float64(kpts)
        U_kpts = np.asarray(U_kpts)

    return kpts, U_kpts


def check_unitarity_u(u_mat: np.ndarray) -> bool:
    """Check unitarity of U matrix
    u_mat: U matrix at different k-point with shape (nwann, nwann, nkpts)
    """
    assert u_mat.ndim == 2
    assert u_mat.shape[0] == u_mat.shape[1]
    nwann = u_mat.shape[0]

    return np.allclose(u_mat @ u_mat.T.conj(), np.eye(nwann))


if __name__ == "__main__":
    filename_umat_mg = os.path.join("wannier90_files", "magnesium_u.mat")

    kpt_to_u_mg = read_u_mat(filename=filename_umat_mg)

    filename_hr_mg = os.path.join("wannier90_files", "calc", "magnesium_hr.dat")

    hr_mat_mg = read_hr_dat(filename=filename_hr_mg)
    # KS eigenvalues and eigenvectors in the Wannier basis
    eigvals, eigvecs = np.linalg.eig(hr_mat_mg[:, :, 0])

    # Check unitarity
    u_mat_mg = np.stack(list(kpt_to_u_mg.values()), axis=2)
    print(f"Unitary: {check_unitarity_u(u_mat_mg)}")
