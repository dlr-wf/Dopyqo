import os
import numpy as np
import dopyqo
import dopyqo.wannier90

if __name__ == "__main__":
    base_folder = os.path.join("qe_files", "Be_wannier")
    filename_umat = os.path.join(base_folder, "Be_u.mat")

    kpt_to_u = dopyqo.read_u_mat(filename=filename_umat)

    filename_hr = os.path.join(base_folder, "Be_hr.dat")

    hr_mat = dopyqo.wannier90.read_hr_dat(filename=filename_hr)
    # KS eigenvalues and eigenvectors in the Wannier basis
    eigvals, eigvecs = np.linalg.eig(hr_mat[:, :, 0])

    # Check unitarity
    u_mat = np.stack(list(kpt_to_u.values()), axis=2)
    print(f"Unitary: {dopyqo.wannier90.check_unitarity_u(u_mat[:, :, 0])}")
