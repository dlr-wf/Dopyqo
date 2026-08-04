import os

import numpy as np

import dopyqo
from dopyqo.calc_matrix_elements import nuclear_repulsion_energy_ewald
from dopyqo.colors import *


def ewald_test(base_folder: str, prefix: str, rust_impl: bool):
    dat_file = os.path.join(base_folder, f"{prefix}.save", "wfc1.dat")
    xml_file = os.path.join(base_folder, f"{prefix}.save", "data-file-schema.xml")
    pp_files = [
        os.path.join(base_folder, f"{prefix}.save", filename)
        for filename in os.listdir(os.path.join(base_folder, f"{prefix}.save"))
        if filename.lower().endswith(".upf")
    ]
    pps = [dopyqo.Pseudopot(pp_file) for pp_file in pp_files]

    wfc = dopyqo.Wfc.from_file(dat_file, xml_file, pseudopots=pps)

    lattice_vectors = np.array([wfc.a1, wfc.a2, wfc.a3])
    lattice_vectors_reciprocal = np.array([wfc.b1, wfc.b2, wfc.b3])

    # The Ewald energy must be independent of sigma. Checking several sigma values.
    for sigma in [None, 2.8, 0.5, 0.2, 0.1]:
        e_ewald = nuclear_repulsion_energy_ewald(
            wfc.atom_positions_hartree,
            wfc.atomic_numbers_valence,
            lattice_vectors,
            lattice_vectors_reciprocal,
            wfc.cell_volume,
            wfc.gcutrho,
            sigma=sigma,
            rust_impl=rust_impl,
        )

        if not np.isclose(e_ewald, wfc.ewald, atol=1e-6, rtol=1e-6):
            print(f"{RED}Ewald mismatch: {prefix} (rust_impl={rust_impl}, sigma={sigma}): got {e_ewald}, expected {wfc.ewald}{RESET_COLOR}")
        else:
            print(f"{GREEN}Ewald test passed: {prefix} (rust_impl={rust_impl}, sigma={sigma}){RESET_COLOR}")


if __name__ == "__main__":
    # Mg has a non-symmetric (hexagonal) cell matrix: a1, a2, a3 are not equal to their transposed
    # counterparts. This can catch incorrect handling of the real-space translation vectors
    # in the Ewald sum, which a symmetric cell (e.g. LiH) cannot catch.
    ewald_test(base_folder=os.path.join("qe_files", "Mg"), prefix="Mg", rust_impl=False)
    ewald_test(base_folder=os.path.join("qe_files", "Mg"), prefix="Mg", rust_impl=True)
