import os
import sys
import time
import logging
import warnings
import numpy as np
import dopyqo
from dopyqo.colors import *


def snapshot_test(base_folder: str, prefix: str, active_electrons: int, active_orbitals: int, create: bool = False):
    snapshot_file = os.path.join("snapshots", f"eri_{prefix}_{active_electrons}e_{active_orbitals}o.npy")
    if not create:
        if not os.path.isfile(snapshot_file):
            print(f"{RED}Snapshot does not exist: {snapshot_file}{RESET_COLOR}")
            sys.exit(1)
        h_pqrs_ref = np.load(snapshot_file)

    dat_file = os.path.join(base_folder, f"{prefix}.save", "wfc1.dat")
    xml_file = os.path.join(base_folder, f"{prefix}.save", "data-file-schema.xml")
    pp_files = [
        os.path.join(base_folder, f"{prefix}.save", filename)
        for filename in os.listdir(os.path.join(base_folder, f"{prefix}.save"))
        if filename.lower().endswith(".upf")
    ]
    pps = [dopyqo.Pseudopot(pp_file) for pp_file in pp_files]

    binary_occupations = True

    wfc1_ncpp = dopyqo.Wfc.from_file(dat_file, xml_file, pseudopots=pps)

    overlaps_ncpp = wfc1_ncpp.get_overlaps()
    if not np.allclose(overlaps_ncpp, np.eye(overlaps_ncpp.shape[0])):
        warnings.warn("The wavefunctions are not orthonormal!")

    p = wfc1_ncpp.k_plus_G  # shape (#waves, 3)
    c_ip = wfc1_ncpp.evc  # shape (#bands, #waves)

    orbital_indices_core, orbital_indices_active = wfc1_ncpp.active_space(active_electrons=active_electrons, active_orbitals=active_orbitals)

    occs_tmp = wfc1_ncpp.occupations_binary if binary_occupations else wfc1_ncpp.occupations
    occ_str = " ".join(
        [
            (f"|{val}" if i == orbital_indices_active[0] else f"{val}|" if i == orbital_indices_active[-1] else f"{val}")
            for i, val in enumerate(occs_tmp)
        ]
    )
    print(f"The following active space will be considered:\n{occ_str}\tNumber of plane waves: {p.shape[0]}")

    occupations_full, c_ip_full = wfc1_ncpp.get_orbitals_by_index(
        orbital_indices_core + orbital_indices_active, binary_occupations=binary_occupations
    )
    occupations_active, c_ip_active = wfc1_ncpp.get_orbitals_by_index(orbital_indices_active, binary_occupations=binary_occupations)
    with warnings.catch_warnings():  # Ignore warning that all core orbitals are occupied
        warnings.filterwarnings("ignore", category=Warning)
        occupations_core, c_ip_core = wfc1_ncpp.get_orbitals_by_index(orbital_indices_core, binary_occupations=binary_occupations)

    # Calculate ERIs via pair density
    fft_grid = wfc1_ncpp.fft_grid
    # fft_grid = np.array([14, 14, 14])
    start_time = time.perf_counter()
    h_pqrs: np.ndarray = dopyqo.eri(c_ip=c_ip_active, b=wfc1_ncpp.b, mill=wfc1_ncpp.mill, fft_grid=fft_grid) / wfc1_ncpp.cell_volume
    time_eri = time.perf_counter() - start_time

    if not create:
        if not np.allclose(h_pqrs, h_pqrs_ref):
            print(f"{RED}Snapshot mismatch: {prefix}{RESET_COLOR}")
        else:
            print(f"{GREEN}Snapshot passed: {prefix}{RESET_COLOR}")
    else:
        np.save(snapshot_file, h_pqrs)
        print(f"{ORANGE}Snapshot created: {prefix}{RESET_COLOR}")


if __name__ == "__main__":
    create = False
    snapshot_test(base_folder=os.path.join("qe_files", "LiH"), prefix="LiH", active_electrons=4, active_orbitals=6, create=create)
    snapshot_test(base_folder=os.path.join("qe_files", "Mg"), prefix="Mg", active_electrons=20, active_orbitals=15, create=create)
