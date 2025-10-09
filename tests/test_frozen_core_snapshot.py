import os
import sys
import time
import logging
import warnings
import numpy as np
import dopyqo
from dopyqo.colors import *


def snapshot_test(base_folder: str, prefix: str, active_electrons: int, active_orbitals: int, create: bool = False):
    snapshot_file = os.path.join("snapshots", f"frozen_core_{prefix}_{active_electrons}e_{active_orbitals}o.npz")
    if not create:
        if not os.path.isfile(snapshot_file):
            print(f"{RED}Snapshot does not exist: {snapshot_file}{RESET_COLOR}")
            sys.exit(1)
        data = np.load(snapshot_file)
        h_pq_core_ref = data["h_pq_core"]
        frozen_core_energy_ref = data["frozen_core_energy"]

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

    # Calculate pseudopotential
    n_threads_pp = 50
    start_time = time.perf_counter()
    pp_full = dopyqo.calc_pps(
        p,
        c_ip_full,
        wfc1_ncpp.cell_volume,
        wfc1_ncpp.atom_positions_hartree,
        wfc1_ncpp.atomic_numbers,
        pps,
        n_threads=n_threads_pp,
        # os.path.join(base_folder, "pp_pw.npy"),
    )
    time_pp = time.perf_counter() - start_time
    logging.info("Pseudopotentials calc. took %.2fs", time_pp)
    pp_orbitals = pp_full[orbital_indices_active, :][:, orbital_indices_active]
    pp_core = pp_full[orbital_indices_core, :][:, orbital_indices_core]

    # Calculate frozen core potential and energy
    logging.info("Calculate frozen core...")
    if len(c_ip_core) <= 0:
        print(f"Snapshot error: No frozen core with {active_electrons}e, {active_orbitals}o")
        sys.exit(1)
    start_time = time.perf_counter()
    h_pq_core, frozen_core_energy = dopyqo.get_frozen_core_pot_and_energy_given_pp(
        p=p,
        c_ip_core=c_ip_core,
        c_ip_active=c_ip_active,
        b=wfc1_ncpp.b,
        mill=wfc1_ncpp.mill,
        cell_volume=wfc1_ncpp.cell_volume,
        pp_core=pp_core,
        occupations_core=occupations_core,
        fft_grid=wfc1_ncpp.fft_grid,
    )

    if not create:
        failed = False
        failed_lst = []
        if not np.allclose(h_pq_core, h_pq_core_ref):
            failed = True
            failed_lst.append("potential")
        elif not np.allclose(frozen_core_energy, frozen_core_energy_ref):
            failed = True
            failed_lst.append(f"energy {frozen_core_energy} (Ref.: {frozen_core_energy_ref})")
        if failed:
            print(f"{RED}Snapshot mismatch: {prefix} ({', '.join(failed_lst)}){RESET_COLOR}")
        else:
            print(f"{GREEN}Snapshot passed: {prefix}{RESET_COLOR}")
    else:
        np.savez(snapshot_file, h_pq_core=h_pq_core, frozen_core_energy=frozen_core_energy)
        print(f"{ORANGE}Snapshot created: {prefix}{RESET_COLOR}")


if __name__ == "__main__":
    create = False
    snapshot_test(base_folder=os.path.join("qe_files", "LiH"), prefix="LiH", active_electrons=2, active_orbitals=2, create=create)
    snapshot_test(base_folder=os.path.join("qe_files", "Mg"), prefix="Mg", active_electrons=2, active_orbitals=2, create=create)
