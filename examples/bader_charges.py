import os
import copy
import numpy as np
import dopyqo
import warnings
import logging
import pandas as pd
import dopyqo
from dopyqo.colors import *
from qiskit.quantum_info import Statevector
from dopyqo import cube


if __name__ == "__main__":
    # Run QE SCF calculation
    # and run QE pp calculation
    # using the input files Mg2Si.scf.in and Mg2Si.pp.in

    base_folder = os.path.join("qe_files", "Mg2Si")
    prefix = "Mg2Si"
    active_electrons = 8
    active_orbitals = 9
    path_to_bader_executable = {"bader": os.path.join(os.path.expanduser("~"), "bader/bader")}
    config = dopyqo.DopyqoConfig(
        base_folder=base_folder,
        prefix=prefix,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        run_fci=True,
        run_vqe=False,
        vqe_optimizer=dopyqo.VQEOptimizers.L_BFGS_B,
        n_threads=10,
        uccsd_reps=1,
    )

    energy_dict, wfc1_ncpp, h_ks, mats = dopyqo.run(config)

    npz_path = os.path.join(f"{base_folder}", f"statevecs_{active_electrons}e_{active_orbitals}o.npz")
    atoms = []
    atoms = [element["element"] for element in wfc1_ncpp.atoms]
    valence = {x.element: x.z_valence for x in wfc1_ncpp.pseudopots}
    # Extracting data from the ACF.dat file in the pp-x_bader folder. WARNING here pp is related to the pp.x command no PseudoPotential

    reference_energy = wfc1_ncpp.etot

    overlaps_ncpp = wfc1_ncpp.get_overlaps()

    if not np.allclose(overlaps_ncpp, np.eye(overlaps_ncpp.shape[0])):
        warnings.warn("The wavefunctions are not orthonormal!")

    p = wfc1_ncpp.k_plus_G  # shape (#waves, 3)
    c_ip = wfc1_ncpp.evc  # shape (#bands, #waves)

    logging.info("Number of plane waves: %i", p.shape[0])

    ########################################################
    #           Calculating real-space densities           #
    ########################################################
    nbands = wfc1_ncpp.nbnd
    max_min = wfc1_ncpp.fft_grid
    mill = wfc1_ncpp.mill
    tot_elec = wfc1_ncpp.nelec

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

    psi_r = []
    WF_cube = []
    QE_file = cube.read_cube(os.path.join(base_folder, f"{prefix}_pp.cube"))
    QE_cube = QE_file["cube"]
    QE_vol = QE_file["volume"]
    QE_cube_scale = QE_cube * QE_vol

    print(f"nbands: {nbands}")
    for i in range(nbands):
        logging.info("Calculating real space KS %s/%s ...", i + 1, nbands)
        c_ip_shifted = np.fft.ifftshift(c_ip_array[i])
        psi_r_i = np.fft.ifftn(c_ip_shifted) * np.sqrt(np.prod(max_min))

        psi_r.append(psi_r_i)

    psi_r = np.array(psi_r)

    occ_bands = int(tot_elec / 2)
    ks_occupation = []

    for i in range(occ_bands):
        occ = 1
        ks_occupation.append(occ)

    for i in range(nbands - occ_bands):
        occ = 0
        ks_occupation.append(occ)

    ks_single_density = np.abs(psi_r) ** 2

    ks_tot_density = cube.get_density_cube(ks_single_density, wfc1_ncpp.occupations_xml)
    ks_tot_density = ks_tot_density / (wfc1_ncpp.cell_volume / np.prod(wfc1_ncpp.fft_grid) / 2.0)
    ks_total_electrons = wfc1_ncpp.nelec

    save_dir = os.path.join(base_folder, f"Bader_dopyqo")
    cube.make_cube(base_folder, prefix, ks_tot_density, save_dir, path_to_bader_executable, max_min)
    ks_ACF_file = os.path.join(save_dir, "ACF.dat")
    ks_bader = cube.extract_charge(atoms, ks_ACF_file, valence)

    ppx_dir = os.path.join(base_folder, "Bader_QE")
    ppx_file = os.path.join(base_folder, f"{prefix}_pp.cube")
    cube.run_ppx(ppx_file, path_to_bader_executable, ppx_dir, prefix)
    ppx_ACF_file = os.path.join(ppx_dir, "ACF.dat")
    qe_bader = cube.extract_charge(atoms, ppx_ACF_file, valence)

    # print(f"{ks_tot_density=}")
    print(f"{ks_total_electrons=}")

    QE_file = cube.read_cube(os.path.join(base_folder, f"{prefix}_pp.cube"))
    QE_cube = QE_file["cube"]
    QE_vol = QE_file["volume"]
    QE_cube_scale = QE_cube * QE_vol

    statevecs_data = np.load(npz_path)
    merged = pd.DataFrame()
    if statevecs_data["fci_state_data"].shape != (0,):
        fci_state_loaded = Statevector(statevecs_data["fci_state_data"])
        fci_AS_occupation = cube.orb_occupation(fci_state_loaded)
        fci_total_occupation = cube.make_full_occ_from_partial_occ(fci_AS_occupation, active_orbitals, active_electrons, ks_total_electrons, nbands)
        save_dir = os.path.join(base_folder, f"Bader_dopyqo_casci_{active_electrons}e_{active_orbitals}o")
        print(f"Occupation of AS of {prefix}: {fci_AS_occupation}")
        print(f"Occupation of CASCI of {prefix} : {fci_total_occupation}")
        print(f"Occupation KS of {prefix}: {wfc1_ncpp.occupations_xml}")
        fci_tot_density = cube.get_density_cube(ks_single_density, fci_total_occupation)
        fci_tot_density = fci_tot_density / (wfc1_ncpp.cell_volume / np.prod(wfc1_ncpp.fft_grid) / 2.0)
        cube.make_cube(base_folder, prefix, fci_tot_density, save_dir, path_to_bader_executable, max_min)
        fci_ACF_file = os.path.join(save_dir, "ACF.dat")
        fci_bader = cube.extract_charge(atoms, fci_ACF_file, valence)

        merged = pd.DataFrame(
            {
                "ELEMENT": atoms,
                "Bader charge QE": qe_bader["EFFECTIVE CHARGE"].values,
                "Bader charge Dopyqo": ks_bader["EFFECTIVE CHARGE"].values,
                "Bader charge Dopyqo CASCI": fci_bader["EFFECTIVE CHARGE"].values,
            }
        )
    if statevecs_data["vqe_state_data"].shape != (0,):
        vqe_state_loaded = Statevector(statevecs_data["vqe_state_data"])
        vqe_AS_occupation = cube.orb_occupation(vqe_state_loaded)
        vqe_total_occupation = cube.make_full_occ_from_partial_occ(vqe_AS_occupation, active_orbitals, active_electrons, ks_total_electrons, nbands)
        save_dir = os.path.join(base_folder, f"Bader_dopyqo_vqe/s_{active_electrons}e_{active_orbitals}o_")
        print(f"Occupation of AS of {prefix}: {vqe_AS_occupation}")
        print(f"Occupation of VQE of {prefix} : {vqe_total_occupation}")
        print(f"Occupation KS of {prefix}: {wfc1_ncpp.occupations_xml}")
        vqe_tot_density = cube.get_density_cube(ks_single_density, vqe_total_occupation)
        vqe_tot_density = vqe_tot_density / (wfc1_ncpp.cell_volume / np.prod(wfc1_ncpp.fft_grid) / 2.0)
        cube.make_cube(base_folder, prefix, vqe_tot_density, save_dir, path_to_bader_executable, max_min)
        vqe_ACF_file = os.path.join(save_dir, "ACF.dat")
        vqe_bader = cube.extract_charge(atoms, vqe_ACF_file, valence)

        merged = pd.DataFrame(
            {
                "ELEMENT": atoms,
                "Bader charge QE": qe_bader["EFFECTIVE CHARGE"].values,
                "Bader charge Dopyqo": ks_bader["EFFECTIVE CHARGE"].values,
                "Bader charge Dopyqo VQE": vqe_bader["EFFECTIVE CHARGE"].values,
            }
        )

    print(f"{GREEN}Bader charge results:{RESET_COLOR}")
    print(merged)
