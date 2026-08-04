import os
import sys
import warnings

import numpy as np

import dopyqo
from dopyqo import calc_pseudo_pot as cpp
from dopyqo import units
from dopyqo.colors import *


def _load_wfc(base_folder: str, prefix: str):
    dat_file = os.path.join(base_folder, f"{prefix}.save", "wfc1.dat")
    xml_file = os.path.join(base_folder, f"{prefix}.save", "data-file-schema.xml")
    pp_files = [
        os.path.join(base_folder, f"{prefix}.save", filename)
        for filename in os.listdir(os.path.join(base_folder, f"{prefix}.save"))
        if filename.lower().endswith(".upf")
    ]
    pps = [dopyqo.Pseudopot(pp_file) for pp_file in pp_files]
    wfc = dopyqo.Wfc.from_file(dat_file, xml_file, pseudopots=pps)
    return wfc, pps


def test_ewald_gradient_wrt_atom_positions_fd(base_folder: str, prefix: str, h: float = 1e-6):
    wfc, _ = _load_wfc(base_folder, prefix)

    #  Displace atoms away from equilibrium first so the gradient being tested is not near-zero.
    offset = np.array([0.1, -0.05, 0.07])
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(wfc.atom_positions_hartree.shape[0])])
    atom_positions = wfc.atom_positions_hartree + signs[:, None] * offset[None, :]
    atomic_numbers_valence = wfc.atomic_numbers_valence
    lattice_vectors = np.array([wfc.a1, wfc.a2, wfc.a3])
    lattice_vectors_reciprocal = np.array([wfc.b1, wfc.b2, wfc.b3])
    cell_volume = wfc.cell_volume
    gcutrho = wfc.gcutrho

    analytic_grad = dopyqo.gradients_atom_positions.gradient_nuclear_repulsion_energy_ewald_wrt_atom_positions(
        atom_positions=atom_positions,
        atomic_numbers=atomic_numbers_valence,
        lattice_vectors=lattice_vectors,
        lattice_vectors_reciprocal=lattice_vectors_reciprocal,
        cell_volume=cell_volume,
        gcutrho=gcutrho,
    )

    def ewald_energy(positions):
        return dopyqo.calc_matrix_elements.nuclear_repulsion_energy_ewald(
            positions,
            atomic_numbers_valence,
            lattice_vectors,
            lattice_vectors_reciprocal,
            cell_volume,
            gcutrho,
            rust_impl=False,
        )

    all_passed = True
    for i in range(atom_positions.shape[0]):
        for j in range(3):
            pos_fwd = atom_positions.copy()
            pos_bwd = atom_positions.copy()
            pos_fwd[i, j] += h
            pos_bwd[i, j] -= h
            fd_grad = (ewald_energy(pos_fwd) - ewald_energy(pos_bwd)) / (2 * h)
            if not np.isclose(fd_grad, analytic_grad[i, j], atol=1e-6, rtol=1e-6):
                print(f"{RED}Ewald gradient mismatch: {prefix}, atom {i}, coord {j}: FD={fd_grad}, analytic={analytic_grad[i, j]}{RESET_COLOR}")
                all_passed = False

    if all_passed:
        print(f"{GREEN}Ewald gradient FD test passed: {prefix}{RESET_COLOR}")
    else:
        print(f"{RED}Ewald gradient FD test failed: {prefix}{RESET_COLOR}")
        sys.exit(1)


def test_pp_gradient_wrt_atom_positions_fd(base_folder: str, prefix: str, active_electrons: int, active_orbitals: int, h: float = 1e-6):
    wfc, pps = _load_wfc(base_folder, prefix)

    _orbital_indices_core, orbital_indices_active = wfc.active_space(active_electrons=active_electrons, active_orbitals=active_orbitals)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        _occupations_active, c_ip_active = wfc.get_orbitals_by_index(orbital_indices_active, binary_occupations=True)

    p = wfc.k_plus_G
    atom_positions = wfc.atom_positions_hartree
    atomic_numbers = wfc.atomic_numbers
    cell_volume = wfc.cell_volume

    analytic_grad = dopyqo.gradients_atom_positions.calc_gradient_pps_wrt_atom_positions(
        p,
        c_ip_active,
        cell_volume,
        atom_positions,
        atomic_numbers,
        pps,
        rust_impl=False,
    )

    def pp_matrix(positions):
        return cpp.calc_pps(p, c_ip_active, cell_volume, positions, atomic_numbers, pps, rust_impl=False)

    all_passed = True
    for i in range(atom_positions.shape[0]):
        for j in range(3):
            pos_fwd = atom_positions.copy()
            pos_bwd = atom_positions.copy()
            pos_fwd[i, j] += h
            pos_bwd[i, j] -= h
            fd_grad = (pp_matrix(pos_fwd) - pp_matrix(pos_bwd)) / (2 * h)
            if not np.allclose(fd_grad, analytic_grad[i, j], atol=1e-6, rtol=1e-6):
                print(f"{RED}PP gradient mismatch: {prefix}, atom {i}, coord {j}{RESET_COLOR}")
                all_passed = False

    if all_passed:
        print(f"{GREEN}PP gradient FD test passed: {prefix}{RESET_COLOR}")
    else:
        print(f"{RED}PP gradient FD test failed: {prefix}{RESET_COLOR}")
        sys.exit(1)


def test_frozen_core_gradient_wrt_atom_positions_fd(base_folder: str, prefix: str, active_electrons: int, active_orbitals: int, h: float = 1e-6):
    wfc, pps = _load_wfc(base_folder, prefix)

    orbital_indices_core, _orbital_indices_active = wfc.active_space(active_electrons=active_electrons, active_orbitals=active_orbitals)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        occupations_core, c_ip_core = wfc.get_orbitals_by_index(orbital_indices_core, binary_occupations=True)

    if c_ip_core.shape[0] == 0:
        print(f"{RED}Frozen core gradient FD test skipped: {prefix} has no core orbitals for this active space{RESET_COLOR}")
        return

    p = wfc.k_plus_G
    atomic_numbers = wfc.atomic_numbers
    cell_volume = wfc.cell_volume

    # Displace atoms away from equilibrium so the gradient is not near-zero.
    offset = np.array([0.1, -0.05, 0.07])
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(wfc.atom_positions_hartree.shape[0])])
    atom_positions = wfc.atom_positions_hartree + signs[:, None] * offset[None, :]

    analytic_grad = dopyqo.gradients_atom_positions.gradient_frozen_core_energy_wrt_atom_positions(
        p,
        c_ip_core,
        cell_volume,
        atom_positions,
        atomic_numbers,
        pps,
        rust_impl=False,
    )

    def frozen_core_energy(positions):
        return dopyqo.get_frozen_core_energy_pp(
            p=p,
            c_ip_core=c_ip_core,
            b=wfc.b,
            mill=wfc.mill,
            cell_volume=cell_volume,
            atom_positions=positions,
            atomic_numbers=atomic_numbers,
            occupations_core=occupations_core,
            pseudopots=pps,
            fft_grid=wfc.fft_grid,
            use_gpu=False,
            n_threads=1,
        ).real

    all_passed = True
    for i in range(atom_positions.shape[0]):
        for j in range(3):
            pos_fwd = atom_positions.copy()
            pos_bwd = atom_positions.copy()
            pos_fwd[i, j] += h
            pos_bwd[i, j] -= h
            fd_grad = (frozen_core_energy(pos_fwd) - frozen_core_energy(pos_bwd)) / (2 * h)
            if not np.isclose(fd_grad, analytic_grad[i, j], atol=1e-6, rtol=1e-6):
                print(f"{RED}Frozen core gradient mismatch: {prefix}, atom {i}, coord {j}: FD={fd_grad}, analytic={analytic_grad[i, j]}{RESET_COLOR}")
                all_passed = False

    if all_passed:
        print(f"{GREEN}Frozen core gradient FD test passed: {prefix}{RESET_COLOR}")
    else:
        print(f"{RED}Frozen core gradient FD test failed: {prefix}{RESET_COLOR}")
        sys.exit(1)


def test_total_gradient_wrt_atom_positions_fd(base_folder: str, prefix: str, active_electrons: int, active_orbitals: int, h: float = 1e-6):
    wfc, _pps = _load_wfc(base_folder, prefix)
    # Displace atoms so the gradient is not near-zero.
    offset = np.array([0.1, -0.05, 0.07])
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(wfc.atom_positions_hartree.shape[0])])
    atom_positions = wfc.atom_positions_hartree + signs[:, None] * offset[None, :]

    config = dopyqo.DopyqoConfig(
        base_folder=base_folder,
        prefix=prefix,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        run_fci=True,
        calculate_atom_gradient=True,
        atom_positions=atom_positions,
        unit=units.Unit.HARTREE,
    )
    _, _, h_ks, mats = dopyqo.run(config, return_h=True, return_wfc=False, return_mats=True, verbosity=0, show_banner=False)

    h_pqrs_zero = np.zeros(mats.h_pqrs.shape, dtype=np.float64)

    def total_energy(positions):
        config_tmp = dopyqo.DopyqoConfig(
            base_folder=base_folder,
            prefix=prefix,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            run_fci=True,
            atom_positions=positions,
            unit=units.Unit.HARTREE,
        )
        energy_dict, _, _, _ = dopyqo.run(config_tmp, return_h=False, return_wfc=False, return_mats=False, verbosity=0, show_banner=False)
        return energy_dict["fci_energy"]

    all_passed = True
    for i in range(atom_positions.shape[0]):
        for j in range(3):
            analytic_grad = (
                dopyqo.energy_from_fci_civector(mats.d_h_pq_pp[i, j], h_pqrs_zero, h_ks.fci_evcs[0], h_ks.norb, h_ks.nelec)
                + mats.d_energy_ewald_atom[i, j]
                + mats.d_energy_frozen_core_atom[i, j]
            )

            pos_fwd = atom_positions.copy()
            pos_bwd = atom_positions.copy()
            pos_fwd[i, j] += h
            pos_bwd[i, j] -= h
            fd_grad = (total_energy(pos_fwd) - total_energy(pos_bwd)) / (2 * h)

            if not np.isclose(fd_grad, analytic_grad, atol=1e-6, rtol=1e-6):
                print(f"{RED}Total gradient mismatch: {prefix}, atom {i}, coord {j}: FD={fd_grad}, analytic={analytic_grad}{RESET_COLOR}")
                all_passed = False

    if all_passed:
        print(f"{GREEN}Total gradient FD test passed: {prefix}{RESET_COLOR}")
    else:
        print(f"{RED}Total gradient FD test failed: {prefix}{RESET_COLOR}")
        sys.exit(1)


if __name__ == "__main__":
    test_ewald_gradient_wrt_atom_positions_fd(base_folder=os.path.join("qe_files", "LiH"), prefix="LiH")
    test_ewald_gradient_wrt_atom_positions_fd(base_folder=os.path.join("qe_files", "Mg"), prefix="Mg")

    test_pp_gradient_wrt_atom_positions_fd(base_folder=os.path.join("qe_files", "LiH"), prefix="LiH", active_electrons=4, active_orbitals=6)
    test_pp_gradient_wrt_atom_positions_fd(base_folder=os.path.join("qe_files", "Mg"), prefix="Mg", active_electrons=20, active_orbitals=15)

    # active_electrons=2, active_orbitals=5 results in 1 core orbital for LiH to test frozen-core derivatives.
    test_frozen_core_gradient_wrt_atom_positions_fd(base_folder=os.path.join("qe_files", "LiH"), prefix="LiH", active_electrons=2, active_orbitals=5)

    test_total_gradient_wrt_atom_positions_fd(base_folder=os.path.join("qe_files", "LiH"), prefix="LiH", active_electrons=2, active_orbitals=5)
