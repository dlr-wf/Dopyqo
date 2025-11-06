import sys
import os
import time
import logging
import warnings
import argparse
import re
import tomllib
import textwrap
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import numpy as np
import xmltodict
import termplotlib as tplt
from qiskit.quantum_info import Statevector
import dopyqo
from dopyqo.fci_vector_matrix import statevector_from_fci
from dopyqo import units
from dopyqo.scripts import banners
from dopyqo.colors import *
from dopyqo import DopyqoConfig
import dopyqo.wannier90
import dopyqo.wfc


def has_save_folder(folder: str) -> str | None:
    for item in os.listdir(None if folder == "" else folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path) and item.endswith(".save"):
            return item[:-5]
    return None


def run(
    config: DopyqoConfig | dict,
    mats: dopyqo.MatrixElements = dopyqo.MatrixElements(),
    return_h: bool = True,
    return_wfc: bool = True,
    return_mats: bool = True,
    show_banner=True,
    verbosity: int = 2,
    show_mat_calc_progress: bool = False,
) -> tuple[dict, dopyqo.Wfc | None, dopyqo.Hamiltonian | None, dopyqo.MatrixElements | None]:
    """Run a Dopyqo calculation

    Args:
        config (DopyqoConfig | dict): Dopyqo config or dictionary that can be converted into a DopyqoConfig object
        mats (dopyqo.MatrixElements, optional): dopyqo.MatrixElements object. Matrix elements that are not None are not calculate and used from this parameter instead.
                                                Defaults to dopyqo.MatrixElements().
        return_h (bool, optional): If the the created dopyqo.Hamiltonian object is returned. Defaults to True.
        return_wfc (bool, optional): If the the created dopyqo.Wftc object is returned. Defaults to True.
        return_mats (bool, optional): If the the created dopyqo.MatrixElements object is returned. Defaults to True.
        show_banner (bool, optional): If the Dopyqo banner is shown. Defaults to True.
        verbosity (int, optional): Verbosity of the printed output. Higher is more verbose. 0 is minimal prints, 1 prints the active space,
                                   2 prints summaries of everything. Defaults to 2.
        show_mat_calc_progress (bool, optional): Show the progress of the matrix element calculation using spinners and timers. Defaults to False.

    Returns:
        tuple[dict, dopyqo.Wfc | None, dopyqo.Hamiltonian | None, dopyqo.MatrixElements | None]: Dictionary of energies, and dopyqo.Wfc object,
                                                                                                 dopyqo.Hamiltonian object, dopyqo.MatrixElements object
    """
    block_color = SOFT_GREEN
    yes_color = NO_COLOR
    no_color = NO_COLOR
    verbosity_summary = 2
    verbosity_active_space = 1

    if show_mat_calc_progress:
        try:
            from rich.progress import Progress, TextColumn, TimeElapsedColumn, SpinnerColumn
        except ImportError:
            show_mat_calc_progress = False
            print(f"{ORANGE}Import warning: Could not import rich package. Matrix calculation progress will not be shown.{RESET_COLOR}")

    def make_progress():
        return Progress(
            SpinnerColumn(spinner_name="dots", style="magenta"),
            TextColumn("[bold cyan]{task.description}"),
            TimeElapsedColumn(),
            # transient=True,
        )

    start_time_all = time.perf_counter()
    ######################### BANNER #########################
    if show_banner:
        banner_attributes = vars(banners)
        banner_strings = [value for name, value in banner_attributes.items() if isinstance(value, str) and not name.startswith("_")]
        rng = np.random.default_rng()
        random_banner = rng.choice(banner_strings)
        dopyqo.print_banner(random_banner, flush=True)

    ######################### CHECK CONFIG #########################
    if isinstance(config, dict):
        config = DopyqoConfig(**config)
    if config.run_vqe:
        if config.uccsd_reps < 1:
            print(f"{RED}Config error: uccsd_reps must be greater or equal to 1 but is {config.uccsd_reps}!{RESET_COLOR}", file=sys.stderr)
            sys.exit(1)

    ######################### SETTINGS SUMMARY #########################
    if config.logging_flag:
        logging.basicConfig(level=logging.INFO)
    if verbosity >= verbosity_summary:
        print()
        dopyqo.print_block("Settings summary", color=block_color, flush=True)
        print(f"Active space of ({config.active_electrons}e, {config.active_orbitals}o)")
        if config.kpoint_idx is not None:
            if isinstance(config.kpoint_idx, int):
                print(f"k-point index:             {config.kpoint_idx}")
            else:
                print(f"k-points:                  {config.kpoint_idx}")
        if config.wannier_transform:
            msg_tmp = f"U-matrix path: {config.wannier_umat}"
            if config.wannier_input_file is not None:
                msg_tmp += f", Wannier input file: {config.wannier_input_file}"
            print(f"Using Wannier functions:   {yes_color}YES{RESET_COLOR} ({msg_tmp})")
        else:
            print(f"Using Wannier functions:   {no_color}NO{RESET_COLOR}")
        if config.run_fci:
            print(f"Running a FCI calculation: {yes_color}YES{RESET_COLOR}")
        else:
            print(f"Running a FCI calculation: {no_color}NO{RESET_COLOR}")
        if config.run_vqe:
            reps_str = f"{config.uccsd_reps} " + ("repetition" if config.uccsd_reps == 1 else "repetitions") + " of the UCCSD ansatz"
            vqe_simulator_str = "Qiskit" if config.use_qiskit else "TenCirChem"
            occ_str = f"and using occupations {config.occupations}" if config.occupations is not None else ""
            if config.vqe_parameters is not None:
                print(
                    f"Running a VQE calculation: {yes_color}YES{RESET_COLOR} by executing the ansatz with {vqe_simulator_str} with the given parameters and {reps_str} {occ_str}"
                )
            else:
                print(
                    f"Running a VQE calculation: {yes_color}YES{RESET_COLOR} with {vqe_simulator_str} using the {config.vqe_optimizer} optimizer and {reps_str} {occ_str}"
                )
        else:
            print(f"Running a VQE calculation: {no_color}NO{RESET_COLOR}")
        if config.logging_flag:
            logging.basicConfig(level=logging.INFO)
            print(f"Showing logging:           {yes_color}YES{RESET_COLOR}")
        else:
            print(f"Showing logging:           {no_color}NO{RESET_COLOR}")
        if config.n_threads > 1:
            print(f"CPU threads:               {config.n_threads}")
        sys.stdout.flush()

    binary_occupations = True  # TODO: Put this in DopyqoConfig?

    ######################### READING FILES #########################
    xml_file = os.path.join(config.base_folder, f"{config.prefix}.save", "data-file-schema.xml")
    pp_files = [
        os.path.join(config.base_folder, f"{config.prefix}.save", filename)
        for filename in os.listdir(os.path.join(config.base_folder, f"{config.prefix}.save"))
        if filename.lower().endswith(".upf")
    ]
    pps = [dopyqo.pseudopot.Pseudopot(pp_file) for pp_file in pp_files]

    logging.info("Reading data from files...")
    if config.kpoint_idx == "all":
        wfc_files = [x for x in os.listdir(os.path.join(config.base_folder, f"{config.prefix}.save")) if x.startswith("wfc")]
        wfc_files = sorted(wfc_files, key=lambda x: float(x[3:].split(".")[0]))
        dat_files = [os.path.join(config.base_folder, f"{config.prefix}.save", x) for x in wfc_files]
        wfc_obj_lst = dopyqo.Wfc.from_file_all_kpoints(dat_files, xml_file, pseudopots=pps)
    else:
        wfc_files = [x for x in os.listdir(os.path.join(config.base_folder, f"{config.prefix}.save")) if x.startswith("wfc")]
        if config.kpoint_idx is None:
            wfc_file = os.path.join(config.base_folder, f"{config.prefix}.save", wfc_files[0])
        else:
            wfc_file = os.path.join(config.base_folder, f"{config.prefix}.save", wfc_files[config.kpoint_idx])
        wfc_obj_lst = [dopyqo.Wfc.from_file(wfc_file, xml_file, pseudopots=pps, kpoint_idx=config.kpoint_idx)]
    if len(wfc_obj_lst) == 1:
        str_tmp = ""
        kpoint_tmp = wfc_obj_lst[0].kpoint
        if np.allclose(kpoint_tmp, 0.0):
            str_tmp += " (Γ-point)"
        if verbosity >= verbosity_summary:
            print(f"k-point:                   {kpoint_tmp}{str_tmp}", flush=True)
    sys.stdout.flush()

    ######################### CHANGING ATOM POSITIONS AND LATTICE VECTORS #########################
    block_title = ""
    if config.atom_positions is not None:
        block_title += "atomic positions"
        atom_positions_hartree_old = wfc_obj_lst[0].atom_positions_hartree.copy()
        for wfc_obj in wfc_obj_lst:
            wfc_obj.set_atom_positions(config.atom_positions, config.unit)
    if config.lattice_vectors is not None:
        block_title += " and lattice vectors"
        lattice_vectors_old = wfc_obj_lst[0].a.copy()
        for wfc_obj in wfc_obj_lst:
            wfc_obj.set_lattice_vectors(config.lattice_vectors, config.unit)
    if (config.atom_positions is not None or config.lattice_vectors is not None) and verbosity >= verbosity_summary:
        print()
        dopyqo.print_block(f"Changing {block_title}", color=block_color, flush=True)
        if config.atom_positions is not None:
            elements_tmp = [dopyqo.elements_to_atomic_number[x] for x in wfc_obj_lst[0].atomic_numbers]
            old_str_tmp = textwrap.indent(np.array2string(atom_positions_hartree_old), "\t")
            new_str_tmp = textwrap.indent(np.array2string(wfc_obj_lst[0].atom_positions_hartree), "\t")
            old_str_tmp_max_len = max([len(x) for x in old_str_tmp.split("\n")])
            old_str_tmp = "\n".join(
                [x + (old_str_tmp_max_len - len(x)) * " " + " <- " + elements_tmp[i] for i, x in enumerate(old_str_tmp.split("\n"))]
            )
            new_str_tmp_max_len = max([len(x) for x in new_str_tmp.split("\n")])
            new_str_tmp = "\n".join(
                [x + (new_str_tmp_max_len - len(x)) * " " + " <- " + elements_tmp[i] for i, x in enumerate(new_str_tmp.split("\n"))]
            )

            print(f"Move atomic positions (in Hartree units) from \n{old_str_tmp}\nto\n{new_str_tmp}")
        if config.lattice_vectors is not None:
            postfix_tmp = ["a1", "a2", "a3"]
            old_str_tmp = textwrap.indent(np.array2string(lattice_vectors_old), "\t")
            new_str_tmp = textwrap.indent(np.array2string(wfc_obj_lst[0].a), "\t")
            old_str_tmp_max_len = max([len(x) for x in old_str_tmp.split("\n")])
            old_str_tmp = "\n".join(
                [x + (old_str_tmp_max_len - len(x)) * " " + " <- " + postfix_tmp[i] for i, x in enumerate(old_str_tmp.split("\n"))]
            )
            new_str_tmp_max_len = max([len(x) for x in new_str_tmp.split("\n")])
            new_str_tmp = "\n".join(
                [x + (new_str_tmp_max_len - len(x)) * " " + " <- " + postfix_tmp[i] for i, x in enumerate(new_str_tmp.split("\n"))]
            )
            print(f"Change lattice vectors (in Hartree units) from \n{old_str_tmp}\nto\n{new_str_tmp}")
    sys.stdout.flush()

    ######################### RUN CALCULATION FOR EVERY K-POINT #########################
    reference_energy = wfc_obj_lst[0].etot
    mats_lst = []
    h_ks_lst = []
    vqe_result_lst = []
    hf_energy_lst = []
    hf_energy_pyscf_lst = []
    fci_energy_lst = []
    vqe_energy_lst = []
    kpoint_weights = []
    for wfc_idx, wfc_obj in enumerate(wfc_obj_lst):
        ######################### LOADING QE DATA #########################
        if len(wfc_obj_lst) > 1 and verbosity >= verbosity_summary:
            print()
            dopyqo.print_block(
                f"k-point {wfc_idx+1}/{len(wfc_obj_lst)}\n{wfc_obj.kpoint}\nweight: {wfc_obj.kpoint_weight}", color=block_color, flush=True
            )
        overlaps_ncpp = wfc_obj.get_overlaps()
        if not np.allclose(overlaps_ncpp, np.eye(overlaps_ncpp.shape[0])):
            print(
                f"{RED}Wavefunction error: The wavefunctions are not orthonormal!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)

        p = wfc_obj.k_plus_G  # shape (#waves, 3)
        c_ip = wfc_obj.evc  # shape (#bands, #waves)

        logging.info("Number of plane waves: %i", p.shape[0])

        orbital_indices_core, orbital_indices_active = wfc_obj.active_space(
            active_electrons=config.active_electrons, active_orbitals=config.active_orbitals
        )

        occs_tmp = wfc_obj.occupations_binary if binary_occupations else wfc_obj.occupations
        occ_str = dopyqo.active_space_string(occs_tmp, orbital_indices_active)

        if verbosity >= verbosity_active_space:
            print(f"The following active space will be considered:\n{occ_str}")
            print(f"Number of plane waves:            {p.shape[0]}")
            sys.stdout.flush()

        occupations_full, c_ip_full = wfc_obj.get_orbitals_by_index(
            orbital_indices_core + orbital_indices_active, binary_occupations=binary_occupations
        )
        occupations_active, c_ip_active = wfc_obj.get_orbitals_by_index(orbital_indices_active, binary_occupations=binary_occupations)
        occupations_active_xml, _ = wfc_obj.get_orbitals_by_index(orbital_indices_active, binary_occupations=False)
        with warnings.catch_warnings():  # Ignore warning that all core orbitals are occupied
            warnings.filterwarnings("ignore", category=Warning)
            occupations_core, c_ip_core = wfc_obj.get_orbitals_by_index(orbital_indices_core, binary_occupations=binary_occupations)

        # Calculate matrix elements
        my_context = nullcontext()
        if show_mat_calc_progress:
            my_context = make_progress()
        with my_context as progress:
            if show_mat_calc_progress:
                task = progress.add_task("Calculating kinetic energy", total=None)
            else:
                print("Calculating kinetic energy...", flush=True)
            start_time_ham = time.perf_counter()
            ######################### KINETIC ENERGY #########################
            iTj_orbitals = mats.h_pq_kin
            if iTj_orbitals is None:
                logging.info("Calculate kinetic energy...")
                start_time = time.perf_counter()
                iTj_orbitals = dopyqo.calc_matrix_elements.iTj(p, c_ip_active)
                time_kin = time.perf_counter() - start_time
                time_kin_str = f"{time_kin:.2f}s"
                logging.info("Kinetic energy calc. took %.2fs", time_kin)
            else:
                logging.info("Loading kinetic energy...")
                time_kin_str = "- Given by " + f"{mats.h_pq_kin=}".split("=")[0] + " parameter"

        ######################### PSEUDOPOTENTIAL #########################
        # Progress bar does not update because of blocking Rust function execution,
        #   need to put rust function in separate thread and properly catch the return value :$
        my_context = nullcontext()
        if show_mat_calc_progress:
            my_context = make_progress()
        with my_context as progress:
            if show_mat_calc_progress:
                task = progress.add_task("Calculating pseudopotential", total=None)
            else:
                print("Calculating pseudopotential...", flush=True)
            pp_orbitals = mats.h_pq_pp
            pp_core = None
            if pp_orbitals is None:
                # print("Calculating pseudopotential ", end="", flush=True)
                n_threads_pp = config.n_threads
                start_time = time.perf_counter()
                pp_full = dopyqo.calc_pseudo_pot.calc_pps(
                    p,
                    c_ip_full,
                    wfc_obj.cell_volume,
                    wfc_obj.atom_positions_hartree,
                    wfc_obj.atomic_numbers,
                    pps,
                    n_threads=n_threads_pp,
                    # os.path.join(config.base_folder, "pp_pw.npy"),
                )
                time_pp = time.perf_counter() - start_time
                time_pp_str = f"{time_pp:.2f}s"
                pp_orbitals = pp_full[orbital_indices_active, :][:, orbital_indices_active]
                pp_core = pp_full[orbital_indices_core, :][:, orbital_indices_core]
                hours = int(time_pp // 3600)
                minutes = int(time_pp // 60 - hours * 60)
                seconds = int(time_pp - minutes * 60 - hours * 3600)
                # print(f"{hours:01d}:{minutes:02d}:{seconds:02d}")
                # print(time_pp_str, flush=True)
            else:
                logging.info("Loading pseudopotentials...")
                time_pp_str = "- Given by " + f"{mats.h_pq_pp=}".split("=")[0] + " parameter"

        ######################### ELECTRON REPULSION INTEGRALS #########################
        my_context = nullcontext()
        if show_mat_calc_progress:
            my_context = make_progress()
        with my_context as progress:
            if show_mat_calc_progress:
                task = progress.add_task("Calculating electron repulsion integrals", total=None)
            else:
                print("Calculating electron repulsion integrals...", flush=True)
            h_pqrs = mats.h_pqrs
            if h_pqrs is None:

                logging.info("Calculate ERIs...")
                start_time = time.perf_counter()
                h_pqrs: np.ndarray = (
                    dopyqo.eri(
                        c_ip=c_ip_active,
                        b=wfc_obj.b,
                        mill=wfc_obj.mill,
                        fft_grid=wfc_obj.fft_grid,
                        use_gpu=config.use_gpu,
                    )
                    / wfc_obj.cell_volume
                )

                time_eri = time.perf_counter() - start_time
                time_eri_str = f"{time_eri:.2f}s"
                logging.info("ERI calc. took %.2fs", time_eri)
            else:
                logging.info("Loading ERIs...")
                time_eri_str = "- Given by " + f"{mats.h_pqrs=}".split("=")[0] + " parameter"

        ######################### FROZEN CORE POTENTIAL AND ENERGY #########################
        my_context = nullcontext()
        if show_mat_calc_progress:
            my_context = make_progress()
        with my_context as progress:
            if show_mat_calc_progress:
                task = progress.add_task("Calculating frozen core", total=None)
            else:
                print("Calculating frozen core...", flush=True)
            h_pq_core = np.zeros_like(iTj_orbitals)
            energy_frozen_core = 0.0
            if len(c_ip_core) > 0:
                h_pq_core = mats.h_pq_core
                energy_frozen_core = mats.energy_frozen_core
                if h_pq_core is None and energy_frozen_core is None:
                    logging.info("Calculate frozen core potential and frozen core energy...")
                    start_time = time.perf_counter()
                    h_pq_core, energy_frozen_core = dopyqo.get_frozen_core_pot_and_energy_given_pp(
                        p=p,
                        c_ip_core=c_ip_core,
                        c_ip_active=c_ip_active,
                        b=wfc_obj.b,
                        mill=wfc_obj.mill,
                        cell_volume=wfc_obj.cell_volume,
                        pp_core=pp_core,
                        occupations_core=occupations_core,
                        fft_grid=wfc_obj.fft_grid,
                        use_gpu=config.use_gpu,
                    )
                    energy_frozen_core = energy_frozen_core.real
                    time_energy_core = time.perf_counter() - start_time
                    time_energy_core_str = f"{time_energy_core:.2f}s"
                    logging.info("Frozen core potential and energy calc. took %.2fs", time_energy_core)
                elif h_pq_core is None:
                    logging.info("Calculate frozen core potential...")
                    start_time = time.perf_counter()
                    h_pq_core = dopyqo.get_frozen_core_pot(
                        c_ip_core=c_ip_core,
                        c_ip_active=c_ip_active,
                        b=wfc_obj.b,
                        mill=wfc_obj.mill,
                        fft_grid=wfc_obj.fft_grid,
                        cell_volume=wfc_obj.cell_volume,
                        use_gpu=config.use_gpu,
                    )
                    time_energy_core = time.perf_counter() - start_time
                    time_energy_core_str = (
                        f"{time_energy_core:.2f}s (potential) - Energy given by " + f"{mats.energy_frozen_core=}".split("=")[0] + " parameter"
                    )
                    logging.info("Frozen core potential and energy calc. took %.2fs", time_energy_core)
                elif energy_frozen_core is None:
                    logging.info("Calculate frozen core energy...")
                    start_time = time.perf_counter()
                    energy_frozen_core = dopyqo.get_frozen_core_energy_pp(
                        p=p,
                        c_ip_core=c_ip_core,
                        b=wfc_obj.b,
                        mill=wfc_obj.mill,
                        cell_volume=wfc_obj.cell_volume,
                        atom_positions=wfc_obj.atom_positions_hartree,
                        atomic_numbers=wfc_obj.atomic_numbers,
                        occupations_core=occupations_core,
                        pseudopots=pps,
                        fft_grid=wfc_obj.fft_grid,
                        use_gpu=config.use_gpu,
                    ).real
                    time_energy_core = time.perf_counter() - start_time
                    time_energy_core_str = (
                        f"{time_energy_core:.2f}s (energy) - Potential given by " + f"{mats.h_pq_core=}".split("=")[0] + " parameter"
                    )
                else:
                    logging.info("Loading frozen core...")
                    time_energy_core_str = (
                        "- Given by " + f"{mats.h_pq_core=}".split("=")[0] + " and " + f"{mats.energy_frozen_core=}".split("=")[0] + " parameters"
                    )

        ######################### NUCLEAR REPULSION #########################
        my_context = nullcontext()
        if show_mat_calc_progress:
            my_context = make_progress()
        with my_context as progress:
            if show_mat_calc_progress:
                task = progress.add_task("Calculating ewald energy", total=None)
            else:
                print("Calculating ewald energy...", flush=True)
            h_pq_ewald = mats.h_pq_ewald
            energy_ewald = mats.energy_ewald
            if energy_ewald is None:
                if wfc_obj.ewald != 0.0 and config.qe_ewald:
                    logging.info("Using Ewald energy from QE (%s)...", wfc_obj.ewald)
                    overlap = np.einsum("ij, kj -> ik", c_ip_active.conj(), c_ip_active)
                    h_pq_ewald = overlap * wfc_obj.ewald
                else:
                    logging.info("Calculate nuclear repulsion...")
                    start_time = time.perf_counter()
                    lattice_vectors = np.array([wfc_obj.a1, wfc_obj.a2, wfc_obj.a3])
                    lattice_vectors_reciprocal = np.array([wfc_obj.b1, wfc_obj.b2, wfc_obj.b3])
                    energy_ewald = dopyqo.nuclear_repulsion_energy_ewald(
                        wfc_obj.atom_positions_hartree,
                        wfc_obj.atomic_numbers_valence,
                        lattice_vectors,
                        lattice_vectors_reciprocal,
                        wfc_obj.cell_volume,
                        gcut=wfc_obj.gcutrho,
                    )
                    time_nucl_rep = time.perf_counter() - start_time
                    time_nucl_rep_str = f"{time_nucl_rep:.2f}s"
                    logging.info("Nuclear repulsion calc. took %.2fs", time_nucl_rep)
                    logging.info(
                        "Ewald energy: %s (QE: %s, Diff.: %s)",
                        energy_ewald,
                        wfc_obj.ewald,
                        np.abs(energy_ewald - wfc_obj.ewald),
                    )
                    if config.atom_positions is None and config.lattice_vectors is None and not np.isclose(energy_ewald, wfc_obj.ewald):
                        print(
                            f"{ORANGE}Ewald warning: Calculated Ewald energy is not close to the QE Ewald energy (QE: {wfc_obj.ewald}, Dopyqo: {energy_ewald.real}, Diff. {np.abs(energy_ewald-wfc_obj.ewald)})!{RESET_COLOR}"
                        )
                    overlap = np.einsum("ij, kj -> ik", c_ip_active.conj(), c_ip_active)
                    h_pq_ewald = overlap * energy_ewald

                # If nucl. rep. energy (E_nuc) is a matrix it needs to be rescaled
                # Imagine E_nuc is a number and we want to build a matrix from that.
                # A diagonal matrix with E_nuc on its diagonal will lead to the wrong energy
                # contribution. The following rescales it correctly.
                if isinstance(h_pq_ewald, np.ndarray):
                    h_pq_ewald = h_pq_ewald / (2 * np.sum(occupations_active))
            else:
                overlap = np.einsum("ij, kj -> ik", c_ip_active.conj(), c_ip_active)
                h_pq_ewald = overlap * energy_ewald
                if config.atom_positions is not None or config.lattice_vectors is not None:
                    print(
                        f"{ORANGE}Ewald warning: Ewald matrix elements are given by "
                        + f"{mats.h_pq_ewald=}".split("=")[0]
                        + f" parameter but atom positions or lattice vectors were changed! Are you sure you want to do this?{RESET_COLOR}"
                    )
                logging.info("Loading nuclear repulsion...")
                time_nucl_rep_str = "- Given by " + f"{mats.energy_ewald=}".split("=")[0] + " parameter"

        h_pq = iTj_orbitals + pp_orbitals + h_pq_core  # + h_pq_ewald

        ######################### Electron self-energy #########################
        # Source: Fraser et al. 1996, "Finite-size effects and Coulomb interactions
        #                              in quantum Monte Carlo calculations for homogeneous
        #                              systems with periodic boundary conditions",
        #       Equations (19, 28). Basically describing the interaction energy, one electron has
        #       with itself in neighbouring cells.
        # \sum_{T \neq 0} 1/|T|
        energy_e_self = mats.energy_e_self
        if energy_e_self is None:
            print("Calculating electron self-energy...", flush=True)
            start_time = time.perf_counter()
            energy_e_self = wfc_obj.nelec * dopyqo.nuclear_repulsion_energy_ewald(
                atom_positions=np.array([[0.0, 0.0, 0.0]]),
                atomic_numbers=np.array([1]),
                lattice_vectors=np.array([wfc_obj.a1, wfc_obj.a2, wfc_obj.a3]),
                lattice_vectors_reciprocal=np.array([wfc_obj.b1, wfc_obj.b2, wfc_obj.b3]),
                cell_volume=wfc_obj.cell_volume,
                gcut=wfc_obj.gcutrho,
            )
            time_e_self = time.perf_counter() - start_time
            time_e_self_str = f"{time_e_self:.2f}s"
        else:
            logging.info("Loading electron self-energy...")
            time_e_self_str = "- Given by " + f"{mats.energy_e_self=}".split("=")[0] + " parameter"

        time_ham = time.perf_counter() - start_time_ham
        logging.info("Calculation of all matrix elements finished. Took %.2fs", time_ham)

        if verbosity >= verbosity_summary:
            print()
            dopyqo.print_block("Hamiltonian calculation summary", color=block_color)
            print(f"Number of plane waves:            {p.shape[0]}")
            print(f"Kinetic energy:                   {time_kin_str}")
            print(f"Pseudopotentials:                 {time_pp_str}")
            print(f"ERIs:                             {time_eri_str}")
            if len(c_ip_core) > 0:
                print(f"Frozen core potential and energy: {time_energy_core_str}")
            if not (wfc_obj.ewald != 0.0 and config.qe_ewald):
                print(f"Nuclear repulsion:                {time_nucl_rep_str}")
            print(f"Electron self-energy:             {time_e_self_str}")
            print(f"Total:                            {time_ham:.2f}s")
            # print(f"Matrix elements saved to {filename}")
            sys.stdout.flush()

        ######################### WANNIER TRANSFORMATION #########################
        if config.wannier_transform:
            # Read Wannier90 transform matrix
            transform_matrix = dopyqo.read_u_mat(config.wannier_umat)[tuple(wfc_obj.kpoint)]
            if not dopyqo.wannier90.check_unitarity_u(transform_matrix):
                print(f"{RED}Wannier error: Transformation matrix is not unitary{RESET_COLOR}")
                sys.exit(1)
            if config.active_orbitals != transform_matrix.shape[0]:
                print(
                    f"{RED}Wannier error: Number of active orbitals ({config.active_orbitals}) "
                    + f"does not match number of Wannier orbitals ({transform_matrix.shape[0]})!{RESET_COLOR}"
                )
                sys.exit(1)
            ######################### READING WANNIER INPUT FILE #########################
            if config.wannier_input_file is not None:
                results = {}
                keys = ["num_wann", "exclude_bands"]

                with open(config.wannier_input_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()

                        for key in keys:
                            if line.startswith(key):
                                if "=" in line:
                                    _, sep, val = line.partition("=")
                                elif ":" in line:
                                    _, sep, val = line.partition(":")
                                else:  # whitespace split
                                    parts = line.split(None, 1)
                                    val = parts[1] if len(parts) == 2 else None
                                results[key] = val.strip()
                                break

                num_wann = int(results["num_wann"])
                exclude_bands = []
                if "exclude_bands" in results:
                    for x in results["exclude_bands"].split(","):
                        if "-" not in x:
                            exclude_bands.append(int(x))
                        else:
                            start_stop = [int(val) for val in x.split("-")]
                            start = start_stop[0]
                            stop = start_stop[1] + 1
                            exclude_bands.extend(list(range(start, stop)))
                    exclude_bands = [x - 1 for x in exclude_bands]  # Wannier90 1-indexing to python 0-indexing
                orbital_indices_active_wannier = list(set(exclude_bands) ^ set(range(wfc_obj.nbnd)))
                if orbital_indices_active != orbital_indices_active_wannier:
                    print(
                        f"{RED}Wannier error: Orbitals in the active space ({orbital_indices_active}) are not the "
                        + f"same orbitals transformed with Wannier90 ({orbital_indices_active_wannier})!{RESET_COLOR}"
                    )
                    sys.exit(1)

            ######################### CALCULATE OCCUPATIONS AND TRANSFORM MATRICES #########################
            # density_mat = dopyqo.to_density_matrix(config.active_electrons, config.active_orbitals) / 2.0
            density_mat = np.diag(occupations_active_xml)
            # density_mat = np.diag(occupations_active)

            wf_density_matrix = transform_matrix.T.conj() @ density_mat @ transform_matrix
            if not np.allclose(wf_density_matrix, wf_density_matrix.real):
                print(f"{ORANGE}Wannier warning: Calculated density matrix in the Wannier basis is not real!{RESET_COLOR}")
            else:
                wf_density_matrix = wf_density_matrix.real

            occupations_new = wf_density_matrix.real.diagonal()
            make_binary = False
            for occ in occupations_new:
                if (not np.isclose(occ, 0.0)) and (not np.isclose(occ, 1.0)):
                    make_binary = True
                    break
            occupations_new_binary = occupations_new.copy().round().astype(np.int64)
            if make_binary:
                occupations_new_binary = np.concatenate(
                    (
                        np.ones((round(np.sum(occupations_new)),), dtype=np.int64),
                        np.zeros(
                            (len(occupations_new) - round(np.sum(occupations_new)),),
                            dtype=np.int64,
                        ),
                    ),
                    dtype=np.int64,
                )
            if not np.isclose(sum(occupations_new) * 2, config.active_electrons) or not np.isclose(
                sum(occupations_new_binary) * 2, config.active_electrons
            ):
                print(
                    f"{RED}Wannier error: Sum of Wannier occupations ({sum(occupations_new) * 2} or {sum(occupations_new_binary) * 2}) does not match number of active electrons ({config.active_electrons}). "
                    + f"This should not happen. Please, contact a developer!{RESET_COLOR}"
                )
                sys.exit(1)

            overlaps_wannier = np.einsum("mq, np, nm -> pq", transform_matrix, transform_matrix.conj(), wfc_obj.get_overlaps(orbital_indices_active))
            if not np.allclose(overlaps_wannier, np.eye(overlaps_wannier.shape[0])):
                print(f"{ORANGE}Wannier warning: The wannier wavefunctions are not orthonormal!{RESET_COLOR}")

            # Calculate Hamiltonian and eigenenergies in Wannier basis
            ks_energies = wfc_obj.ks_energies[orbital_indices_active]
            wannier_h_mat = np.einsum(
                "mp, mq, m -> pq",
                transform_matrix.conj(),
                transform_matrix,
                ks_energies,
            )
            wannier_energies, wannier_eigvecs = np.linalg.eig(wannier_h_mat)
            if not np.allclose(np.sort(wannier_energies), np.sort(ks_energies)):
                print(f"{ORANGE}Wannier warning: The wannier energies do not coincide with the KS-energies!{RESET_COLOR}")
            d = np.linalg.inv(wannier_eigvecs) @ wannier_h_mat @ wannier_eigvecs
            if not np.allclose(d.diagonal(), wannier_energies):
                print(
                    f"{RED}Wannier error: Eigenvectors of DFT hamiltonian in Wannier basis do not "
                    + f"diagonalize DFT hamiltonian in Wannier basis! This should never happen. Please, contact a developer{RESET_COLOR}"
                )
                sys.exit(1)

            h_pq_wannier = dopyqo.transform_one_body_matrix(h_pq, transform_matrix)
            h_pqrs_wannier = dopyqo.transform_two_body_matrix(h_pqrs, transform_matrix)

            # h_pq_ks = h_pq.copy()
            # h_pqrs_ks = h_pqrs.copy()

            h_pq = h_pq_wannier
            h_pqrs = h_pqrs_wannier

            if verbosity >= verbosity_summary:
                print()
                dopyqo.print_block("Wannier transformation summary", color=block_color, flush=True)
                print(f"Kohn-Sham active-space occupations: {occupations_active}")
                print(f"Kohn-Sham xml occupations:          {occupations_active_xml}")
                str_tmp = "Kohn-Sham density matrix:           "
                ks_density_matrix_str = np.array2string(
                    density_mat, prefix=str_tmp, max_line_width=1000, precision=3, suppress_small=True, separator=" "
                )
                print(str_tmp + ks_density_matrix_str)
                str_tmp = (
                    ""
                    if not make_binary
                    else f" (non-binary: {np.array2string(occupations_new, max_line_width=1000, precision=3, suppress_small=True, separator=' ')})"
                )
                print(f"Wannier occupations:                {occupations_new_binary}{str_tmp}")
                str_tmp = "Wannier density matrix:             "
                wf_density_matrix_str = np.array2string(
                    wf_density_matrix, prefix=str_tmp, max_line_width=1000, precision=3, suppress_small=True, separator=" "
                )
                print(str_tmp + wf_density_matrix_str)
                sys.stdout.flush()
            # IDEA: Wannier VQE initialization based on density matrix using approriate single- and double exications?

            occupations_active = np.array(occupations_new_binary)

        if config.occupations is not None:
            print(f"Using custom occupations:           {config.occupations}")
            occupations_active = config.occupations

        ######################### SAVE MATRIX ELEMENTS #########################
        filename = f"matrix_elements_ks_{config.active_electrons}e_{config.active_orbitals}o_kpoint{wfc_obj.kpoint}.npz"
        # TODO: Save MatrixElements and Wfc object together with energy_dict instead of saving indiviual matrices separately. Do it like this:
        # filename = f"energydict_wfcobj_mats_{config.active_electrons}e_{config.active_orbitals}o.npz"
        # np.savez(
        #     os.path.join(config.base_folder, filename),
        #     active_electrons=config.active_electrons,
        #     active_orbitals=config.active_orbitals,
        #     energy_dict=energy_dict,
        #     wfc_obj_ks=wfc_obj,
        #     mats=mats,
        # )
        np.savez(
            os.path.join(config.base_folder, filename),
            active_electrons=config.active_electrons,
            active_orbitals=config.active_orbitals,
            h_pq_kin=iTj_orbitals,
            h_pq_pp=pp_orbitals,
            h_pq_core=h_pq_core,
            h_pqrs=h_pqrs,
            h_pq_ewald=h_pq_ewald,
            energy_frozen_core=energy_frozen_core,
            energy_ewald=energy_ewald,
            energy_e_self=energy_e_self,
            transform_matrix=None if not config.wannier_transform else transform_matrix,
            mats=dopyqo.MatrixElements(
                h_pq_kin=iTj_orbitals,
                h_pq_pp=pp_orbitals,
                h_pq_core=h_pq_core,
                h_pqrs=h_pqrs,
                h_pq_ewald=h_pq_ewald,
                energy_frozen_core=energy_frozen_core,
                energy_ewald=energy_ewald,
                energy_e_self=energy_e_self,
                transform_matrix=None if not config.wannier_transform else transform_matrix,
            ),
        )

        ######################### CREATE HAMILTONIAN OBJECT #########################
        h_ks = dopyqo.Hamiltonian(
            h_pq.copy(),
            h_pqrs.copy(),
            occupations_active.copy(),
            reference_energy=reference_energy,
            constants=energy_frozen_core + energy_e_self + energy_ewald,
        )

        ######################### RUN PySCF HF CALCULATION #########################
        if config.run_hf:
            hf_pyscf_energy = h_ks.solve_hf()
            mo_coeff = h_ks.mo_coeff
            #
            h_ks = dopyqo.Hamiltonian(
                dopyqo.transform_one_body_matrix(h_pq.copy(), mo_coeff),
                dopyqo.transform_two_body_matrix(h_pqrs.copy(), mo_coeff),
                occupations_active.copy(),
                reference_energy=reference_energy,
                constants=energy_frozen_core + energy_e_self + energy_ewald,
            )

        ######################### RUN FCI CALCULATION #########################
        fci_state = Statevector(np.array([]))
        if config.run_fci:
            if verbosity >= verbosity_summary:
                print()
                dopyqo.print_block("Running FCI", color=block_color, flush=True)
            start_time_fci = time.perf_counter()

            if wfc_obj.gamma_only:
                fci_energy = h_ks.solve_fci_spin1(n_energies=config.n_fci_energies).real
                if config.n_fci_energies == 1:
                    fci_energy = fci_energy[0]
                fci_state = h_ks.fci_statevector()
            else:
                fci_energy = h_ks.solve_fci(n_energies=config.n_fci_energies).real
                if config.n_fci_energies == 1:
                    fci_energy = fci_energy[0]

            time_fci = time.perf_counter() - start_time_fci
            h_ks.reference_energy = fci_energy
            print(f"FCI energy: {fci_energy}")

            if verbosity >= verbosity_summary:
                print(f"\tTook {time_fci:.2f}s")
            # TODO: Generalize fci_statevector to support statevector generated from solve_fci_general
            if wfc_obj.gamma_only:
                fci_state = h_ks.fci_statevector()
            else:
                print(
                    f"{ORANGE}FCI warning: The FCI statevector will not be saved. The FCI statevector cannot be calculated for a non-gamma-only calculation, yet!{RESET_COLOR}"
                )

        ######################### RUN VQE CALCULATION #########################
        vqe_state = Statevector(np.array([]))
        if config.run_vqe:
            ######################### VQE #########################
            if not config.vqe_adapt:
                if config.vqe_parameters is not None:
                    if verbosity >= verbosity_summary:
                        print()
                        dopyqo.print_block("Executing VQE circuit with given parameters", color=block_color, flush=True)
                    start_time_vqe = time.perf_counter()
                    if config.use_qiskit:
                        vqe_energy = h_ks.run_qiskit_ansatz(config.vqe_parameters, config.uccsd_reps)
                    else:
                        vqe_energy = h_ks.run_tcc_ansatz(
                            config.vqe_parameters, config.uccsd_reps, excitations=config.vqe_excitations, qiskit_equivalent=False
                        )
                    time_vqe = time.perf_counter() - start_time_vqe
                    if verbosity >= verbosity_summary:
                        print(f"\nVQE energy: {vqe_energy}")
                        print(f"\tTook {time_vqe:.2f}s")
                else:
                    if verbosity >= verbosity_summary:
                        print()
                        dopyqo.print_block("Running VQE", color=block_color, flush=True)
                    if config.use_qiskit:  # Qiskit
                        if config.vqe_initial_parameters is not None:
                            print(
                                f"{RED}Config error: Setting vqe_initial_parameters is not supported when using qiskit!{RESET_COLOR}", file=sys.stderr
                            )
                            sys.exit(1)
                        start_time_vqe = time.perf_counter()
                        h_ks.solve_vqe_qiskit(UCCSD_reps=config.uccsd_reps, optimizer=config.vqe_optimizer)
                        time_vqe = time.perf_counter() - start_time_vqe
                    else:  # TenCirChem
                        optimizer = config.vqe_optimizer
                        start_time_vqe = time.perf_counter()
                        h_ks.solve_vqe(
                            UCCSD_reps=config.uccsd_reps,
                            optimizer=optimizer,
                            initial_params=config.vqe_initial_parameters,
                            maxiter=config.vqe_maxiter,
                            excitations=config.vqe_excitations,
                        )
                        time_vqe = time.perf_counter() - start_time_vqe
                    vqe_energy = h_ks.vqe_values[-1]

                    if verbosity >= verbosity_summary:
                        print(f"\nVQE energy: {vqe_energy}")
                        print(f"\tTook {time_vqe:.2f}s")
                        print(f"VQE Result:\n{h_ks.vqe_result}")

                    vqe_state = h_ks.vqe_statevector_qiskit() if config.use_qiskit else h_ks.vqe_statevector()
                    if verbosity >= verbosity_summary:
                        print(f"VQE Statevector: {vqe_state.draw('latex_source')}")

                    if verbosity >= verbosity_summary:
                        fig = tplt.figure()
                        if config.run_fci:
                            print()
                            dopyqo.print_block("VQE convergence plot", color=block_color, flush=True)
                            print("Y-axis shows the absolute difference to the FCI energy and is logarithmic (x -> 10^x).")
                            abs_diff = np.abs(fci_energy - h_ks.vqe_values)
                            if np.any(abs_diff < 1e-15):
                                abs_diff[abs_diff < 1e-15] = 1e-15  # np.log10 raises "divide by zero" error if values are 0.0
                            fig.plot(
                                h_ks.vqe_counts,
                                np.log10(abs_diff),
                            )
                        else:
                            print("VQE convergence plot below. Y-axis shows VQE energy.")
                            fig.plot(
                                h_ks.vqe_counts,
                                h_ks.vqe_values,
                            )
                        fig.show()
            ######################### ADAPT-VQE #########################
            else:
                if verbosity >= verbosity_summary:
                    print()
                    dopyqo.print_block("Running VQE", color=block_color, flush=True)
                if config.use_qiskit:
                    print(
                        f"{RED}Dopyqo error: Both "
                        + f"{config.vqe_adapt=}".split("=")[0]
                        + f" and "
                        + f"{config.use_qiskit=}".split("=")[0]
                        + f" are set to True. ADAPT-VQE is not supported when using Qiskit!{RESET_COLOR}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                else:
                    optimizer = config.vqe_optimizer
                    start_time_vqe = time.perf_counter()
                    h_ks.solve_vqe_adapt(
                        optimizer=optimizer,
                        maxiter=config.vqe_maxiter,
                        excitation_pool=config.vqe_excitations,
                        drain_pool=config.vqe_adapt_drain_pool,
                    )
                    time_vqe = time.perf_counter() - start_time_vqe
                vqe_energy = h_ks.vqe_values[-1]

                if verbosity >= verbosity_summary:
                    print(f"\nVQE energy: {vqe_energy}")
                    print(f"\tTook {time_vqe:.2f}s")
                    print(f"VQE Result:\n{h_ks.vqe_result}")

                if config.use_qiskit:
                    print(f"{RED}Qiskit cannot be used with ADAPT-VQE, yet. Cannot calculate statevector!{RESET_COLOR}")
                    sys.exit(1)
                # vqe_state = h_ks.vqe_statevector_qiskit() if config.use_qiskit else h_ks.vqe_statevector()
                vqe_state = h_ks.vqe_statevector()
                if verbosity >= verbosity_summary:
                    print(f"VQE Statevector: {vqe_state.draw('latex_source')}")

                if verbosity >= verbosity_summary:
                    fig = tplt.figure()
                    if config.run_fci:
                        print()
                        dopyqo.print_block("VQE convergence plot", color=block_color, flush=True)
                        print("Y-axis shows the absolute difference to the FCI energy and is logarithmic (x -> 10^x).")
                        abs_diff = np.abs(fci_energy - h_ks.vqe_values)
                        if np.any(abs_diff < 1e-15):
                            abs_diff[abs_diff < 1e-15] = 1e-15  # np.log10 raises "divide by zero" error if values are 0.0
                        fig.plot(
                            h_ks.vqe_counts,
                            np.log10(abs_diff),
                        )
                    else:
                        print("VQE convergence plot below. Y-axis shows VQE energy.")
                        fig.plot(
                            h_ks.vqe_counts,
                            h_ks.vqe_values,
                        )
                    fig.show()

        ######################### SAVE STATEVECTORS AND APPENDING RESULTS #########################
        if config.run_fci or config.run_vqe:
            logging.info("Saving statevectors...")
            filename = f"statevecs_{config.active_electrons}e_{config.active_orbitals}o.npz"
            np.savez(os.path.join(config.base_folder, filename), fci_state_data=fci_state.data, vqe_state_data=vqe_state.data)
            if verbosity >= verbosity_summary:
                print(f"Statevectors saved to {filename}")

        # IDEA: If error occurs after matrix calculation, do not exit with sys.exit(1) but stop and return
        #       everything, i.e., wfc objects, matrix elements for every processed k-point
        #       return success flag to signal if error occured?
        hf_energy_lst.append(h_ks.hf_energy())
        if config.run_hf:
            hf_energy_pyscf_lst.append(hf_pyscf_energy)
        if config.run_fci:
            fci_energy_lst.append(fci_energy)
        if config.run_vqe:
            vqe_energy_lst.append(vqe_energy)
        if return_h:
            h_ks_lst.append(h_ks)
        vqe_result_lst.append(h_ks.vqe_result)
        if return_mats:
            mats_lst.append(
                dopyqo.MatrixElements(
                    h_pq_kin=iTj_orbitals,
                    h_pq_pp=pp_orbitals,
                    h_pq_core=h_pq_core,
                    h_pqrs=h_pqrs,
                    h_pq_ewald=h_pq_ewald,
                    energy_frozen_core=energy_frozen_core,
                    energy_ewald=energy_ewald,
                    energy_e_self=energy_e_self,
                    transform_matrix=None if not config.wannier_transform else transform_matrix,
                )
            )

        kpoint_weights.append(wfc_obj.kpoint_weight)

    if verbosity >= verbosity_summary:
        print()
        msg_tmp = "Energy summary"
        if len(wfc_obj_lst) > 1:
            msg_tmp += f"\n(weighted over {len(wfc_obj_lst)} k-points)"
        dopyqo.print_block(msg_tmp, color=block_color, flush=True)
        print(f"DFT energy:        {wfc_obj_lst[0].etot}")

    ######################### WEIGHT ENERGIES #########################
    # Weighted HF energies
    hf_energy = 0.0
    for hf_nrg, weight in zip(hf_energy_lst, kpoint_weights):
        hf_energy += weight * hf_nrg.real
    hf_energy /= sum(kpoint_weights)
    hf_energy_pyscf = 0.0
    if config.run_hf:
        for hf_nrg, weight in zip(hf_energy_pyscf_lst, kpoint_weights):
            hf_energy_pyscf += weight * hf_nrg.real
        hf_energy_pyscf /= sum(kpoint_weights)
        if verbosity >= verbosity_summary:
            print(f"HF energy (PySCF): {hf_energy_pyscf}")
    if verbosity >= verbosity_summary:
        print(f'"HF" energy:       {hf_energy}')
    energy_dict = {"dft_energy": wfc_obj_lst[0].etot, "hf_energy": hf_energy, "hf_energy_pyscf": hf_energy_pyscf}

    if len(wfc_obj_lst) > 1:
        energy_dict["hf_energy_per_kpoint"] = hf_energy_lst
        energy_dict["hf_energy_pyscf_per_kpoint"] = hf_energy_pyscf_lst
    if config.run_fci:
        # Weighted FCI energies
        fci_energy = 0.0
        for fci_nrg, weight in zip(fci_energy_lst, kpoint_weights):
            fci_energy += weight * fci_nrg.real
        fci_energy /= sum(kpoint_weights)
        if verbosity >= verbosity_summary:
            print(f"FCI energy:        {fci_energy}")
            print(f"\tTook {time_fci:.2f}s")  # TODO: sum times for multiple kpoints
        energy_dict["fci_energy"] = fci_energy
        if len(wfc_obj_lst) > 1:
            energy_dict["fci_energy_per_kpoint"] = fci_energy_lst
    if config.run_vqe:
        # Weighted VQE energies
        vqe_energy = 0.0
        for vqe_nrg, weight in zip(vqe_energy_lst, kpoint_weights):
            vqe_energy += weight * vqe_nrg.real
        vqe_energy /= sum(kpoint_weights)
        energy_dict["vqe_energy"] = vqe_energy
        if len(wfc_obj_lst) > 1:
            energy_dict["vqe_energy_per_kpoint"] = vqe_energy_lst
        if config.vqe_parameters is None:
            if len(wfc_obj_lst) > 1:
                energy_dict["vqe_result"] = vqe_result_lst
            else:
                energy_dict["vqe_result"] = vqe_result_lst[0]
        if verbosity >= verbosity_summary:
            if config.run_fci:
                print(f"VQE energy: {vqe_energy} | Diff. to FCI: {np.abs(vqe_energy-fci_energy)}")
            else:
                print(f"VQE energy: {vqe_energy}")
            print(f"\tTook {time_vqe:.2f}s")
    sys.stdout.flush()

    filename = f"energies_{config.active_electrons}e_{config.active_orbitals}o.npz"
    np.savez(os.path.join(config.base_folder, filename), **energy_dict)
    if verbosity >= verbosity_summary:
        print(f"Energies saved to {filename}")

    time_all = time.perf_counter() - start_time_all
    print(f"{GREEN}All done in {time_all:.2f}s! Exiting...{RESET_COLOR}", flush=True)

    return_tuple = (energy_dict,)
    if return_wfc:
        return_tuple += (wfc_obj_lst[0] if len(wfc_obj_lst) == 1 else wfc_obj_lst,)
    else:
        return_tuple += (None,)
    if return_h:
        return_tuple += (h_ks_lst[0] if len(wfc_obj_lst) == 1 else h_ks_lst,)
    else:
        return_tuple += (None,)
    if return_mats:
        return_tuple += (mats_lst[0] if len(wfc_obj_lst) == 1 else mats_lst,)
    else:
        return_tuple += (None,)
    return return_tuple


def main():
    ######################### BANNER AND ARGUMENT PARSING #########################
    banner_attributes = vars(banners)
    banner_strings = [value for name, value in banner_attributes.items() if isinstance(value, str) and not name.startswith("_")]
    # random_banner = banner_strings[np.random.randint(0, len(banner_strings) + 1)]
    rng = np.random.default_rng()
    random_banner = rng.choice(banner_strings)
    dopyqo.print_banner(random_banner)
    parser = argparse.ArgumentParser(
        epilog="Either provide an input file or at least the number of electrons and number of active orbitals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input_file", type=str, default=argparse.SUPPRESS, help="Input file")
    parser.add_argument("-e", "--active_electrons", type=int, default=argparse.SUPPRESS, help="Number of electrons in the active space")
    parser.add_argument("-o", "--active_orbitals", type=int, default=argparse.SUPPRESS, help="Number of spatial orbitals in the active space")
    parser.add_argument("-fci", required=False, action="store_true", default=argparse.SUPPRESS, help="Running a FCI calculation")
    parser.add_argument("-vqe", required=False, action="store_true", default=argparse.SUPPRESS, help="Running a VQE calculation")
    parser.add_argument("-uccsd_reps", type=int, default=argparse.SUPPRESS, help="Number of repetitions of the UCCSD in the VQE circuit")
    parser.add_argument("-vqe_opt", type=str, default=argparse.SUPPRESS, help="Optimizer used in the VQE")
    parser.add_argument("-qiskit", required=False, action="store_true", default=argparse.SUPPRESS, help="Using qiskit instead of TenCirChem")
    parser.add_argument("-logging", required=False, action="store_true", default=argparse.SUPPRESS, help="Show logging")
    args = parser.parse_args()
    n_given_args = len(args.__dict__)
    if "input_file" not in args and "i" not in args:
        args.input_file = None
    if "active_electrons" not in args and "e" not in args:
        args.active_electrons = None
    if "active_orbitals" not in args and "o" not in args:
        args.active_orbitals = None
    if "fci" not in args:
        args.fci = False
    if "vqe" not in args:
        args.vqe = False
    if "uccsd_reps" not in args:
        args.uccsd_reps = 1
    if "vqe_opt" not in args:
        args.vqe_opt = "L-BFGS-B"
    if "qiskit" not in args:
        args.qiskit = False
    if "logging" not in args:
        args.logging = False
    #
    if args.input_file is None and (args.active_electrons is None or args.active_orbitals is None):
        parser.print_help(sys.stderr)
        sys.exit(1)

    ######################### READING INPUT FILE #########################
    if args.input_file is not None:
        if n_given_args != 1:
            argument_s_str = "arguments were" if n_given_args - 1 > 1 else "argument was"
            print(
                f"{RED}Error in arguments: If input file is given no other arguments are allowed but {n_given_args-1} other {argument_s_str} given!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        input_filename = args.input_file
        config_dict = {}
        print(f"Reading {input_filename} input file...")
        config_dict = dopyqo.read_input_toml(input_filename)
        if config_dict.get("base_folder") is None:
            base_folder = os.path.join(*os.path.split(input_filename)[:-1])  # os.getcwd()
            base_folder_tmp = "./" if len(base_folder) == 0 else base_folder
            print(f"base_folder not given in input file, defaulting to folder of input file: {base_folder_tmp}")
            config_dict["base_folder"] = base_folder
        if config_dict.get("prefix") is None:
            base_folder_tmp = "./" if base_folder is None or len(base_folder) == 0 else base_folder
            print(f"prefix not given in input file. Searching for '.save' folder in base_folder ({base_folder_tmp})...")
            prefix = has_save_folder(base_folder)
            if prefix is None:
                print(
                    f"{RED}Error finding files: Could not find a folder ending with '.save' in base_folder ({base_folder_tmp}){RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"prefix set to: {prefix}")
            config_dict["prefix"] = prefix
        config = DopyqoConfig(**config_dict)
    ######################### READING CLI ARGUMENTS #########################
    else:
        base_folder = os.getcwd()
        prefix = has_save_folder(base_folder)
        if prefix is None:
            print(
                f"{RED}Error finding files: Could not find a folder ending with '.save' in current folder ({base_folder}){RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)

        # parser = argparse.ArgumentParser(description="")
        # parser.add_argument("-e", "--active_electrons", type=int, required=True, help="Number of electrons in the active space")
        # parser.add_argument("-o", "--active_orbitals", type=int, required=True, help="Number of spatial orbitals in the active space")
        # parser.add_argument("-fci", required=False, action="store_true", help="Running a FCI calculation")
        # parser.add_argument("-vqe", required=False, action="store_true", help="Running a VQE calculation")
        # parser.add_argument("-uccsd_reps", type=int, default=1, required=False, help="Number of repetitions of the UCCSD in the VQE circuit")
        # parser.add_argument("-vqe_opt", type=str, default="L-BFGS-B", required=False, help="Optimizer used in the VQE")
        # parser.add_argument("-qiskit", action="store_true", required=False, help="Using qiskit instead of TenCirChem")
        # parser.add_argument("-logging", action="store_true", required=False, help="Show logging")
        # args = parser.parse_args()

        active_electrons = args.active_electrons
        active_orbitals = args.active_orbitals
        run_fci_flag = args.fci
        run_vqe_flag = args.vqe
        uccsd_reps = args.uccsd_reps
        vqe_opt = args.vqe_opt
        use_qiskit_flag = args.qiskit
        logging_flag = args.logging

        config = DopyqoConfig(
            base_folder,
            prefix,
            active_electrons,
            active_orbitals,
            logging_flag,
            run_vqe_flag,
            run_fci_flag,
            use_qiskit_flag,
            vqe_optimizer=vqe_opt,
            uccsd_reps=uccsd_reps,
        )

    # print(f"asdict(Config(**config_dict)): {asdict(DopyqoConfig(**config_dict))}")
    # print(f"config_dict: {config_dict}")
    _ = run(config, show_banner=False)
