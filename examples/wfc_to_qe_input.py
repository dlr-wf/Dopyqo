import os
import copy
import numpy as np
import dopyqo
from dopyqo.colors import *
from dopyqo.units import Unit
from dopyqo import DopyqoConfig, MatrixElements, print_block
from dopyqo.scripts.main import run as run_dopyqo

if __name__ == "__main__":
    # From QE output
    base_folder = os.path.join("qe_files", "Be")
    prefix = "Be"
    xml_file = os.path.join(base_folder, f"{prefix}.save", "data-file-schema.xml")
    wfc_file = os.path.join(base_folder, f"{prefix}.save", "wfc1.dat")
    wfc_obj = dopyqo.Wfc.from_file(wfc_file, xml_file)
    wfc_obj.set_atom_positions(wfc_obj.atom_positions_hartree + 0.01, unit=Unit.HARTREE)
    wfc_obj.set_lattice_vectors(wfc_obj.a + 0.5, unit=Unit.HARTREE)
    wfc_obj.nbnd = 20
    input_file = os.path.join("qe_files", "Be", "Be_from_dopyqo.scf.in")
    wfc_obj.to_qe_input(input_file, prefix="Be")

    # # Or from dopyqo calculation
    # config = DopyqoConfig(
    #     base_folder="Be",
    #     prefix="Be",
    #     active_electrons=2,
    #     active_orbitals=2,
    #     run_fci=False,
    #     run_vqe=False,
    #     n_threads=10,
    # )
    # energy_dict, wfc_obj, h_ks, mats = run_dopyqo(config, return_h=True, return_wfc=True, return_mats=True)

    # wfc_obj.set_atom_positions(wfc_obj.atom_positions_hartree + 0.01, unit=Unit.HARTREE)
    # wfc_obj.to_qe_input(os.path.join("Be_simple", "Be_simple.scf.in"), prefix="Be_simple")

    # # Run QE SCF calculation
    # dopyqo.runQE(input_file)
