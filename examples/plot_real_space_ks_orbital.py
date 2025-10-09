import os
import numpy as np
import dopyqo
from dopyqo.colors import *

if __name__ == "__main__":
    config = dopyqo.DopyqoConfig(
        # base_folder=os.path.join("qe_files", "Be"),
        # prefix="Be",
        base_folder=os.path.join("qe_files", "Mg2Si"),
        prefix="Mg2Si",
        active_electrons=2,
        active_orbitals=2,
        n_threads=10,
    )

    # # We could run a dopyqo calculation, which would also calculate the matrix elements, but we just need the Wfc object for plotting
    # energy_dict, wfc_obj, h, mats = run_dopyqo(config, show_banner=False, verbosity=0)

    # Calculating the matrix elements is not necessary, we just need the Wfc object. Let's create this manually!
    xml_file = os.path.join(config.base_folder, f"{config.prefix}.save", "data-file-schema.xml")
    wfc_file = os.path.join(config.base_folder, f"{config.prefix}.save", "wfc1.dat")
    wfc_obj = dopyqo.Wfc.from_file(wfc_file, xml_file, kpoint_idx=config.kpoint_idx)

    plotter = wfc_obj.plot_real_space(
        band_idc=[8],
        isosurfaces=5,
        opacity=0.6,
        extend_data=False,
        extend_atoms=False,
        plot_lattice_vectors=True,
        html_filename=os.path.join(config.base_folder, "plot_wfc_ks"),
    )
