import os
import numpy as np
import matplotlib.pyplot as plt
import dopyqo
from dopyqo.colors import *

if __name__ == "__main__":
    vqe_optimizer = dopyqo.VQEOptimizers.L_BFGS_B
    config = dopyqo.DopyqoConfig(
        base_folder=os.path.join("qe_files", "Be"),
        prefix="Be",
        active_electrons=2,
        active_orbitals=4,
        run_fci=True,  # Run FCI calculation
        run_vqe=True,  # Run VQE calculation
        vqe_optimizer=vqe_optimizer,
        vqe_excitations=dopyqo.ExcitationPools.SINGLES_DOUBLES,
        n_threads=10,
    )
    energy_dict, wfc_obj, h_ks, mats = dopyqo.run(config)
    dft_energy = energy_dict["dft_energy"]
    fci_energy = energy_dict["fci_energy"]
    vqe_energy = energy_dict["vqe_energy"]

    plt.plot(h_ks.vqe_counts, np.abs(h_ks.vqe_values - fci_energy), linestyle="-", marker="x")
    plt.yscale("log")
    plt.grid()
