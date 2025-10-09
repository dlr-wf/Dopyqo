import os
import numpy as np
import matplotlib.pyplot as plt
from tencirchem.static.hamiltonian import random_integral
import dopyqo
from dopyqo.colors import *

if __name__ == "__main__":
    # Defining custom core potential and custom electron-electron interaction
    active_orbitals = 4
    h1e_core_custom, h2e_custom = random_integral(active_orbitals)
    h2e_custom = h2e_custom.transpose((0, 2, 3, 1))  # Tranform from chemist index order to physicist index order

    mats_custom = dopyqo.MatrixElements(
        h_pq_core=h1e_core_custom,
        h_pqrs=h2e_custom,
        # Other matrix elements and energies are automatically set to None and are calculated
    )

    vqe_optimizer = dopyqo.VQEOptimizers.L_BFGS_B
    config = dopyqo.DopyqoConfig(
        base_folder=os.path.join("qe_files", "Be"),
        prefix="Be",
        active_electrons=2,
        active_orbitals=active_orbitals,
        run_fci=True,  # Run FCI calculation
        run_vqe=True,  # Run VQE calculation
        vqe_optimizer=vqe_optimizer,
        vqe_excitations=dopyqo.ExcitationPools.SINGLES_DOUBLES,
        n_threads=10,
    )
    energy_dict, wfc_obj, h_ks, mats = dopyqo.run(config, mats_custom)
    dft_energy = energy_dict["dft_energy"]
    fci_energy = energy_dict["fci_energy"]
    vqe_energy = energy_dict["vqe_energy"]

    plt.plot(h_ks.vqe_counts, np.abs(h_ks.vqe_values - fci_energy), linestyle="-", marker="x")
    plt.yscale("log")
    plt.grid()
