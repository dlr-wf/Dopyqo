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
        n_threads=10,
    )
    energy_dict, wfc_obj, h_ks, mats = dopyqo.run(config)

    h_pqrs = mats.h_pqrs
    h_pq = mats.h_pq
    energy_frozen_core = mats.energy_frozen_core
    energy_ewald = mats.energy_ewald
    energy_e_self = mats.energy_e_self

    print()
    print(f"{h_pqrs.shape=}")
    print(f"{h_pq.shape=}")
    print(f"{energy_frozen_core=}")
    print(f"{energy_ewald=}")
    print(f"{energy_e_self=}")

    # Solve manually with FCI
    fci_energies = h_ks.solve_fci()
    print(f"{fci_energies=}")
