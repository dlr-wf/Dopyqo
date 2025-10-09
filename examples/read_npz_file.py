import os
import numpy as np

if __name__ == "__main__":
    base_folder = os.path.join("qe_files", "Be")
    prefix = "Be"

    active_electrons = 2
    active_orbitals = 2

    data = np.load(os.path.join(base_folder, f"energies_{active_electrons}e_{active_orbitals}o.npz"), allow_pickle=True)
    for key, val in data.items():
        print(f"{key}: {val}")
