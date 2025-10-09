import os
import copy
from dataclasses import replace
import numpy as np
from dopyqo.units import Unit
import dopyqo

if __name__ == "__main__":
    # A basic config
    config = dopyqo.DopyqoConfig(
        base_folder=os.path.join("qe_files", "Be"),
        prefix="Be",
        active_electrons=2,
        active_orbitals=2,
        run_fci=True,
        n_threads=10,
    )

    # A new config where we want to change the atom positions
    # Besides that it is equal to the config above
    atom_positions = np.array([0.0, 0.5])
    config_new_pos = dopyqo.DopyqoConfig(
        base_folder=os.path.join("qe_files", "Be"),
        prefix="Be",
        active_electrons=2,
        active_orbitals=2,
        run_fci=True,
        n_threads=10,
        atom_positions=atom_positions,
        unit=Unit.CRYSTAL,
    )

    # # Following will fail with a FrozenInstanceError,
    # # since we cannot assign new values to a existing DopyqoConfig.
    # config_new_pos_copy = copy.copy(config)
    # config_new_pos_copy.atom_positions = atom_positions
    # config_new_pos_copy.unit = Unit.CRYSTAL

    # replace will copy the config and assign new values.
    # When using replace all new values will also be checked for their validity.
    # So this is completely equivalent to config_new_pos.
    config_new_pos_replace = replace(
        config,
        atom_positions=atom_positions,
        unit=Unit.CRYSTAL,
    )
