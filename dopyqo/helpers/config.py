import os
import sys
from dataclasses import dataclass
import xmltodict
import tomllib
import numpy as np
from dopyqo.colors import *
from dopyqo import units
from dopyqo.helpers.vqe_helpers import ExcitationPools, VQEOptimizers


@dataclass(frozen=True)
class DopyqoConfig:
    """Config for running a Dopyqo calculation

    Objects of this class are immutable (frozen instances) and variables of an object cannot be set manually after creation.
    Therefore, copying a object and changing values does not work. For this, use the dataclasses.replace function to create a new
    object from an existing one and simultaneously changing values.

    NOTE: None is the default of optional entries since we parse them as None from a toml-input file

    FIXME: If None is read from input file, just don't pass it into DopyqoConfig and set the defaults to non-None values?!

    Args:
        base_folder (str): Path to the folder in which the prefix.save folder is. For setting the prefix see argument prefix.
        prefix (str): Prefix of the Quantum ESPRESSO calculation. See also argument base_folder.
        active_electrons (int): Number of electrons in the active space.
        active_orbitals (int): Number of orbitals in the active space.
        kpoint_idx (int | str | None): Specify the k-point(s) that are calculated (integer or string).
                                       If integer than it specifies the index of the k-point in the k-point list used in Quantum ESPRESSO.
                                       If string than only "all" is allowed and all k-points are calculated. Set to 0, if None.
                                       Defaults to None.
        logging_flag (bool | None): If set to true, logging information will be shown. If None or False, no logging information will be shown. Defaults to None.
        n_threads (int | None): Number of threads used to calculate the pseudopotential when using the kohn-sham-matrix-elements-rs package. Set to 1 if None. Defaults to None.
        use_gpu (bool | None): If set to True, the GPU is used for ERI and frozen core matrix elements, if the cupy package is available.
                               If set to True and cupy package is not available, numpy will be used instead. Set to True if None. Defaults to None.
        wannier_transform (bool | None): If True, the orbitals in the active space are transformed into Wannier functions. See also wannier_umat and wannier_input_file.
                                         Set to False if None. Defaults to None.
        wannier_umat (str | None): Path to file holding the Wannier transformation matrix in Wannier90 format. Used only if wannier_transform is True.
                                   If None and wannier_transform is True will be set to base_folder/prefix_u.mat. Defaults to None.
        wannier_input_file (str | None): Path to the Wannier90 input file used to generate the u.mat file for argument wannier_umat.
                                         Used to validate if active space matches the transformed orbitals. Ignored if None. Defaults to None.
        occupations (list[int] | np.ndarray | None): NOT FULLY IMPLEMENTED, YET. List or numpy array of occupations used as the initial state in the VQE ansatz.
                                                     Only values of 0 and 1 are allowed. Defaults to None.
        run_vqe (bool | None): If set to True, a VQE calculation will be performed. Set to False if None. Defaults to None.
        run_fci (bool | None): If set to True, a FCI calculation will be performed. Set to False if None. Defaults to None.
        n_fci_energies (int | None): Number of energies calculated by the FCI solver. Set to 1 if None. Defaults to None.
        use_qiskit (bool | None): If set to True, qiskit is used to perform the VQE calculation. Set to False if None. Defaults to None.
        unit (units.Unit | None): Defines the unit used for the atom positions (see atom_positions) and lattice vectors (see lattice_vectors).
                                  Must not be None if either atom_positions or lattice_vectors are set. Defaults to None.
        atom_positions (np.ndarray | None): Set the atom positions with numpy array of shape (N, 3) where N is the number of atoms.
                                            Atoms are in the same order as in the output-xml file from Quantum ESPRESSO. Defaults to None.
        lattice_vectors (np.ndarray | None): Set lattice vectors with numpy array of shape (3, 3). Each line defines one lattice vector. Defaults to None.
        vqe_parameters (np.ndarray | None): Calculate energy of the VQE ansatz using fixed parameters. If given, no VQE optimization is performed.
                                            VQE optimization is performed if None. Defaults to None.
        vqe_initial_parameters (np.ndarray | None): Initial parameters of the VQE optimization. Set to zeros if None. Defaults to None.
        vqe_optimizer (VQEOptimizers | None): Optimizer used for the VQE optimization. Needs to be set if run_vqe is set to True. Defaults to None.
        vqe_maxiter (int | None): Number of VQE optimization steps before stopping the optimization. Set to large default values depending on the optimizer if None.
                                  Defaults to None.
        vqe_adapt (bool | None): If set to True, a APAPT-VQE calculation will be performed. If False, a VQE calculation will be performed. Set to False if None.
                                 Defaults to None.
        vqe_adapt_drain_pool (bool | None): If set to True the operator pool is drained when a operator is appended to the ansatz.
                                            If False, an operator can be appended to the ansatz multiple times. Set to True if None. Defaults to None.
        vqe_excitations (list[tuple[int, ...]] | ExcitationPools | None): Excitations used in the VQE ansatz our ADAPT-VQE pool.
                                                                          If given as list of tuples, each tuple represents one excitation.
                                                                          The tuples follow the notation used in TenCirChem:
                                                                          - An excitation operator is represented by a tuple of integers. Each integer corresponds to one spin-orbital.
                                                                          - The integers start from zero and denote first all up-spin orbitals then all down-spin orbitals
                                                                          - The first half of the tuple contains the indices for creation operator and the second half is for annilation operator.
                                                                          - Hermitian conjugation is handled internally.
                                                                           For example, (6, 2, 0, 4) corresponds to a†_6 a†_2 a_0 a_4 - a†_4 a†_0 a_2 a_6
                                                                          Set to dopyqo.ExcitationPools.SINGLES_DOUBLES if None. Defaults to None.
        uccsd_reps (int | None): Number of repetitions of the UCCSD ansatz used in VQE. Set to 1 if None. Defaults to None.
        qe_ewald (bool | None): If True, the nuclear-repulsion (Ewald) energy is not calculated but read from the xml-file outputted by Quantum ESPRESSO.
                                Set to False if None. Defaults to None.
        run_hf (bool | None): If True, a Hartree-Fock (HF) calculation is performed with PySCF and the Hamiltonian is transformed into the HF basis for the VQE/FCI.
                              Set to False if None. Defaults to None.
    """

    base_folder: str
    prefix: str
    active_electrons: int
    active_orbitals: int
    kpoint_idx: int | str | None = None
    logging_flag: bool | None = None
    n_threads: int | None = None
    use_gpu: bool | None = None
    wannier_transform: bool | None = None
    wannier_umat: str | None = None
    wannier_input_file: str | None = None
    occupations: list[int] | np.ndarray | None = None
    run_vqe: bool | None = None
    run_fci: bool | None = None
    n_fci_energies: int | None = None  # TODO: Add this to toml-input-file
    use_qiskit: bool | None = None
    unit: units.Unit | None = None
    atom_positions: np.ndarray | None = None
    lattice_vectors: np.ndarray | None = None
    vqe_parameters: np.ndarray | None = None  # No VQE optimization, just run circuit with these parameters
    vqe_initial_parameters: np.ndarray | None = None  # TODO: Add this to toml-input-file
    vqe_optimizer: VQEOptimizers | None = None
    vqe_maxiter: int | None = None
    vqe_adapt: bool | None = None
    vqe_adapt_drain_pool: bool | None = None
    vqe_excitations: list[tuple[int, ...]] | ExcitationPools | None = None
    uccsd_reps: int | None = None
    qe_ewald: bool | None = None
    run_hf: bool | None = None  # TODO: Add this to toml-input-file

    def __post_init__(self):
        if len(self.base_folder) > 0 and not os.path.isdir(self.base_folder):
            print(f"{RED}Config error: Base folder ({self.base_folder}) does not exist!{RESET_COLOR}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(os.path.join(self.base_folder, f"{self.prefix}.save")):
            print(
                f"{RED}Config error: QE save folder ({os.path.join(self.base_folder, f'{self.prefix}.save')}) does not exist!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if self.active_electrons is None:
            print(f"{RED}Config error: Number of active electrons (active_electrons) has to be set but is not!{RESET_COLOR}", file=sys.stderr)
            sys.exit(1)
        if self.active_orbitals is None:
            print(f"{RED}Config error: Number of active orbitals (active_orbitals) has to be set but is not!{RESET_COLOR}", file=sys.stderr)
            sys.exit(1)
        if self.kpoint_idx is None:
            object.__setattr__(self, "kpoint_idx", 0)
        if (not isinstance(self.kpoint_idx, int) or isinstance(self.kpoint_idx, str)) and self.kpoint_idx != "all":
            print(
                f'{RED}Config error: Invalid k-point index ({self.kpoint_idx})! Either set to integer starting from zero or set to "all"!{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)
        if isinstance(self.kpoint_idx, int) and self.kpoint_idx < 0:
            print(
                f'{RED}Config error: Given k-point index ("{self.kpoint_idx}") negative but needs to be positive!{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)
        n_wfc_files = len([x for x in os.listdir(os.path.join(self.base_folder, f"{self.prefix}.save")) if x.startswith("wfc")])
        if isinstance(self.kpoint_idx, int) and self.kpoint_idx + 1 > n_wfc_files:
            print(
                f"{RED}Config error: Given k-point index ({self.kpoint_idx}, starting at zero) is to large! Found only {n_wfc_files} wavefunction (wfc) files!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        xml_file = os.path.join(self.base_folder, f"{self.prefix}.save", "data-file-schema.xml")
        with open(xml_file, encoding="utf-8") as file:
            xml_dict = xmltodict.parse(file.read())
        n_kpoints = int(xml_dict["qes:espresso"]["output"]["band_structure"]["nks"])
        if n_kpoints != 1 and self.kpoint_idx is None:
            print(
                f"{RED}Config error: You try loading a DFT calculation involving multiple k-points ({self.n_kpoints}) "
                + "without specifying the k-point you want to load (kpoint_idx=None). "
                + "Specify a k-point by setting kpoint_idx to the corresponding index of the k-point, starting at zero. "
                + f'You can also load all k-points by setting kpoint_idx to "all"!{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)
        if self.run_vqe is None:
            object.__setattr__(self, "run_vqe", False)
        if self.run_fci is None:
            object.__setattr__(self, "run_fci", False)
        if self.run_fci and self.n_fci_energies is None:
            object.__setattr__(self, "n_fci_energies", 1)
        if self.atom_positions is not None:
            if self.unit is None:
                allowed_units_str = "\n\t".join(units.ALLOWED_UNITS)
                print(
                    f"{RED}Config error: Atom positions defined but unit not defined. Please define unit! Supported units are:\n\t{allowed_units_str}\n{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if self.lattice_vectors is not None:
            if self.lattice_vectors.shape != (3, 3):
                print(
                    f"{RED}Config error: Lattice vectors are invalid. Expected a numpy array with shape (3, 3) but got {self.lattice_vectors.shape}.{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if self.unit is None:
                allowed_units_str = "\n\t".join(units.ALLOWED_UNITS_LATTICE)
                print(
                    f"{RED}Config error: Atom positions defined but unit not defined. Please define unit! Supported units are:\n\t{allowed_units_str}\n{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if self.unit.name.lower() not in units.ALLOWED_UNITS_LATTICE:
                allowed_units_str = "\n\t".join(units.ALLOWED_UNITS_LATTICE)
                print(
                    f"{RED}Config error: Defined unit ({self.unit}) cannot be used when defining lattice vectors. Supported units are:\n\t{allowed_units_str}\n{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if self.wannier_umat is not None and not self.wannier_transform:
            print(
                f"{ORANGE}Config warning: Wannier U-matrix path (wannier_umat) set but no Wannier transformation will be performed (wannier_transform not True).{RESET_COLOR}",
                file=sys.stdout,
            )
        if self.wannier_input_file is not None and not self.wannier_transform:
            print(
                f"{ORANGE}Config warning: Wannier input file path (wannier_input_file) set but no Wannier transformation will be performed (wannier_transform not True).{RESET_COLOR}",
                file=sys.stdout,
            )
        if self.wannier_input_file is not None and not os.path.isfile(self.wannier_input_file):
            print(
                f"{RED}Config error: Wannier input file ({self.wannier_input_file}) does not exist!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)

        if self.vqe_adapt is not None and self.vqe_adapt:
            if self.uccsd_reps is not None:
                print(
                    f"{ORANGE}Config warning: Both "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" and "
                    + f"{self.uccsd_reps=}".split("=")[0][5:]
                    + f" are set. "
                    + f"{self.uccsd_reps=}".split("=")[0][5:]
                    + f" has no effect and will be ignored!{RESET_COLOR}",
                )
            if self.vqe_initial_parameters is not None:
                print(
                    f"{ORANGE}Config warning: Both "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" and "
                    + f"{self.vqe_initial_parameters=}".split("=")[0][5:]
                    + f" are set. "
                    + f"{self.vqe_initial_parameters=}".split("=")[0][5:]
                    + f" has no effect and will be ignored!{RESET_COLOR}",
                )
        if self.vqe_optimizer is None and self.run_vqe:
            print(
                f"{RED}Config error: "
                + f"{self.run_vqe=}".split("=")[0][5:]
                + f" is True but "
                + f"{self.vqe_optimizer=}".split("=")[0][5:]
                + f"is None but has to be set if "
                + f"{self.run_vqe=}".split("=")[0][5:]
                + f" is True!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not isinstance(self.vqe_optimizer, VQEOptimizers) and self.vqe_optimizer is not None:
            print(
                f"{RED}Config error: "
                + f"{self.vqe_optimizer=}".split("=")[0][5:]
                + f" has to be of type dopyqo.VQEOptimizers but is of type {type(self.vqe_optimizer)}!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if self.uccsd_reps is None:
            object.__setattr__(self, "uccsd_reps", 1)
        if self.qe_ewald is None:
            object.__setattr__(self, "qe_ewald", False)
        if self.run_hf is None:
            object.__setattr__(self, "pyscf_hf", False)
        if self.n_threads is None or self.n_threads <= 0:
            object.__setattr__(self, "n_threads", 1)
        if self.use_gpu is None:
            object.__setattr__(self, "use_gpu", True)
        if self.vqe_adapt is None:
            object.__setattr__(self, "vqe_adapt", False)
        if self.vqe_excitations is None:
            object.__setattr__(self, "vqe_excitations", ExcitationPools.SINGLES_DOUBLES)
        if not isinstance(self.vqe_excitations, list) and not isinstance(self.vqe_excitations, ExcitationPools):
            print(
                f"{RED}Config error: VQE excitations parameter has to be of type list or of type dopyqo.ExcitationPools but is of type {type(self.vqe_excitations)}!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if self.vqe_adapt:
            if not self.run_vqe:
                print(
                    f"{ORANGE}Config warning: "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" is set to True but "
                    + f"{self.run_vqe=}".split("=")[0][5:]
                    + f" is set to False. "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" has no effect and will be ignored!{RESET_COLOR}",
                )
        if self.vqe_adapt_drain_pool is not None and self.vqe_adapt_drain_pool:
            if not self.vqe_adapt:
                print(
                    f"{ORANGE}Config warning: "
                    + f"{self.vqe_adapt_drain_pool=}".split("=")[0][5:]
                    + f" is set to True but "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" is set to False. "
                    + f"{self.vqe_adapt_drain_pool=}".split("=")[0][5:]
                    + f" has no effect and will be ignored!{RESET_COLOR}",
                )
        if self.vqe_adapt_drain_pool is None:
            if self.vqe_adapt:
                object.__setattr__(self, "vqe_adapt_drain_pool", True)
            else:
                object.__setattr__(self, "vqe_adapt_drain_pool", False)
        if self.vqe_excitations is not None:
            if self.use_qiskit:
                print(
                    f"{ORANGE}Config warning: VQE excitations set but Qiskit will be used. Costum VQE excitations currently only supported when using TenCirChem. Given VQE excitations will be ignored!{RESET_COLOR}"
                )
            if isinstance(self.vqe_excitations, list):
                for exc in self.vqe_excitations:
                    if not isinstance(exc, tuple):
                        print(
                            f"{RED}Config error: VQE excitations parameter has to be a list of tuples but is a list that contains elements of type {type(exc)}!{RESET_COLOR}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    for idx in exc:
                        if not isinstance(idx, int):
                            print(
                                f"{RED}Config error: VQE excitations parameter has to be a list of tuples of ints but is a list of tuples coantining elements of type {type(idx)} in tuple {exc}!{RESET_COLOR}",
                                file=sys.stderr,
                            )
                            sys.exit(1)
                    if len(exc) % 2 != 0:
                        print(
                            f"{RED}Config error: VQE excitations parameter has to be a list of tuples of an even number of ints but found tuple of length {len(exc)} (tuple: {exc})!{RESET_COLOR}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
        if self.vqe_adapt:
            if self.use_qiskit:
                print(
                    f"{RED}Config error: Both "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" and "
                    + f"{self.use_qiskit=}".split("=")[0][5:]
                    + f" are set to True. ADAPT-VQE is not supported when using Qiskit!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if self.vqe_parameters is not None:
            if self.vqe_adapt:
                print(
                    f"{RED}Config error: VQE parameters given by "
                    + f"{self.vqe_parameters=}".split("=")[0][5:]
                    + f" but "
                    + f"{self.vqe_adapt=}".split("=")[0][5:]
                    + f" is set to True. "
                    + f"{self.vqe_parameters=}".split("=")[0][5:]
                    + f" can only be set for non-ADAPT VQE calculations!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if isinstance(self.vqe_excitations, list):
                n_exc = len(self.vqe_excitations)
            elif isinstance(self.vqe_excitations, ExcitationPools):
                print(
                    f"{RED}Config error: Type dopyqo.ExcitationPools currently not supported for "
                    + f"{self.vqe_excitations=}".split("=")[0][5:]
                    + f" if  "
                    + f"{self.vqe_parameters=}".split("=")[0][5:]
                    + f" are given. Provide a list of excitations instead!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            else:
                no = self.active_electrons // 2
                norb = self.active_orbitals
                nv = norb - no

                n_singles = no * nv * 2
                n_doubles = 2 * (no * (no - 1) // 2) * (nv * (nv - 1) // 2)
                for i in range(no):
                    for j in range(i + 1):
                        for a in range(nv):
                            for b in range(a + 1):
                                if i == j and a == b:
                                    n_doubles += 1
                                    continue
                                n_doubles += 2
                                if (i != j) and (a != b):
                                    n_doubles += 2
                n_exc = (n_singles + n_doubles) * self.uccsd_reps
            if len(self.vqe_parameters) != n_exc:
                print(
                    f"{RED}Config error: Expected {n_exc} VQE parameters but got {len(self.vqe_parameters)}!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if self.vqe_initial_parameters is not None:
            if self.vqe_parameters is not None:
                print(f"{RED}Config error: vqe_initial_parameters set but also vqe_parameters is set. Choose one!{RESET_COLOR}", file=sys.stderr)
                sys.exit(1)

            if self.vqe_excitations is not None:
                n_exc = len(self.vqe_excitations)
            else:
                no = self.active_electrons // 2
                norb = self.active_orbitals
                nv = norb - no

                n_singles = no * nv * 2
                n_doubles = 2 * (no * (no - 1) // 2) * (nv * (nv - 1) // 2)
                for i in range(no):
                    for j in range(i + 1):
                        for a in range(nv):
                            for b in range(a + 1):
                                if i == j and a == b:
                                    n_doubles += 1
                                    continue
                                n_doubles += 2
                                if (i != j) and (a != b):
                                    n_doubles += 2
                n_exc = (n_singles + n_doubles) * self.uccsd_reps
            if len(self.vqe_initial_parameters) != n_exc:
                print(
                    f"{RED}Config error: Expected {n_exc} VQE initial parameters but got {len(self.vqe_initial_parameters)}!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if self.wannier_transform:
            if self.wannier_umat is None:
                object.__setattr__(self, "wannier_umat", os.path.join(self.base_folder, f"{self.prefix}_u.mat"))
            if not os.path.isfile(self.wannier_umat):
                print(
                    f"{RED}Config error: Wannier U-matrix ({self.wannier_umat}) does not exist!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if self.wannier_input_file is None:
                object.__setattr__(self, "wannier_input_file", os.path.join(self.base_folder, f"{self.prefix}.win"))
            if not os.path.isfile(self.wannier_input_file):
                print(
                    f"{ORANGE}Config warning: Wannier input file (automatically set to {self.wannier_input_file}) does not exist! ",
                    f"If the Wannier-transformed orbitals match the orbitals in the active space cannot be checked.{RESET_COLOR}",
                )
        if self.occupations is not None:
            if not self.run_vqe:
                print(
                    f"{ORANGE}Config warning: occupations are specified but no VQE calculation will be performed. occupations will not be used!{RESET_COLOR}"
                )
            if self.use_qiskit:
                print(
                    f"{ORANGE}Config warning: occupations are specified but qiskit will be used for the VQE calculation. occupations are currently only supported when using TenCirChem. occupations will not be used!{RESET_COLOR}"
                )
            if not isinstance(self.occupations, list) and not isinstance(self.occupations, np.ndarray):
                print(
                    f"{RED}Config error: occupations has to be a list or numpy array, but is {type(self.occupations)}!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if isinstance(self.occupations, np.ndarray) and self.occupations.size != self.active_orbitals:
                print(
                    f"{RED}Config error: occupations must have {self.active_orbitals} elements (number of orbitals) but has {self.occupations.size} elements (shape {self.occupations.shape})!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if len(self.occupations) != self.active_orbitals:
                print(
                    f"{RED}Config error: occupations must have {self.active_orbitals} elements (number of orbitals) but has {len(self.occupations)} elements!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
            for x in self.occupations:
                if not np.isclose(x, int(x)):
                    print(
                        f"{RED}Config error: occupations must be integers but found value {x}!{RESET_COLOR}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            object.__setattr__(self, "occupations", np.array([int(x) for x in self.occupations]))
            if not np.isclose(2.0 * np.sum(self.occupations), self.active_electrons):
                print(
                    f"{RED}Config error: Two times the occupations must sum to the number of electrons ({self.active_electrons}) but sum to {2.0*np.sum(self.occupations)}!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)


def read_control(control: dict) -> tuple:
    base_folder = control.get("base_folder")
    prefix = control.get("prefix")
    active_electrons = control.get("active_electrons")
    active_orbitals = control.get("active_orbitals")
    kpoint_idx = control.get("kpoint_idx")
    logging_flag = control.get("logging")
    n_threads = control.get("n_threads")
    run_vqe = control.get("run_vqe")
    run_fci = control.get("run_fci")
    use_qiskit = control.get("use_qiskit")

    if active_electrons is None:
        print(
            f'{RED}Error in input file: Number of active electrons ("active_electrons") has to be defined in section "control" but is not!{RESET_COLOR}',
            file=sys.stderr,
        )
        sys.exit(1)
    if active_orbitals is None:
        print(
            f'{RED}Error in input file: Number of active orbitals ("active_orbitals") has to be defined in section "control" but is not!{RESET_COLOR}',
            file=sys.stderr,
        )
        sys.exit(1)

    return base_folder, prefix, active_electrons, active_orbitals, kpoint_idx, logging_flag, n_threads, run_vqe, run_fci, use_qiskit


def read_wannier(wannier: dict) -> tuple:
    wannier_transform = wannier.get("transform")
    wannier_umat = wannier.get("umat")
    wannier_input_file = wannier.get("input_file")

    return wannier_transform, wannier_umat, wannier_input_file


def read_geometry(geometry: dict) -> tuple[units.Unit | None, np.ndarray | None, np.ndarray | None]:
    allowed_units_str = "\n\t".join(units.ALLOWED_UNITS)
    unit = None

    def get_unit(unit: str) -> units.Unit:
        match unit:
            case "angstrom":
                unit = units.Unit.ANGSTROM
            case "bohr":
                unit = units.Unit.HARTREE
            case "meter":
                unit = units.Unit.METER
            case "alat":
                unit = units.Unit.ALAT
            case "crystal":
                unit = units.Unit.CRYSTAL
            case _:
                print(
                    f"{RED}Error in input file: Unit {unit} not supported! Supported units are:\n\t{allowed_units_str}\n{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        return unit

    ######################### COORDINATES #########################
    # IDEA: dict instead of list? Element: coordinates e.g. {"H":[0.0, 0.1, 1.5]}?
    atom_positions = geometry.get("coordinates")
    if atom_positions is not None:
        if "unit" not in geometry.keys():
            print(
                f'{RED}Error in input file: Atom positions defined in section "coordinates" but "unit" not defined. Please define "unit"! Supported units are:\n\t{allowed_units_str}\n{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)
        unit = get_unit(str(geometry["unit"]).lower())
        atom_positions = np.array(atom_positions)

    ######################### LATTICE_VECTORS #########################
    allowed_units_lattice_str = "\n\t".join(units.ALLOWED_UNITS_LATTICE)
    lattice_vectors = geometry.get("lattice_vectors")
    if lattice_vectors is not None:
        lattice_vectors = np.array(lattice_vectors)
        if lattice_vectors.shape != (3, 3):
            print(
                f'{RED}Error in input file: Lattice vectors defined in section "lattice_vectors" are invalid. Expected a list of lists with shape (3, 3) but got {lattice_vectors.shape}.{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)
        if "unit" not in geometry.keys():
            print(
                f'{RED}Error in input file: Lattice vectors defined in section "lattice_vectors" but "unit" not defined. Please define "unit"! Supported units are:\n\t{allowed_units_str}\n{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)
        unit = get_unit(str(geometry["unit"]).lower())
        if unit.name.lower() not in units.ALLOWED_UNITS_LATTICE:
            print(
                f'{RED}Error in input file: Defined unit ({geometry["unit"]}) cannot be used when defining lattice vectors in section "lattice_vectors". Supported units are:\n\t{allowed_units_lattice_str}\n{RESET_COLOR}',
                file=sys.stderr,
            )
            sys.exit(1)

    return unit, atom_positions, lattice_vectors


def read_vqe(vqe: dict) -> tuple[np.ndarray | None, str | None, int | None]:
    ######################### PARAMETERS #########################
    parameters = vqe.get("parameters")
    parameters = np.array(parameters) if parameters is not None else None

    ######################### OPTIMIZER #########################
    optimizer = vqe.get("optimizer")
    if optimizer is not None:
        optimizer = str(optimizer)
        if optimizer.lower() == "l-bfgs-b":
            optimizer = VQEOptimizers.L_BFGS_B
        elif optimizer.lower() == "cobyla":
            optimizer = VQEOptimizers.COBYLA
        elif optimizer.lower() == "excitationsolve":
            optimizer = VQEOptimizers.ExcitationSolve
        else:
            allowed_opt = [x.name.lower() for x in VQEOptimizers]
            print(f"{RED}VQE optimizer {optimizer} is not supported. Please use on of {allowed_opt}!{RESET_COLOR}")

    ######################### UCCSD_REPS #########################
    uccsd_reps = vqe.get("uccsd_reps")
    uccsd_reps = int(uccsd_reps) if uccsd_reps is not None else None

    return parameters, optimizer, uccsd_reps


def read_input_toml(file) -> DopyqoConfig:
    allowed_sections = ["control", "wannier", "geometry", "vqe"]
    with open(file, "rb") as f:
        config_toml = tomllib.load(f)

    present_sections = config_toml.keys()
    unallowed_sections_present = False
    for sec_tmp in present_sections:
        if sec_tmp not in allowed_sections:
            print(f"Section {sec_tmp} in TOML-file is not supported and ignored!")
            unallowed_sections_present = True
    if unallowed_sections_present:
        allowed_str_tmp = "\n\t".join(allowed_sections)
        print(f"Supported sections are:\n\t{allowed_str_tmp}\n")

    config_dict = {}
    ######################### CONTROL #########################
    control = config_toml.get("control")
    if control is not None:
        base_folder, prefix, active_electrons, active_orbitals, kpoint_idx, logging_flag, n_threads, run_vqe, run_fci, use_qiskit = read_control(
            control
        )
        config_dict["base_folder"] = base_folder
        config_dict["prefix"] = prefix
        config_dict["active_electrons"] = active_electrons
        config_dict["active_orbitals"] = active_orbitals
        config_dict["logging_flag"] = logging_flag
        config_dict["n_threads"] = n_threads
        config_dict["kpoint_idx"] = kpoint_idx
        config_dict["run_vqe"] = run_vqe
        config_dict["run_fci"] = run_fci
        config_dict["use_qiskit"] = use_qiskit

    ######################### WANNIER #########################
    wannier = config_toml.get("wannier")
    if wannier is not None:
        wannier_transform, wannier_umat, wannier_input_file = read_wannier(wannier)
        config_dict["wannier_transform"] = wannier_transform
        config_dict["wannier_umat"] = wannier_umat
        config_dict["wannier_input_file"] = wannier_input_file

    ######################### GEOMETRY #########################
    geometry = config_toml.get("geometry")
    if geometry is not None:
        unit, atom_positions, lattice_vectors = read_geometry(geometry)
        config_dict["unit"] = unit
        config_dict["atom_positions"] = atom_positions
        config_dict["lattice_vectors"] = lattice_vectors

    ######################### VQE #########################
    vqe = config_toml.get("vqe")
    if vqe is not None:
        parameters, optimizer, uccsd_reps = read_vqe(vqe)
        config_dict["vqe_parameters"] = parameters
        config_dict["vqe_optimizer"] = optimizer
        config_dict["uccsd_reps"] = uccsd_reps

    return config_dict
