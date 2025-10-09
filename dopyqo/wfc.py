import sys
import os
import logging
import subprocess
import textwrap
from warnings import warn
from enum import Enum
from collections import Counter
import numpy as np
import itertools
from scipy.spatial import Voronoi
import h5py
import xmltodict
import pyvista as pv
import dopyqo
from dopyqo import plotting
from dopyqo.colors import *


class Wfc:
    """Class storing wavefunction and crystal information from Quantum ESPRESSO output files

    See from_file(...) function to create object from Quantum ESPRESSO output files.
    """

    def __init__(
        self,
        ik: int,
        xk: np.ndarray,
        ispin: int,
        gamma_only: bool,
        scalef: float,
        ngw: int,
        igwx: int,
        npol: int,
        nbnd: int,
        b1: np.ndarray,
        b2: np.ndarray,
        b3: np.ndarray,
        mill: np.ndarray,
        evc: np.ndarray,
        output_xml: str,
        pseudopots: list[dopyqo.Pseudopot] = None,
        kpoint_idx: int | None = None,
    ):
        """Class storing wavefunction and crystal information from Quantum ESPRESSO output files

        Args:
            ik (int): k-point index starting from 1
            xk (np.ndarray): k-point coordinates
            ispin (int): Spin index for LSDA case: ispin=1 for spin-up, ispin=2 for spin-down.
                          For unpolarized or non-colinear cases, ispin=1
            gamma_only (bool): True if a gamma-only calculation was performed, then only half of the plane waves are written to file
            scalef (float): Scale factor applied to wavefunctions
            ngw (int): Number of plane waves
            igwx (int): Max number of plane waves
            npol (int): Number of spin states for plane waves: 2 for non-colinear case, 1 otherwise
            nbnd (int): Number of Kohn-Sham orbitals, i.e., number of bands
            b1 (np.ndarray): First reciprocal lattice vector
            b2 (np.ndarray): Second reciprocal lattice vector
            b3 (np.ndarray): Third reciprocal lattice vector
            mill (np.ndarray): Miller indices for each plane wave. Shape (# of plane waves, 3)
            evc (np.ndarray): Coefficients describing the Kohn-Sham orbitals in terms of plane waves.
                              Shape (# of plane waves, # of Kohn-Sham orbitals)
            output_xml (str): Path to the XML-file Quantum ESPRESSO outputs
            pseudopots (list[Pseudopot], optional): List of dopyqo.Pseudopot objects, one object per atom type. Defaults to None.
            kpoint_idx (int | None, optional): k-point index starting from 0. Is set to 0, if None and Quantum ESPRESSO calculation
                                               only involved one k-point. Has to match ik+1, if Quantum ESPRESSO calculation
                                               involved multiple k-points. Defaults to None.
        """
        # NOTE: For gamma-only calculations the KS-orbitals are real in real-space.
        #       For k-point calculations, the gamma-point KS-orbitals are NOT real in real-space.
        #       This is because the KS-orbitals at the gamma-point do not have to be real in real-space,
        #       they can just be chosen to be real!
        self.ik = ik
        # k-point vector is in units of 1/bohr = 1/(5.29177210544e-11 meter)
        self.kpoint = xk
        self.ispin = ispin
        self.gamma_only = gamma_only
        self.scalef = scalef
        if not np.isclose(self.scalef, 1.0):
            print(f"{ORANGE}Warning: Scale-factor is not 1.0. Please contact a developer!{RESET_COLOR}")
        self.ngw = ngw
        self.igwx = igwx
        self.npol = npol
        self.nbnd = nbnd
        # Reciprocal lattice vectors are in units of 1/bohr = 1/(5.29177210544e-11 meter)
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.mill = mill
        self.evc = evc
        self.pseudopots = pseudopots
        self.kpoint_idx = kpoint_idx

        if self.kpoint_idx is not None:
            assert isinstance(
                self.kpoint_idx, int
            ), f"kpoint_idx invalid! Expected integer or None but got: {self.kpoint_idx} of type {type(self.kpoint_idx)}"

        self.b = np.array([self.b1, self.b2, self.b3])  # b1 in first row, b2 in second row, b3 in third row.

        self.G = np.einsum("ij,jk->ik", mill, self.b)

        self.k_plus_G = xk + self.G
        self.k_plus_G_norm = np.linalg.norm(self.k_plus_G, ord=2, axis=1)

        # If only Γ point is sampled, only positive half of
        # the plane wave expansion coefficients are saved. Generate and append negative half here.
        # See https://docs.abinit.org/theory/wavefunctions/#plane wave-basis-set-sphere
        if self.gamma_only:
            self.evc_org = self.evc.copy()
            self.G_org = self.G.copy()
            self.mill_org = self.mill.copy()

            assert (self.G[0] == np.array([0.0, 0.0, 0.0])).all(), (
                f"Expected first G-vector to be the zero-vector but found {self.G[0]} in order" " to generate coefficients of negative G-vectors!"
            )
            self.G = np.append(self.G, -self.G[1:], axis=0)
            self.mill = np.append(self.mill, -self.mill[1:], axis=0)
            self.k_plus_G = xk + self.G
            self.k_plus_G_norm = np.linalg.norm(self.k_plus_G, ord=2, axis=1)
            self.evc = np.append(
                self.evc, self.evc[:, 1:].conj(), axis=1
            )  # conj() correct, see https://docs.abinit.org/theory/wavefunctions/#plane wave-basis-set-sphere

        with open(output_xml, encoding="utf-8") as file:
            self.xml_dict = xmltodict.parse(file.read())

        # | k + G | < G_cut with E_cut = G_cut^2/2 in atomic units (-> G_cut = \sqrt(2*E_cut))
        # see: https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/structureoptimization/stress/stress.html
        self.ecutwfc = float(self.xml_dict["qes:espresso"]["input"]["basis"]["ecutwfc"])
        self.gcutwfc = np.sqrt(self.ecutwfc * 2)
        self.ecutrho = float(self.xml_dict["qes:espresso"]["input"]["basis"]["ecutrho"])
        self.gcutrho = np.sqrt(self.ecutrho * 2)
        self.occupations_scf = self.xml_dict["qes:espresso"]["input"]["bands"]["occupations"]
        self.degauss = None
        self.smearing = None
        if self.occupations_scf == "smearing":
            smearing = self.xml_dict["qes:espresso"]["input"]["bands"]["smearing"]
            self.degauss = float(smearing["@degauss"])
            self.smearing = smearing["#text"]
        self.conv_thr = float(self.xml_dict["qes:espresso"]["input"]["electron_control"]["conv_thr"])
        self.mixing_beta = float(self.xml_dict["qes:espresso"]["input"]["electron_control"]["mixing_beta"])
        self.electron_maxstep = int(self.xml_dict["qes:espresso"]["input"]["electron_control"]["max_nstep"])

        # Read k-points
        self.n_kpoints = int(self.xml_dict["qes:espresso"]["output"]["band_structure"]["nks"])
        # If multiple k-points are available, you have to specify kpoint_idx
        if self.n_kpoints != 1 and self.kpoint_idx is None:
            print(
                f"{RED}Wavefunction error: You try loading a DFT calculation involving multiple k-points ({self.n_kpoints}) "
                + "without specifying the k-point you want to load (kpoint_idx=None). "
                + "Specify a k-point by setting kpoint_idx to the corresponding index of the k-point, starting at zero. "
                + "You can also directly load all k-points with the "
                + f"{self.__class__.__name__}.from_file_all_kpoints method.{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Set kpoint_idx to zero if only one k-point is available and kpoint_idx is None
        if self.n_kpoints == 1 and self.kpoint_idx is None:
            self.kpoint_idx = 0
        if self.kpoint_idx >= self.n_kpoints or self.kpoint_idx < 0 or not np.isclose(self.kpoint_idx, int(self.kpoint_idx)):
            raise ValueError(
                f"kpoint_idx ({self.kpoint_idx}) is invalid. It has to be an integer greater than "
                + f"zero and smaller than the number of k-points ({self.n_kpoints})."
            )
        assert self.kpoint_idx == self.ik - 1, (
            f"kpoint_idx ({self.kpoint_idx}) does not match the k-point index in the wavefunction "
            + f"({self.ik-1}). The wavefunction does not belong to the specified k-point index kpoint_idx ({self.kpoint_idx})."
        )

        # If calculation had more than one k-point xml_dict[...]["ks_energies"] is a list instead of a dict
        # self.kpoint_xml is in units of 2*np.pi/alat_out, NOT in 1/bohr = 1/(5.29177210544e-11 meter)
        if self.n_kpoints == 1:
            self.kpoint_xml = np.fromstring(
                self.xml_dict["qes:espresso"]["output"]["band_structure"]["ks_energies"]["k_point"]["#text"],
                sep=" ",
                dtype=np.float64,
            )
            self.kpoint_weight = float(self.xml_dict["qes:espresso"]["output"]["band_structure"]["ks_energies"]["k_point"]["@weight"])
        else:
            self.kpoint_xml = np.fromstring(
                self.xml_dict["qes:espresso"]["output"]["band_structure"]["ks_energies"][self.kpoint_idx]["k_point"]["#text"],
                sep=" ",
                dtype=np.float64,
            )
            self.kpoint_weight = float(
                self.xml_dict["qes:espresso"]["output"]["band_structure"]["ks_energies"][self.kpoint_idx]["k_point"]["@weight"]
            )

        fft_grid_x = int(self.xml_dict["qes:espresso"]["output"]["basis_set"]["fft_grid"]["@nr1"])
        fft_grid_y = int(self.xml_dict["qes:espresso"]["output"]["basis_set"]["fft_grid"]["@nr2"])
        fft_grid_z = int(self.xml_dict["qes:espresso"]["output"]["basis_set"]["fft_grid"]["@nr3"])
        self.fft_grid = np.array([fft_grid_x, fft_grid_y, fft_grid_z])

        self.spin = 1
        if (
            "nbnd_up" in self.xml_dict["qes:espresso"]["output"]["band_structure"]
            and "nbnd_dw" in self.xml_dict["qes:espresso"]["output"]["band_structure"]
        ):
            self.spin = 2

        self.etot = float(self.xml_dict["qes:espresso"]["output"]["total_energy"]["etot"])
        self.eband = float(self.xml_dict["qes:espresso"]["output"]["total_energy"]["eband"])
        self.ehart = float(self.xml_dict["qes:espresso"]["output"]["total_energy"]["ehart"])
        self.vtxc = float(self.xml_dict["qes:espresso"]["output"]["total_energy"]["vtxc"])
        self.etxc = float(self.xml_dict["qes:espresso"]["output"]["total_energy"]["etxc"])
        self.ewald = float(self.xml_dict["qes:espresso"]["output"]["total_energy"]["ewald"])
        atoms_dict = self.xml_dict["qes:espresso"]["input"]["atomic_structure"]["atomic_positions"]["atom"]

        self.atomic_species = self.xml_dict["qes:espresso"]["input"]["atomic_species"]["species"]
        if not isinstance(self.atomic_species, list):
            self.atomic_species = [self.atomic_species]
        self.atomic_species = {x["@name"]: {"mass": float(x["mass"]), "pseudo_file": x["pseudo_file"]} for x in self.atomic_species}

        # If the lattice vectors are defined in units of alat (CELL_PARAMETERS alat) the "input" and "output"
        # values of alat are different and represent alat (\sqrt(a1 @ a1)) calculated with the lattice vectors
        # in hartree/bohr units and alat defined in the input file (whyever this should be written in the "output" section)
        # via celldem(1) or A, respectively
        # The k-points in the wfc files are in units of the "output"-alat, but the alat of the unit cell is the
        # "input"-alat. Very confusing!
        self.alat = float(self.xml_dict["qes:espresso"]["input"]["atomic_structure"]["@alat"])
        alat_out = float(self.xml_dict["qes:espresso"]["output"]["atomic_structure"]["@alat"])
        assert np.allclose(self.kpoint_xml, self.kpoint / (2 * np.pi) * alat_out), (
            f"The k-point with index {self.kpoint_idx} from the xml-file ({self.kpoint_xml}) does not match the "
            + f"k-point in the wavefunction file ({self.kpoint / (2*np.pi) * alat_out}). "
            + "This should not be possible! Please, contact a developer!"
        )

        if isinstance(atoms_dict, dict):
            atoms_dict = [atoms_dict]

        atoms = []
        for atom in atoms_dict:
            atoms.append(
                {
                    "element": atom["@name"],
                    "mass": self.atomic_species[atom["@name"]]["mass"],
                    "pseudo_file": self.atomic_species[atom["@name"]]["pseudo_file"],
                    "position_bohr": np.fromstring(atom["#text"], sep=" ", dtype=np.float64),
                    "position_hartree": np.fromstring(atom["#text"], sep=" ", dtype=np.float64),
                    "position_meter": np.fromstring(atom["#text"], sep=" ", dtype=np.float64) * dopyqo.BOHR_TO_METER,
                    "position_angstrom": np.fromstring(atom["#text"], sep=" ", dtype=np.float64) * dopyqo.BOHR_TO_ANGSTROM,
                    "atomic_number": dopyqo.elements_to_atomic_number[atom["@name"]],
                }
            )
        self.atoms = atoms

        # shape (#atoms, 3)
        self.atom_positions_hartree = np.array([atom["position_hartree"] for atom in self.atoms], dtype=np.float64)
        self.atomic_numbers = np.array([atom["atomic_number"] for atom in self.atoms])

        self.atomic_numbers_valence = self.atomic_numbers.copy()
        if pseudopots is not None:
            # Mapping atomic numbers to list of atom positions
            atoms_dict = {num: [] for num in self.atomic_numbers}
            atoms_dict = {key: np.array(val) for key, val in atoms_dict.items()}
            pp_dict = {pp.atomic_number: pp for pp in pseudopots}

            # Checking for duplicate PPs
            atomic_nums_pp_duplicates = [
                atomic_num_pp for atomic_num_pp, count_val in Counter([pp.atomic_number for pp in pseudopots]).items() if count_val > 1
            ]
            atomic_el_pp_duplicates = [dopyqo.elements_to_atomic_number[x] for x in atomic_nums_pp_duplicates]
            assert len(atomic_nums_pp_duplicates) == 0, "More than one pseudopotential given for atoms " + ", ".join(
                str(x) + f" (Z={atomic_nums_pp_duplicates[i]})" for i, x in enumerate(atomic_el_pp_duplicates)
            )

            # Checking for atoms without PP
            atomic_num_wo_pp = [atomic_num for atomic_num in atoms_dict.keys() if atomic_num not in pp_dict.keys()]
            atomic_el_wo_pp = [dopyqo.elements_to_atomic_number[x] for x in atomic_num_wo_pp]
            assert len(atomic_num_wo_pp) == 0, "No pseudopotentials given for atoms " + ", ".join(
                str(x) + f" (Z={atomic_num_wo_pp[i]})" for i, x in enumerate(atomic_el_wo_pp)
            )

            for i, z in enumerate(self.atomic_numbers):
                self.atomic_numbers_valence[i] = pp_dict[z].z_valence

        assert len(self.atom_positions_hartree) == len(self.atomic_numbers), (
            f"Number of atomic positions ({len(self.atom_positions_hartree)}) is not equal "
            + f"to number of atomic numbers ({len(self.atomic_numbers)})! "
            + "Both where read from the xml-file."
        )

        self.atom_positions_hartree_mean = np.mean(self.atom_positions_hartree, axis=0)
        self.atom_center_of_mass = np.sum(self.atomic_numbers[:, None] * self.atom_positions_hartree, axis=0) / np.sum(self.atomic_numbers)

        # Calculate Lattice vectors in bohr/hartree units from reciprocal lattice vectors
        self.cell_volume_reciprocal = self.b1.dot(np.cross(self.b2, self.b3))
        self.a1 = 2 * np.pi / self.cell_volume_reciprocal * np.cross(self.b2, self.b3)
        self.a2 = 2 * np.pi / self.cell_volume_reciprocal * np.cross(self.b3, self.b1)
        self.a3 = 2 * np.pi / self.cell_volume_reciprocal * np.cross(self.b1, self.b2)
        self.cell_volume = np.abs(self.a1.dot(np.cross(self.a2, self.a3)))
        self.cell_volume_reciprocal = np.abs(self.cell_volume_reciprocal)

        # Read reciprocal lattice vectors from xml file
        # are in units of 2*np.pi/alat_out, NOT in 1/bohr = 1/(5.29177210544e-11 meter)
        reciprocal_cell_xml = self.xml_dict["qes:espresso"]["output"]["basis_set"]["reciprocal_lattice"]
        self.b1_xml = np.fromstring(reciprocal_cell_xml["b1"], sep=" ", dtype=np.float64)
        self.b2_xml = np.fromstring(reciprocal_cell_xml["b2"], sep=" ", dtype=np.float64)
        self.b3_xml = np.fromstring(reciprocal_cell_xml["b3"], sep=" ", dtype=np.float64)

        assert np.allclose(
            np.array([self.b1, self.b2, self.b3]), np.array([self.b1_xml, self.b2_xml, self.b3_xml]) * (2 * np.pi) / alat_out
        ), f"Reciprocal lattice vectors in {output_xml} do not match the given reciprocal lattice vectors b1, b2, b3 (possibly from dat/hdf5 file)"

        # Read lattice vectors from xml file
        cell = self.xml_dict["qes:espresso"]["input"]["atomic_structure"]["cell"]
        a1 = np.fromstring(cell["a1"], sep=" ", dtype=np.float64)
        a2 = np.fromstring(cell["a2"], sep=" ", dtype=np.float64)
        a3 = np.fromstring(cell["a3"], sep=" ", dtype=np.float64)

        assert np.allclose(
            np.array([self.a1, self.a2, self.a3]), np.array([a1, a2, a3])
        ), f"Lattice vectors in {output_xml} (\n{np.array([a1, a2, a3])}\n) do not match the given reciprocal lattice vectors b1, b2, b3 (possibly from dat/hdf5 file) and their corresponding lattice vectors (\n{np.array([self.a1, self.a2, self.a3])}\n)"
        self.a = np.array([self.a1, self.a2, self.a3])  # a1 in first row, a2 in second row, a3 in third row.

        ks_energies_entry = self.xml_dict["qes:espresso"]["output"]["band_structure"]["ks_energies"]
        if self.n_kpoints == 1:
            self.ks_energies = np.array([float(x) for x in ks_energies_entry["eigenvalues"]["#text"].split()])
        else:
            self.ks_energies = [np.array([float(x) for x in y["eigenvalues"]["#text"].split()]) for y in ks_energies_entry][self.kpoint_idx]

        # TODO: Define nelec tuple of number of electrons for spin-up and spin-down
        self.nelec = int(float(self.xml_dict["qes:espresso"]["output"]["band_structure"]["nelec"]))

        if self.nelec % 2 == 1:
            assert self.spin == 2, (
                f"The number of electrons ({self.nelec}) is odd but a spin-unpolarized DFT calculation was performed! "
                + "Odd number of electrons are only supported for spin-polarized DFT calculations!"
            )

        if self.n_kpoints == 1:
            self.occupations = np.array([float(x) for x in ks_energies_entry["occupations"]["#text"].split()])
        else:
            self.occupations = [np.array([float(x) for x in y["occupations"]["#text"].split()]) for y in ks_energies_entry][self.kpoint_idx]
        self.occupations_xml = self.occupations.copy()

        # Negative occupations are non-physical and are filtered out
        # but keeping the sum of occupations invariant by adding the sum
        # of negative occupations to the largest occupation
        # This hopefully does not change the physics
        sum_of_neg = np.sum(self.occupations[self.occupations < 0.0])
        self.occupations[self.occupations < 0.0] = 0.0
        self.occupations[np.argmax(self.occupations)] += sum_of_neg

        # Same for occupations greater than 1
        sum_of_gt1 = np.sum(self.occupations[self.occupations > 1.0] - 1.0)
        self.occupations[self.occupations > 1.0] = 1.0
        self.occupations[np.argmin(self.occupations)] += sum_of_gt1

        # Assert the previous
        assert not any(n < 0 for n in self.occupations), f"The occupations cannot be smaller than 0 but are {self.occupations}."
        assert not any(n > 1 for n in self.occupations), f"The occupations cannot be greater than 1 but are {self.occupations}."

        if self.n_kpoints > 1:
            # For k-point calculations the occupations are define wrt to the Fermi energy. This can result in the fact that
            # two times the sum of occupations does not match the number of electrons.
            # For k-point calculations we set to occupations to binary occupations with the correct sum.
            logging.debug("Calculation with more than one k-point. Setting occupations to binary occupations!")
            self.occupations = np.concatenate(
                (
                    np.ones((self.nelec // 2,), dtype=np.int64),
                    np.zeros(
                        (len(self.occupations) - self.nelec // 2,),
                        dtype=np.int64,
                    ),
                ),
                dtype=np.int64,
            )

        assert np.isclose(self.nelec, occ_sum := np.sum(self.occupations) * 2), (
            f"Two times the occupations ({self.occupations}) do not sum up (sum={occ_sum}) "
            + f"to the number of electrons from the xml-file ({self.nelec})!"
        )
        assert np.isclose(round(np.sum(self.occupations)), np.sum(self.occupations)), (
            f"Sum of occupations ({np.sum(self.occupations)}) is not close to an integer, "
            + f"but matches half the number of electrons ({self.nelec/2}). "
            + "This should not be possible! Please, contact a developer!"
        )

        # If occupations are not only 0 and 1 then we do build the occupations_binary vector from scratch
        # Note: Logic not implementable with for-else since:
        #       "If the loop finishes without executing the break, the else clause executes."
        make_binary = False
        for occ in self.occupations:
            if (not np.isclose(occ, 0.0)) and (not np.isclose(occ, 1.0)):
                make_binary = True
                break

        self.occupations_binary = self.occupations.copy().round().astype(np.int64)
        if make_binary:
            self.occupations_binary = np.concatenate(
                (
                    np.ones((round(np.sum(self.occupations)),), dtype=np.int64),
                    np.zeros(
                        (len(self.occupations) - round(np.sum(self.occupations)),),
                        dtype=np.int64,
                    ),
                ),
                dtype=np.int64,
            )

        assert np.isclose(
            np.sum(self.occupations_binary), np.sum(self.occupations)
        ), "Sum of binary occupations is not equal to the sum of occupations. This should not be possible! Please, contact a developer!"

        self.ks_energies_up = self.ks_energies[:nbnd]
        self.occupations_up = self.occupations[:nbnd]
        self.occupations_binary_up = self.occupations_binary[:nbnd]
        if self.spin == 2:
            self.ks_energies_dw = self.ks_energies[nbnd:]
            self.occupations_dw = self.occupations[nbnd:]
            self.occupations_binary_dw = self.occupations_binary[nbnd:]
        else:
            self.ks_energies_dw = self.ks_energies_up
            self.occupations_dw = self.occupations_up
            self.occupations_binary_dw = self.occupations_binary_up

    @classmethod
    def from_file(
        cls,
        file: str,
        output_xml: str,
        pseudopots: list[dopyqo.Pseudopot] = None,
        kpoint_idx: int | None = None,
    ):
        """Create Wfc object from a .dat or .hdf5 Quantum ESPRESSO output file.
        Calls Wfc.from_dat_file or Wfc.from_hdf5_file depending on the provided file.

        Args:
            file (str): Path to the .dat or .hdf5 file.
            output_xml (str): Path to the XML-file Quantum ESPRESSO outputs.
            pseudopots (list[Pseudopot], optional): List of dopyqo.Pseudopot objects, one object per atom type. Defaults to None.
            kpoint_idx (int | None, optional): k-point index starting from 0. Is set to 0, if None and Quantum ESPRESSO calculation
                                               only involved one k-point. Has to match ik+1, if Quantum ESPRESSO calculation
                                               involved multiple k-points. Defaults to None.

        Raises:
            NotImplementedError: If the provided file is not a .dat and not .hdf5 file.

        Returns:
            Wfc: Wfc object
        """
        if file.endswith("dat"):
            return Wfc.from_dat_file(file, output_xml, pseudopots, kpoint_idx)
        elif file.endswith("hdf5"):
            return Wfc.from_hdf5_file(file, output_xml, pseudopots, kpoint_idx)
        else:
            raise NotImplementedError(f"File extension {file.split('.')[-1]} not supported!")

    @classmethod
    def from_hdf5_file(
        cls,
        hdf5_file: str,
        output_xml: str,
        pseudopots: list[dopyqo.Pseudopot] = None,
        kpoint_idx: int | None = None,
    ):
        """Create Wfc object from a .hdf5 Quantum ESPRESSO output file.

        Args:
            hdf5_file (str): Path to the .hdf5 file.
            output_xml (str): Path to the XML-file Quantum ESPRESSO outputs.
            pseudopots (list[Pseudopot], optional): List of dopyqo.Pseudopot objects, one object per atom type. Defaults to None.
            kpoint_idx (int | None, optional): k-point index starting from 0. Is set to 0, if None and Quantum ESPRESSO calculation
                                               only involved one k-point. Has to match ik+1, if Quantum ESPRESSO calculation
                                               involved multiple k-points. Defaults to None.

        Returns:
            Wfc: Wfc object
        """
        f = h5py.File(hdf5_file, "r")  # Works like a python dictionary
        # HDF5 files contain datasets which have a shape and a dtype attribute.
        # HDF5 files and all containing datasets also contain attributes which
        # can be obtained with f.attrs.keys().

        # The coefficients alternate between the real and imaginary part
        evc_real_imag = np.array(f["evc"])
        mill = np.array(f["MillerIndices"])
        b1 = np.array(f["MillerIndices"].attrs["bg1"])
        b2 = np.array(f["MillerIndices"].attrs["bg2"])
        b3 = np.array(f["MillerIndices"].attrs["bg3"])
        gamma_only = "TRUE" in str(f.attrs["gamma_only"])
        igwx = f.attrs["igwx"]
        ik = f.attrs["ik"]
        ispin = f.attrs["ispin"]
        nbnd = f.attrs["nbnd"]
        ngw = f.attrs["ngw"]
        npol = f.attrs["npol"]
        scale_factor = f.attrs["scale_factor"]
        xk = f.attrs["xk"]
        f.close()

        evc = np.zeros(
            shape=(evc_real_imag.shape[0], evc_real_imag.shape[1] // 2),
            dtype=np.complex128,
        )
        evc = evc_real_imag[:, 0::2] + 1j * evc_real_imag[:, 1::2]

        return cls(
            ik=ik,
            xk=xk,
            ispin=ispin,
            gamma_only=gamma_only,
            scalef=scale_factor,
            ngw=ngw,
            igwx=igwx,
            npol=npol,
            nbnd=nbnd,
            b1=b1,
            b2=b2,
            b3=b3,
            mill=mill,
            evc=evc,
            output_xml=output_xml,
            pseudopots=pseudopots,
            kpoint_idx=kpoint_idx,
        )

    @classmethod
    def from_dat_file(
        cls,
        dat_file: str,
        output_xml: str,
        pseudopots: list[dopyqo.Pseudopot] = None,
        kpoint_idx: int | None = None,
    ):
        """Create Wfc object from a .dat Quantum ESPRESSO output file.

        Args:
            dat_file (str): Path to the .dat file.
            output_xml (str): Path to the XML-file Quantum ESPRESSO outputs.
            pseudopots (list[Pseudopot], optional): List of dopyqo.Pseudopot objects, one object per atom type. Defaults to None.
            kpoint_idx (int | None, optional): k-point index starting from 0. Is set to 0, if None and Quantum ESPRESSO calculation
                                               only involved one k-point. Has to match ik+1, if Quantum ESPRESSO calculation
                                               involved multiple k-points. Defaults to None.

        Returns:
            Wfc: Wfc object
        """
        # Taken from https://mattermodeling.stackexchange.com/questions/9149/how-to-read-qes-wfc-dat-files-with-python
        # INTEGER :: ik
        # !! k-point index (1 to number of k-points)
        # REAL(8) :: xk(3)
        # !! k-point coordinates
        # INTEGER :: ispin
        # !! spin index for LSDA case: ispin=1 for spin-up, ispin=2 for spin-down
        # !! for unpolarized or non-colinear cases, ispin=1 always
        # LOGICAL :: gamma_only
        # !! if .true. write or read only half of the plane waves
        # REAL(8) :: scalef
        # !! scale factor applied to wavefunctions
        # INTEGER :: ngw
        # !! number of plane waves (PW)
        # INTEGER :: igwx
        # !! max number of PW (may be larger than ngw, not sure why)
        # INTEGER :: npol
        # !! number of spin states for PWs: 2 for non-colinear case, 1 otherwise
        # INTEGER :: nbnd
        # !! number of wavefunctions
        # REAL(8) :: b1(3), b2(3), b3(3)
        # !! primitive reciprocal lattice vectors
        # INTEGER :: mill(3,igwx)
        # !! miller indices: h=mill(1,i), k=mill(2,i), l=mill(3,i)
        # !! the i-th PW has wave vector (k+G)(:)=xk(:)+h*b1(:)+k*b2(:)+ l*b3(:)
        # COMPLEX(8) :: evc(npol*igwx,nbnd)
        # !! wave functions in the PW basis set
        # !! The first index runs on PW components,
        # !! the second index runs on band states.
        # !! For non-colinear case, each PW has a spin component
        # !! first  igwx components have PW with   up spin,
        # !! second igwx components have PW with down spin
        with open(dat_file, "rb") as f:
            # Moves the cursor 4 bytes to the right
            f.seek(4)

            ik = np.fromfile(f, dtype="int32", count=1)[0]
            xk = np.fromfile(f, dtype="float64", count=3)
            ispin = np.fromfile(f, dtype="int32", count=1)[0]
            gamma_only = bool(np.fromfile(f, dtype="int32", count=1)[0])
            scalef = np.fromfile(f, dtype="float64", count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            ngw = np.fromfile(f, dtype="int32", count=1)[0]
            igwx = np.fromfile(f, dtype="int32", count=1)[0]
            npol = np.fromfile(f, dtype="int32", count=1)[0]
            nbnd = np.fromfile(f, dtype="int32", count=1)[0]

            # Move the cursor 8 byte to the right
            f.seek(8, 1)

            b1 = np.fromfile(f, dtype="float64", count=3)
            b2 = np.fromfile(f, dtype="float64", count=3)
            b3 = np.fromfile(f, dtype="float64", count=3)

            f.seek(8, 1)

            mill = np.fromfile(f, dtype="int32", count=3 * igwx)
            mill = mill.reshape((igwx, 3))

            evc = np.zeros((nbnd, npol * igwx), dtype="complex128")

            f.seek(8, 1)
            for i in range(nbnd):
                evc[i, :] = np.fromfile(f, dtype="complex128", count=npol * igwx)
                f.seek(8, 1)

        return cls(
            ik=ik,
            xk=xk,
            ispin=ispin,
            gamma_only=gamma_only,
            scalef=scalef,
            ngw=ngw,
            igwx=igwx,
            npol=npol,
            nbnd=nbnd,
            b1=b1,
            b2=b2,
            b3=b3,
            mill=mill,
            evc=evc,
            output_xml=output_xml,
            pseudopots=pseudopots,
            kpoint_idx=kpoint_idx,
        )

    @classmethod
    def from_file_all_kpoints(
        cls,
        files: list[str],
        output_xml: str,
        pseudopots: list[dopyqo.Pseudopot] = None,
        verbose: bool = False,
    ):
        """Create Wfc object from a list of .dat or .hdf5 Quantum ESPRESSO output files and return a list of Wfc objects.
        Calls Wfc.from_file for each file in the list of files.

        Args:
            files (list[str]): List of .dat or .hdf5 files
            output_xml (str): Path to the XML-file Quantum ESPRESSO outputs.
            pseudopots (list[dopyqo.Pseudopot], optional): List of dopyqo.Pseudopot objects, one object per atom type. Defaults to None.

        Returns:
            list[Wfc]: List of Wfc objects
        """
        assert isinstance(files, list), f"Expected list of files but got {files.__class__.__name__}!"

        with open(output_xml, encoding="utf-8") as file:
            xml_dict = xmltodict.parse(file.read())
        # Read number of k-points
        n_kpoints = int(xml_dict["qes:espresso"]["output"]["band_structure"]["nks"])

        # Create Wfc object for every k-point
        wfc_obj_list = []
        for i in range(n_kpoints):
            if verbose:
                print(f"Loading k-point {i+1}/{n_kpoints}...", end="\r", flush=True)
            wfc_obj_list.append(
                Wfc.from_file(
                    files[i],
                    output_xml,
                    pseudopots,
                    kpoint_idx=i,
                )
            )
        return wfc_obj_list

    def get_overlaps(self, orbitals=None):
        if orbitals is None:
            overlaps = np.einsum("ij, kj -> ik", self.evc.conj(), self.evc)
        else:
            overlaps = np.einsum("ij, kj -> ik", self.evc[orbitals].conj(), self.evc[orbitals])

        return overlaps

    def check_norm(self):
        assert np.allclose(self.get_overlaps(), np.identity(self.evc.shape[0])), "Overlap matrix is not diagonal (Wavefunctions are not orthonormal)!"

        logging.info("Overlap matrix is diagonal (Wavefunctions are orthonormal)!")

    def get_orbitals_by_index(self, indices: list | np.ndarray | None = None, binary_occupations=True):
        if indices is None:
            indices = list(range(len(self.occupations)))

        indices = indices.copy()

        if binary_occupations:
            occupations = self.occupations_binary[indices]
        else:
            occupations = self.occupations[indices]

        c_ip_orbitals = self.evc[indices]

        if len(indices) == 1:
            indices.append(indices[0])

        warn_msg = "The selected orbitals for the active space are all fully occupied or unoccupied!"
        occ_str = " ".join([(f"|{val}" if i == indices[0] else f"{val}|" if i == indices[1] else f"{val}") for i, val in enumerate(occupations)])
        warn_msg += f" You selected the following active space: {occ_str}"
        if np.any(occupations == 2):
            if not np.any(occupations == 1):
                warn(warn_msg)
        elif np.all(occupations == 0):
            warn(warn_msg)
        else:
            if not np.any(occupations == 0):
                warn(warn_msg)

        return occupations, c_ip_orbitals

    # Copied and adjusted from the pennylane function (e.g. pennylane version 0.40.0)
    # https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.active_space.html
    def active_space(self, active_electrons: int, active_orbitals: int) -> tuple[list[int], list[int]]:
        if self.spin == 1:
            mult = 1
        elif self.spin == 2:
            mult = np.abs(np.sum(self.occupations_up) - np.sum(self.occupations_dw))
            assert np.isclose(round(mult), mult), f"Calculated spin multiplicity ({mult}) is not close to an integer!"
        else:
            raise ValueError(f"Spin is neither 1 nor 2, but {self.spin}")
        if active_electrons is None:
            ncore_orbs = 0
            core = []
        else:
            if active_electrons <= 0:
                raise ValueError(f"The number of active electrons ({active_electrons}) " f"has to be greater than 0.")

            if active_electrons > self.nelec:
                raise ValueError(
                    f"The number of active electrons ({active_electrons}) "
                    f"can not be greater than the total "
                    f"number of electrons ({self.nelec})."
                )

            if active_electrons < mult - 1:
                raise ValueError(
                    f"For a reference state with multiplicity {mult}, "
                    f"the number of active electrons ({active_electrons}) should be "
                    f"greater than or equal to {mult - 1}."
                )

            if mult % 2 == 1:
                if active_electrons % 2 != 0:
                    raise ValueError(
                        f"For a reference state with multiplicity {mult}, " f"the number of active electrons ({active_electrons}) should be even."
                    )
            else:
                if active_electrons % 2 != 1:
                    raise ValueError(
                        f"For a reference state with multiplicity {mult}, " f"the number of active electrons ({active_electrons}) should be odd."
                    )

            ncore_orbs = (self.nelec - active_electrons) // 2
            core = list(range(ncore_orbs))

        if active_orbitals is None:
            active = list(range(ncore_orbs, self.nbnd))
        else:
            if active_orbitals <= 0:
                raise ValueError(f"The number of active orbitals ({active_orbitals}) " f"has to be greater than 0.")

            if ncore_orbs + active_orbitals > self.nbnd:
                raise ValueError(
                    f"The number of core ({ncore_orbs}) + active orbitals ({active_orbitals}) cannot "
                    f"be greater than the total number of orbitals ({self.nbnd})"
                )

            homo = (self.nelec + mult - 1) / 2
            if ncore_orbs + active_orbitals <= homo:
                raise ValueError(f"For n_active_orbitals={active_orbitals}, there are no virtual orbitals " f"in the active space.")

            active = list(range(ncore_orbs, ncore_orbs + active_orbitals))

        return core, active

    def set_atom_positions(self, atom_positions: np.ndarray, unit: dopyqo.Unit):
        """Change the atom positions

        Args:
            atom_positions (np.ndarray): New atom positions. Shape (# of atoms, 3).
            unit (dopyqo.Unit): Unit in which the new atom positions are given

        Raises:
            ValueError: If the unit is not supported.
        """
        assert len(atom_positions) == len(self.atomic_numbers), (
            f"Number of new atomic positions ({len(atom_positions)}) is not equal " + f"to number of atomic numbers ({len(self.atomic_numbers)})!"
        )
        assert atom_positions.shape[1] == 3, (
            "New atomic positions should define x,y, and z coordinates " + f"but they have {atom_positions.shape[1]} coordinates!"
        )

        atom_postions_hartree = None
        allowed_units_str = "\n\t".join(dopyqo.ALLOWED_UNITS)
        match unit:
            case dopyqo.Unit.HARTREE:
                atom_postions_hartree = atom_positions
            case dopyqo.Unit.ANGSTROM:
                atom_postions_hartree = atom_positions / dopyqo.BOHR_TO_ANGSTROM
            case dopyqo.Unit.METER:
                atom_postions_hartree = atom_positions / dopyqo.BOHR_TO_METER
            case dopyqo.Unit.ALAT:
                atom_postions_hartree = atom_positions * self.alat
            case dopyqo.Unit.CRYSTAL:
                atom_postions_hartree = np.array([np.sum(pos * self.a, axis=0) for pos in atom_positions])
            case _:
                raise ValueError(f"Unit {unit} not supported for setting atom positions! Supported units are:\n\t{allowed_units_str}\n")

        atoms = []
        for i, atom in enumerate(self.atoms):
            atom["position_hartree"] = atom_postions_hartree[i]
            atom["position_bohr"] = atom_postions_hartree[i]
            atom["position_meter"] = atom_postions_hartree[i] * dopyqo.BOHR_TO_METER
            atom["position_angstrom"] = atom_postions_hartree[i] * dopyqo.BOHR_TO_ANGSTROM
            atoms.append(atom)
        self.atoms = atoms
        # shape (#atoms, 3)
        self.atom_positions_hartree = np.array([atom["position_hartree"] for atom in self.atoms], dtype=np.float64)

        self.atom_positions_hartree_mean = np.mean(self.atom_positions_hartree, axis=0)
        self.atom_center_of_mass = np.sum(self.atomic_numbers[:, None] * self.atom_positions_hartree, axis=0) / np.sum(self.atomic_numbers)

    def set_lattice_vectors(self, lattice_vectors: np.ndarray, unit: dopyqo.Unit):
        """Change the lattice vectors

        Args:
            lattice_vectors (np.ndarray): New lattice vectors. Shape (3, 3). Each row is one lattice vector.
            unit (dopyqo.Unit):  Unit in which the new atom positions are given

        Raises:
            ValueError:  If the unit is not supported.
        """
        if lattice_vectors.shape != (3, 3):
            raise ValueError(f"Lattice vectors are invalid. Expected a list of array of shape (3, 3) but got {lattice_vectors.shape}.")

        lattice_vectors_hartree = None
        allowed_units_lattice = dopyqo.ALLOWED_UNITS_LATTICE
        allowed_units_lattice_str = "\n\t".join(allowed_units_lattice)
        match unit:
            case dopyqo.Unit.HARTREE:
                lattice_vectors_hartree = lattice_vectors
            case dopyqo.Unit.ANGSTROM:
                lattice_vectors_hartree = lattice_vectors / dopyqo.BOHR_TO_ANGSTROM
            case dopyqo.Unit.METER:
                lattice_vectors_hartree = lattice_vectors / dopyqo.BOHR_TO_METER
            case _:
                raise ValueError(f"Unit {unit} not supported for setting lattice vectors! Supported units are:\n\t{allowed_units_lattice_str}\n")

        self.a1 = lattice_vectors_hartree[0]
        self.a2 = lattice_vectors_hartree[1]
        self.a3 = lattice_vectors_hartree[2]
        self.cell_volume = np.abs(self.a1.dot(np.cross(self.a2, self.a3)))

        self.a = np.array([self.a1, self.a2, self.a3])
        self.b = np.array([self.b1, self.b2, self.b3])

        self.b1 = 2 * np.pi / self.cell_volume * np.cross(self.a2, self.a3)
        self.b2 = 2 * np.pi / self.cell_volume * np.cross(self.a3, self.a1)
        self.b3 = 2 * np.pi / self.cell_volume * np.cross(self.a1, self.a2)
        self.cell_volume_reciprocal = np.abs(self.b1.dot(np.cross(self.b2, self.b3)))

        # See https://www.quantum-espresso.org/Doc/INPUT_PW.html CARD: CELL_PARAMETERS description
        self.alat = np.sqrt(self.a1 @ self.a1)

    def wigner_seitz_cell(self) -> list[list[np.ndarray]]:
        """Calculate all facets of the Wigner-Seitz cell. See dopyqo.wfc.plot_wigner_seitz_cell to plot the return values of this function.

        Returns:
            list[list[np.ndarray]]: List of facets
        """
        facets = wigner_seitz_cell(self.a1, self.a2, self.a3)
        return facets

    def wigner_seitz_in_radius(self) -> float:
        """Calculate the inner radius of the Wigner-Seitz cell.

        Returns:
            float: Inner radius of the Wigner-Seitz cell
        """
        facets = wigner_seitz_cell(self.a1, self.a2, self.a3)
        r_in = wigner_seitz_in_radius(facets)
        return r_in

    def plot_real_space(
        self,
        isosurfaces: int | list[float],
        band_idc: list | np.ndarray | None = None,
        plot_abs: bool = True,
        plot_lattice_vectors: bool = True,
        extend_data: bool = False,
        plotter: pv.Plotter = None,
        **kwargs,
    ) -> pv.Plotter:
        """Plot the real-space representation of the Kohn-Sham orbitals

        Args:
            isosurfaces (int | list[float]): Number of isosurfaces, if int. Values of isosurfaces if list of floats
            band_idc (list | np.ndarray | None, optional): Indices of Kohn-Sham orbitals to plot. Plots all Kohn-Sham orbitals if None. Defaults to None.
            plot_abs (bool, optional): If the absolute values of the Kohn-Sham orbitals are plotted. Defaults to True.
            plot_lattice_vectors (bool, optional): If the lattice vectors are plotted as arrows. Defaults to True.
            extend_data (bool, optional): If the orbitals are also plotted for all neighbouring cells besides the computational cell. Defaults to False.
            plotter (pv.Plotter, optional): PyVista plotter object used for plotting. A new one is created if None. Defaults to None.
            kwargs: Arguments passed to dopyqo.plotting.plot_3d_data

        Returns:
            pv.Plotter: PyVista plotter object
        """
        if band_idc is None:
            band_idc = list(range(self.nbnd))

        # Real space wavecfunctions
        max_min = self.fft_grid
        mill = self.mill
        tot_elec = self.nelec
        occupations_active, c_ip_active = self.get_orbitals_by_index(band_idc, binary_occupations=True)
        c_ip = c_ip_active

        return self.plot_real_space_from_c_ip(
            c_ip=c_ip,
            plot_abs=plot_abs,
            plot_lattice_vectors=plot_lattice_vectors,
            extend_data=extend_data,
            isosurfaces=isosurfaces,
            plotter=plotter,
            **kwargs,
        )

    def plot_real_space_from_c_ip(
        self,
        c_ip: np.ndarray,
        isosurfaces: int | list[float],
        plot_abs: bool = True,
        plot_lattice_vectors: bool = True,
        extend_data: bool = False,
        extend_atoms: bool = False,
        plotter: pv.Plotter | None = None,
        **kwargs,
    ) -> pv.Plotter:
        """Plot the real-space representation of the Kohn-Sham orbitals, given their coefficients in the plane wave basis

        Args:
            c_ip (np.ndarray): Numpy array of plane wave coefficients of shape (nbnd, npw), where nbnd is the number of bands and npw is the number of plane waves
            isosurfaces (int | list[float]): Number of isosurfaces, if int. Values of isosurfaces if list of floats
            plot_abs (bool, optional): If the absolute values of the Kohn-Sham orbitals are plotted. Defaults to True.
            plot_lattice_vectors (bool, optional): If the lattice vectors are plotted as arrows. Defaults to True.
            extend_data (bool, optional): If the orbitals are also plotted for all neighbouring cells besides the computational cell. Defaults to False.
            extend_atoms (bool, optional): If the atoms are also plotted for all neighbouring cells besides the computational cell. Defaults to False.
            plotter (pv.Plotter | None, optional): PyVista plotter object used for plotting. A new one is created if None. Defaults to None.
            kwargs: Arguments passed to dopyqo.plotting.plot_3d_data

        Returns:
            pv.Plotter: PyVista plotter object
        """
        max_min = self.fft_grid
        mill = self.mill
        nbands = c_ip.shape[0]
        c_ip_array = np.zeros(
            (
                nbands,
                *max_min,
            ),
            dtype=c_ip.dtype,
        )
        #
        # Set \psi_i(p) on given grid points
        assert mill.shape[0] == c_ip.shape[1]
        for idx, mill_idx in enumerate(mill):
            x, y, z = mill_idx
            i, j, k = (
                x + max_min[0] // 2,
                y + max_min[1] // 2,
                z + max_min[2] // 2,
            )
            c_ip_array[:, i, j, k] = c_ip[:, idx]
        #
        psi_r = []
        for i in range(nbands):
            c_ip_shifted = np.fft.ifftshift(c_ip_array[i])
            psi_r_i = np.fft.ifftn(c_ip_shifted) * np.sqrt(np.prod(max_min))
            # psi_r_i = np.fft.fftshift(psi_r_i)
            psi_r.append(psi_r_i)
        psi_r = np.array(psi_r)
        if plot_abs:
            psi_r = np.abs(psi_r) ** 2

        nx, ny, nz = max_min
        origin = np.zeros(3)
        # Looked good for diamond but yielded wrong plot for H2
        coordinates = (
            np.arange(nx)[:, None] * self.a[0] / nx
            + np.arange(ny)[:, None, None] * self.a[1] / ny
            + np.arange(nz)[:, None, None, None] * self.a[2] / nz
            + origin
        )  # .reshape((-1, 3))
        # Following two are equivalent and yield correct plot for H2 compared to the XSF-files
        coordinates = (
            np.arange(nx)[:, None, None, None] * self.a[0] / nx
            + np.arange(ny)[:, None, None] * self.a[1] / ny
            + np.arange(nz)[:, None] * self.a[2] / nz
            + origin
        )  # .reshape((-1, 3))
        #
        # coordinates = (
        #     np.arange(nz)[:, None] * self.a[2] / nz
        #     + np.arange(ny)[:, None, None] * self.a[1] / ny
        #     + np.arange(nx)[:, None, None, None] * self.a[0] / nx
        #     + origin
        # )  # .reshape((-1, 3))

        return self.plot_real_space_from_3d_data(
            coordinates=coordinates,
            psi_r=psi_r,
            plot_lattice_vectors=plot_lattice_vectors,
            extend_data=extend_data,
            extend_atoms=extend_atoms,
            isosurfaces=isosurfaces,
            plotter=plotter,
            **kwargs,
        )

    def plot_real_space_from_3d_data(
        self,
        coordinates: np.ndarray,
        isosurfaces: int | list[float],
        psi_r: list[np.ndarray],
        plot_lattice_vectors: bool = True,
        extend_data: bool = False,
        extend_atoms: bool = False,
        plotter: pv.Plotter | None = None,
        **kwargs,
    ) -> pv.Plotter:
        """_summary_

        Args:
            coordinates (np.ndarray): Numpy array of coordinates of shape (nx,ny,nz,3) or of shape (nx*ny*nz,3)
            isosurfaces (int | list[float]): Number of isosurfaces, if int. Values of isosurfaces if list of floats
            psi_r (list[np.ndarray]): List of numpy arrays. Each numpy array represents scalar field data of shape (nx,ny,nz) or of shape (nx*ny*nz)
            plot_lattice_vectors (bool, optional): If the lattice vectors are plotted as arrows. Defaults to True.
            extend_data (bool, optional): If the orbitals are also plotted for all neighbouring cells besides the computational cell. Defaults to False.
            extend_atoms (bool, optional): If the atoms are also plotted for all neighbouring cells besides the computational cell. Defaults to False.
            plotter (pv.Plotter | None, optional): _description_. Defaults to None.
            kwargs: Arguments passed to dopyqo.plotting.plot_3d_data

        Returns:
            pv.Plotter: PyVista plotter object
        """
        max_min = psi_r[0].shape

        origin = np.zeros(3)

        ################################
        atom_positions_hartree = self.atom_positions_hartree.copy()
        # # atom positions in crystal coordinates
        # atom_positions_crystal = np.array([np.linalg.solve(self.a.T, x) for x in atom_positions_hartree])
        # # atom positions in datagrid coordinates
        # atom_positions_crystal = np.array([x * (max_min) for x in atom_positions_crystal])

        # Move atom positions into home cell spanned by lattice vectors starting from origin
        atom_positions_hartree_in_home_cell = []
        for pos_tmp in atom_positions_hartree:
            pos_new_tmp = pos_tmp.copy()
            inside, coeffs = is_point_inside_parallelepiped(pos_tmp - origin, self.a[0], self.a[1], self.a[2])
            if not inside:
                move_coeffs = -np.sign(coeffs) * np.ceil(np.abs(np.minimum(coeffs, 0)))
                pos_new_tmp = pos_tmp - origin + self.a.T @ move_coeffs

                inside_new, _ = is_point_inside_parallelepiped(pos_new_tmp, self.a[0], self.a[1], self.a[2])
                if not inside_new:
                    print(
                        f"{ORANGE}Plot warning: Could not move atom at position {pos_tmp} into home cell. This should never happen. Please, contact a developer!"
                        + f"\n\tMore information: {inside=} | {coeffs=} | {move_coeffs=} | {pos_new_tmp=} | {self.a=}{RESET_COLOR}"
                    )
                    pos_new_tmp = pos_tmp
            atom_positions_hartree_in_home_cell.append(pos_new_tmp)
        atom_positions_hartree = np.array(atom_positions_hartree_in_home_cell)

        atom_labels = [f"{self.atoms[idx_tmp]['element']} [{idx_tmp}]" for idx_tmp, _ in enumerate(atom_positions_hartree)]

        if extend_atoms:
            n_max_x = 2
            n_max_y = 2
            n_max_z = 2
            ranges = itertools.product(
                range(-1, n_max_x + 1),
                range(-1, n_max_y + 1),
                range(-1, n_max_z + 1),
            )
        else:
            n_max_x = 1
            n_max_y = 1
            n_max_z = 1
            ranges = itertools.product(
                range(-n_max_x - 0, n_max_x + 1),
                range(-n_max_y - 0, n_max_y + 1),
                range(-n_max_z - 0, n_max_z + 1),
            )

        t_vecs_unordered = [[np.dot(self.a.T, np.array([nx, ny, nz])), [nx, ny, nz]] for nx, ny, nz in ranges]
        t_vecs = sorted(t_vecs_unordered, key=lambda x: np.linalg.norm(x[0], ord=2))
        sphere_centers_hartree = []
        sphere_labels = []
        for t_vec, (nx, ny, nz) in t_vecs:
            for idx_tmp, pos in enumerate(atom_positions_hartree):
                sphere_centers_hartree.append(pos + t_vec)
                sphere_labels.append(f"{self.atoms[idx_tmp]['element']} [{idx_tmp}]")
        sphere_centers_hartree_filtered = []
        sphere_labels_filtered = []
        for x, label in zip(sphere_centers_hartree, sphere_labels):
            inside, _ = is_point_inside_parallelepiped(x - origin, self.a[0] * n_max_x, self.a[1] * n_max_y, self.a[2] * n_max_z, tol=1e-2)
            if inside:
                sphere_centers_hartree_filtered.append(x)
                sphere_labels_filtered.append(label)

        sphere_centers_hartree = sphere_centers_hartree_filtered
        sphere_labels = sphere_labels_filtered

        sphere_centers_hartree = np.array(sphere_centers_hartree)

        lattice_vectors_meshes = [pv.Arrow(direction=v, shaft_radius=2e-2, tip_radius=5e-2, scale="auto") for v in self.a]
        meshes_tmp = []
        if plot_lattice_vectors:
            meshes_tmp.extend(lattice_vectors_meshes)

        for data in psi_r:
            if not np.allclose(data.imag, 0.0):
                print(f"{ORANGE}Plot warning: Data is complex. Only real part is plotted, imaginary part is discarded!{RESET_COLOR}")
            data = data.real
            if "meshes" in kwargs:
                kwargs["meshes"].extend(meshes_tmp)
            else:
                kwargs["meshes"] = meshes_tmp
            plotter = plotting.plot_3d_data(
                coordinates=coordinates,
                data=data,
                grid_vectors=self.a.copy(),
                plotter=plotter,
                atom_positions=sphere_centers_hartree,
                atom_labels=sphere_labels,
                isosurfaces=isosurfaces,
                extend_data=extend_data,
                **kwargs,
            )
        return plotter


# Adjusted from pymatgen.core.lattice.get_wigner_seitz_cell
# https://github.com/materialsproject/pymatgen/blob/v2025.5.28/src/pymatgen/core/lattice.py#L1304-L1326
def wigner_seitz_cell(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> list[list[np.ndarray]]:
    """Calculate facets of Wigner-Seitz cell. See plot_wigner_seitz_cell to plot the return values of this function

    Args:
        a1 (np.ndarray): First real-space lattice vector
        a2 (np.ndarray): Second real-space lattice vector
        a3 (np.ndarray): Third real-space lattice vector

    Returns:
        list[list[np.ndarray]]: List of facets. Each facet is a list of coordinates. Each coordinate is a numpy array.
                                Each facets is a face of the Wigner-Seitz cell
    """
    list_k_points = []
    for ii, jj, kk in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        list_k_points.append(ii * a1 + jj * a2 + kk * a3)

    tess = Voronoi(list_k_points)
    out = []
    for r in tess.ridge_dict:
        if r[0] == 13 or r[1] == 13:
            out.append([tess.vertices[i] for i in tess.ridge_dict[r]])

    return out


def plot_wigner_seitz_cell(facets):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect([1, 1, 1])

    poly = Poly3DCollection(facets, facecolors="lightblue", edgecolors="k", alpha=0.5)
    ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def wigner_seitz_in_radius(facets) -> float:
    def plane_distance_from_origin(facet):
        v0, v1, v2 = map(np.array, facet[:3])
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            return None  # degenerate facet
        nhat = n / norm
        return abs(np.dot(nhat, v0))

    # Compute distances
    dists = [plane_distance_from_origin(f) for f in facets]
    dists = [d for d in dists if d is not None]

    # The in-radius
    r_in = min(dists)

    return r_in


def is_point_inside_parallelepiped(point: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, tol=1e-8) -> tuple[bool, np.ndarray]:
    """Check if point is in a parallelepiped spanned by three vectors

    Args:
        point (np.ndarray): Point
        v1 (np.ndarray): First spanning vector
        v2 (np.ndarray): Second spanning vector
        v3 (np.ndarray): Third spanning vector

    Returns:
        tuple[bool, np.ndarray]: Tuple of: Is point in parallelepiped? [\alpha, \beta a2, \gamma] of point = \alpha v1 + \beta v2 + \gamma v3
    """
    M = np.vstack([v1, v2, v3]).T
    lambdas = np.linalg.solve(M, point)
    return np.all(lambdas >= 0 - tol) and np.all(lambdas <= 1 + tol), lambdas
