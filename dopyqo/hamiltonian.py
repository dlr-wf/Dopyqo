import sys
import logging
from warnings import warn
import os
import time
import copy
from enum import Enum
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.problems.electronic_structure_result import (
    ElectronicStructureResult,
)
from qiskit_algorithms.minimum_eigensolvers.vqe import VQEResult
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_algorithms.minimum_eigensolvers import VQE, VQEResult
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
import qiskit_algorithms.optimizers as optim
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.primitives import Estimator, StatevectorEstimator
import pyscf.fci
from pyscf import gto, scf, cc
from pyscf.fci.cistring import str2addr
import tencirchem
import scipy
import dopyqo
from dopyqo import calc_matrix_elements
import dopyqo.fci_vector_matrix
from dopyqo.helpers import tcc_helpers
from dopyqo import fci_vector_matrix
from dopyqo.colors import *


def transform_index(i, n_orbitals):
    """From tequila order (up, down, up, down) to TenCirChem order (up, up, ..., down, down)"""
    i_orbital = i // 2
    i_spin = i % 2
    return i_orbital + n_orbitals * i_spin


class Hamiltonian:
    """Defines a second-quanitzation hamiltonian of the form

    H = \sum_{pq} h_{pq} a_p^\dagger a_q + 1/2 \sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s + E_{const.}

    with h_{pqrs} = \int \int \phi^*_p(r_1)  \phi^*_q(r_2) \phi_r(r_2)  \phi_s(r_1) / |r_1-r_2| dr_1 dr_2

    h_{pq} and h_{pqrs} are defined in terms of N spatial orbitals.

    Args:
        h_pq (np.ndarray): One-electron matrix elements. Shape (N, N).
        h_pqrs (np.ndarray): Two-electron matrix elements. Shape (N, N, N, N).
        occupations (np.ndarray): Occupations of the N spatial orbitals taking values either 0 or 1, and summing up to half of the number of electrons. Shape (N,)
        constants (float | dict[str, float], optional): Energy offset E_{const.} as float or dict of str, float pairs. Defaults to 0.0.
        reference_energy (float, optional): Reference energy used for VQE calculations. Defaults to 0.0.
    """

    def __init__(
        self,
        h_pq: np.ndarray,
        h_pqrs: np.ndarray,
        occupations: np.ndarray,
        constants: float | dict[str, float] = 0.0,
        reference_energy: float = 0.0,
    ):
        # TODO: Currently no support for spin-polarized h_pq, h_pqrs
        assert h_pq.ndim == 2
        assert h_pqrs.ndim == 4

        assert h_pq.shape[0] == h_pq.shape[1] == h_pqrs.shape[0] == h_pqrs.shape[1] == h_pqrs.shape[2] == h_pqrs.shape[3]

        # assert occupations.ndim == 1
        # assert occupations.shape[0] == h_pq.shape[0]

        self.norb = h_pq.shape[0]

        self.nspin = 1
        self.occupations = occupations
        for x in self.occupations:
            if not (np.isclose(int(x), 0) or np.isclose(int(x), 1)):
                print(
                    f"{RED}Hamiltonian error: occupations must be either 0 or 1 but found value {x}!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        self.nelec = round(np.sum(self.occupations))
        if self.nelec >= self.norb:
            print(
                f"{RED}Hamiltonian error: Half the number of electrons ({self.nelec}) is equal or greater, but has to be less, ",
                f"than the number of spatial orbitals ({self.norb})!{RESET_COLOR}",
            )
            sys.exit(1)

        logging.info("Checking matrix element symmetries...")
        if calc_matrix_elements.check_symmetry_one_body_matrix(h_pq) is False:
            print(f"{ORANGE}Hamiltonian warning: h_pq does not obey one body matrix symmetries (hermiticity)!{RESET_COLOR}")
        if False in (sym := calc_matrix_elements.check_symmetry_two_body_matrix(h_pqrs)):
            print(
                f"{ORANGE}Hamiltonian warning: h_pqrs does not obey two body matrix symmetries (swap symmetry {'fulfilled' if sym[0] else 'not fulfilled'}, "
                f"hermiticity {'fulfilled' if sym[1] else 'not fulfilled'}, hermiticity+swap {'fulfilled' if sym[2] else 'not fulfilled'})!{RESET_COLOR}"
            )

        self.h_pq = h_pq
        self.h_pqrs = h_pqrs
        if not isinstance(constants, dict) and not isinstance(constants, float):
            print(
                f"{RED}Hamiltonian error: Provided value for 'constants' is of type {type(constants)} but only dict or float are supported!{RESET_COLOR}"
            )
            sys.exit(1)
        if isinstance(constants, dict):
            key_types = set(map(type, constants.keys()))
            value_types = set(map(type, constants.values()))
            if key_types != {str} or value_types == {float}:
                print(
                    f"{RED}Hamiltonian error: Provided value for 'constants' is dict with keys of type {key_types} and values of type {value_types} "
                    + f"but only keys of type str and values of type float are supported!{RESET_COLOR}"
                )
                sys.exit(1)
        self.constants = constants
        self.reference_energy = reference_energy

        self.fci_energy = np.nan
        self.fci_solver: pyscf.fci.direct_spin1.FCIBase = pyscf.fci.direct_spin1.FCIBase()
        self.fci_evs: float = np.nan
        self.fci_evcs: list = []
        self.qiskit_elec_struc_result = ElectronicStructureResult()
        self.qiskit_vqe_result = VQEResult()
        self.qiskit_problem = None
        self.qiskit_ansatz = QuantumCircuit()
        self.qiskit_pauli_sum_op = SparsePauliOp(["I"])
        self.qiskit_vqe_counts = {}
        self.qiskit_vqe_values = {}
        self.qiskit_vqe_initial_state = None
        self.qiskit_vqe_ansatz = {}
        self.qiskit_vqe_solver = {}
        self.qiskit_vqe_result = None

        self.tcc_vqe_counts = None
        self.tcc_vqe_values = None
        self.tcc_vqe_result = None
        self.tcc_ansatz = None

        # Take the values of the latest ran VQE result, either from TenCirChem or qiskit
        self.vqe_counts = None
        self.vqe_values = None
        self.vqe_result = None

        # Used in solve_vqe when using TenCirChem and set in solve_hf
        self.mo_coeff = None
        self.mf = None

    def to_qiskit_problem(self, auto_index_order=True) -> ElectronicStructureProblem:
        """Create Qiskit ElectronicStructureProblem object.

        Args:
            auto_index_order (bool, optional): auto_index_order parameter passed to Qiskit ElectronicIntegrals.from_raw_integrals. Defaults to True.

        Returns:
            ElectronicStructureProblem: Created Qiskit ElectronicStructureProblem
        """
        num_particles = (
            self.nelec,
            self.nelec,
        )

        # Qiskit calculation
        integrals = ElectronicIntegrals.from_raw_integrals(self.h_pq, self.h_pqrs, auto_index_order=auto_index_order)
        qiskit_energy = ElectronicEnergy(integrals)

        if isinstance(self.constants, dict):
            for key, val in self.constants.items():
                qiskit_energy.constants[key] = val
        else:
            qiskit_energy.constants["energy_offset"] = self.constants
        # qiskit_energy.constants["frozen_core_energy"] = self.constants
        qiskit_problem = ElectronicStructureProblem(qiskit_energy)

        # number of particles for spin-up, spin-down
        qiskit_problem.num_particles = num_particles
        qiskit_problem.num_spatial_orbitals = self.norb

        qiskit_problem.reference_energy = self.reference_energy

        self.qiskit_problem = qiskit_problem

        return qiskit_problem

    def hf_energy(self) -> float:
        """Calculate the mean-field energy of the single Slater determinant given by self.occupations.
        If this Slater determinant is the Hartree-Fock state, the energy is the Hartree-Fock energy.

        The calculated energy is:
        E = 2 \sum_i h_{ii} + \sum_{ij) (2 h_{ijji} - h_{ijij}) + E_{const.}
        where the sums go over the occupied orbitals

        Returns:
            float: The calculated mean-field energy
        """
        h_pq = self.h_pq
        h_pqrs = self.h_pqrs

        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        # Same as frozen core energy contribution. Note that we only sum over occupied orbitals
        h_pq_energy = 2 * np.sum(h_pq.diagonal() * self.occupations)

        h_pqrs_energy = 0
        for i in range(self.nelec):
            for j in range(self.nelec):
                # Hartree term and exchange term
                h_pqrs_energy += (2 * h_pqrs[i, j, j, i] - h_pqrs[i, j, i, j]) * self.occupations[i] * self.occupations[j]

        return (h_pq_energy + h_pqrs_energy + energy_offset).real

    def hf_energy_qiskit(self, mapper: FermionicMapper | None = JordanWignerMapper()):
        """Calculate the mean-field energy of the single Slater determinant given by self.occupations.
        If this Slater determinant is the Hartree-Fock state, the energy is the Hartree-Fock energy.

        The energy is calculated using Qiskit by using its HartreeFock class and statevector simulation.

        The calculated energy is:
        E = 2 \sum_i h_{ii} + \sum_{ij) (2 h_{ijji} - h_{ijij}) + E_{const.}
        where the sums go over the occupied orbitals

        Args:
            mapper (FermionicMapper | None, optional): _description_. Defaults to JordanWignerMapper().

        Returns:
            float: The calculated mean-field energy
        """
        problem = self.to_qiskit_problem()
        problem.reference_energy = self.fci_energy

        if mapper is None:
            mapper = JordanWignerMapper()
        energy = problem.hamiltonian
        fermionic_op = energy.second_q_op()
        pauli_sum_op = mapper.map(fermionic_op)

        logging.info("pauli_sum_op.num_qubits: %i", pauli_sum_op.num_qubits)

        # Qubit mapping
        logging.info("Using Hartree-Fock state")
        initial_state = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        )
        # estimator = Estimator()
        estimator = StatevectorEstimator()

        hf_energy = estimator.run([(initial_state, pauli_sum_op)]).result()[0].data.evs
        hf_energy += sum((energy.constants.values())).real
        logging.info("HF-result: %f", hf_energy)

        return hf_energy

    def fci_statevector(self) -> Statevector:
        """Return the statevector of the FCI state

        Returns:
            Statevector: Qiskit statevector object of the FCI state
        """
        if len(self.fci_evcs) == 0:
            self.solve_fci()
        # TODO: Not optimal to calculate nelec manually. Add this to Wfc class?
        nelec = (self.nelec, self.nelec)
        norb = self.norb
        fci_state = dopyqo.fci_vector_matrix.statevector_from_fci(fci_vector=self.fci_evcs[0], nelec=nelec, norb=norb)
        return fci_state

    def vqe_statevector(self, vqe_params: np.ndarray | None = None) -> Statevector:
        """Return the statevector of the state prepared by the TenCirChem VQE ansatz

        Args:
            vqe_params (np.ndarray | None, optional): Parameters used for the VQE ansatz. If None, the parameters determined of a prior VQE optimization are used. Defaults to None.

        Raises:
            ValueError: If no TenCirChem VQE optimization has been performed and no parameters are given.

        Returns:
            Statevector: Qiskit statevector object of the state prepared by the TenCirChem VQE ansatz
        """
        if self.tcc_vqe_result is not None or vqe_params is not None:
            logging.info("Calculating VQE statevector from TenCirChem result!")
            ci_vec = self.tcc_ansatz.civector(self.tcc_vqe_params if vqe_params is None else vqe_params)
            nelec = (
                self.nelec,
                self.nelec,
            )
            # state = fci_vector_matrix.statevector_from_civector_restricted(ci_vec, nelec, self.norb)

            dim_spin = np.sqrt(ci_vec.size)
            if not np.isclose(int(dim_spin), dim_spin):
                print(
                    f"{RED}Statevector error: Size of TenCirChem statevector ({ci_vec.size}) is not n^2 (n={dim_spin}) where n should be a squared integer. This should not happen. Please, contact a developer!{RESET_COLOR}"
                )
                sys.exit(1)
            dim_spin = int(dim_spin)
            new_shape = (dim_spin,) * 2
            state = fci_vector_matrix.statevector_from_fci(ci_vec.reshape(new_shape), nelec, self.norb)
        else:
            raise ValueError("No TenCirChem VQE optimization has been performed and no parameters are given. Cannot calculate VQE statevector!")

        return state

    def vqe_statevector_qiskit(self, vqe_params: np.ndarray | None = None) -> Statevector:
        """Return the statevector of the state prepared by the Qiskit VQE ansatz

        Args:
            vqe_params (np.ndarray | None, optional): Parameters used for the VQE ansatz. If None, the parameters determined of a prior VQE optimization are used. Defaults to None.

        Raises:
            ValueError: If no Qiskit VQE optimization has been performed and no parameters are given.

        Returns:
            Statevector: Qiskit statevector object of the state prepared by the Qiskit VQE ansatz
        """
        if self.qiskit_vqe_result is not None or vqe_params is not None:
            logging.info("Calculating VQE statevector from qiskit result!")
            qc = self.qiskit_ansatz
            if vqe_params is None:
                vqe_params = self.qiskit_vqe_result.optimal_parameters
            else:
                # param_vec_elements = ParameterVector("t", length=len(vqe_params)).params
                # vqe_params = {vec_elem: val for vec_elem, val in zip(param_vec_elements, vqe_params)}
                vqe_params = vqe_params

            bc = qc.assign_parameters(vqe_params)

            n_qbits = self.qiskit_pauli_sum_op.num_qubits

            state = Statevector.from_int(i=0, dims=2**n_qbits)
            state = state.evolve(bc)
        else:
            raise ValueError("No Qiskit VQE optimization has been performed and no parameters are given. Cannot calculate VQE statevector!")

        return state

    def run_qiskit_ansatz(
        self,
        parameters: np.ndarray,
        UCCSD_reps: int = 1,
        mapper: FermionicMapper | None = JordanWignerMapper(),
    ) -> float:
        """Run the Qiskit UCCSD ansatz and return the estimated energy

        Args:
            parameters (np.ndarray): Parameters used for the ansatz.
            UCCSD_reps (int, optional): Number of times the UCCSD ansatz is repeated with new parameters. Defaults to 1.
            mapper (FermionicMapper | None, optional): Used Fermion-to-qubit mapping. Defaults to JordanWignerMapper().

        Returns:
            float: The energy of the ansatz
        """
        problem = self.to_qiskit_problem()

        if mapper is None:
            mapper = JordanWignerMapper()
        energy = problem.hamiltonian
        energy_offset = np.sum(list(energy.constants.values()))
        fermionic_op = energy.second_q_op()
        pauli_sum_op = mapper.map(fermionic_op)

        logging.info("pauli_sum_op.num_qubits: %f", pauli_sum_op.num_qubits)

        # Qubit mapping
        logging.info("Using Hartree-Fock state")
        initial_state = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        )

        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=initial_state,
            reps=UCCSD_reps,
        )

        estimator = Estimator()

        job = estimator.run([ansatz], [pauli_sum_op], [parameters])
        estimator_result = job.result().values  #  + energy_offset

        self.qiskit_vqe_initial_state = initial_state
        self.qiskit_problem = problem
        self.qiskit_ansatz = ansatz
        self.qiskit_pauli_sum_op = pauli_sum_op

        return estimator_result

    def run_tcc_ansatz(
        self, parameters: np.ndarray, UCCSD_reps: int = 1, excitations: list[tuple[int, ...]] | None = None, qiskit_equivalent: bool = False
    ) -> float:
        """
        Run the TenCirChem UCCSD ansatz and return the estimated energy

        Args:
            parameters (np.ndarray): Parameters used for the VQE ansatz.
            UCCSD_reps (int, optional): Number of times the UCCSD ansatz is repeated with new parameters. Defaults to 1.
            excitations (list[tuple[int, ...]] | None, optional): Excitations used in the ansatz. If None all double and single excitations are used. Defaults to None.
            qiskit_equivalent (bool, optional): If set to True, the same excitations and ordering as in Qiskit is used.
                                                With this, the provided parameters give the same result as the run_qiskit_ansatz function. Defaults to False.

        Returns:
            float: The energy of the ansatz
        """
        # NOTE: TenCirChem can only simulate spin-restricted systems!
        nelec = (
            self.nelec,
            self.nelec,
        )

        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        two_mo = self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3)

        assert np.allclose(one_mo.real, one_mo), "Single electron matrix elements are complex. TenCirChem only supports real matrix elements!"
        assert np.allclose(two_mo.real, two_mo), "Two electron matrix elements are complex. TenCirChem only supports real matrix elements!"
        assert np.allclose(energy_offset.real, energy_offset), "Constant energy offset is complex. TenCirChem only supports real offsets!"
        one_mo = one_mo.real
        two_mo = two_mo.real
        constant = energy_offset.real

        kwargs = {"mo_coeff": np.eye(self.norb), "init_method": "zeros", "run_hf": False, "run_mp2": False, "run_ccsd": False, "run_fci": False}
        tcc_uccsd = tencirchem.UCCSD.from_integral(one_mo, two_mo, nelec, constant, **kwargs)
        # See https://tensorcircuit.github.io/TenCirChem-NG/faq.html#why-are-the-number-of-excitation-operators-and-the-number-of-parameters-different-what-is-param-ids
        tcc_uccsd.param_ids = None
        if excitations is not None:
            tcc_uccsd.ex_ops = excitations
        else:
            singles_tcc = tcc_helpers.get_ex1_ops(self.norb, nelec)
            doubles_tcc = tcc_helpers.get_ex2_ops(self.norb, nelec)
            # NOTE: Sort indices of excitaions like qiskit
            singles_tcc_sorted = sorted(singles_tcc)
            doubles_tcc_sorted = sorted([(*sorted(x[:2]), *sorted(x[2:])) for x in doubles_tcc])
            tcc_uccsd.ex_ops = doubles_tcc_sorted + singles_tcc_sorted

            logging.info("TenCirChem UCCSD has %s single and %s doubles.", len(singles_tcc_sorted), len(doubles_tcc_sorted))

            if qiskit_equivalent:
                #####################################################################################
                ##                                  Qiskit UCCSD                                   ##
                #####################################################################################
                integrals = ElectronicIntegrals.from_raw_integrals(one_mo, two_mo, auto_index_order=True)
                qiskit_energy = ElectronicEnergy(integrals)
                qiskit_problem = ElectronicStructureProblem(qiskit_energy)
                qiskit_problem.num_particles = nelec
                qiskit_problem.num_spatial_orbitals = self.norb
                mapper = JordanWignerMapper()
                initial_state = HartreeFock(
                    num_spatial_orbitals=qiskit_problem.num_spatial_orbitals,
                    num_particles=qiskit_problem.num_particles,
                    qubit_mapper=mapper,
                )
                qiskit_uccsd = UCCSD(
                    qiskit_problem.num_spatial_orbitals,
                    qiskit_problem.num_particles,
                    mapper,
                    initial_state=initial_state,
                    reps=1,
                )

                #####################################################################################
                ##               Making TenCirChem UCCSD equivalent to Qiskit UCCSD                ##
                #####################################################################################
                # The indices of the creation/annihilation operators in the double excitations are sorted in qiskit.
                # We do the same for the TenCirChem excitations
                doubles_tcc_sorted = [(*sorted(x[:2]), *(sorted(x[2:]))) for x in doubles_tcc]
                tcc_uccsd.ex_ops = singles_tcc + doubles_tcc_sorted

                # Bringing the qiskit excitations into the same notation as now used in tcc_uccsd.ex_ops to make them comparable
                qiskit_ex_list = [
                    (x[1][0], x[0][0]) if len(x[0]) == 1 else (x[1][0], x[1][1], x[0][0], x[0][1]) for x in qiskit_uccsd.excitation_list
                ]

                # We map the indices of qiskit excitations to indices of TenCirChem excitations
                tcc_idc = [tcc_uccsd.ex_ops.index(ex_qiskit) for ex_qiskit in qiskit_ex_list]

                # With this mapping we now reorder the excitations in the TenCirChem UCCSD to have the same ordering as in the qiskit UCCSD
                tcc_uccsd.ex_ops = [tcc_uccsd.ex_ops[idx] for idx in tcc_idc]

            tcc_uccsd.ex_ops *= UCCSD_reps

        #####################################################################################
        ##                                 Running Ansatz                                  ##
        #####################################################################################
        assert len(parameters) == tcc_uccsd.n_params, f"TenCirChem ansatz expects {tcc_uccsd.n_params} parameters but {len(parameters)} were given!"
        tcc_energy = tcc_uccsd.energy(parameters)

        self.tcc_ansatz = tcc_uccsd

        return tcc_energy

    def solve_vqe(
        self,
        optimizer: dopyqo.VQEOptimizers = dopyqo.VQEOptimizers.L_BFGS_B,
        UCCSD_reps: int = 1,
        qiskit_equivalent: bool = False,
        initial_params: np.ndarray | None = None,
        maxiter: int | None = None,
        excitations: list[tuple[int, ...]] | dopyqo.ExcitationPools = dopyqo.ExcitationPools.SINGLES_DOUBLES,
    ) -> scipy.optimize.OptimizeResult:
        """Run a VQE optimization with TenCirChem and a UCCSD ansatz. The initial state, that is prepared before the parametrized ansatz is applied,
        is a Slater determinant matching self.occupations.

        Args:
            optimizer (dopyqo.VQEOptimizers, optional): Classical optimizer used in the VQE. Defaults to dopyqo.VQEOptimizers.L_BFGS_B.
            UCCSD_reps (int, optional): Number of times the UCCSD ansatz is repeated with new parameters. Defaults to 1.
            qiskit_equivalent (bool, optional): If set to True, the same excitations and ordering as in Qiskit is used.
                                                With this, the provided parameters give the same result as the solve_vqe_qiskit function. Defaults to False.
            initial_params (np.ndarray | None, optional): Parameters to start the VQE optimization with. Defaults to None.
            maxiter (int | None, optional): Maximum number of optimizer iterations before stopping the optimization. If None, large defaults are used. Defaults to None.
            excitations (list[tuple[int, ...]] | dopyqo.ExcitationPools, optional): Excitations used in the ansatz. Defaults to dopyqo.ExcitationPools.SINGLES_DOUBLES where all double and single excitations are used.

        Returns:
            scipy.optimize.OptimizeResult: Scipy optimization result
        """
        # tencirchem.set_backend("cupy")
        # NOTE: TenCirChem can only simulate spin-restricted systems!

        nelec = (
            self.nelec,
            self.nelec,
        )

        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        two_mo = self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3)

        if not np.allclose(one_mo.real, one_mo):
            print(
                f"{RED}Hamiltonian error: Single electron matrix elements are complex. TenCirChem only supports real matrix elements!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(two_mo.real, two_mo):
            print(
                f"{RED}Hamiltonian error: Two electron matrix elements are complex. TenCirChem only supports real matrix elements!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(energy_offset.real, energy_offset):
            print(
                f"{RED}Hamiltonian error: Constant energy offset is complex. TenCirChem only supports real offsets!!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        one_mo = one_mo.real
        two_mo = two_mo.real
        energy_offset = energy_offset.real

        # mo_coeff = np.eye(self.norb) if self.mo_coeff is None else self.mo_coeff
        # print(f"{mo_coeff=}")
        mo_coeff = np.eye(self.norb)
        kwargs = {
            "mo_coeff": mo_coeff,
            "init_method": "zeros",
            "run_hf": False,
            "run_mp2": False,
            "run_ccsd": False,
            "run_fci": False,
            # "engine": "civector",
        }
        tcc_uccsd = tencirchem.UCCSD.from_integral(one_mo, two_mo, nelec, energy_offset, **kwargs)
        # print(f"\t Using TenCirChem engine {tcc_uccsd.engine}")
        # See https://tensorcircuit.github.io/TenCirChem-NG/faq.html#why-are-the-number-of-excitation-operators-and-the-number-of-parameters-different-what-is-param-ids
        tcc_uccsd.param_ids = None
        if isinstance(excitations, list):
            tcc_uccsd.ex_ops = excitations
        else:
            singles_tcc = tcc_helpers.get_ex1_ops(self.norb, nelec)
            doubles_tcc = tcc_helpers.get_ex2_ops(self.norb, nelec)
            # Sort indices of excitations like qiskit
            singles_tcc_sorted = sorted(singles_tcc)
            doubles_tcc_sorted = sorted([(*sorted(x[:2]), *sorted(x[2:])) for x in doubles_tcc])

            match excitations:
                case dopyqo.ExcitationPools.SINGLES:
                    excitation_list = singles_tcc_sorted
                case dopyqo.ExcitationPools.DOUBLES:
                    excitation_list = doubles_tcc_sorted
                case dopyqo.ExcitationPools.SINGLES_DOUBLES:
                    excitation_list = doubles_tcc_sorted + singles_tcc_sorted
                case _:
                    print(
                        f"{RED}VQE error: Parameter {excitations=} is not supported. Use a list of excitation tuples or dopyqo.ExcitationPools types!{RESET_COLOR}"
                    )
                    sys.exit(1)

            tcc_uccsd.ex_ops = excitation_list
            logging.info("TenCirChem UCCSD has %s excitations.", len(excitation_list))

            if qiskit_equivalent:
                #####################################################################################
                ##                                  Qiskit UCCSD                                   ##
                #####################################################################################
                integrals = ElectronicIntegrals.from_raw_integrals(one_mo, two_mo, auto_index_order=True)
                qiskit_energy = ElectronicEnergy(integrals)
                qiskit_problem = ElectronicStructureProblem(qiskit_energy)
                qiskit_problem.num_particles = nelec
                qiskit_problem.num_spatial_orbitals = self.norb
                mapper = JordanWignerMapper()
                initial_state = HartreeFock(
                    num_spatial_orbitals=qiskit_problem.num_spatial_orbitals,
                    num_particles=qiskit_problem.num_particles,
                    qubit_mapper=mapper,
                )
                qiskit_uccsd = UCCSD(
                    qiskit_problem.num_spatial_orbitals,
                    qiskit_problem.num_particles,
                    mapper,
                    initial_state=initial_state,
                    reps=1,
                )
                # Fix UCCSD operator order: first doubles, then singles
                doubles = [op for (op, excitation) in zip(qiskit_uccsd.operators, qiskit_uccsd.excitation_list) if len(excitation[0]) == 2]
                singles = [op for (op, excitation) in zip(qiskit_uccsd.operators, qiskit_uccsd.excitation_list) if len(excitation[0]) == 1]
                qiskit_uccsd.operators = doubles + singles
                qiskit_uccsd._invalidate()

                #####################################################################################
                ##               Making TenCirChem UCCSD equivalent to Qiskit UCCSD                ##
                #####################################################################################
                # Bringing the qiskit excitations into the same notation as now used in tcc_uccsd.ex_ops to make them comparable
                qiskit_ex_list = [
                    (x[1][0], x[0][0]) if len(x[0]) == 1 else (x[1][0], x[1][1], x[0][0], x[0][1]) for x in qiskit_uccsd.excitation_list
                ]

                # We map the indices of qiskit excitations to indices of TenCirChem excitations
                tcc_idc = [tcc_uccsd.ex_ops.index(ex_qiskit) for ex_qiskit in qiskit_ex_list if ex_qiskit in tcc_uccsd.ex_ops]

                # With this mapping we now reorder the excitations in the TenCirChem UCCSD to have the same ordering as in the qiskit UCCSD
                tcc_uccsd.ex_ops = [tcc_uccsd.ex_ops[idx] for idx in tcc_idc]

            tcc_uccsd.ex_ops *= UCCSD_reps

        # Setting tcc_uccsd.init_state to match self.occupations
        binary_string_tmp = "0b" + "".join(map(str, map(int, reversed(self.occupations)))) * 2
        binary_string_tmp = int(binary_string_tmp, base=2)
        ci_strings_tmp = tcc_uccsd.get_ci_strings()  # .get()
        ci_string_idx_tmp = np.where(ci_strings_tmp == binary_string_tmp)[0][0]
        #
        init_statevector_tmp = np.zeros(len(ci_strings_tmp))
        init_statevector_tmp[ci_string_idx_tmp] = 1.0
        tcc_uccsd.init_state = list(init_statevector_tmp)

        #####################################################################################
        ##                                  Running VQE                                    ##
        #####################################################################################
        # Manual optimization
        times_tcc = []
        eval_count_tcc = 0
        counts_tcc = []
        values_tcc = []
        tcc_vqe_params_lst = []

        def cost(x):
            nonlocal times_tcc
            nonlocal eval_count_tcc
            nonlocal counts_tcc
            nonlocal values_tcc

            tcc_energy = tcc_uccsd.energy(x)
            times_tcc.append(time.perf_counter())

            eval_count_tcc += 1
            print(
                f"Optimizer evaluation #{eval_count_tcc}, Diff. to ref.: {np.abs(tcc_energy - self.reference_energy)}",
                end="\r",
                flush=True,
            )
            counts_tcc.append(eval_count_tcc)
            values_tcc.append(tcc_energy)
            tcc_vqe_params_lst.append(x)

            return tcc_energy

        n_params_tmp = tcc_uccsd.n_params
        if initial_params is None:
            initial_params = np.zeros(n_params_tmp)
        else:
            if len(initial_params) != n_params_tmp:
                print(
                    f"{RED}VQE error: initial_params has length {len(initial_params)} but VQE ansatz has {n_params_tmp} parameters!{RESET_COLOR}",
                    file=sys.stderr,
                )
                sys.exit(1)
        logging.info("Using optimizer %s", optimizer)

        match optimizer:
            case dopyqo.VQEOptimizers.COBYLA:
                maxiter = 1e6 if maxiter is None else maxiter
                options = dict(maxiter=maxiter, tol=1e-10)
                optimizer_func = "COBYLA"
            case dopyqo.VQEOptimizers.L_BFGS_B:
                maxiter = 1e6 if maxiter is None else maxiter
                options = dict(maxfun=maxiter * 4 * tcc_uccsd.n_params, maxiter=maxiter, ftol=1e-13)
                optimizer_func = "L-BFGS-B"
            case dopyqo.VQEOptimizers.ExcitationSolve:
                from excitationsolve import ExcitationSolveScipy

                maxiter = 100 if maxiter is None else maxiter
                excsolve_obj = ExcitationSolveScipy(maxiter=maxiter, tol=1e-10, save_parameters=True)
                optimizer_func = excsolve_obj.minimize
                options = dict(print_instead_logging=False, reference_energy=self.reference_energy)
            case _:
                print(f"{RED}VQE error: Parameter {optimizer=} is not supported. Use dopyqo.VQEOptimizers types!{RESET_COLOR}")
                sys.exit(1)
        res_tcc = scipy.optimize.minimize(cost, initial_params, method=optimizer_func, options=options)
        params_tcc = res_tcc.x

        match optimizer:
            case dopyqo.VQEOptimizers.ExcitationSolve:
                counts_tcc = excsolve_obj.nfevs
                values_tcc = excsolve_obj.energies

        self.tcc_vqe_params = params_tcc
        self.tcc_vqe_params_lst = tcc_vqe_params_lst
        # To have the same parameter values in the qiskit UCCSD circuit we have to flip the sign
        if qiskit_equivalent:
            self.qiskit_vqe_params = -params_tcc

        self.tcc_vqe_counts = np.array(counts_tcc)
        self.tcc_vqe_values = np.array(values_tcc)

        self.tcc_vqe_result = res_tcc
        self.tcc_ansatz = tcc_uccsd

        self.vqe_counts = self.tcc_vqe_counts
        self.vqe_values = self.tcc_vqe_values
        self.vqe_result = self.tcc_vqe_result

        return self.tcc_vqe_result

    def solve_vqe_qiskit(
        self,
        optimizer: dopyqo.VQEOptimizers = dopyqo.VQEOptimizers.L_BFGS_B,
        UCCSD_reps: int = 1,
        mapper: FermionicMapper | None = JordanWignerMapper(),
        maxiter: int | None = None,
        excitations: list[tuple[int, ...]] | dopyqo.ExcitationPools = dopyqo.ExcitationPools.SINGLES_DOUBLES,
    ) -> VQEResult:
        """Run a VQE optimization with Qiskit and a UCCSD ansatz. The initial state, that is prepared before the parametrized ansatz is applied,
        is a Slater determinant matching self.occupations.

        Args:
            mapper (FermionicMapper | None, optional): Used Fermion-to-qubit mapping. Defaults to JordanWignerMapper().
            optimizer (dopyqo.VQEOptimizers, optional): Classical optimizer used in the VQE. Defaults to dopyqo.VQEOptimizers.L_BFGS_B.
            maxiter (int | None, optional): Maximum number of optimizer iterations before stopping the optimization. If None, large defaults are used. Defaults to None.
            UCCSD_reps (int, optional): Number of times the UCCSD ansatz is repeated with new parameters. Defaults to 1.

        Returns:
            VQEResult: Qiskit VQEResult object
        """
        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        two_mo = self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3)
        # if not np.allclose(one_mo.real, one_mo):
        #     print(
        #         f"{RED}Hamiltonian error: Single electron matrix elements are complex. Qiskit only supports real matrix elements!{RESET_COLOR}",
        #         file=sys.stderr,
        #     )
        #     sys.exit(1)
        # if not np.allclose(two_mo.real, two_mo):
        #     print(
        #         f"{RED}Hamiltonian error: Two electron matrix elements are complex. Qiskit only supports real matrix elements!{RESET_COLOR}",
        #         file=sys.stderr,
        #     )
        #     sys.exit(1)
        auto_index_order = True
        if not np.allclose(one_mo.real, one_mo):
            # print(
            #     f"{ORANGE}Hamiltonian error: Single electron matrix elements are complex. Qiskit only supports real matrix elements!{RESET_COLOR}",
            # )
            auto_index_order = False
        if not np.allclose(two_mo.real, two_mo):
            # print(
            #     f"{ORANGE}Hamiltonian error: Two electron matrix elements are complex. Qiskit only supports real matrix elements!{RESET_COLOR}",
            # )
            auto_index_order = False
        if not np.allclose(energy_offset.real, energy_offset):
            print(
                f"{RED}Hamiltonian error: Constant energy offset is complex. Qiskit only supports real offsets!!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)

        problem = self.to_qiskit_problem(auto_index_order=auto_index_order)
        problem.reference_energy = self.fci_energy

        if mapper is None:
            mapper = JordanWignerMapper()
        energy = problem.hamiltonian
        fermionic_op = energy.second_q_op()
        pauli_sum_op = mapper.map(fermionic_op)

        logging.info("pauli_sum_op.num_qubits: %f", pauli_sum_op.num_qubits)

        # Qubit mapping
        logging.info("Using Hartree-Fock state")
        initial_state = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        )
        # print("Initial state:")
        # print(initial_state.draw())
        # from qiskit.quantum_info import Statevector

        # print(Statevector(initial_state).draw("latex_source"))
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=initial_state,
            reps=UCCSD_reps,
        )

        doubles = [op for (op, excitation) in zip(ansatz.operators, ansatz.excitation_list) if len(excitation[0]) == 2]
        singles = [op for (op, excitation) in zip(ansatz.operators, ansatz.excitation_list) if len(excitation[0]) == 1]
        match excitations:
            case dopyqo.ExcitationPools.SINGLES:
                ansatz.operators = singles
            case dopyqo.ExcitationPools.DOUBLES:
                ansatz.operators = doubles
            case dopyqo.ExcitationPools.SINGLES_DOUBLES:
                ansatz.operators = doubles + singles
            case _:
                print(f"{RED}VQE error: Parameter {excitations=} is not supported. Use dopyqo.ExcitationPools types!{RESET_COLOR}")
                sys.exit(1)
        ansatz._invalidate()

        logging.info("ansatz.num_qubits: %i", ansatz.num_qubits)

        # initial_state.draw(
        #     "mpl", filename=os.path.join("results", "UCCSD_initial_state")
        # )

        # ansatz.draw("mpl", filename=os.path.join("results", "vqe_ansatz"))

        match optimizer:
            case dopyqo.VQEOptimizers.COBYLA:
                maxiter = 1e6 if maxiter is None else maxiter
                optimizer_obj = optim.COBYLA(maxiter=maxiter, tol=1e-13)
            case dopyqo.VQEOptimizers.L_BFGS_B:
                maxiter = 1e6 if maxiter is None else maxiter
                optimizer_obj = optim.L_BFGS_B(maxfun=maxiter * 4 * len(ansatz.operators), maxiter=maxiter, ftol=1e-13)
            case dopyqo.VQEOptimizers.ExcitationSolve:
                from excitationsolve.excitation_solve_qiskit import ExcitationSolveQiskit

                maxiter = 100 if maxiter is None else maxiter
                optimizer_obj = ExcitationSolveQiskit(maxiter=maxiter, tol=1e-12)
            case _:
                print(f"{RED}VQE error: Parameter {optimizer=} is not supported. Use dopyqo.VQEOptimizers types!{RESET_COLOR}")
                sys.exit(1)
        estimator = Estimator()
        # estimator = StatevectorEstimator()

        # logging.info("HF-result: %s", estimator.run(initial_state, pauli_sum_op).result().values)

        offset = np.sum(list(energy.constants.values()))
        counts = []
        values = []

        def store_intermediate_result(eval_count, parameters, mean, std):
            print(
                f"Optimizer evaluation #{eval_count}, Diff. to ref.: {np.abs(mean + offset - self.reference_energy)}",
                end="\r",
            )
            counts.append(eval_count)
            values.append(mean)

        solver = VQE(
            estimator,
            ansatz,
            optimizer_obj,
            callback=store_intermediate_result,
            initial_point=np.zeros((ansatz.num_parameters,)),
        )
        logging.info("VQE defined")
        logging.info("Solving VQE...")
        vqe_result = solver.compute_minimum_eigenvalue(pauli_sum_op)
        logging.info("Solved")
        elec_struc_result = problem.interpret(vqe_result)

        match optimizer:
            case dopyqo.VQEOptimizers.ExcitationSolve:
                counts = optimizer_obj.nfevs
                values = optimizer_obj.energies

        self.qiskit_vqe_solver = solver
        self.qiskit_vqe_result = vqe_result

        self.qiskit_vqe_initial_state = initial_state
        self.qiskit_vqe_ansatz = solver.ansatz

        self.qiskit_vqe_counts = np.array(counts)
        self.qiskit_vqe_values = np.array(values) + offset

        self.qiskit_elec_struc_result = elec_struc_result
        self.qiskit_vqe_result = vqe_result
        self.qiskit_problem = problem
        self.qiskit_ansatz = ansatz
        self.qiskit_pauli_sum_op = pauli_sum_op

        self.vqe_counts = self.qiskit_vqe_counts
        self.vqe_values = self.qiskit_vqe_values
        self.vqe_result = self.qiskit_elec_struc_result

        return self.qiskit_vqe_result

    def solve_vqe_adapt(
        self,
        optimizer: dopyqo.VQEOptimizers = dopyqo.VQEOptimizers.L_BFGS_B,
        maxiter: int | None = None,
        excitation_pool: list[tuple[int, ...]] | dopyqo.ExcitationPools = dopyqo.ExcitationPools.SINGLES_DOUBLES,
        drain_pool: bool = True,
        conv_threshold: float = 1e-13,
        selection_criterion: dopyqo.AdaptSelectionCriterion = dopyqo.AdaptSelectionCriterion.ENERGY,
    ) -> scipy.optimize.OptimizeResult:
        """Run a ADAPT-VQE optimization with TenCirChem given a pool of excitations. The initial state,
        that is prepared before the parametrized ansatz is applied, is a Slater determinant matching self.occupations.
        All appended excitations are optimized after every appended excitation.
        The criterion with which a excitation is selected from the pool can be set with the selection_criterion parameter.

        Args:
            optimizer (dopyqo.VQEOptimizers, optional): Classical optimizer used in the VQE. Defaults to dopyqo.VQEOptimizers.L_BFGS_B.
            maxiter (int | None, optional): Maximum number of optimizer iterations used for optimizing all appended excitations before
                                            stopping the optimization. If None, large defaults are used. Defaults to None.
            excitation_pool (list[tuple[int, ...]] | dopyqo.ExcitationPools, optional): Excitations in the ADAPT pool. Defaults to
                                                                                        dopyqo.ExcitationPools.SINGLES_DOUBLES where all double and
                                                                                        single excitations are in the pool.
            drain_pool (bool, optional): If set to True an excitation is removed from the pool if it was appended to the ansatz. This means each
                                         operator can only appear once in the ansatz. Defaults to True.
            conv_threshold (float, optional): Excitations are appended to the ansatz until their initial impact drops below this threshold value.
                                              Defaults to 1e-13.
            selection_criterion (dopyqo.AdaptSelectionCriterion, optional): Used operator selection criterion.
                                                                            When dopyqo.AdaptSelectionCriterion.ENERGY is used, every appended
                                                                            excitation is initialized with its optimal parameter.Defaults to
                                                                            dopyqo.AdaptSelectionCriterion.ENERGY.

        Returns:
            scipy.optimize.OptimizeResult: Scipy optimization result
        """
        # NOTE: TenCirChem can only simulate spin-restricted systems!

        nelec = (
            self.nelec,
            self.nelec,
        )

        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        two_mo = self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3)  # To TenCirChem/PySCF notation

        if not np.allclose(one_mo.real, one_mo):
            print(
                f"{RED}Hamiltonian error: Single electron matrix elements are complex. TenCirChem only supports real matrix elements!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(two_mo.real, two_mo):
            print(
                f"{RED}Hamiltonian error: Two electron matrix elements are complex. TenCirChem only supports real matrix elements!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(energy_offset.real, energy_offset):
            print(
                f"{RED}Hamiltonian error: Constant energy offset is complex. TenCirChem only supports real offsets!!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        one_mo = one_mo.real
        two_mo = two_mo.real
        energy_offset = energy_offset.real

        # mo_coeff = np.eye(self.norb) if self.mo_coeff is None else self.mo_coeff
        # print(f"{mo_coeff=}")
        mo_coeff = np.eye(self.norb)
        kwargs = {
            "mo_coeff": mo_coeff,
            "init_method": "zeros",
            "run_hf": False,
            "run_mp2": False,
            "run_ccsd": False,
            "run_fci": False,
            # "engine": "civector",
        }
        ansatz = tencirchem.UCCSD.from_integral(one_mo, two_mo, nelec, energy_offset, **kwargs)
        print(f"{ansatz.engine=}")
        # See https://tensorcircuit.github.io/TenCirChem-NG/faq.html#why-are-the-number-of-excitation-operators-and-the-number-of-parameters-different-what-is-param-ids
        ansatz.param_ids = None
        ansatz.ex_ops = []

        if not isinstance(excitation_pool, list):
            singles_tcc = tcc_helpers.get_ex1_ops(self.norb, nelec)
            doubles_tcc = tcc_helpers.get_ex2_ops(self.norb, nelec)
            # NOTE: Sort indices of excitaions like qiskit
            singles_tcc_sorted = sorted(singles_tcc)
            doubles_tcc_sorted = sorted([(*sorted(x[:2]), *sorted(x[2:])) for x in doubles_tcc])

            match excitation_pool:
                case dopyqo.ExcitationPools.SINGLES:
                    excitation_pool = singles_tcc_sorted
                case dopyqo.ExcitationPools.DOUBLES:
                    excitation_pool = doubles_tcc_sorted
                case dopyqo.ExcitationPools.SINGLES_DOUBLES:
                    excitation_pool = singles_tcc_sorted + doubles_tcc_sorted
                case _:
                    print(
                        f"{RED}ADAPT-VQE error: Parameter {excitation_pool=} is not supported. Use a list of excitation tuples or dopyqo.ExcitationPools types!{RESET_COLOR}"
                    )
                    sys.exit(1)

        # Setting tcc_uccsd.init_state to match self.occupations
        binary_string_tmp = "0b" + "".join(map(str, map(int, reversed(self.occupations)))) * 2
        binary_string_tmp = int(binary_string_tmp, base=2)
        ci_strings_tmp = ansatz.get_ci_strings()
        ci_string_idx_tmp = np.where(ci_strings_tmp == binary_string_tmp)[0][0]
        #
        init_statevector_tmp = np.zeros(len(ci_strings_tmp))
        init_statevector_tmp[ci_string_idx_tmp] = 1.0
        ansatz.init_state = list(init_statevector_tmp)

        #####################################################################################
        ##                           Cost function on parameter                            ##
        #####################################################################################
        times_tcc = []
        eval_count_tcc = 0
        counts_tcc = []
        values_tcc = []
        tcc_vqe_params_lst = []

        # Tracking values and counts has to be handled differently for ExcSolve,
        # since we want to manually set the values and counts from the ExcSolve object
        eval_count_tcc_excsolve = 0
        counts_tcc_excsolve = []
        values_tcc_excsolve = []
        tcc_vqe_params_lst_excsolve = []

        def cost_one_exc(x):
            nonlocal times_tcc
            nonlocal eval_count_tcc
            nonlocal counts_tcc
            nonlocal values_tcc
            nonlocal current_params

            tcc_energy = ansatz.energy(np.append(current_params, x))
            times_tcc.append(time.perf_counter())

            eval_count_tcc += 1
            print(
                f"Optimizer evaluation #{eval_count_tcc}, Diff. to ref.: {np.abs(tcc_energy - self.reference_energy)}",
                end="\r",
                flush=True,
            )
            counts_tcc.append(eval_count_tcc)
            values_tcc.append(tcc_energy)
            tcc_vqe_params_lst.append(x)

            return tcc_energy

        #####################################################################################
        ##                        Cost function for all parameters                         ##
        #####################################################################################
        def cost(x):
            nonlocal times_tcc
            nonlocal eval_count_tcc
            nonlocal counts_tcc
            nonlocal values_tcc

            tcc_energy = ansatz.energy(x)
            times_tcc.append(time.perf_counter())

            eval_count_tcc += 1
            print(
                f"Optimizer evaluation #{eval_count_tcc}, Diff. to ref.: {np.abs(tcc_energy - self.reference_energy)}",
                end="\r",
                flush=True,
            )
            counts_tcc.append(eval_count_tcc)
            values_tcc.append(tcc_energy)
            tcc_vqe_params_lst.append(x)

            return tcc_energy

        #####################################################################################
        ##                               Running ADAPT-VQE                                 ##
        #####################################################################################
        converged = False
        pool_drained = False
        current_params = np.zeros((0,))
        initial_param = 0.0
        current_energy = self.hf_energy()
        match selection_criterion:
            case dopyqo.AdaptSelectionCriterion.ENERGY:
                try:
                    from excitationsolve import ExcitationSolveScipy
                except ImportError:
                    print(
                        f"{ORANGE}ADAPT error: Could not import excitationsolve package which is needed for the energy-based operator selection. ",
                        f"Please install the excitationsolve package.{RESET_COLOR}",
                    )
                    sys.exit(1)

        while len(excitation_pool) > 0:
            # Check every excitation in the pool and compute their impact on the energy
            impact_and_param_per_excitation = {}
            current_ansatz_excs = ansatz.ex_ops.copy()
            for exc in excitation_pool:
                ansatz.ex_ops = current_ansatz_excs + [exc]

                match selection_criterion:
                    case dopyqo.AdaptSelectionCriterion.ENERGY:
                        # Using ExcitationSolve for operator selection
                        excsolve_obj = ExcitationSolveScipy(maxiter=1, tol=1e-10, save_parameters=True)
                        optimizer_func = excsolve_obj.minimize
                        options = dict(print_instead_logging=False, reference_energy=self.reference_energy)
                        res_tcc = scipy.optimize.minimize(cost_one_exc, initial_param, method=optimizer_func, options=options)
                        impact_and_param_per_excitation[exc] = (np.abs(current_energy - res_tcc.fun), res_tcc.x)
                        #
                        eval_count_tcc_excsolve += excsolve_obj.nfevs[-1]
                        counts_tcc_excsolve.append(eval_count_tcc_excsolve)
                        values_tcc_excsolve.append(res_tcc.fun)
                        tcc_vqe_params_lst_excsolve.append(res_tcc.x)
                        counts_tcc = counts_tcc[: -excsolve_obj.nfevs[-1]]
                        counts_tcc.append(eval_count_tcc)
                        values_tcc = values_tcc[: -excsolve_obj.nfevs[-1]]
                        values_tcc.append(res_tcc.fun)
                        tcc_vqe_params_lst = tcc_vqe_params_lst[: -excsolve_obj.nfevs[-1]]
                        tcc_vqe_params_lst.append(res_tcc.x)
                    case dopyqo.AdaptSelectionCriterion.GRADIENT:
                        # Compute d/dθ <ψ(θ)|H|ψ(θ)> at θ=0: Equivalent to <ψ(θ)|[H, A]|ψ(θ)> where A is the excitation exc
                        _nrg, grad = ansatz.energy_and_grad(np.append(current_params, 0.0))
                        grad = grad[-1]
                        impact_and_param_per_excitation[exc] = (np.abs(grad), 0.0)

                        for _ in range(4):  # 4 energy evaluations per gradient/s
                            eval_count_tcc += 1
                            counts_tcc.append(eval_count_tcc)
                            values_tcc.append(current_energy)
                    case _:
                        print(
                            f"{RED}ADAPT-VQE error: Parameter {selection_criterion=} is not supported. Use dopyqo.AdaptSelectionCriterion types!{RESET_COLOR}"
                        )
                        sys.exit(1)

            # Append operator with largest impact
            sorted_exc = sorted(impact_and_param_per_excitation.items(), key=lambda x: x[1][0], reverse=True)
            exc_to_append, imp_param_tmp = sorted_exc[0]
            # Stop if impact is less than threshold
            if imp_param_tmp[0] < conv_threshold:
                converged = True
                ansatz.ex_ops = current_ansatz_excs
                print(f"{GREEN}ADAPT-VQE converged!{RESET_COLOR}")
                break
            ansatz.ex_ops = current_ansatz_excs + [exc_to_append]
            if drain_pool:
                excitation_pool = [exc for exc in excitation_pool if exc != exc_to_append]
            current_params = np.append(current_params, imp_param_tmp[1])

            # Optimize all parameters
            logging.info("Using optimizer %s", optimizer)
            match optimizer:
                case dopyqo.VQEOptimizers.COBYLA:
                    maxiter = 1e6 if maxiter is None else maxiter
                    options = dict(maxiter=maxiter, tol=1e-10)
                    optimizer_func = "COBYLA"
                case dopyqo.VQEOptimizers.L_BFGS_B:
                    maxiter = 1e6 if maxiter is None else maxiter
                    options = dict(maxfun=maxiter * 4 * ansatz.n_params, maxiter=maxiter, ftol=1e-13)
                    optimizer_func = "L-BFGS-B"
                case dopyqo.VQEOptimizers.ExcitationSolve:
                    # TODO: If ExcSolve is used and there is only one operator in the ansatz,
                    #       we do not need to optimize here since ExcSolve already initialized the parameter in its optimal value
                    maxiter = 100 if maxiter is None else maxiter
                    excsolve_obj = ExcitationSolveScipy(maxiter=maxiter, tol=1e-10, save_parameters=True)
                    optimizer_func = excsolve_obj.minimize
                    options = dict()
                case _:
                    print(f"{RED}ADAPT-VQE error: Parameter {optimizer=} is not supported. Use dopyqo.VQEOptimizers types!{RESET_COLOR}")
                    sys.exit(1)
            res_tcc = scipy.optimize.minimize(cost, current_params, method=optimizer_func, options=options)
            match optimizer:
                case dopyqo.VQEOptimizers.ExcitationSolve:
                    counts_tcc_excsolve.extend(np.array(excsolve_obj.nfevs) + eval_count_tcc_excsolve)
                    eval_count_tcc_excsolve += excsolve_obj.nfevs[-1]
                    values_tcc_excsolve.extend(excsolve_obj.energies)
                    tcc_vqe_params_lst_excsolve.extend(excsolve_obj.params)

            current_params = res_tcc.x.copy()
            current_energy = res_tcc.fun

            ops_left_str = f" ({len(excitation_pool)} operators left in pool)" if drain_pool else ""
            # print(
            #     f"Done optimizing all {len(current_params)} parameters in the ansatz{ops_left_str}. Initial energy impact of last appended operator ({exc_to_append}): {imp_param_tmp[0]:.2e} (threshold: {conv_threshold}). Current diff. to ref.: {(np.abs(self.reference_energy - res_tcc.fun)):.2e}\n"
            # )
        else:  # Pool is drained. Not executed if break was used to exit while loop
            pool_drained = True
            print(f"{ORANGE}ADAPT-VQE pool is drained!{RESET_COLOR}")

        print(f"{GREEN}ADAPT ansatz consits of {ansatz.n_params} operators!{RESET_COLOR}")

        params_tcc = current_params

        match optimizer:
            case dopyqo.VQEOptimizers.ExcitationSolve:
                counts_tcc = counts_tcc_excsolve
                values_tcc = values_tcc_excsolve
                tcc_vqe_params_lst = tcc_vqe_params_lst_excsolve
        self.tcc_vqe_params = params_tcc
        self.tcc_vqe_params_lst = tcc_vqe_params_lst

        self.tcc_vqe_counts = np.array(counts_tcc)
        self.tcc_vqe_values = np.array(values_tcc)

        self.tcc_vqe_result = res_tcc
        self.tcc_ansatz = ansatz

        self.vqe_counts = self.tcc_vqe_counts
        self.vqe_values = self.tcc_vqe_values
        self.vqe_result = self.tcc_vqe_result

        return self.tcc_vqe_result

    def solve_hf(self) -> float:
        """Perform a Hartree-Fock (HF) calculation with PySCF.
        The PySCF RHF object is saved in self.mf.

        Returns:
            float: HF energy
        """
        # https://github.com/pyscf/pyscf/blob/master/examples/scf/40-customizing_hamiltonian.py

        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        mol = pyscf.gto.M()
        nelec = 0
        nelec = (self.nelec, self.nelec)
        nelec = sum(nelec)
        mol.nelectron = nelec
        mf = scf.RHF(mol)
        # mf.verbose = 5

        mf.get_hcore = lambda *args: one_mo.real
        mf.get_ovlp = lambda *args: np.eye(self.norb)
        mf._eri = pyscf.ao2mo.restore(8, self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3).real, self.norb)
        mf.energy_nuc = lambda *args: energy_offset

        mol.incore_anyway = True
        res = mf.kernel()
        self.mo_coeff = mf.mo_coeff
        self.mf = mf
        return res

    def solve_fci(self, n_energies=1) -> np.ndarray:
        """Perform a FCI calculation using PySCF. If the matrix elements self.h_pq and self.h_pqrs are real,
        the function self.solve_fci_spin1 is executed which uses the PySCF solver pyscf.fci.direct_spin1.FCISolver().
        If the matrix elements are complex, the function self.solve_fci_general is executed which uses the PySCF solver pyscf.fci.direct_spin1.FCISolver().

        Args:
            n_energies (int, optional): Number of energies the FCI solver will compute. Defaults to 1.

        Returns:
            np.ndarray: Computed FCI energies as a numpy array.
        """
        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        if not np.allclose(self.h_pq.real, self.h_pq) or not np.allclose(self.h_pqrs.real, self.h_pqrs):
            return self.solve_fci_general(n_energies=n_energies)
        return self.solve_fci_spin1(n_energies=n_energies)

    def solve_fci_spin1(self, n_energies=1) -> np.ndarray:
        """Perform a FCI calculation using PySCF solver pyscf.fci.direct_spin1.FCISolver().

        Args:
            n_energies (int, optional): Number of energies the FCI solver will compute. Defaults to 1.

        Returns:
            np.ndarray: Computed FCI energies as a numpy array.
        """
        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        two_mo = self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3)
        if not np.allclose(one_mo.real, one_mo):
            print(
                f"{RED}Hamiltonian error (solve_fci_spin1): Single electron matrix elements are complex. Use solve_fci_general method instead!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(two_mo.real, two_mo):
            print(
                f"{RED}Hamiltonian error (solve_fci_spin1): Two electron matrix elements are complex. Use solve_fci_general method instead!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(energy_offset.real, energy_offset):
            print(
                f"{RED}Hamiltonian error (solve_fci_spin1): Constant energy offset is complex. Use solve_fci_general method instead!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)

        if self.h_pq.shape[0] == 0:
            return energy_offset

        nelec = (self.nelec, self.nelec)

        # FCI calculation
        nroots = n_energies  # number of states to calculate
        self.fci_solver = pyscf.fci.direct_spin1.FCISolver()
        h_pq_real = self.h_pq.real
        h_pqrs_real = self.h_pqrs.real.swapaxes(1, 2).swapaxes(1, 3)

        self.fci_evs, self.fci_evcs = self.fci_solver.kernel(
            h1e=h_pq_real,
            eri=h_pqrs_real,
            norb=self.norb,
            nelec=nelec,
            nroots=nroots,
        )
        # Full Hamiltonian matrix
        # h_fci = pyscf.fci.direct_spin1.pspace(h_pq_real, h_pqrs_real, self.norb, nelec, np=100000000)[1]

        # Save eigenvalues and -vectors in lists
        if n_energies == 1:
            self.fci_evs = np.array([self.fci_evs])
            self.fci_evcs = [self.fci_evcs]

        fci_energy = self.fci_evs + energy_offset

        self.fci_energy = fci_energy
        return fci_energy

    def solve_fci_uhf(self, n_energies=1) -> np.ndarray:
        """Perform a FCI calculation using the PySCF solver pyscf.fci.direct_uhf.FCISolver().

        Args:
            n_energies (int, optional): Number of energies the FCI solver will compute. Defaults to 1.

        Returns:
            np.ndarray: Computed FCI energies as a numpy array.
        """
        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        one_mo = self.h_pq
        two_mo = self.h_pqrs.swapaxes(1, 2).swapaxes(1, 3)
        if not np.allclose(one_mo.real, one_mo):
            print(
                f"{RED}Hamiltonian error (solve_fci_uhf): Single electron matrix elements are complex. Use solve_fci_general method instead!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(two_mo.real, two_mo):
            print(
                f"{RED}Hamiltonian error (solve_fci_uhf): Two electron matrix elements are complex. Use solve_fci_general method instead!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.allclose(energy_offset.real, energy_offset):
            print(
                f"{RED}Hamiltonian error (solve_fci_uhf): Constant energy offset is complex. Use solve_fci_general method instead!{RESET_COLOR}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.h_pq, self.h_pq
        eri_up, eri_dw, eri_dw_up, eri_up_dw = (
            self.h_pqrs,
            self.h_pqrs,
            self.h_pqrs,
            self.h_pqrs,
        )

        # Transform ERIs to chemist's index order
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        # TODO: Not optimal to calculate nelec manually. Add this to Wfc class?
        nelec = (self.nelec, self.nelec)
        # FCI calculation
        nroots = n_energies  # number of states to calculate
        norb = self.norb

        h_ij_up = h_ij_up.real
        h_ij_dw = h_ij_dw.real
        eri_up = eri_up.real
        eri_dw = eri_dw.real
        eri_up_dw = eri_up_dw.real

        self.fcisolver = pyscf.fci.direct_uhf.FCISolver()
        # Ordering of parameters from direct_uhf.make_hdiag
        self.fci_evs, self.fci_evcs = self.fcisolver.kernel(
            h1e=(h_ij_up, h_ij_dw),  # a, b (a=up, b=down)
            eri=(
                eri_up,
                eri_up_dw,
                eri_dw,
            ),  # aa, ab, bb (a=up, b=down)
            norb=norb,
            nelec=nelec,
            nroots=nroots,
        )

        # h_ij_up = h_ij_up.real
        # h_ij_up_diag = np.diag(h_ij_up.diagonal())
        # h_ij_up = np.zeros((4, 4))
        # h_ij_up[0, 0] = h_ij_up_diag[0, 0]
        # h_ij_up[1, 1] = h_ij_up_diag[1, 1]
        # h_ij_up[2, 2] = h_ij_up_diag[0, 0]
        # h_ij_up[3, 3] = h_ij_up_diag[1, 1]
        # eri_up = np.zeros((4, 4, 4, 4))
        # self.fcisolver = pyscf.fci.fci_dhf_slow.FCI()
        # self.fci_evs, self.fci_evcs = pyscf.fci.fci_dhf_slow.kernel(
        #     h1e=h_ij_up,
        #     eri=eri_up,
        #     norb=norb,
        #     nelec=sum(nelec),
        #     nroots=nroots,
        # )

        # self.fci_evs, self.fci_evcs = pyscf.fci.fci_slow.kernel(
        #     h1e=h_ij_up,
        #     eri=eri_up,
        #     norb=norb,
        #     nelec=sum(nelec),
        # )

        # Save eigenvalues and -vectors in lists
        if n_energies == 1:
            self.fci_evs = np.array([self.fci_evs])
            self.fci_evcs = [self.fci_evcs]

        fci_energy = self.fci_evs + energy_offset

        self.fci_energy = fci_energy
        return fci_energy

    def solve_fci_general(self, n_energies=1) -> np.ndarray:
        """Perform a FCI calculation using the PySCF solver pyscf.fci.direct_spin1.FCISolver().

        Args:
            n_energies (int, optional): Number of energies the FCI solver will compute. Defaults to 1.

        Returns:
            np.ndarray: Computed FCI energies as a numpy array.
        """
        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.h_pq, self.h_pq
        eri_up, eri_dw, eri_dw_up, eri_up_dw = (
            self.h_pqrs,
            self.h_pqrs,
            self.h_pqrs,
            self.h_pqrs,
        )
        # Transform ERIs to chemist's index order
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        # TODO: Not optimal to calculate nelec manually. Add this to Wfc class?
        nelec = (self.nelec, self.nelec)
        # FCI calculation
        nroots = n_energies  # number of states to calculate
        norb = self.norb

        energy_offset = self.constants
        if isinstance(self.constants, dict):
            energy_offset = sum(self.constant.values())

        # # Transforming matrices into fully spin-mixed form used in pyscf.scf.GHF and pyscf.fci.fci_dhf_slow
        # # We have separate matrices for up- and down-spin,
        # #       shape (norb, norb) for h1e and (norb, norb, norb, norb) for eri each
        # # We want one matrix for both spin-channels allowing also for spin-mixing,
        # #       shape (2*norb, 2*norb) for h1e and (2*norb, 2*norb, 2*norb, 2*norb) for eri each.
        # #       We set all spin-mixing matrix elements to zero, e.g. <\psi_{1,up}|h|\psi_{1,down}> = 0.0
        # # Orbital order is (α (up) and β (down) spin-channels):
        # #       β1, α1, β2, α2, β3, α3, β4, α4, β5, α5, .... where αn is the n-orbital with spin α
        # id_up = np.array([[1.0, 0.0], [0.0, 0.0]])
        # id_dw = np.array([[0.0, 0.0], [0.0, 1.0]])
        # h1e_custom_up = np.kron(h_ij_up, id_up)
        # h1e_custom_dw = np.kron(h_ij_dw, id_dw)
        # h1e_custom_spin = h1e_custom_up + h1e_custom_dw
        # eri_custom_up = np.kron(np.kron(eri_up, id_up).T, id_up).T
        # eri_custom_dw = np.kron(np.kron(eri_dw, id_dw).T, id_dw).T
        # eri_custom_dw_up = np.kron(np.kron(eri_dw_up, id_dw).T, id_up).T
        # eri_custom_up_dw = np.kron(np.kron(eri_up_dw, id_up).T, id_dw).T
        # eri_custom_spin = eri_custom_up + eri_custom_dw + eri_custom_dw_up + eri_custom_up_dw

        h1e_custom_spin = np.zeros((2 * norb, 2 * norb), dtype=np.complex128)
        for i in range(norb):
            for j in range(norb):
                h1e_custom_spin[2 * i, 2 * j] = h_ij_up[i, j]  # spin-↑
                h1e_custom_spin[2 * i + 1, 2 * j + 1] = h_ij_dw[i, j]  # spin-↓
        eri_custom_spin = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=np.complex128)
        # (i,l) have same spin and (j,k) have same spin
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l in range(norb):
                        # += same as =
                        eri_custom_spin[2 * i, 2 * j, 2 * k, 2 * l] = self.h_pqrs[i, j, k, l]  # ↑↑
                        eri_custom_spin[2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1] = self.h_pqrs[i, j, k, l]  # ↓↓
                        eri_custom_spin[2 * i, 2 * j + 1, 2 * k + 1, 2 * l] = self.h_pqrs[i, j, k, l]  # ↑↓
                        eri_custom_spin[2 * i + 1, 2 * j, 2 * k, 2 * l + 1] = self.h_pqrs[i, j, k, l]  # ↓↑
        eri_custom_spin = eri_custom_spin.transpose(0, 3, 1, 2)  #  / (-2.0)  # physicist to chemist order

        # self.fci_evs, self.fci_evcs = pyscf.fci.fci_dhf_slow.kernel(
        #     h1e=h1e_custom_spin,
        #     eri=eri_custom_spin,
        #     norb=norb * 2,
        #     nelec=sum(nelec),
        #     nroots=nroots,
        # )

        # self.fcisolver = pyscf.fci.fci_dhf_slow.FCI()
        # self.fci_evs, self.fci_evcs = self.fcisolver.kernel(
        #     h1e=h1e_custom_spin.real,  # TODO: REMOVE .real!
        #     eri=eri_custom_spin.real,
        #     norb=norb * 2,
        #     nelec=sum(nelec),
        #     nroots=nroots,
        #     # verbose=5,
        # )

        self.fcisolver = pyscf.fci.fci_dhf_slow.FCI()
        na = pyscf.fci.cistring.num_strings(norb * 2, sum(nelec))
        # random initial guess for robust results, see https://github.com/pyscf/pyscf/issues/2827
        # ci0 = [np.random.random((na,)) + np.random.random((na,)) * 1j]
        # self.fci_evs, self.fci_evcs = self.fcisolver.kernel(h1e_custom_spin, eri_custom_spin, norb * 2, nelec=sum(nelec), ci0=ci0)

        # TODO: Use following only if using updated pyscf.fci.fci_dhf_slow.py from
        #       https://github.com/pyscf/pyscf/commit/623f6f0a1d94972ead2d432d1e76e0590c021bbe
        self.fci_evs, self.fci_evcs = self.fcisolver.kernel(h1e_custom_spin, eri_custom_spin, norb * 2, nelec=sum(nelec))

        # Save eigenvalues and -vectors in lists
        if n_energies == 1:
            self.fci_evs = np.array([self.fci_evs])
            self.fci_evcs = [self.fci_evcs]

        fci_energy = self.fci_evs + energy_offset

        self.fci_energy = fci_energy
        return fci_energy
