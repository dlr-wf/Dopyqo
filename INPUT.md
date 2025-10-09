The input file is expected to be in the [TOML format](https://toml.io/en/).
In the following all sections of the input file are documented. See [example section](#example-input-files) for examples of input files.

## Table of contents

- [control](#control)
    - [base_folder](#base_folder)
    - [prefix](#prefix)
    - [active_electrons](#active_electrons)
    - [active_orbitals](#active_orbitals)
    - [kpoint_idx](#kpoint_idx)
    - [logging](#logging)
    - [run_vqe](#run_vqe)
    - [run_fci](#run_fci)
    - [use_qiskit](#use_qiskit)

- [wannier](#wannier)
    - [transform](#transform)
    - [umat](#umat)
    - [input_file](#input_file)

- [geometry](#geometry)
    - [unit](#unit)
    - [coordinates](#coordinates)
    - [lattice_vectors](#lattice_vectors)

- [vqe](#vqe)
    - [parameters](#parameters)
    - [optimizer](#optimizer)
    - [uccsd_reps](#uccsd_reps)

------------------------------------------------------------------------------------------------------------------------------

## Control
(required)
### base_folder
**STRING** (optional):

Output folder of the Quantum ESPRESSO calculation. Defaults to the folder from which the program is executed.

### prefix
**STRING** (optional):

Prefix of the Quantum ESPRESSO calculation. Defaults to the name of the first folder thats ends with `.save` in the [base_folder](#base_folder), e.g. `Si` for the folder `Si.save`.

### active_electrons
**INTEGER**:

Number of electrons in the active space.

### active_orbitals
**INTEGER**:

Number of spatial orbitals in the active space.

### kpoint_idx
**INTEGER | STRING** (required if Quantum ESPRESSO calculation involved more than one k-point):

Index of the k-point you want to load, starting from zero. If you want to load all k-points set to `"all"`.

### logging
**BOOLEAN** (optional):

Whether or not to show logging output. Defaults to `false`.

### run_vqe
**BOOLEAN** (optional):

Whether or not to run a VQE calculation. Defaults to `false`. See [vqe](#vqe) for more information.

### run_fci
**BOOLEAN** (optional):

Whether or not to run a FCI calculation. Defaults to `false`. If set to `false` no FCI related calculation will be performed.

### use_qiskit
**BOOLEAN** (optional):

Whether or not to use qiskit for VQE related calculations. Defaults to `false` for which TenCirChem will be used.

------------------------------------------------------------------------------------------------------------------------------

## Wannier
(optional):

If given the Hamiltonian is unitarily-transformed into a Wannier-orbital basis

### transform
**BOOLEAN** (optional):

Whether or not to perform the Wannier transformation

### umat
**STRING** (optional):

Path to the Wannier90 transformation matrix (U-matrix). If [transform](#transform) is `true` defaults to [prefix](#prefix)_u.mat in the [base_folder](#base_folder).

### input_file
**STRING** (optional):

Path to the Wannier90 input file. If given, it is checked that the Wannier orbitals in the U-matrix ([umat](#umat)) match with the selected active space ([active_electrons](#active_electrons) and [active_orbitals](#active_orbitals))

------------------------------------------------------------------------------------------------------------------------------

## Geometry
(optional):

If not given the geometry from the Quantum ESPRESSO calculation is used. The geometry is currently not checked for its validity, i.e. atomic positions outside of the unit cell can result in unwanted behaviour.

### unit
**STRING** (required if either [atomic positions](#coordinates) or [lattice vectors](#lattice_vectors) are given):

Unit in which the [atomic positions](#coordinates) and/or [lattice vectors](#lattice_vectors) are given. Supported units depend on whether or not [lattice_vectors](#lattice_vectors) are given. If they are **not** given the supported units are:
- `"angstrom"`
- `"bohr"`
- `"meter"`
- `"alat"`
- `"crystal"`
If [lattice_vectors](#lattice_vectors) are given the supported units are
- `"angstrom"`
- `"bohr"`
- `"meter"`

### coordinates
**LIST OF LISTS** (optional):

Atomic positions as a list of lists in the specified [unit](#unit). Each inner list specifies one atom in the unit cell.

### lattice_vectors
**LIST OF LISTS** (optional):

Lattice vectors as a list of lists in the specified [unit](#unit). Each inner list specifies one lattice vector.

------------------------------------------------------------------------------------------------------------------------------

## VQE
(required if [#run_vqe](#run_vqe) is set to `true`):

### parameters
**LIST** (optional):

Each value in the list is one parameter value. The parameters are order as in the used UCCSD ansatz where we first apply all double excitations and then all single excitations. If given, no VQE optimization will be performed, only the energy of the VQE ansatz at the given parameter values is evaluated.

### optimizer
**STRING** (required):

Optimizer used for the VQE optimization. Supported optimizers are:
- `"L-BFGS-B"`
- `"COBYLA"`
- `"ExcitationSolve"`

### uccsd_reps
**INTEGER** (optional):

Number of repetitions of the UCCSD ansatz used in the VQE ansatz. Defaults to `1`.

## Example input files
Running a calculation with an active space of (2e, 2o) with redefined atomic positions and lattice vectors, solving with the FCI solver and estimating the energy of the VQE ansatz using costum parameters.
```toml
[control]
active_electrons=2
active_orbitals=2
run_vqe = true
run_fci = true

[geometry]
unit = "angstrom"
coordinates = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5]
]
lattice_vectors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

[vqe]
parameters = [0.4, 0.3, 0.1]
```
