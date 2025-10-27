[![DOI](https://zenodo.org/badge/1072921992.svg)](https://doi.org/10.5281/zenodo.17457083)

<div style="text-align: center;">
<pre>
oooooooooo.                                                         
`888'   `Y8b                                                        
 888      888  .ooooo.  oo.ooooo.  oooo    ooo  .ooooo oo  .ooooo.  
 888      888 d88' `88b  888' `88b  `88.  .8'  d88' `888  d88' `88b 
 888      888 888   888  888   888   `88..8'   888   888  888   888 
 888     d88' 888   888  888   888    `888'    888   888  888   888 
o888bood8P'   `Y8bod8P'  888bod8P'     .8'     `V8bod888  `Y8bod8P' 
                         888       .o..P'            888.           
                        o888o      `Y8P'             8P'            
                                                     "              
&nbsp;
   Many-body analysis on top of Quantum ESPRESSO calculations   
</pre>
</div>


## Introduction
This python package provides an interface between the [Quantum ESPRESSO (QE)](https://www.quantum-espresso.org/) DFT software and many-body calculations. We extract the Kohn-Sham orbitals from a QE calculation to construct a many-body Hamiltonian and solve it using classical and quantum algorithms. Classical solvers are implemented using the [PySCF](https://pyscf.org/) package. Quantum solvers are implemented using the [TenCirChem](https://tensorcircuit.github.io/TenCirChem-NG/index.html) and [Qiskit](https://www.ibm.com/quantum/qiskit) packages. After this we can calculate Bader charges on all atoms in the computational cell using the charge density and the [Bader charge analysis code](https://theory.cm.utexas.edu/henkelman/code/bader/).

Dopyqo reads the output files of a QE calculation to build the Hamiltonian. The needed files are located in the .save folder that QE outputs and are the `data-file-schema.xml` and the `wfc.dat` or `wfc.hdf5` files.

## Installation
Install via pip using
```bash
pip install dopyqo
```

To use a [more performant rust implementation for some matrix element calculations](https://github.com/dlr-wf/dopyqo-rs) you can install
```bash
pip install dopyqo-rs
```
or
```bash
pip install dopyqo[rs]
```
For this you need the following installed on your machine:
- [Rust programming language](https://rustup.rs/)
- [GNU scientific library](https://www.gnu.org/software/gsl/). On a debian (or debian-derived) OS ask your admin to run `sudo apt install libgsl-dev`.

If `dopyqo-rs` is not installed, a slower python implementation using numpy and scipy is used.

Further, `Dopyqo` can utilize GPUs for performing FFTs using the [`cupy`](https://docs.cupy.dev/en/stable/index.html) package to calculate matrix elements related to the electron repulsion integrals (ERIs):
```bash
pip install cupy
```
or
```bash
pip install dopyqo[cupy]
```
For this make sure to install the correct `cupy` version matching your installed CUDA version.
If `cupy` is not installed, a python implementation using numpy is used. If `cupy` is installed, the python implementation can still be used by setting `use_gpu=False` in the `dopyqo.DopyqoConfig`.


## Usage

`Dopyqo` can be used via a python interface:
```python
import os
import dopyqo

config = dopyqo.DopyqoConfig(
    base_folder=os.path.join("path/to/your/qe/calculation"),
    prefix="PrefixUsedInYourQECalculiaton",
    active_electrons=2,
    active_orbitals=4,
    run_fci=True, # Solve ground state with PySCF FCI solver
    run_vqe=True, # Solve ground state with VQE solver
    use_qiskit=False, # If False use TenCirChem to simulate the VQE, if True use Qiskit
    vqe_optimizer=dopyqo.VQEOptimizers.L_BFGS_B,
    vqe_excitations=dopyqo.ExcitationPools.SINGLES_DOUBLES,
    n_threads=10,
)
energy_dict, wfc_obj, h_ks, mats = dopyqo.run(config)
```
This reads the output of the specified QE calculation (`base_folder` and `prefix`), computes the many-body Hamiltonian, and solves for the ground-state using a FCI and VQE calculation in an active space of two electrons in four orbitals.

Alternatively, you can use the providend script called `dopyqo` and provide an input file in the toml format:
```bash
dopyqo -i your_input_file.toml
```
For information about the input file see [INPUT.md](INPUT.md). The latter currently provides only part of the functionality. Therefore, we encourage the use of the python interface.

See the [examples](examples/) folder for example scripts and an example input toml-file.

## Requirements
- Tested Python version: `3.11`
- Tested QE version: `7.1` compiled with and without HDF5 `1.14.0`
- Tested with QE SCF calculations. General k-point calculations only work when using Qiskit or PySCF. Gamma-only calculations also work with TenCirChem.
- Only norm-conserving pseudopotentials are currently supported.
- Only spin-restricted QE calculations are currently supported.
- To calculate Bader charges, the [Bader charge analysis code](https://theory.cm.utexas.edu/henkelman/code/bader/) needs to be installed.
- See [pyproject.toml](pyproject.toml) for required Python packages.


## Issues
### VQE calculation takes a lot of CPU resources
If using Linux set the environment variable `OMP_NUM_THREADS` to limit the number of used CPU threads.

## Authors
- Gabriel Breuil
- Erik Hansen
- Jasper Nickelsen
- Alexander Rehn
- Erik Schultheis

## Contact
Feel free to contact [David Melching](mailto:David.Melching@dlr.de) if you have any questions.

## Citation
If you use portions of this code please cite our [paper](https://arxiv.org/abs/2510.12887):
```bibtex
@article{Schultheis2025ManyBody,
      title={Many-body post-processing of density functional calculations using the variational quantum eigensolver for Bader charge analysis},
      url={https://arxiv.org/abs/2510.12887},
      DOI={10.48550/ARXIV.2510.12887},
      publisher={arXiv},
      author={Schultheis, Erik and Rehn, Alexander and Breuil, Gabriel},
      year={2025}
}
```

## Acknowledgment
This project was made possible by the DLR Quantum Computing Initiative and the Federal Ministry for Economic Affairs and Climate Action; https://qci.dlr.de/quanticom.
