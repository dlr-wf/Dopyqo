import os
import numpy as np
from qiskit.quantum_info import Statevector
import subprocess as sp
import shutil
import pandas as pd

# Author: Alexander Rehn


def orb_occupation(state_vector: Statevector):
    orb_occ_prob_list = []
    num_qubits = state_vector.num_qubits
    n_orb = num_qubits // 2
    # using inbuild method to get expectation values
    for i in range(num_qubits):
        orb_occ_prob_list.append(state_vector.probabilities([i])[1])
    orb_occ_prob_list = np.array(orb_occ_prob_list)

    up = orb_occ_prob_list[:n_orb]
    down = orb_occ_prob_list[n_orb:]
    spatial_occ = (up + down) / 2
    # transformation in numpy array
    return spatial_occ


def save_cube(n_at, origin, size, vectors, atoms, values, path):
    lines = []
    lines.append("Cube generated with the sum of squared KS orbitals of the QE calculation")
    lines.append(" ")
    # n_at and origin
    lines.append(f'{n_at} {"{:= #6.6g}".format(origin[0])} {"{:= #6.6g}".format(origin[1])} {"{:= #6.6g}".format(origin[2])}')
    # voxel vectors
    for i in range(3):
        lines.append(f'{size[i]} {"{:= #6.6g}".format(vectors[i,0])} {"{:= #6.6g}".format(vectors[i,1])} {"{:= #6.6g}".format(vectors[i,2])}')
    # atoms
    for atom in atoms:
        lines.append(atom.removesuffix("\n").removeprefix("  "))
    # values
    for ix in range(size[0]):
        for iy in range(size[1]):
            i = 0
            line = ""
            for iz in range(size[2]):
                if i == 6:
                    lines.append(line)
                    i = 0
                    line = ""
                line = line + "{: = #.5E}  ".format(values[ix, iy, iz])
                i += 1
            lines.append(line)

    lines = list(map(lambda s: "  " + s + "\n", lines))
    with open(path, "w") as f:
        f.writelines(lines)
    lines.append("")


def read_cube(path):
    # Cube file format:
    # The first two lines of the header are comments
    # The third line has the number of atoms included in the file followed by the position of the origin of the volumetric data.
    # The next three lines give the number of voxels along each axis (x, y, z) followed by the axis vector.
    #   Note this means the volume need not be aligned with the coordinate axis,
    #   indeed it also means it may be sheared although most volumetric packages won't support that.
    #   The length of each vector is the length of the side of the voxel thus allowing non cubic volumes.
    #   If the sign of the number of voxels in a dimension is positive then the units are Bohr, if negative then Angstroms.
    # The last section in the header is one line for each atom consisting of 5 numbers,
    #   the first is the atom number,
    #   the second is the charge,
    #   and the last three are the x,y,z coordinates of the atom center.
    # Volumetric data. The volumetric data is straightforward, one floating point number for each volumetric element.
    #   The original Gaussian format arranged the values in the format shown below in the example,
    #   most parsing programs can read any white space separated format.
    #   Traditionally the grid is arranged with the x axis as the outer loop and the z axis as the inner loop, for example, written as
    #     for (ix=0;ix<NX;ix++) {
    #       for (iy=0;iy<NY;iy++) {
    #          for (iz=0;iz<NZ;iz++) {
    #             printf("%g ",data[ix][iy][iz]);
    #             if (iz % 6 == 5)
    #                printf("\n");
    #          }
    #          printf("\n");
    #       }
    #    }

    data_dict = dict()
    data_dict["path"] = path

    with open(path, "r") as f:
        file_content = f.readlines()
    # 3. line (n_at and origin)
    origin = np.genfromtxt([file_content[2]])
    n_at = int(origin[0])
    origin = origin[1:]
    # print('origin', origin)
    data_dict["n_at"] = n_at
    data_dict["origin"] = origin
    # 4.-6. line n_voxel_vectors and voxel_vectors
    vectors = np.genfromtxt(file_content[3:6])
    # print('vectors', vectors)
    nx = int(vectors[0, 0])
    # print('nx', nx)
    ny = int(vectors[1, 0])
    # print('ny', ny)
    nz = int(vectors[2, 0])
    # print('nz', nz)
    data_dict["vectors"] = vectors[0:3, 1:]
    # print(np.linalg.norm(nx*vectors[0,1:]))
    # 7. - 7+n_at atoms and their position
    data_dict["atoms"] = np.genfromtxt(file_content[6 : 6 + n_at])
    data_dict["atoms_txt"] = file_content[6 : 6 + n_at]
    # 8+n_at.- end voxel data
    xyz_cube = np.zeros((nx, ny, nz))
    coord = np.zeros((nx, ny, nz, 3))
    if nz % 6 == 0:
        n_lines_of_z_point = nz / 6
        cube = np.genfromtxt(file_content[6 + n_at :])
        cube_flat = cube.flatten()
    else:
        small_line_contains_n_numbers = nz % 6
        n_lines_of_z_point = int(nz / 6) + 1
        j = 0
        cube_flat = np.zeros((nx * ny * nz))
        for x in range(nx):
            for y in range(ny):
                cube_part_1 = np.genfromtxt(file_content[6 + n_at + j * n_lines_of_z_point : 6 + n_at + (j + 1) * n_lines_of_z_point - 1])
                cube_part_2 = np.genfromtxt([file_content[6 + n_at + (j + 1) * n_lines_of_z_point - 1]])
                cube_flat[j * (nz) : (j + 1) * nz - small_line_contains_n_numbers] = cube_part_1.flatten()
                cube_flat[(j + 1) * nz - small_line_contains_n_numbers : (j + 1) * nz] = cube_part_2
                j += 1

    i = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                xyz_cube[x, y, z] = cube_flat[i]
                coord[x, y, z] = x * vectors[0, 1:] + y * vectors[1, 1:] + z * vectors[2, 1:] + origin
                i += 1

    data_dict["cube"] = xyz_cube
    data_dict["coord"] = coord
    data_dict["size"] = xyz_cube.shape
    # print('xyz_cube.shape', xyz_cube.shape)
    volume = np.linalg.det(vectors[0:3, 1:])
    data_dict["volume"] = volume
    # print('volume', volume)
    return data_dict


def make_cube(calculation_dir, prefix, cube_qiskit, save_dir, path_to_executable, fft):

    os.makedirs(save_dir, exist_ok=True)
    tot_DFT_cube = read_cube(os.path.join(calculation_dir, f"{prefix}_pp.cube"))
    fft = tuple(fft)
    save_cube(
        tot_DFT_cube["n_at"],
        tot_DFT_cube["origin"],
        fft,
        tot_DFT_cube["vectors"],
        tot_DFT_cube["atoms_txt"],
        cube_qiskit,
        os.path.join(save_dir, "chr_dens.cube"),
    )
    path_before = os.getcwd()
    # bader calculation
    os.chdir(save_dir)
    sp.run([path_to_executable["bader"], "chr_dens.cube"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    os.chdir(path_before)


def run_ppx(pp_file, path_to_executable, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(pp_file, save_dir)
    path_before = os.getcwd()
    os.chdir(save_dir)
    sp.run([path_to_executable["bader"], f"{prefix}_pp.cube"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    os.chdir(path_before)


def extract_charge(atoms, ACF_file, valence):
    bader = pd.read_csv(ACF_file, sep=r"\s+", skiprows=2, nrows=len(atoms), names=["#", "X", "Y", "Z", "CHARGE", "MIN_DIST", "ATOMIC_VOL"])
    bader["ELEMENT"] = atoms
    bader["VALENCE"] = bader["ELEMENT"].map(valence)
    bader["EFFECTIVE CHARGE"] = bader["VALENCE"] - bader["CHARGE"]
    # print(bader)
    return bader


def get_density_cube(sq_psi, occupation):

    occupation = np.array(occupation)
    single_density = sq_psi * occupation[:, np.newaxis, np.newaxis, np.newaxis]
    total_density = np.sum(single_density, axis=(0))
    return total_density


def make_full_occ_from_partial_occ(spatial_occ: np.ndarray, active_orbitals: int, active_electrons: int, total_electrons: int, nbands: int):

    passive_electrons = total_electrons - active_electrons
    passive_filled_orbitals = int(passive_electrons / 2)
    empty_orbitals = nbands - passive_filled_orbitals - active_orbitals
    full_occupation = []

    full_occupation = np.concatenate([np.ones(passive_filled_orbitals), spatial_occ, np.zeros(empty_orbitals)])

    return full_occupation
