import sys
import itertools
import numpy as np
from itertools import product
import pyvista as pv
from dopyqo.colors import *

# http://www.xcrysden.org/doc/XSF.html
# Column-major order


def load_xsf(filename):
    with open(filename) as f:
        lines = f.readlines()

    data = None
    grid_vecs = None
    origin = None
    atom_pos = None
    primvecs = None
    convvecs = None
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            continue
        if line.strip().startswith("PRIMVEC"):
            str_tmp = " ".join([x.strip() for x in lines[i + 1 : i + 4]]).strip()
            primvecs = np.fromstring(str_tmp, dtype=float, sep=" ").reshape((3, 3))
            # print(f"{primvecs=}")
        if line.strip().startswith("CONVVEC"):
            str_tmp = " ".join([x.strip() for x in lines[i + 1 : i + 4]]).strip()
            convvecs = np.fromstring(str_tmp, dtype=float, sep=" ").reshape((3, 3))
            # print(f"{convvecs=}")
        if line.strip().startswith("PRIMCOORD"):
            natm = int(lines[i + 1].strip().split()[0])
            # print(f"{natm=}")
            str_tmp = [x.strip() for x in lines[i + 2 : i + natm + 2]]
            str_tmp = " ".join([" ".join(x.split()[1:]) for x in str_tmp])  # 0 element is element string, e.g. Mg
            atom_pos = np.fromstring(str_tmp, dtype=float, sep=" ").reshape((natm, 3))  # In Angstrom
            # print(f"{atom_pos=}")
        if line.strip().startswith("BEGIN_DATAGRID"):
            nx, ny, nz = map(int, lines[i + 1].split())
            # print(f"{nx=} {ny=} {nz=}")
            origin = np.fromstring(lines[i + 2].strip(), dtype=float, sep=" ")
            # print(f"{origin=}")
            str_tmp = " ".join([x.strip() for x in lines[i + 3 : i + 6]]).strip()
            grid_vecs = np.fromstring(str_tmp, dtype=float, sep=" ").reshape((3, 3))
            # print(f"{grid_vecs=}")

            for i_tmp, line_tmp in enumerate(lines[i + 6 :]):
                if line_tmp.strip().startswith("END"):
                    break

            data_lst = []
            for data_line in lines[i + 6 : i + 6 + i_tmp]:
                data_lst.extend(map(float, data_line.strip().split()))
            # print(f"{len(data_lst)=}")
            # print(f"{nx * ny * nz =}")
            assert len(data_lst) == nx * ny * nz, f"Number of data points ({len(data_lst)}) is not equal to number of grid points ({nx*ny*nz})"

            data = np.array(data_lst).reshape((nx, ny, nz), order="C")
            # NOTE: Order C is needed to get correct plots for H2 compared to plots with VESTA.
            #       But for diamond order F looked fine.

    if data is None:
        print(f"{RED}XSF error: Could not read data from xsf file {filename}.{RESET_COLOR}")
        sys.exit(1)
    if grid_vecs is None:
        print(f"{RED}XSF error: Could not read vectors spanning the data grid from xsf file {filename}.{RESET_COLOR}")
        sys.exit(1)
    if origin is None:
        print(f"{RED}XSF error: Could not read origin of the data grid from xsf file {filename}.{RESET_COLOR}")
        sys.exit(1)
    if atom_pos is None:
        print(f"{RED}XSF error: Could not read atom positions from xsf file {filename}.{RESET_COLOR}")
        sys.exit(1)
    if primvecs is None:
        print(f"{RED}XSF error: Could not read primitive lattice vectors from xsf file {filename}.{RESET_COLOR}")
        sys.exit(1)
    if convvecs is None:
        print(
            f"{ORANGE}XSF warning: Could not read conventional lattice vectors from xsf file {filename}. "
            + f"Setting conventional lattice vectors equal to primitive lattice vectors.{RESET_COLOR}"
        )
        convvecs = primvecs.copy()

    return data, grid_vecs, origin, atom_pos, primvecs, convvecs


def lattice_points(a, b, c, counts, origin=None):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    n_a, n_b, n_c = counts
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.array(origin)

    coords = []
    for i, j, k in product(range(n_a), range(n_b), range(n_c)):
        pt = origin + i * a + j * b + k * c
        coords.append(pt)
    return np.array(coords).reshape(n_a, n_b, n_c, 3)
