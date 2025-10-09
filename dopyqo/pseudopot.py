import logging
import os
import xmltodict
from dataclasses import dataclass
import numpy as np
import dopyqo


@dataclass
class BetaProjector:
    r"""Projector r \beta^i(r) (note the factor r)

    Args:
        idx (int): index i of projector starting at 1. Written as PP_BETA.idx in PP file
        angular_momentum (int): Angular momentum value of projector
        values (np.ndarray): Values of r \beta^i(r) for different values of r (note the factor r)
    """

    idx: int
    angular_momentum: int
    values: np.ndarray


@dataclass
class rGrid:
    r"""A real-space grid where a pseudopotential is defined on

    Args:
        r (np.ndarray): radial grid points
        rab (np.ndarray): factor required for discrete integration: \int f(r) dr = \sum_i f_i rab_i.
    """

    def __init__(self, r: np.ndarray, rab: np.ndarray):
        assert (lr := len(r)) == (
            lrab := len(rab)
        ), f"Real-space grid invalid. Number radial grid point ({lr}) is not equal to number of rab factors ({lrab})!"

        self.r = r
        self.rab = rab


class Pseudopot:
    r"""Pseudopotential (PP) class holding information from the PP file in RYDBERG atomic units

    Args:
        pp_file (str): Path to the pseudopotential file
    """

    def __init__(self, pp_file: str):
        with open(pp_file, "r", encoding="utf-8") as file:
            xml_dict = xmltodict.parse(file.read())["UPF"]

        self.version = xml_dict["@version"]
        self.pp_info = xml_dict["PP_INFO"]
        self.pp_input = self.pp_info["PP_INPUTFILE"]
        # ref_inp_columns = [
        #     "#",
        #     "atsym",
        #     "z",
        #     "nc",
        #     "nv",
        #     "iexc",
        #     "psfile",
        # ]
        # assert (
        #     pp_input.split("\n")[1].split() == ref_inp_columns
        # ), "Cannot read information about atom in pseudopotential!"
        # assert len(pp_input.split("\n")[2].split()) == len(ref_inp_columns) - 1
        # atsym, z, nc, nv, iexc, psfile = pp_input.split("\n")[2].split()

        self.pp_header = xml_dict["PP_HEADER"]
        self.element = self.pp_header["@element"].strip()
        self.atomic_number = dopyqo.elements_to_atomic_number[self.element]
        self.z_valence = float(self.pp_header["@z_valence"])
        self.z_core = self.atomic_number - self.z_valence
        self.n_proj = int(self.pp_header["@number_of_proj"])

        logging.info(f"Pseudopotential for {self.element} with core of {int(self.z_core)} electrons.")

        assert self.pp_header["@pseudo_type"] == "NC", f"Only fully nonlocal NCPPs supported but found {self.pp_header['@pseudo_type']} PP!"
        assert self.pp_header["@is_coulomb"] == "F", "Only non-coulombic potentials supported but PP is coulombic!"

        self.pp_mesh = xml_dict["PP_MESH"]
        self.pp_mesh_r = np.array(list(map(float, self.pp_mesh["PP_R"]["#text"].split())))
        # \int f(r) dr = \sum_i f_i rab_i
        self.pp_mesh_rab = np.array(list(map(float, self.pp_mesh["PP_RAB"]["#text"].split())))
        self.r_grid = rGrid(self.pp_mesh_r, self.pp_mesh_rab)

        # NOTE: The loaded potentials are centered at the origin. To have them atom-centered
        #       use \vec{r} -> \vec{r} - \vec{R}, where \vec{R} is the position of the atom.
        #       The local potential is only dependent on the norm so |r| -> |r-R|

        # local potential (Ry a.u.) sampled on the radial grid
        # is spherical symmetric and does not involve e.g. spherical harmonics,
        # i.e. V_loc(\vec{r}) = V_loc(|r|)
        self.pp_local = np.array(list(map(float, xml_dict["PP_LOCAL"]["#text"].split())))

        # BETA: projector r_i \beta(r_i) (note the factor r)
        # DIJ:  Dij factors of the nonlocal PP in row-major order:
        #       V_{NL} = \sum_{i,j} D_{i,j} |\beta_i><\beta_j|
        # Angular part is described by spherical harmonics
        self.pp_nonlocal = {}
        for key, val in xml_dict["PP_NONLOCAL"].items():
            self.pp_nonlocal[key] = np.array(list(map(float, val["#text"].split())))
        # Save Dij in numpy array using row-major order
        self.pp_nonlocal["PP_DIJ"] = np.array(self.pp_nonlocal["PP_DIJ"]).reshape((int(np.sqrt(len(self.pp_nonlocal["PP_DIJ"]))),) * 2, order="C")
        self.pp_betas = [  # Same as pp_nonlocal["PP_BETA.x"] but with angular momentum information
            BetaProjector(
                idx=int(key.split(".")[-1]),
                angular_momentum=int(val["@angular_momentum"]),
                values=np.array(list(map(float, val["#text"].split()))),
            )
            for key, val in xml_dict["PP_NONLOCAL"].items()
            if "BETA" in key  # and int(key.split(".")[-1]) <= n_proj
        ]

        self.pp_dij = self.pp_nonlocal["PP_DIJ"]

        # Non-local potential V_NL
        # V_NL(r) = \sum_{l,m} \sum_{i,j} \beta_{i,l}(r) Y_{lm}(r) D_{i,j} \beta*_{j,l}(r) Y*_{lm}(r)
        # self.v_nl_r = np.zeros_like(self.pp_mesh_r)
        # for i in range(self.pp_nonlocal["PP_DIJ"].shape[0]):
        #     for j in range(self.pp_nonlocal["PP_DIJ"].shape[1]):
        #         self.v_nl_r += (
        #             self.pp_nonlocal["PP_DIJ"][i, j]
        #             * self.pp_nonlocal[f"PP_BETA.{i+1}"]
        #             / np.array(self.pp_mesh_r)
        #             * self.pp_nonlocal[f"PP_BETA.{j+1}"]
        #             / np.array(self.pp_mesh_r)
        #         )

        self.pp_pswfc = xml_dict["PP_PSWFC"]

        # radial atomic (pseudo-)charge. This is 4\pi r^2 times the true charge.
        self.pp_rhoatom = list(map(float, xml_dict["PP_RHOATOM"]["#text"].split()))


if __name__ == "__main__":

    xml_file = os.path.join("qe_files", "pseudo", "H_ONCV_PBE-1.2.upf")
    pp = Pseudopot(xml_file)
    print(pp.__dict__)
