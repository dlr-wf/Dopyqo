import numpy as np
import h5py
import xmltodict
import dopyqo


bohr_to_meter = 5.29177210903e-11  # Bohr radius in meter
angstrom_to_meter = 1e-10  # Angstrom in meter
bohr_to_angstrom = bohr_to_meter / angstrom_to_meter  # Conversion from Bohr radius to Angstrom


class Wfc:
    def __init__(
        self,
        ik,
        xk,
        ispin,
        gamma_only,
        scalef,
        ngw,
        igwx,
        npol,
        nbnd,
        b1,
        b2,
        b3,
        mill,
        evc,
        output_xml,
    ):
        self.ik = ik
        self.xk = xk
        self.ispin = ispin
        self.gamma_only = gamma_only
        self.scalef = scalef
        self.ngw = ngw
        self.igwx = igwx
        self.npol = npol
        self.nbnd = nbnd
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.mill = mill
        self.evc = evc

        self.b = np.array([b1, b2, b3])  # b1 in first row, b2 in second row, b3 in third row.
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
            xml_dict = xmltodict.parse(file.read())
        self.spin = 1
        if "nbnd_up" in xml_dict["qes:espresso"]["output"]["band_structure"] and "nbnd_dw" in xml_dict["qes:espresso"]["output"]["band_structure"]:
            self.spin = 2
        atoms_dict = xml_dict["qes:espresso"]["input"]["atomic_structure"]["atomic_positions"]["atom"]
        atoms = []
        for atom in atoms_dict:
            atoms.append(
                {
                    "element": atom["@name"],
                    "position_bohr": np.fromstring(atom["#text"], sep=" ", dtype=np.float32),
                    "position_hartree": np.fromstring(atom["#text"], sep=" ", dtype=np.float32),
                    "position_meter": np.fromstring(atom["#text"], sep=" ", dtype=np.float32) * bohr_to_meter,
                    "position_angstrom": np.fromstring(atom["#text"], sep=" ", dtype=np.float32) * bohr_to_angstrom,
                    "atomic_number": dopyqo.elements_to_atomic_number[atom["@name"]],
                }
            )
        self.atoms = atoms

        # Calculate Lattice vectors in bohr/hartree units from reciprocal lattice vectors
        self.cell_volume_reciprocal = self.b1.dot(np.cross(self.b2, self.b3))
        self.a1 = 2 * np.pi * 1 / self.cell_volume_reciprocal * np.cross(self.b2, self.b3)
        self.a2 = 2 * np.pi * 1 / self.cell_volume_reciprocal * np.cross(self.b3, self.b1)
        self.a3 = 2 * np.pi * 1 / self.cell_volume_reciprocal * np.cross(self.b1, self.b2)
        self.cell_volume = self.a1.dot(np.cross(self.a2, self.a3))

        # Read lattice vectors from xml file
        cell = xml_dict["qes:espresso"]["input"]["atomic_structure"]["cell"]
        a1 = np.fromstring(cell["a1"], sep=" ", dtype=np.float32)
        a2 = np.fromstring(cell["a2"], sep=" ", dtype=np.float32)
        a3 = np.fromstring(cell["a3"], sep=" ", dtype=np.float32)

        assert np.allclose(
            np.array([self.a1, self.a2, self.a3]), np.array([a1, a2, a3])
        ), f"Lattice vectors in {output_xml} do not match to given reciprocal lattice vectors b1, b2, b3"

        ks_energies_dict = xml_dict["qes:espresso"]["output"]["band_structure"]["ks_energies"]
        self.ks_energies = np.array([float(x) for x in ks_energies_dict["eigenvalues"]["#text"].split()])
        self.occupations = np.array([float(x) for x in ks_energies_dict["occupations"]["#text"].split()])
        self.occupations_binary = self.occupations.copy()
        self.occupations_binary[np.abs(self.occupations_binary - 1.0) < np.abs(self.occupations_binary)] = 1.0
        self.occupations_binary[np.abs(self.occupations_binary - 1.0) > np.abs(self.occupations_binary)] = 0.0
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

        mill_to_c = {}
        for i, mill_idx in enumerate(self.mill):
            mill_to_c[tuple(mill_idx.tolist())] = self.evc[:, i]
        self.mill_to_c = mill_to_c

    @classmethod
    def from_file(cls, file, output_xml):
        if file.endswith("dat"):
            return Wfc.from_dat_file(file, output_xml)
        elif file.endswith("hdf5"):
            return Wfc.from_hdf5_file(file, output_xml)
        else:
            raise NotImplementedError(f"File extension {file.split('.')[-1]} not supported!")

    @classmethod
    def from_hdf5_file(cls, hdf5_file, output_xml):
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
        )

    @classmethod
    def from_dat_file(cls, dat_file, output_xml):
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
        )

    def get_overlaps(self):
        overlaps = np.einsum("ij, kj -> ik", self.evc.conj(), self.evc)

        return overlaps

    def check_norm(self):
        assert np.allclose(self.get_overlaps(), np.identity(self.evc.shape[0])), "Overlap matrix is not diagonal (Wavefunctions are not orthonormal)!"

        print("Overlap matrix is diagonal (Wavefunctions are orthonormal)!")

    def get_orbitals_by_index(self, indices: list | np.ndarray):
        occupations = self.occupations_binary[indices]
        c_ip_orbitals = self.evc[indices]

        return occupations, c_ip_orbitals
