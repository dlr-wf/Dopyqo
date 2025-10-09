from enum import Enum

BOHR_TO_METER = 5.29177210903e-11  # Bohr radius in meter
ANGSTROM_TO_METER = 1e-10  # Angstrom in meter
BOHR_TO_ANGSTROM = BOHR_TO_METER / ANGSTROM_TO_METER  # Conversion from Bohr radius to Angstrom


class Unit(Enum):
    BOHR = HARTREE = 1
    ANGSTROM = 2
    METER = 3
    ALAT = 4
    CRYSTAL = 5


ALLOWED_UNITS = [x.name.lower() for x in Unit]
ALLOWED_UNITS_LATTICE = [x.name.lower() for x in Unit if x.name.lower() != "alat" and x.name.lower() != "crystal"]
