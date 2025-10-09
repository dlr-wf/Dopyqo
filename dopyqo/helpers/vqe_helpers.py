from enum import Enum


class ExcitationPools(Enum):
    SINGLES = 1
    DOUBLES = 2
    SINGLES_DOUBLES = 3


class VQEOptimizers(Enum):
    L_BFGS_B = 1
    COBYLA = 2
    ExcitationSolve = 3


class AdaptSelectionCriterion(Enum):
    """Types of selection criterions used in ADAPT-VQE"""

    GRADIENT = 1
    """Gradient-based selection as proposed by Grimsley et al. 2018 (https://doi.org/10.1038/s41467-019-10988-2),
    by estimating the operator impact by calculating its energy gradient at parameter value 0.
    """
    ENERGY = 2
    """Energy-based selection as proposed by Jäger et al. 2025 (https://doi.org/10.48550/arXiv.2409.05939),
    by estimating the operator impact by calculating its maximum energy impact at its optimal parameter value.
    """
