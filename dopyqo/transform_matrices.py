import numpy as np


def transform_one_body_matrix(one_body_matrix: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    assert one_body_matrix.ndim == 2
    assert transform_matrix.ndim == 2
    assert one_body_matrix.shape == (one_body_matrix.shape[0],) * 2
    assert transform_matrix.shape == (transform_matrix.shape[0],) * 2

    # <p|V|q> = \sum_{i,j} <i|p>* <i|V|j> <j|q>
    # where <i|p> is transform_matrix
    # and <i|V|j> is one_body_matrix
    res = np.einsum("ip, ij, jq -> pq", transform_matrix.conj(), one_body_matrix, transform_matrix)
    return res


def transform_two_body_matrix(two_body_matrix: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    assert two_body_matrix.ndim == 4
    assert transform_matrix.ndim == 2
    assert two_body_matrix.shape == (two_body_matrix.shape[0],) * 4
    assert transform_matrix.shape == (transform_matrix.shape[0],) * 2

    # <pq|V|rs> = \sum_{i,j,k,l} <i|p>* <j|q>* <ij|V|kl> <k|r> <l|s>
    # where <i|p> is transform_matrix
    # and <ij|V|kl> is two_body_matrix
    res = np.einsum(
        "ip, jq, ijkl, kr, ls -> pqrs",
        transform_matrix.conj(),
        transform_matrix.conj(),
        two_body_matrix,
        transform_matrix,
        transform_matrix,
    )
    return res


def to_density_matrix(active_electrons, active_orbitals):
    diag_density = []
    electrons = active_electrons
    orbitals = active_orbitals
    while electrons > 0 and len(diag_density) < orbitals:
        diag_density.append(2)
        electrons -= 2

    diag_density.extend([0] * (orbitals - len(diag_density)))
    density_mat = np.diag(diag_density)

    return density_mat
