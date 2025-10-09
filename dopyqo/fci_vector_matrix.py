import numpy as np
from qiskit.quantum_info import Statevector
from pyscf.fci import FCIvector


def largest_binary_with_same_number_of_ones(binary_string: str) -> bytes:
    count_ones = binary_string.count("1")
    return "1" * count_ones + "0" * (len(binary_string) - count_ones)


def closest_larger_binary_with_same_number_of_ones(binary_string: str) -> bytes | None:
    # Count the number of 1's in the given binary string
    count_ones = binary_string.count("1")

    # Find the next larger integer with the same number of 1's
    next_integer = int(binary_string, 2) + 1
    while bin(next_integer).count("1") != count_ones:
        next_integer += 1

    if next_integer > int(largest_binary_with_same_number_of_ones(binary_string), base=2):
        return None

    new_binary_string = bin(next_integer)[2:]

    if len(new_binary_string) < len(binary_string):
        n_missing_zeros = len(binary_string) - len(new_binary_string)
        new_binary_string = f"{'0'*n_missing_zeros}{new_binary_string}"

    assert (l1 := len(new_binary_string)) == (l2 := len(binary_string)), (
        f"New binary string {new_binary_string} (length: {l1}) has not the same length "
        + f"as the original binary string binary_string {binary_string} (length: {l2}). "
        + "This should not be possible! Please, contact a developer!"
    )

    # Convert the next integer to binary and return it
    return new_binary_string


def has_same_spin(binary_string: str, binary_string_reference: str) -> bool:
    count_ones = binary_string_reference.count("1")
    ones_correct = binary_string.count("1") == count_ones
    if not ones_correct:
        return False

    count_spin_up = binary_string_reference[::2].count("1")
    count_spin_dw = binary_string_reference[1::2].count("1")
    spin_up_correct = binary_string[::2].count("1") == count_spin_up
    spin_dw_correct = binary_string[1::2].count("1") == count_spin_dw

    return ones_correct and spin_up_correct and spin_dw_correct


def closest_larger_binary_with_same_spin(binary_string: str) -> bytes | None:
    # Find the next larger integer with the same spin
    next_integer = int(binary_string, 2) + 1
    while not has_same_spin(bin(next_integer)[2:], binary_string) and next_integer < int(
        largest_binary_with_same_number_of_ones(binary_string), base=2
    ):
        next_integer += 1

    if next_integer > int(largest_binary_with_same_number_of_ones(binary_string), base=2):
        return None

    new_binary_string = bin(next_integer)[2:]

    if len(new_binary_string) < len(binary_string):
        n_missing_zeros = len(binary_string) - len(new_binary_string)
        new_binary_string = f"{'0'*n_missing_zeros}{new_binary_string}"

    assert (l1 := len(new_binary_string)) == (l2 := len(binary_string)), (
        f"New binary string {new_binary_string} (length: {l1}) has not the same length "
        + f"as the original binary string binary_string {binary_string} (length: {l2}). "
        + "This should not be possible! Please, contact a developer!"
    )

    # Convert the next integer to binary and return it
    return new_binary_string


def statevector_from_fci(fci_vector: FCIvector, nelec: tuple[int, int], norb: int) -> Statevector:
    base_str_up = "0" * (norb - nelec[0]) + "1" * nelec[0]
    base_str_dw = "0" * (norb - nelec[1]) + "1" * nelec[1]

    # to int
    base_int_up = int(base_str_up, base=2)
    base_int_dw = int(base_str_dw, base=2)

    # to binary with length of norb
    # bin(base_int_up)[2:].zfill(norb)

    strs_up = [base_str_up]
    while (val := closest_larger_binary_with_same_number_of_ones(strs_up[-1])) is not None:
        # print(f"Appending {val}")
        strs_up.append(val)

    strs_dw = [base_str_dw]
    while (val := closest_larger_binary_with_same_number_of_ones(strs_dw[-1])) is not None:
        # print(f"Appending {val}")
        strs_dw.append(val)

    state_vec = np.zeros((2 ** (norb * 2),))
    for i, str_up in enumerate(strs_up):
        for j, str_dw in enumerate(strs_dw):
            idx = int(str_up + str_dw, base=2)
            state_vec[idx] = fci_vector[i, j]

    return Statevector(state_vec, dims=state_vec.shape[0])

    # strs_up = [base_str_up]
    # strs_up = strs_up + [val for val in iter(lambda: closest_larger_binary_with_same_number_of_ones(strs_up[-1]), None)]


def statevector_from_civector_restricted(ffsim_state: np.ndarray, nelec: tuple[int, int], norb: int) -> np.ndarray:
    base_str = "0" * (norb * 2 - np.sum(nelec)) + "1" * np.sum(nelec)
    # print(f"base_str: {base_str}")

    strs = [base_str]
    while (val := closest_larger_binary_with_same_spin(strs[-1])) is not None:
        # print(f"Appending {val}")
        strs.append(val)

    if len(strs) < len(ffsim_state):
        raise ValueError(
            f"The provided ffsim state describes more states than expected. {len(strs)} spin-restricted states found but ffsim state has {len(ffsim_state)}. Maybe you wanted to use statevector_from_ffsim_state_general?"
        )
    elif len(strs) > len(ffsim_state):
        raise ValueError(
            f"The provided ffsim state describes less states than expected. {len(strs)} spin-restricted states found but ffsim state has {len(ffsim_state)}."
        )

    state_vec = np.zeros((2 ** (norb * 2),))
    for i, str in enumerate(strs):
        idx = int(str, base=2)
        state_vec[idx] = ffsim_state[i]

    return Statevector(state_vec, dims=state_vec.shape[0])


def statevector_from_civector_general(ffsim_state: np.ndarray, nelec: tuple[int, int], norb: int) -> np.ndarray:
    base_str = "0" * (norb * 2 - np.sum(nelec)) + "1" * np.sum(nelec)
    # print(f"base_str: {base_str}")

    strs = [base_str]
    while (val := closest_larger_binary_with_same_number_of_ones(strs[-1])) is not None:
        # print(f"Appending {val}")
        strs.append(val)

    if len(strs) < len(ffsim_state):
        raise ValueError(
            f"The provided ffsim state describes more states than expected. {len(strs)} spin-restricted states found but ffsim state has {len(ffsim_state)}."
        )
    elif len(strs) > len(ffsim_state):
        raise ValueError(
            f"The provided ffsim state describes less states than expected. {len(strs)} spin-restricted states found but ffsim state has {len(ffsim_state)}. Maybe you wanted to use statevector_from_ffsim_state_restricted?"
        )

    state_vec = np.zeros((2 ** (norb * 2),))
    for i, str in enumerate(strs):
        idx = int(str, base=2)
        state_vec[idx] = ffsim_state[i]

    return Statevector(state_vec, dims=state_vec.shape[0])
