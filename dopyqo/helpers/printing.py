import re
from collections.abc import Iterable
import dopyqo
from dopyqo.colors import *


def active_space_string(
    occupations: Iterable[float],
    indices_active_space: Iterable[int],
    occs_per_line: int = 8,
    color_active_space: str = SOFT_RED,
    color_rest: str = MEDIUM_GRAY,
    separator: str = " ",
) -> str:
    """Generate string of occupations where an active space is marked

    Args:
        occupations (Iterable[float]): Iterable holding the occupations
        indices_active_space (Iterable[int]): Iterable holding the indices of the active space
        occ_per_line (int): Number of occupations listed in one line in the output string. 8 is used in a QE output file. Defaults to 8.
        color_active_space (str): Color of occupations in the active space
        color_rest (str): Color of all occupations not in the active space
        separator (str): Separator used to enclose the active space
    Returns:
        str: String of occupations where the active space is marked
    """
    occ_str = " ".join(
        [
            (f"|{val}" if i == indices_active_space[0] else f"{val}|" if i == indices_active_space[-1] else f"{val}")
            for i, val in enumerate(occupations)
        ]
    )
    occ_str = re.sub(r"\s*\|\s*", "|", occ_str)  # remove spaces around every |
    occ_str_split = re.findall(r"\|\d|\d", occ_str)  # create list by splitting at " ", and make "a|b" into "a" and "|b"
    # Now split string into lines containing group_size number of elements
    # Create sublists containing group_size number of elements
    occ_str_grouped = [occ_str_split[i : i + occs_per_line] for i in range(0, len(occ_str_split), occs_per_line)]
    # Join sublists with empty spaces
    occ_str_grouped = [(" " * len(separator)).join(group) for group in occ_str_grouped]
    # Remove spaces around every |
    occ_str_grouped = [re.sub(r"\s*\|\s*", "|", x) for x in occ_str_grouped]
    # Indent every sublist/line by an empty space, do not indent if line starts with first |, if line starts with second | append this | to line before
    # We indent so that we can place the first | at the start of the line if line starts with the first |
    occ_str_split = []
    act_spc_start_found = False
    for line in occ_str_grouped:
        if line.startswith("|") and not act_spc_start_found:  # the | here signals the start of the active space
            occ_str_split.append(line)
        elif line.startswith("|") and act_spc_start_found:  # the | here signals the end of the active space
            occ_str_split[-1] = occ_str_split[-1] + "|"  # add | to end of previous line
            occ_str_split.append(len(separator) * " " + line[1:])  # add indented line without starting |
        else:
            occ_str_split.append(len(separator) * " " + line)  # add indented line
        if "|" in line and not act_spc_start_found:  # | in line but not at the start
            act_spc_start_found = True
    # Add color to active space and rest
    occ_str_split_w_color = []
    first_pipe_found = False
    for line in occ_str_split:  # Add color to active space
        parts = line.split("|")  # there can be at most two | in one line, yielding 3 parts
        if len(parts) == 3:  # both | are in the line
            # line = parts[0] + f"{color_active_space}|" + parts[1] + f"|{color_rest}" + parts[2] # with |
            # line = parts[0] + f"{color_active_space} " + parts[1] + f" {color_rest}" + parts[2]  # without |
            line = parts[0] + f"{color_active_space}{separator}" + parts[1] + f"{separator}{color_rest}" + parts[2]
        elif len(parts) == 2 and not first_pipe_found:  # first | is in the line
            first_pipe_found = True
            # line = parts[0] + f"{color_active_space}|" + parts[1] # with |
            # line = parts[0] + f"{SOFT_RED} " + parts[1]  # without |
            line = parts[0] + f"{color_active_space}{separator}" + parts[1]
        elif len(parts) == 2 and first_pipe_found:  # second | is in the line
            # line = parts[0] + f"|{color_rest}" + parts[1] # with |
            # line = parts[0] + f" {color_rest}" + parts[1]  # without |
            line = parts[0] + f"{separator}{color_rest}" + parts[1]
        else:  # no | is in the line
            line = line
        occ_str_split_w_color.append(line)
    occ_str = color_rest + "\n".join(occ_str_split_w_color) + RESET_COLOR

    return occ_str


def print_block(text: str, width: int = 60, color: str = NO_COLOR, flush: bool = False):
    text_max_len = max([len(x) for x in text.split("\n")])
    text_lines = len(text.split("\n"))
    if width < text_max_len + 2:
        width = text_max_len + 2
    prefix = color
    postfix = RESET_COLOR if color != NO_COLOR else ""
    print(prefix + "\t+" + "-" * width + "+")
    for line in text.split("\n"):
        print("\t|" + (width - len(line)) // 2 * " " + line + (width - len(line) - (width - len(line)) // 2) * " " + "|")
    print("\t+" + "-" * width + "+" + postfix, flush=flush)


def print_banner(banner, flush: bool = False):
    second_title = f"\nMany-body analysis on top of Quantum ESPRESSO calculations\nVersion: {dopyqo.__version__}"
    print_block("\n".join([banner.strip("\n"), second_title]), color=BRIGHT_CYAN, flush=flush)
