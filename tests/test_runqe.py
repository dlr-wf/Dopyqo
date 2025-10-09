import os
import dopyqo

input_file = os.path.join("qe_files", "LiH.scf.in")
cwd_before = os.getcwd()
print("Running QE...")
dopyqo.runQE(input_file=input_file, num_cpus=1, nk=1, nb=1, nt=1)
cwd_after = os.getcwd()
assert cwd_before == cwd_after
print("Test passed!")
