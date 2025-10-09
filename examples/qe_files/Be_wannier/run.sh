#!/bin/bash

mpirun -n 1 pw.x -i Be.scf.in > Be.scf.out
#mpirun -n 1 pw.x -i Be.nscf.in > Be.nscf.out
wannier90.x -pp Be
mpirun -n 1 pw2wannier90.x -i Be.pw2wan > Be.pw2wan.out
wannier90.x Be
