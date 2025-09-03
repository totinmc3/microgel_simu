#!/usr/bin/env bash

# BASH program to execute a a single pypresso process for given parameter p value and process number proc_iter
# p and proc_iter must be parsed
# p can be: boxsize L, number of anionic (cationic) groups N_an (N_cat), salinity, etc. 


# execute by: $ ./microgel_exe.sh <p> <proc_iter>


mpiexec -n 6 /home/tobias/trabajo/espresso_md/espresso/build/pypresso /home/tobias/trabajo/geles/microgel_simu/main.py ALPHA BOX NBEADS
exit
