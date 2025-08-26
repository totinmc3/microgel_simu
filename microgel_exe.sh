#!/usr/bin/env bash

# BASH program to execute a a single pypresso process for given parameter p value and process number proc_iter
# p and proc_iter must be parsed
# p can be: boxsize L, number of anionic (cationic) groups N_an (N_cat), salinity, etc. 


# execute by: $ ./microgel_exe.sh <p> <proc_iter>

LOGFILE="log.log"
plabel="alpha_n" # etiqueta para el parámetro

# Lista de valores de j
for j in $(seq 0.0 0.1 0.9); do
    echo "$plabel = $j"

    dir="${plabel}_$j"
    if [ -d "$dir" ]; then
        cd "$dir"
    else
        mkdir "$dir" && cd "$dir"
    fi

    mpiexec -n 6 /home/tobias/trabajo/espresso_md/espresso/build/pypresso /home/tobias/trabajo/geles/microgel_simu/main.py "$j"

    cd ..  # Volver al directorio base para la siguiente iteración
done
