#!/usr/bin/env bash

# BASH program to execute a a single pypresso process for given parameter p value and process number proc_iter
# p and proc_iter must be parsed
# p can be: boxsize L, number of anionic (cationic) groups N_an (N_cat), salinity, etc. 


# execute by: $ ./microgel_exe.sh <p> <proc_iter>

LOGFILE="log.log"
plabel="alpha_n" # etiqueta para el parámetro
NPROCS=6 # Número de procesos MPI (puede cambiarse según necesidad)
# Check if mpiexec exists
if ! command -v mpiexec &> /dev/null; then
    echo "Error: mpiexec not found. Please install MPI." >&2
    exit 1
fi

PY_SCRIPT="/home/tobias/trabajo/geles/microgel_simu/main.py"
if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: Python script $PY_SCRIPT not found." >&2
    exit 1
fi

# Lista de valores de j
for j in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "$plabel = $j"

    dir="${plabel}_$j"
    if [ -d "$dir" ]; then
        cd "$dir"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to change to directory $dir" >&2
            exit 1
        fi
    else
        mkdir "$dir" && cd "$dir"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create or change to directory $dir" >&2
            exit 1
        fi
    fi

    mpiexec -n 6 /home/tobias/trabajo/espresso_md/espresso/build/pypresso "$PY_SCRIPT" "$j" >> "../$LOGFILE" 2>&1

    cd ..  # Volver al directorio base para la siguiente iteración
done
